"""Typed, serializable patches for :func:`crashlab.fork`.

A patch says what to change in a forked run. Every patch is a small Pydantic model with a
``kind`` discriminator, so a list of patches round-trips through JSON / YAML (and a
diagnosis agent can emit them):

==========================  ============================================================
``SystemPromptPatch``       edit the system prompt: replace, append, prepend, insert /
                            replace / delete a line, substitute text
``MessagePatch``            edit one message of the request sent at a chat step
``InjectMessagePatch``      insert an extra message into the request at a chat step
``ToolResultPatch``         "what if the tool had returned X" at a tool step
``ToolDescriptionPatch``    change a tool's description in the requests
``ToolSchemaPatch``         change a tool's JSON-schema parameters in the requests
``ModelParamPatch``         change the model and / or sampling parameters
==========================  ============================================================

Patches are applied at the capture interceptor, to the provider-native request just
before it is sent, so user code is never touched. Request-wide patches (system prompt,
tools, model) apply to every LLM call from the fork step on (or from ``from_step``);
message patches apply at their ``step`` and stay applied to later requests that still
carry the original message, so the conversation stays consistent.

Compose with ``+`` (or :class:`PatchSet`); ``describe()`` gives a one-line summary and
:meth:`PatchSet.render_diff` a unified diff against a recorded run.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

if TYPE_CHECKING:
    from .models import Run


class PatchError(ValueError):
    """A patch cannot be applied (bad target, anchor not found, wrong step kind ...)."""


def _q(value: Any, width: int = 70) -> str:
    """Strings quoted, other values as compact JSON; one line, at most ``width`` chars."""
    s = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    s = " ".join(s.split())
    s = s if len(s) <= width else s[: width - 1] + "…"
    return json.dumps(s, ensure_ascii=False) if isinstance(value, str) else s


def _canon(value: Any) -> str:
    from .capture.normalize import to_jsonable

    return json.dumps(to_jsonable(value), sort_keys=True, ensure_ascii=False, default=str)


# --- provider-native request editing --------------------------------------------------

_SYSTEM_ROLES = ("system", "developer")


def _family(api: str) -> str:
    if api.startswith("anthropic"):
        return "anthropic"
    if api == "openai.responses":
        return "responses"
    return "chat"  # openai.chat.completions, litellm.completion, OpenAI-compatible


def _normalizer(api: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    from .capture import normalize as N

    return {
        "anthropic": N.normalize_anthropic_request,
        "responses": N.normalize_openai_responses_request,
        "chat": N.normalize_openai_chat_request,
    }[_family(api)]


def normalize_request(api: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Provider-native request kwargs -> crashlab's normalized request shape."""
    return _normalizer(api)(copy.deepcopy(kwargs))


def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                out.append(part["text"])
        return "\n\n".join(out)
    return str(content)


class RequestEditor:
    """Edits a deep copy of one provider-native request (``kwargs`` of the SDK call)."""

    def __init__(self, api: str, kwargs: dict[str, Any]) -> None:
        self.api = api
        self.family = _family(api)
        self.kwargs: dict[str, Any] = copy.deepcopy(dict(kwargs))
        self.changed = False

    # messages ----------------------------------------------------------------------
    @property
    def _msg_key(self) -> str:
        return "input" if self.family == "responses" else "messages"

    def native_messages(self) -> list[Any]:
        """The native message list (system messages included for chat), made mutable."""
        msgs = self.kwargs.get(self._msg_key)
        if isinstance(msgs, list):  # already our deep copy: edit in place
            return msgs
        if isinstance(msgs, str):  # Responses: input="..."
            msgs = [{"role": "user", "content": msgs}]
        msgs = list(msgs or [])
        self.kwargs[self._msg_key] = msgs
        return msgs

    def _leading_system(self, msgs: list[Any]) -> int:
        if self.family == "anthropic":
            return 0
        n = 0
        while n < len(msgs) and isinstance(msgs[n], dict) and msgs[n].get("role") in _SYSTEM_ROLES:
            n += 1
        return n

    def normalized_count(self, native: list[Any]) -> int:
        kw = dict(self.kwargs)
        kw[self._msg_key] = native
        return len(normalize_request(self.api, kw).get("messages") or [])

    def native_index(self, index: int, *, insert: bool = False) -> int:
        """Native list position of normalized message ``index`` (``request.messages``)."""
        msgs = self.native_messages()
        total = self.normalized_count(msgs)
        if index < 0:
            index += total + (1 if insert else 0)
        if insert and index == total:
            return len(msgs)
        if not 0 <= index < total:
            raise PatchError(f"message index {index} out of range: the request has {total}")
        for j in range(1, len(msgs) + 1):
            if self.normalized_count(msgs[:j]) > index:
                return j - 1
        raise PatchError(f"cannot locate message {index}")  # pragma: no cover

    # system prompt -------------------------------------------------------------------
    def get_system(self) -> str | None:
        if self.family == "anthropic":
            system = self.kwargs.get("system")
            return None if system is None else _text_of(system)
        msgs = self.kwargs.get(self._msg_key)
        lead: list[str] = []
        if isinstance(msgs, list):
            n = self._leading_system(msgs)
            lead = [_text_of(m.get("content")) for m in msgs[:n]]
        if self.family == "responses" and self.kwargs.get("instructions") is not None:
            lead.insert(0, _text_of(self.kwargs["instructions"]))
        return "\n\n".join(lead) if lead else None

    def set_system(self, text: str) -> None:
        self.changed = True
        if self.family == "anthropic":
            old = self.kwargs.get("system")
            if isinstance(old, list) and old and isinstance(old[0], dict):
                block = {"type": "text", "text": text}
                if "cache_control" in old[-1]:
                    block["cache_control"] = old[-1]["cache_control"]
                self.kwargs["system"] = [block]
            else:
                self.kwargs["system"] = text
            return
        if self.family == "responses":
            msgs = self.kwargs.get("input")
            if isinstance(msgs, list):
                n = self._leading_system(msgs)
                self.kwargs["input"] = msgs[n:]
            self.kwargs["instructions"] = text
            return
        msgs = self.native_messages()
        n = self._leading_system(msgs)
        role = msgs[0].get("role", "system") if n else "system"
        msgs[:n] = [{"role": role, "content": text}]

    # tools ---------------------------------------------------------------------------
    def tool(self, name: str) -> dict[str, Any]:
        for t in self.kwargs.get("tools") or []:
            if not isinstance(t, dict):
                continue
            if self.family == "chat" and isinstance(t.get("function"), dict):
                if t["function"].get("name") == name:
                    return t["function"]
            elif t.get("name") == name:
                return t
        names = [
            (t.get("function") or {}).get("name") or t.get("name")
            for t in self.kwargs.get("tools") or []
            if isinstance(t, dict)
        ]
        raise PatchError(f"tool {name!r} is not in the request (tools: {names})")

    @property
    def schema_key(self) -> str:
        return "input_schema" if self.family == "anthropic" else "parameters"

    def set_param(self, key: str, value: Any) -> None:
        self.changed = True
        self.kwargs[key] = value

    def unset_param(self, key: str) -> None:
        if key in self.kwargs:
            self.changed = True
            del self.kwargs[key]


# --- patch models ---------------------------------------------------------------------


class _Patch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: "request": applies to every chat call from max(fork step, from_step) on;
    #: "step": bound to one step (``step``)
    scope: ClassVar[str] = "request"

    def describe(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def target_step(self) -> int | None:
        """The step this patch is bound to, or ``from_step`` (None = the fork step)."""
        return getattr(self, "step", None) if self.scope == "step" else self.from_step  # type: ignore[attr-defined]

    def active_at(self, idx: int, fork_step: int) -> bool:
        start = self.target_step()
        return idx >= max(fork_step, start if start is not None else fork_step)

    def apply_chat(self, ed: RequestEditor, idx: int, fork_step: int, state: dict) -> bool:
        return False

    def __add__(self, other: Any) -> PatchSet:
        return PatchSet.of(self) + other


class SystemPromptPatch(_Patch):
    """Edit the system prompt (OpenAI system/developer messages, Anthropic ``system``,
    Responses ``instructions``).

    Lines are 1-based. ``insert_line`` takes ``line`` (insert before it; one past the end
    appends) or an ``after`` / ``before`` anchor (a substring of an existing line);
    ``replace_line`` / ``delete_line`` take ``line`` or ``match``; ``substitute`` replaces
    ``old`` with ``text``.
    """

    kind: Literal["system_prompt"] = "system_prompt"
    op: Literal[
        "replace", "append", "prepend", "insert_line", "replace_line", "delete_line", "substitute"
    ] = "append"
    text: str = ""
    line: int | None = None
    after: str | None = None
    before: str | None = None
    match: str | None = None
    old: str | None = None
    from_step: int | None = None

    @model_validator(mode="after")
    def _check(self) -> SystemPromptPatch:
        anchors = [x for x in (self.line, self.after, self.before) if x is not None]
        if self.op == "insert_line" and len(anchors) != 1:
            raise ValueError("insert_line needs exactly one of line, after, before")
        if self.op in ("replace_line", "delete_line") and (self.line is None) == (
            self.match is None
        ):
            raise ValueError(f"{self.op} needs exactly one of line, match")
        if self.op == "substitute" and not self.old:
            raise ValueError("substitute needs old (the text to replace)")
        if self.line is not None and self.line < 1:
            raise ValueError("line numbers start at 1")
        return self

    def _find(self, lines: list[str], needle: str) -> int:
        hits = [i for i, ln in enumerate(lines) if needle in ln]
        if not hits:
            raise PatchError(f"system prompt has no line containing {_q(needle, 50)}")
        if len(hits) > 1:
            raise PatchError(
                f"{_q(needle, 50)} matches {len(hits)} system prompt lines; be more specific"
            )
        return hits[0]

    def edit(self, system: str | None) -> str:
        """Apply to a system prompt string (None = no system prompt)."""
        cur = system or ""
        if self.op == "replace":
            return self.text
        if self.op == "append":
            return f"{cur}\n{self.text}" if cur and not cur.endswith("\n") else cur + self.text
        if self.op == "prepend":
            return f"{self.text}\n{cur}" if cur else self.text
        if self.op == "substitute":
            assert self.old is not None
            if self.old not in cur:
                raise PatchError(f"system prompt does not contain {_q(self.old, 50)}")
            return cur.replace(self.old, self.text)
        lines = cur.split("\n") if cur else []
        if self.op == "insert_line":
            if self.line is not None:
                if self.line > len(lines) + 1:
                    raise PatchError(f"line {self.line}: the system prompt has {len(lines)} lines")
                pos = self.line - 1
            elif self.after is not None:
                pos = self._find(lines, self.after) + 1
            else:
                pos = self._find(lines, self.before or "")
            lines[pos:pos] = self.text.split("\n")
            return "\n".join(lines)
        if self.line is not None:
            if self.line > len(lines):
                raise PatchError(f"line {self.line}: the system prompt has {len(lines)} lines")
            pos = self.line - 1
        else:
            pos = self._find(lines, self.match or "")
        lines[pos : pos + 1] = [] if self.op == "delete_line" else self.text.split("\n")
        return "\n".join(lines)

    def apply_chat(self, ed: RequestEditor, idx: int, fork_step: int, state: dict) -> bool:
        if not self.active_at(idx, fork_step):
            return False
        ed.set_system(self.edit(ed.get_system()))
        return True

    def describe(self) -> str:
        where = {
            "replace": "replace with",
            "append": "append",
            "prepend": "prepend",
            "substitute": f"replace {_q(self.old, 40)} with",
        }.get(self.op)
        if where is None:
            if self.line is not None:
                at = f"line {self.line}"
            elif self.after is not None:
                at = f"after the line with {_q(self.after, 40)}"
            elif self.before is not None:
                at = f"before the line with {_q(self.before, 40)}"
            else:
                at = f"the line with {_q(self.match, 40)}"
            verb = {"insert_line": "insert", "replace_line": "replace", "delete_line": "delete"}
            where = f"{verb[self.op]} {at}" + (":" if self.op != "delete_line" else "")
        text = "" if self.op == "delete_line" else f" {_q(self.text)}"
        frm = f" (from step {self.from_step})" if self.from_step is not None else ""
        return f"system prompt: {where}{text}{frm}"


class MessagePatch(_Patch):
    """Replace the content (and / or role) of message ``index`` of ``request.messages``
    (as shown by ``crashlab show``; system messages are not counted) in the request sent
    at chat step ``step``. Later requests that still carry the original message get the
    same edit."""

    kind: Literal["message"] = "message"
    scope: ClassVar[str] = "step"
    step: int
    index: int
    content: Any = None
    role: str | None = None

    @model_validator(mode="after")
    def _check(self) -> MessagePatch:
        if self.content is None and self.role is None:
            raise ValueError("MessagePatch needs content and / or role")
        return self

    def _edited(self, msg: Any) -> Any:
        msg = dict(msg) if isinstance(msg, dict) else {"role": "user", "content": msg}
        if self.content is not None:
            msg["content"] = self.content
        if self.role is not None:
            msg["role"] = self.role
        return msg

    def apply_chat(self, ed: RequestEditor, idx: int, fork_step: int, state: dict) -> bool:
        msgs = ed.native_messages()
        if idx == self.step:
            pos = ed.native_index(self.index)
            state.update(pos=pos, original=_canon(msgs[pos]))
        elif idx > self.step and "pos" in state:
            pos = state["pos"]
            if pos >= len(msgs) or _canon(msgs[pos]) != state["original"]:
                return False
        else:
            return False
        msgs[pos] = self._edited(msgs[pos])
        ed.changed = True
        return True

    def describe(self) -> str:
        what = []
        if self.role is not None:
            what.append(f"role -> {self.role}")
        if self.content is not None:
            what.append(f"content -> {_q(self.content)}")
        return f"message {self.index} at step {self.step}: " + ", ".join(what)


class InjectMessagePatch(_Patch):
    """Insert a message into the request at chat step ``step``: before ``request.messages``
    index ``index`` (default: at the end). Later requests get it at the same position."""

    kind: Literal["inject_message"] = "inject_message"
    scope: ClassVar[str] = "step"
    step: int
    role: str = "user"
    content: Any
    index: int | None = None

    def apply_chat(self, ed: RequestEditor, idx: int, fork_step: int, state: dict) -> bool:
        msgs = ed.native_messages()
        if idx == self.step:
            pos = len(msgs) if self.index is None else ed.native_index(self.index, insert=True)
            state["pos"] = pos
        elif idx > self.step and "pos" in state and state["pos"] <= len(msgs):
            pos = state["pos"]
        else:
            return False
        msgs.insert(pos, {"role": self.role, "content": self.content})
        ed.changed = True
        return True

    def describe(self) -> str:
        at = "at the end" if self.index is None else f"before message {self.index}"
        return f"inject {self.role} message at step {self.step} ({at}): {_q(self.content)}"


class ToolResultPatch(_Patch):
    """At tool step ``step``, do not run the tool: return ``result`` instead (or raise
    ``error`` as the tool's exception). ``tool`` optionally guards the tool name."""

    kind: Literal["tool_result"] = "tool_result"
    scope: ClassVar[str] = "step"
    step: int
    result: Any = None
    error: str | None = None
    tool: str | None = None

    def describe(self) -> str:
        name = f" ({self.tool})" if self.tool else ""
        if self.error is not None:
            return f"tool result at step {self.step}{name}: raise {_q(self.error)}"
        return f"tool result at step {self.step}{name} -> {_q(self.result)}"


class ToolDescriptionPatch(_Patch):
    """Replace (or append to) the description of tool ``tool`` in every request."""

    kind: Literal["tool_description"] = "tool_description"
    tool: str
    description: str
    op: Literal["replace", "append"] = "replace"
    from_step: int | None = None

    def apply_chat(self, ed: RequestEditor, idx: int, fork_step: int, state: dict) -> bool:
        if not self.active_at(idx, fork_step):
            return False
        t = ed.tool(self.tool)
        old = t.get("description") or ""
        t["description"] = (
            f"{old} {self.description}".strip() if self.op == "append" else self.description
        )
        ed.changed = True
        return True

    def describe(self) -> str:
        verb = "append to description" if self.op == "append" else "description ->"
        return f"tool {self.tool}: {verb} {_q(self.description)}"


class ToolSchemaPatch(_Patch):
    """Replace the JSON-schema parameters of tool ``tool`` in every request (or merge
    ``properties`` into the existing schema with ``merge=True``)."""

    kind: Literal["tool_schema"] = "tool_schema"
    tool: str
    parameters: dict[str, Any]
    merge: bool = False
    from_step: int | None = None

    def apply_chat(self, ed: RequestEditor, idx: int, fork_step: int, state: dict) -> bool:
        if not self.active_at(idx, fork_step):
            return False
        t = ed.tool(self.tool)
        key = ed.schema_key
        if self.merge and isinstance(t.get(key), dict):
            new = copy.deepcopy(t[key])
            for k, v in self.parameters.items():
                if isinstance(v, dict) and isinstance(new.get(k), dict):
                    new[k] = {**new[k], **v}
                else:
                    new[k] = v
            t[key] = new
        else:
            t[key] = copy.deepcopy(self.parameters)
        ed.changed = True
        return True

    def describe(self) -> str:
        verb = "merge into parameters" if self.merge else "parameters ->"
        return f"tool {self.tool}: {verb} {_q(self.parameters)}"


class ModelParamPatch(_Patch):
    """Change the model and / or request parameters (temperature, max_tokens, top_p,
    seed, reasoning_effort ...); ``unset`` removes parameters."""

    kind: Literal["model_params"] = "model_params"
    model: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    unset: list[str] = Field(default_factory=list)
    from_step: int | None = None

    @model_validator(mode="after")
    def _check(self) -> ModelParamPatch:
        if self.model is None and not self.params and not self.unset:
            raise ValueError("ModelParamPatch needs model, params or unset")
        if "model" in self.params or "messages" in self.params or "tools" in self.params:
            raise ValueError("use model= for the model; messages / tools have their own patches")
        return self

    def apply_chat(self, ed: RequestEditor, idx: int, fork_step: int, state: dict) -> bool:
        if not self.active_at(idx, fork_step):
            return False
        if self.model is not None:
            ed.set_param("model", self.model)
        for k, v in self.params.items():
            ed.set_param(k, v)
        for k in self.unset:
            ed.unset_param(k)
        return True

    def describe(self) -> str:
        parts = [f"model -> {self.model}"] if self.model else []
        parts += [f"{k}={json.dumps(v, default=str)}" for k, v in self.params.items()]
        parts += [f"unset {k}" for k in self.unset]
        return "model params: " + ", ".join(parts)


AnyPatch = Annotated[
    Union[  # noqa: UP007
        SystemPromptPatch,
        MessagePatch,
        InjectMessagePatch,
        ToolResultPatch,
        ToolDescriptionPatch,
        ToolSchemaPatch,
        ModelParamPatch,
    ],
    Field(discriminator="kind"),
]
_ANY = TypeAdapter(AnyPatch)


class PatchSet(BaseModel):
    """An ordered, composable list of patches (applied in order)."""

    model_config = ConfigDict(extra="forbid")

    patches: list[AnyPatch] = Field(default_factory=list)

    @classmethod
    def of(cls, *items: Any) -> PatchSet:
        out = cls()
        for item in items:
            out = out + item
        return out

    def __add__(self, other: Any) -> PatchSet:
        return PatchSet(patches=[*self.patches, *as_patchset(other).patches])

    def __radd__(self, other: Any) -> PatchSet:
        return as_patchset(other) + self

    def __iter__(self) -> Iterator[Any]:  # type: ignore[override]
        return iter(self.patches)

    def __len__(self) -> int:
        return len(self.patches)

    def __bool__(self) -> bool:
        return bool(self.patches)

    # info ----------------------------------------------------------------------------
    def describe(self) -> str:
        if not self.patches:
            return "no patch"
        return "; ".join(p.describe() for p in self.patches)

    def describe_lines(self) -> list[str]:
        return [p.describe() for p in self.patches]

    def bound_steps(self) -> list[int]:
        """Steps that step-bound patches target (message / inject / tool result)."""
        return [p.step for p in self.patches if p.scope == "step"]

    def tool_result_for(self, idx: int) -> ToolResultPatch | None:
        for p in self.patches:
            if isinstance(p, ToolResultPatch) and p.step == idx:
                return p
        return None

    @property
    def edits_requests(self) -> bool:
        return any(not isinstance(p, ToolResultPatch) for p in self.patches)

    # application ---------------------------------------------------------------------
    def apply_chat(
        self,
        api: str,
        idx: int,
        kwargs: dict[str, Any],
        *,
        fork_step: int = 0,
        state: dict[int, dict] | None = None,
    ) -> tuple[dict[str, Any], list[int]]:
        """Patch one chat request. Returns ``(kwargs, indexes of the patches applied)``.

        ``state`` carries per-patch memory across the calls of one forked run (where a
        message patch was applied, so later requests get the same edit).
        """
        state = {} if state is None else state
        ed = RequestEditor(api, kwargs)
        applied = [
            i
            for i, p in enumerate(self.patches)
            if p.apply_chat(ed, idx, fork_step, state.setdefault(i, {}))
        ]
        return (ed.kwargs if applied else kwargs), applied

    # serialization -------------------------------------------------------------------
    def to_json(self, **kw: Any) -> str:
        """JSON without unset optional fields (what patch files look like)."""
        kw.setdefault("exclude_none", True)
        return self.model_dump_json(**kw)

    def render_diff(self, run: Run | None = None, *, fork_step: int = 0) -> str:
        """Human-readable summary plus a unified diff against ``run``'s recorded requests
        (the first recorded LLM call each patch would change) and tool results."""
        lines = [f"{i}. {d}" for i, d in enumerate(self.describe_lines(), 1)] or ["no patch"]
        if run is None:
            return "\n".join(lines)
        preview = preview_patches(self, run, fork_step=fork_step)
        for title, diff in preview:
            lines += ["", f"--- {title}", diff]
        return "\n".join(lines).rstrip()


def preview_patches(patches: PatchSet, run: Run, *, fork_step: int = 0) -> list[tuple[str, str]]:
    """Dry-run ``patches`` against ``run``'s recorded steps from ``fork_step`` on.

    Returns ``[(title, unified diff)]`` and raises :class:`PatchError` when a patch does
    not apply (anchor not found, unknown tool, a tool-result patch aimed at a chat step).
    """
    from .replay.divergence import diff_values

    out: list[tuple[str, str]] = []
    steps = {s.idx: s for s in run.steps}
    for p in patches.patches:
        if p.scope == "step":
            src = steps.get(p.step)
            want = "execute_tool" if isinstance(p, ToolResultPatch) else "chat"
            if src is not None and src.operation != want:
                kind = "an LLM call" if want == "chat" else "a tool call"
                raise PatchError(
                    f"{p.describe()}: step {p.step} of run {run.run_id} is "
                    f"{src.operation} {src.name!r}, not {kind}"
                )
            if isinstance(p, ToolResultPatch) and src is not None:
                if p.tool and src.name != p.tool:
                    raise PatchError(f"{p.describe()}: step {p.step} calls {src.name!r}")
                before = {"error": src.error.message} if src.error else {"result": src.result}
                after = {"error": p.error} if p.error is not None else {"result": p.result}
                _, diff = diff_values(before, after, fromfile=f"recorded step {p.step}",
                                      tofile=f"patched step {p.step}")  # fmt: skip
                out.append((f"step {p.step} {src.name} result", diff))
    state: dict[int, dict] = {}
    seen: set[int] = set()
    for step in sorted(run.steps, key=lambda s: s.idx):
        if step.idx < fork_step or step.operation != "chat" or not step.raw_request:
            continue
        api = str(step.attributes.get("crashlab.api") or "openai.chat.completions")
        kwargs, applied = patches.apply_chat(
            api, step.idx, step.raw_request, fork_step=fork_step, state=state
        )
        new = [i for i in applied if i not in seen]
        if not new:
            continue
        seen.update(applied)
        _, diff = diff_values(
            normalize_request(api, step.raw_request),
            normalize_request(api, kwargs),
            fromfile=f"recorded step {step.idx}",
            tofile=f"patched step {step.idx}",
        )
        out.append((f"step {step.idx} {step.name} request", diff or "(no visible change)"))
    return out


# --- construction helpers -------------------------------------------------------------


def as_patchset(value: Any) -> PatchSet:
    """``None`` / a patch / a list / a dict / JSON text / a :class:`PatchSet` -> PatchSet."""
    if value is None:
        return PatchSet()
    if isinstance(value, PatchSet):
        return value
    if isinstance(value, _Patch):
        return PatchSet(patches=[value])  # type: ignore[list-item]
    if isinstance(value, str):
        return as_patchset(json.loads(value))
    if isinstance(value, dict):
        if "patches" in value and "kind" not in value:
            return PatchSet.model_validate(value)
        return PatchSet(patches=[_ANY.validate_python(value)])
    if isinstance(value, Iterable):
        out = PatchSet()
        for item in value:
            out = out + item
        return out
    raise TypeError(f"not a patch: {value!r}")


def load(path: str | Path) -> PatchSet:
    """Read patches from a ``.json`` or ``.yaml`` / ``.yml`` file: one patch object, a
    list of them, or ``{"patches": [...]}``."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise PatchError("YAML patch files need PyYAML: pip install 'crashlab[yaml]'") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    try:
        return as_patchset(data)
    except (ValueError, TypeError) as exc:
        raise PatchError(f"{p}: {exc}") from exc


# Short aliases, as in the design doc: crashlab.patch.SystemPrompt(...)
SystemPrompt = SystemPromptPatch
Message = MessagePatch
InjectMessage = InjectMessagePatch
ToolResult = ToolResultPatch
ToolDescription = ToolDescriptionPatch
ToolSchema = ToolSchemaPatch
ModelParams = ModelParamPatch
Model = ModelParamPatch

__all__ = [
    "AnyPatch",
    "InjectMessagePatch",
    "MessagePatch",
    "ModelParamPatch",
    "PatchError",
    "PatchSet",
    "RequestEditor",
    "SystemPromptPatch",
    "ToolDescriptionPatch",
    "ToolResultPatch",
    "ToolSchemaPatch",
    "as_patchset",
    "load",
    "normalize_request",
    "preview_patches",
]
