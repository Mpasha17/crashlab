"""Context-window section debugging and static tool-schema auditing.

``analyze_context`` breaks each LLM chat prompt into sections (system, tools,
messages, tool results) and estimates token share, growth, repeated messages
and which tool outputs flooded the context. ``audit_tool_schemas`` inspects
recorded tool definitions for the kind of schema bugs that break providers
like Groq and Gemini (``required`` without ``properties``, missing types /
descriptions, near-duplicate tools, unused schemas).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Any, Literal

from .models import Run, Step

__all__ = [
    "ContextReport",
    "RepeatedMessage",
    "SectionBreakdown",
    "StepContext",
    "ToolAuditReport",
    "ToolInfo",
    "ToolIssue",
    "ToolOutputSize",
    "analyze_context",
    "audit_tool_schemas",
]

SECTION_NAMES = (
    "system_prompt",
    "tool_definitions",
    "user_messages",
    "assistant_messages",
    "tool_results",
    "other",
)

# A tool JSON body larger than this is resent on every LLM call and is worth flagging.
LARGE_SCHEMA_CHARS = 2000
# Near-identical names or descriptions that confuse tool selection.
NAME_SIMILARITY = 0.85
DESC_SIMILARITY = 0.9
MIN_DESC_FOR_SIMILARITY = 20
# Top-level descriptions shorter than this, or in the denylist, are treated as vague.
MIN_DESCRIPTION_CHARS = 8
_VAGUE_DESCRIPTIONS = frozenset(
    {
        "a function",
        "a tool",
        "default",
        "description",
        "do something",
        "does something",
        "function",
        "helper",
        "lorem ipsum",
        "misc",
        "n/a",
        "na",
        "none",
        "other",
        "placeholder",
        "tba",
        "tbd",
        "temp",
        "test",
        "todo",
        "tool",
        "utility",
    }
)
_ID_KEYS = frozenset({"id", "tool_call_id", "call_id"})

Severity = Literal["info", "warn", "error"]


@dataclass
class SectionBreakdown:
    """Character and estimated-token share of one prompt section."""

    name: str
    chars: int
    estimated_tokens: int
    share_pct: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolOutputSize:
    """How much one tool's results contributed to recorded chat prompts."""

    name: str
    chars: int
    estimated_tokens: int
    occurrences: int
    max_chars: int = 0
    steps: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RepeatedMessage:
    """The same message body appearing more than once in a prompt or across turns."""

    role: str
    preview: str
    count: int
    chars: int
    steps: list[int] = field(default_factory=list)
    within_step: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StepContext:
    """Section breakdown for one LLM chat step."""

    idx: int
    model: str | None
    input_tokens: int | None
    total_chars: int
    estimated_tokens: int
    delta_chars: int | None
    delta_tokens: int | None
    sections: list[SectionBreakdown] = field(default_factory=list)
    tool_outputs: list[ToolOutputSize] = field(default_factory=list)
    repeats: list[RepeatedMessage] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ContextReport:
    """Prompt composition, growth, tool-output leaderboard and repeats for a run."""

    run_id: str
    name: str
    step: int | None = None
    steps: list[StepContext] = field(default_factory=list)
    tool_outputs: list[ToolOutputSize] = field(default_factory=list)
    repeats: list[RepeatedMessage] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolIssue:
    """One schema or usage problem found by :func:`audit_tool_schemas`."""

    kind: str
    severity: Severity
    message: str
    tool: str | None = None
    steps: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolInfo:
    """A tool defined on recorded LLM steps and/or actually called."""

    name: str
    description: str | None
    chars: int
    defined_steps: list[int] = field(default_factory=list)
    call_count: int = 0
    called_steps: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolAuditReport:
    """Static audit of every tool schema and call observed in a run."""

    run_id: str
    name: str
    tools: list[ToolInfo] = field(default_factory=list)
    issues: list[ToolIssue] = field(default_factory=list)
    unused: list[str] = field(default_factory=list)
    called: list[str] = field(default_factory=list)
    ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- context analysis -----------------------------------------------------------------


def analyze_context(run: Run, *, step: int | None = None) -> ContextReport:
    """Break down each LLM chat prompt in ``run`` by section, tokens and growth.

    When ``step`` is set, only that chat step is included (growth is still
    measured against the previous chat step in the run). Raises ``ValueError``
    if ``step`` does not exist or is not a chat step.
    """
    chats = [s for s in sorted(run.steps, key=lambda s: s.idx) if s.operation == "chat"]
    if step is not None:
        match = next((s for s in run.steps if s.idx == step), None)
        if match is None:
            raise ValueError(f"no step {step} in run {run.run_id}")
        if match.operation != "chat":
            raise ValueError(f"step {step} is {match.operation}, not a chat step")
    previous: StepContext | None = None
    steps_out: list[StepContext] = []
    all_repeats: list[RepeatedMessage] = []
    all_fps: dict[str, dict[int, list[tuple[str, str, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for chat in chats:
        sections_text = _prompt_sections(chat)
        char_counts = {name: len(text) for name, text in sections_text.items()}
        total_chars = sum(char_counts.values())
        recorded = _input_tokens(chat)
        token_counts = _scaled_tokens(char_counts, recorded)
        estimated = sum(token_counts.values())
        sections = [
            SectionBreakdown(
                name=name,
                chars=char_counts[name],
                estimated_tokens=token_counts[name],
                share_pct=_share_pct(char_counts[name], total_chars),
            )
            for name in SECTION_NAMES
        ]
        _fix_share_pct(sections)
        tool_outputs = _tool_outputs_for_step(chat)
        fingerprints = _message_fingerprints(chat)
        within: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
        for fp, role, preview, chars in fingerprints:
            within[fp].append((role, preview, chars))
            all_fps[fp][chat.idx].append((role, preview, chars))
        step_repeats: list[RepeatedMessage] = []
        for _fp, copies in within.items():
            role, preview, chars = copies[0]
            if len(copies) > 1:
                step_repeats.append(
                    RepeatedMessage(
                        role=role,
                        preview=preview,
                        count=len(copies),
                        chars=chars,
                        steps=[chat.idx],
                        within_step=True,
                    )
                )

        delta_chars = None if previous is None else total_chars - previous.total_chars
        delta_tokens = None if previous is None else estimated - previous.estimated_tokens
        ctx = StepContext(
            idx=chat.idx,
            model=_step_model(chat),
            input_tokens=recorded,
            total_chars=total_chars,
            estimated_tokens=estimated,
            delta_chars=delta_chars,
            delta_tokens=delta_tokens,
            sections=sections,
            tool_outputs=tool_outputs,
            repeats=step_repeats,
        )
        previous = ctx
        if step is None or chat.idx == step:
            steps_out.append(ctx)
            all_repeats.extend(step_repeats)

    # History that is simply carried forward once per later step is expected.
    # Flag a fingerprint when a later step contains strictly more copies than
    # an earlier one (the same body was re-appended to the prompt).
    already = {(r.role, r.preview, tuple(r.steps)) for r in all_repeats}
    for _fp, by_step in all_fps.items():
        counts = {idx: len(copies) for idx, copies in by_step.items()}
        if step is not None:
            counts = {idx: n for idx, n in counts.items() if idx <= step}
        idxs = sorted(counts)
        if not any(counts[b] > counts[a] for a, b in zip(idxs, idxs[1:], strict=False)):
            continue
        role, preview, chars = by_step[idxs[0]][0]
        rec_steps = idxs
        rec = RepeatedMessage(
            role=role,
            preview=preview,
            count=max(counts[i] for i in rec_steps),
            chars=chars,
            steps=rec_steps,
            within_step=False,
        )
        if (rec.role, rec.preview, tuple(rec.steps)) in already:
            continue
        # A within-step finding already covers a single-step spike.
        if len(rec_steps) <= 1 and any(
            r.within_step and r.role == role and r.preview == preview for r in all_repeats
        ):
            continue
        all_repeats.append(rec)

    # Leaderboard across the reported steps (cumulative context footprint).
    tool_outputs = _merge_tool_outputs([t for ctx in steps_out for t in ctx.tool_outputs])
    return ContextReport(
        run_id=run.run_id,
        name=run.name,
        step=step,
        steps=steps_out,
        tool_outputs=tool_outputs,
        repeats=all_repeats,
    )


def _prompt_sections(step: Step) -> dict[str, str]:
    req = step.request or {}
    buckets: dict[str, list[str]] = {name: [] for name in SECTION_NAMES}
    system = req.get("system")
    if system:
        buckets["system_prompt"].append(_as_text(system))
    for msg in req.get("messages") or []:
        if not isinstance(msg, dict):
            buckets["other"].append(_as_text(msg))
            continue
        buckets[_classify_message(msg)].append(_as_text(msg))
    tools = req.get("tools")
    if tools:
        buckets["tool_definitions"].append(_as_text(tools))
    return {name: "\n".join(part for part in parts if part) for name, parts in buckets.items()}


def _classify_message(msg: dict[str, Any]) -> str:
    role = str(msg.get("role") or "").strip().lower()
    if _is_tool_result(msg, role):
        return "tool_results"
    if role == "system":
        return "system_prompt"
    if role == "user":
        return "user_messages"
    if role == "assistant":
        return "assistant_messages"
    return "other"


def _is_tool_result(msg: dict[str, Any], role: str) -> bool:
    if role in {"tool", "function"}:
        return True
    if msg.get("tool_call_id"):
        return True
    for part in msg.get("parts") or []:
        if isinstance(part, dict) and part.get("type") == "tool_call_response":
            return True
    return False


def _scaled_tokens(char_counts: dict[str, int], recorded: int | None) -> dict[str, int]:
    """``chars // 4``, scaled so the sections sum to ``recorded`` when known."""
    raw = {name: char_counts[name] // 4 for name in SECTION_NAMES}
    total_chars = sum(char_counts.values())
    if recorded is None or recorded <= 0 or total_chars <= 0:
        return raw
    # Proportional to character share, then fix rounding so the parts sum to
    # the recorded input token count.
    tokens = {
        name: int(round(recorded * char_counts[name] / total_chars)) for name in SECTION_NAMES
    }
    delta = recorded - sum(tokens.values())
    if delta and tokens:
        # Adjust the largest section so the total matches the provider count.
        biggest = max(SECTION_NAMES, key=lambda n: (char_counts[n], n))
        tokens[biggest] = max(0, tokens[biggest] + delta)
    return tokens


def _share_pct(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(100.0 * part / total, 2)


def _fix_share_pct(sections: list[SectionBreakdown]) -> None:
    total = sum(s.chars for s in sections)
    if total <= 0:
        return
    delta = round(100.0 - sum(s.share_pct for s in sections), 2)
    if not delta:
        return
    biggest = max(sections, key=lambda s: (s.chars, s.name))
    biggest.share_pct = round(biggest.share_pct + delta, 2)


def _tool_outputs_for_step(step: Step) -> list[ToolOutputSize]:
    call_ids = _tool_call_names(step)
    buckets: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for msg in (step.request or {}).get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").strip().lower()
        if not _is_tool_result(msg, role):
            continue
        name = _tool_message_name(msg, call_ids)
        chars = len(_as_text(msg))
        buckets[name].append((chars, step.idx))
    return _buckets_to_outputs(buckets)


def _buckets_to_outputs(buckets: dict[str, list[tuple[int, int]]]) -> list[ToolOutputSize]:
    out: list[ToolOutputSize] = []
    for name, items in buckets.items():
        chars = sum(c for c, _ in items)
        steps = sorted({i for _, i in items})
        out.append(
            ToolOutputSize(
                name=name,
                chars=chars,
                estimated_tokens=chars // 4,
                occurrences=len(items),
                max_chars=max(c for c, _ in items),
                steps=steps,
            )
        )
    out.sort(key=lambda t: (-t.chars, -t.occurrences, t.name))
    return out


def _merge_tool_outputs(items: list[ToolOutputSize]) -> list[ToolOutputSize]:
    buckets: dict[str, list[tuple[int, int]]] = defaultdict(list)
    max_chars: dict[str, int] = defaultdict(int)
    # Re-expand by occurrences so counts add correctly; each ToolOutputSize may
    # already be an aggregate for one step.
    for item in items:
        # Represent the aggregate as one blob plus occurrence count: we cannot
        # recover per-message sizes except for max_chars. Sum chars and
        # occurrences directly instead.
        max_chars[item.name] = max(max_chars[item.name], item.max_chars)
        # Store a single tuple of (chars, first step) and tack steps on after.
        buckets[item.name].append((item.chars, item.occurrences, tuple(item.steps)))
    out: list[ToolOutputSize] = []
    for name, rows in buckets.items():
        chars = sum(r[0] for r in rows)
        occurrences = sum(r[1] for r in rows)
        steps = sorted({i for r in rows for i in r[2]})
        out.append(
            ToolOutputSize(
                name=name,
                chars=chars,
                estimated_tokens=chars // 4,
                occurrences=occurrences,
                max_chars=max_chars[name],
                steps=steps,
            )
        )
    out.sort(key=lambda t: (-t.chars, -t.occurrences, t.name))
    return out


def _tool_call_names(step: Step) -> dict[str, str]:
    """Map tool_call ids seen in this request's history (and this response) to names."""
    names: dict[str, str] = {}

    def _eat(payload: Any) -> None:
        if isinstance(payload, dict):
            cid = payload.get("id") or payload.get("tool_call_id")
            name = payload.get("name")
            typ = payload.get("type")
            if typ in {"tool_call", "function", None} and cid and name:
                names[str(cid)] = str(name)
            fn = payload.get("function")
            if isinstance(fn, dict) and cid and fn.get("name"):
                names[str(cid)] = str(fn["name"])
            for value in payload.values():
                _eat(value)
        elif isinstance(payload, list):
            for item in payload:
                _eat(item)

    _eat((step.request or {}).get("messages"))
    _eat(step.response)
    return names


def _tool_message_name(msg: dict[str, Any], call_ids: dict[str, str]) -> str:
    name = msg.get("name")
    if isinstance(name, str) and name.strip():
        return name
    for key in ("tool_call_id", "id"):
        cid = msg.get(key)
        if cid and str(cid) in call_ids:
            return call_ids[str(cid)]
    for part in msg.get("parts") or []:
        if not isinstance(part, dict):
            continue
        part_name = part.get("name")
        if isinstance(part_name, str) and part_name.strip():
            return part_name
        for key in ("id", "tool_call_id", "call_id"):
            cid = part.get(key)
            if cid and str(cid) in call_ids:
                return call_ids[str(cid)]
    return "?"


def _message_fingerprints(step: Step) -> list[tuple[str, str, str, int]]:
    """``(fingerprint, role, preview, chars)`` for each message in the prompt."""
    out: list[tuple[str, str, str, int]] = []
    system = (step.request or {}).get("system")
    if system:
        text = _as_text(system)
        out.append((_fingerprint("system", text), "system", _preview(text), len(text)))
    for msg in (step.request or {}).get("messages") or []:
        if not isinstance(msg, dict):
            text = _as_text(msg)
            out.append((_fingerprint("other", text), "other", _preview(text), len(text)))
            continue
        role = str(msg.get("role") or "other").strip().lower() or "other"
        if _is_tool_result(msg, role):
            role = "tool"
        body = _message_body(msg)
        out.append((_fingerprint(role, body), role, _preview(body), len(_as_text(msg))))
    return out


def _message_body(msg: dict[str, Any]) -> str:
    """Stable text of a message, ignoring volatile call ids so copies still match."""
    role = str(msg.get("role") or "").strip().lower()
    name = msg.get("name") or ""
    chunks: list[str] = []
    content = msg.get("content")
    if isinstance(content, str):
        chunks.append(content)
    elif content is not None:
        chunks.append(_as_text(_strip_ids(content)))
    for part in msg.get("parts") or []:
        if not isinstance(part, dict):
            chunks.append(_as_text(part))
            continue
        typ = part.get("type")
        if typ == "text" and part.get("content"):
            chunks.append(str(part["content"]))
        elif typ == "tool_call_response":
            chunks.append(_as_text(_strip_ids(part.get("response", part))))
        elif typ == "tool_call":
            chunks.append(_as_text({"name": part.get("name"), "arguments": part.get("arguments")}))
        else:
            chunks.append(_as_text(_strip_ids(part)))
    if not chunks and msg:
        chunks.append(_as_text(_strip_ids({k: v for k, v in msg.items() if k != "role"})))
    return f"{role}:{name}:" + "\n".join(chunks)


def _strip_ids(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_ids(v) for k, v in value.items() if k not in _ID_KEYS}
    if isinstance(value, list):
        return [_strip_ids(v) for v in value]
    return value


def _fingerprint(role: str, body: str) -> str:
    return f"{role}\n{body}"


def _preview(text: str, limit: int = 80) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _input_tokens(step: Step) -> int | None:
    value = (step.attributes or {}).get("gen_ai.usage.input_tokens")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        usage = (step.response or {}).get("usage") or {}
        if isinstance(usage, dict):
            value = usage.get("input_tokens") or usage.get("prompt_tokens")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _step_model(step: Step) -> str | None:
    for key in ("gen_ai.request.model", "gen_ai.response.model"):
        value = (step.attributes or {}).get(key)
        if isinstance(value, str) and value.strip():
            return value
    model = (step.response or {}).get("model")
    if isinstance(model, str) and model.strip():
        return model
    if isinstance(step.name, str) and step.name.strip():
        return step.name
    return None


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)


# --- tool schema audit ----------------------------------------------------------------


def audit_tool_schemas(run: Run) -> ToolAuditReport:
    """Inspect recorded tool definitions and calls for schema and usage issues."""
    defined: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    chat_steps_with_tools: list[int] = []
    for step in sorted(run.steps, key=lambda s: s.idx):
        if step.operation != "chat":
            continue
        tools = _tools_of(step)
        if tools:
            chat_steps_with_tools.append(step.idx)
        for tool in tools:
            defined[_tool_name(tool)].append((step.idx, tool))

    called: dict[str, list[int]] = defaultdict(list)
    for step in sorted(run.steps, key=lambda s: s.idx):
        for name in _called_tool_names(step):
            called[name].append(step.idx)

    issues: list[ToolIssue] = []
    infos: list[ToolInfo] = []
    # Union of defined and called names, defined first in first-seen order.
    order: list[str] = []
    for name in defined:
        if name not in order:
            order.append(name)
    for name in called:
        if name not in order:
            order.append(name)

    for name in order:
        entries = defined.get(name, [])
        first = entries[0][1] if entries else {}
        desc = _tool_description(first) if entries else None
        chars = max((len(_as_text(tool)) for _, tool in entries), default=0)
        defined_steps = sorted({idx for idx, _ in entries})
        called_steps = sorted(set(called.get(name, [])))
        infos.append(
            ToolInfo(
                name=name,
                description=desc,
                chars=chars,
                defined_steps=defined_steps,
                call_count=len(called.get(name, [])),
                called_steps=called_steps,
            )
        )
        issues.extend(_issues_for_tool(name, entries))

    # Duplicate names inside a single request (already partly handled per tool
    # when the same name is listed twice — emit a dedicated issue).
    issues.extend(_duplicate_name_issues(defined))
    issues.extend(_similar_tool_issues(infos))

    unused: list[str] = []
    if chat_steps_with_tools:
        required_steps = set(chat_steps_with_tools)
        for info in infos:
            defined_set = set(info.defined_steps)
            if not defined_set:
                continue
            # "defined in every step" that carried a tool list, and never called
            if defined_set >= required_steps and info.call_count == 0:
                unused.append(info.name)
                issues.append(
                    ToolIssue(
                        kind="unused_tool",
                        severity="info",
                        tool=info.name,
                        message=(
                            f"{info.name} is defined on every LLM step "
                            f"({_steps_text(info.defined_steps)}) but was never called"
                        ),
                        steps=info.defined_steps,
                    )
                )

    called_names = [n for n, steps in called.items() if steps]
    # Deterministic issue order: error, warn, info, then kind, tool, message.
    rank = {"error": 0, "warn": 1, "info": 2}
    issues.sort(key=lambda i: (rank.get(i.severity, 9), i.kind, i.tool or "", i.message))
    ok = not any(i.severity == "error" for i in issues)
    return ToolAuditReport(
        run_id=run.run_id,
        name=run.name,
        tools=infos,
        issues=issues,
        unused=unused,
        called=called_names,
        ok=ok,
    )


def _issues_for_tool(name: str, entries: list[tuple[int, dict[str, Any]]]) -> list[ToolIssue]:
    if not entries:
        return []
    # Audit the first definition; also re-check later ones if the schema changes.
    issues: list[ToolIssue] = []
    seen: set[tuple[str, str]] = set()
    for idx, tool in entries:
        for issue in _schema_issues(name, tool, idx):
            key = (issue.kind, issue.message)
            if key in seen:
                # Same finding on a later step: just extend the step list.
                existing = next(i for i in issues if (i.kind, i.message) == key)
                if idx not in existing.steps:
                    existing.steps.append(idx)
                continue
            seen.add(key)
            issues.append(issue)
    return issues


def _schema_issues(name: str, tool: dict[str, Any], step_idx: int) -> list[ToolIssue]:
    issues: list[ToolIssue] = []
    steps = [step_idx]
    desc = _tool_description(tool)
    if desc is None:
        issues.append(
            ToolIssue(
                kind="missing_description",
                severity="warn",
                tool=name,
                message=f"{name} has no top-level description",
                steps=steps,
            )
        )
    elif _vague_description(desc):
        preview = _preview(desc, 40) or "(empty)"
        issues.append(
            ToolIssue(
                kind="vague_description",
                severity="warn",
                tool=name,
                message=f"{name} has an empty or vague description: {preview!r}",
                steps=steps,
            )
        )

    schema = _parameters(tool)
    if schema is not None:
        issues.extend(_parameter_issues(name, schema, steps))

    size = len(_as_text(tool))
    if size > LARGE_SCHEMA_CHARS:
        issues.append(
            ToolIssue(
                kind="schema_too_large",
                severity="warn",
                tool=name,
                message=(
                    f"{name}'s schema is {size} characters (limit {LARGE_SCHEMA_CHARS}); "
                    "it is resent on every LLM request"
                ),
                steps=steps,
            )
        )
    return issues


def _parameter_issues(name: str, schema: dict[str, Any], steps: list[int]) -> list[ToolIssue]:
    issues: list[ToolIssue] = []
    required = schema.get("required")
    properties = schema.get("properties")
    has_required = required is not None
    required_keys = _required_keys(required)
    props_ok = isinstance(properties, dict) and bool(properties)

    if has_required and not props_ok:
        issues.append(
            ToolIssue(
                kind="required_missing_properties",
                severity="error",
                tool=name,
                message=(
                    f"{name} lists required parameters "
                    f"({_keys_text(required_keys) or 'empty required'}) but "
                    "properties is missing or empty — providers like Groq and "
                    "Gemini reject this schema"
                ),
                steps=steps,
            )
        )
    elif required_keys and isinstance(properties, dict):
        missing = [k for k in required_keys if k not in properties]
        if missing:
            issues.append(
                ToolIssue(
                    kind="required_missing_properties",
                    severity="error",
                    tool=name,
                    message=(
                        f"{name} required list references {_keys_text(missing)} "
                        "which are missing from properties"
                    ),
                    steps=steps,
                )
            )

    if isinstance(properties, dict):
        for path, prop in _walk_properties(properties):
            if not isinstance(prop, dict):
                issues.append(
                    ToolIssue(
                        kind="param_missing_type",
                        severity="warn",
                        tool=name,
                        message=f"{name} parameter {path} is not a schema object",
                        steps=steps,
                    )
                )
                continue
            if "type" not in prop and "anyOf" not in prop and "$ref" not in prop:
                issues.append(
                    ToolIssue(
                        kind="param_missing_type",
                        severity="warn",
                        tool=name,
                        message=f"{name} parameter {path} has no type",
                        steps=steps,
                    )
                )
            if not str(prop.get("description") or "").strip():
                issues.append(
                    ToolIssue(
                        kind="param_missing_description",
                        severity="warn",
                        tool=name,
                        message=f"{name} parameter {path} has no description",
                        steps=steps,
                    )
                )
    return issues


def _required_keys(required: Any) -> list[str]:
    if required is None:
        return []
    if isinstance(required, str):
        return [required] if required else []
    if isinstance(required, (list, tuple)):
        return [str(k) for k in required]
    return [str(required)]


def _keys_text(keys: list[str]) -> str:
    return ", ".join(repr(k) for k in keys)


def _walk_properties(properties: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for key, prop in properties.items():
        path = f"{prefix}.{key}" if prefix else key
        out.append((path, prop))
        if not isinstance(prop, dict):
            continue
        nested = prop.get("properties")
        if isinstance(nested, dict):
            out.extend(_walk_properties(nested, path))
        items = prop.get("items")
        if isinstance(items, dict) and isinstance(items.get("properties"), dict):
            out.extend(_walk_properties(items["properties"], f"{path}[]"))
    return out


def _vague_description(desc: str) -> bool:
    text = " ".join(desc.split()).strip()
    if not text:
        return True
    if text.lower() in _VAGUE_DESCRIPTIONS:
        return True
    return len(text) < MIN_DESCRIPTION_CHARS


def _duplicate_name_issues(
    defined: dict[str, list[tuple[int, dict[str, Any]]]],
) -> list[ToolIssue]:
    issues: list[ToolIssue] = []
    per_step: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for name, entries in defined.items():
        for idx, _tool in entries:
            per_step[idx][name] += 1
    for idx, counts in sorted(per_step.items()):
        for name, count in counts.items():
            if count > 1:
                issues.append(
                    ToolIssue(
                        kind="duplicate_name",
                        severity="error",
                        tool=name,
                        message=(
                            f"{name} is defined {count} times in the same request (step {idx})"
                        ),
                        steps=[idx],
                    )
                )
    return issues


def _similar_tool_issues(infos: list[ToolInfo]) -> list[ToolIssue]:
    issues: list[ToolIssue] = []
    named = [i for i in infos if i.name and i.name != "?"]
    for i, left in enumerate(named):
        for right in named[i + 1 :]:
            if left.name == right.name:
                continue
            why = _similarity_reason(left, right)
            if not why:
                continue
            issues.append(
                ToolIssue(
                    kind="similar_tools",
                    severity="warn",
                    tool=left.name,
                    message=(f"{left.name} and {right.name} are easy to confuse ({why})"),
                    steps=sorted(set(left.defined_steps + right.defined_steps)),
                )
            )
    return issues


def _similarity_reason(left: ToolInfo, right: ToolInfo) -> str | None:
    a, b = left.name, right.name
    name_ratio = SequenceMatcher(None, a.lower(), b.lower()).ratio()
    compact_a, compact_b = _compact_name(a), _compact_name(b)
    compact_ratio = SequenceMatcher(None, compact_a, compact_b).ratio()
    if compact_a and compact_a == compact_b:
        return f"names {a!r} and {b!r} differ only in punctuation"
    if (
        compact_a
        and compact_b
        and (compact_a.startswith(compact_b) or compact_b.startswith(compact_a))
        and abs(len(compact_a) - len(compact_b)) <= 3
    ):
        return f"names {a!r} and {b!r} are near-identical"
    if max(name_ratio, compact_ratio) >= NAME_SIMILARITY:
        return f"names {a!r} and {b!r} are {max(name_ratio, compact_ratio):.0%} similar"
    da = " ".join((left.description or "").split()).strip()
    db = " ".join((right.description or "").split()).strip()
    if len(da) >= MIN_DESC_FOR_SIMILARITY and len(db) >= MIN_DESC_FOR_SIMILARITY:
        desc_ratio = SequenceMatcher(None, da.lower(), db.lower()).ratio()
        if desc_ratio >= DESC_SIMILARITY:
            return f"descriptions are {desc_ratio:.0%} similar"
    return None


def _compact_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _tools_of(step: Step) -> list[dict[str, Any]]:
    raw = (step.request or {}).get("tools") or []
    if isinstance(raw, dict):
        raw = [raw]
    out: list[dict[str, Any]] = []
    for tool in raw:
        if not isinstance(tool, dict):
            continue
        out.append(_unwrap_tool(tool))
    return out


def _unwrap_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Accept both crashlab-normalized tools and raw OpenAI ``{type, function}``."""
    fn = tool.get("function")
    if isinstance(fn, dict) and (tool.get("type") == "function" or "name" not in tool):
        merged = {
            "type": "function",
            "name": fn.get("name") or tool.get("name"),
            "description": fn.get("description", tool.get("description")),
            "parameters": fn.get("parameters", fn.get("input_schema")),
        }
        # Preserve extra keys from the inner function object.
        for key, value in fn.items():
            merged.setdefault(key, value)
        return {k: v for k, v in merged.items() if v is not None} or tool
    return tool


def _tool_name(tool: dict[str, Any]) -> str:
    name = tool.get("name")
    if isinstance(name, str) and name.strip():
        return name
    fn = tool.get("function")
    if isinstance(fn, dict) and isinstance(fn.get("name"), str) and fn["name"].strip():
        return str(fn["name"])
    typ = tool.get("type")
    if isinstance(typ, str) and typ.strip() and typ != "function":
        return typ
    return "?"


def _tool_description(tool: dict[str, Any]) -> str | None:
    if "description" in tool:
        desc = tool.get("description")
        return None if desc is None else str(desc)
    fn = tool.get("function")
    if isinstance(fn, dict) and "description" in fn:
        desc = fn.get("description")
        return None if desc is None else str(desc)
    return None


def _parameters(tool: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("parameters", "input_schema"):
        value = tool.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError):
                parsed = None
            if isinstance(parsed, dict):
                return parsed
    fn = tool.get("function")
    if isinstance(fn, dict):
        return _parameters(fn)
    return None


def _called_tool_names(step: Step) -> list[str]:
    names: list[str] = []
    if step.operation == "execute_tool":
        name = step.name or (step.attributes or {}).get("gen_ai.tool.name")
        if isinstance(name, str) and name.strip():
            names.append(name)
        return names
    if step.operation != "chat":
        return names
    response = step.response or {}
    for call in response.get("tool_calls") or []:
        if isinstance(call, dict):
            name = call.get("name")
            if not name and isinstance(call.get("function"), dict):
                name = call["function"].get("name")
            if isinstance(name, str) and name.strip():
                names.append(name)
    for msg in response.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        for part in msg.get("parts") or []:
            if isinstance(part, dict) and part.get("type") == "tool_call":
                name = part.get("name")
                if isinstance(name, str) and name.strip():
                    names.append(name)
    return names


def _steps_text(steps: list[int]) -> str:
    return ", ".join(str(i) for i in steps)
