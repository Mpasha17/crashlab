"""First-class adapters for LlamaIndex, smolagents and Haystack.

Zero hard dependency: this module never imports ``llama_index``, ``smolagents``,
``haystack`` or provider SDKs (it only looks at ``sys.modules`` and attributes of the
objects it is handed), so ``import crashlab.integrations.frameworks_extra`` is safe in a
core-only environment and tests can drive every hook with plain fake objects.

- :class:`LlamaIndexHandler` / :func:`llama_index_handler` implement the LlamaIndex
  instrumentation dispatcher's event handler *and* span handler interfaces (duck-typed).
  LLM chat / completion events become ``chat`` steps (model, usage, tool calls),
  retrieval events become a retrieval tool step whose result is
  ``[{id, text, score, metadata}]`` (visible to :func:`crashlab.rag.retrieved`), tool
  spans / ``AgentToolCallEvent`` become tool steps, agent spans attribute steps with
  ``gen_ai.agent.name``, and errors are recorded as failing steps.
  :func:`wrap_query_engine` records ``query`` / ``chat`` calls inside
  :class:`crashlab.record`.
- :func:`wrap_smolagent` / :func:`crashlab_step_callback` record a smolagents agent's
  ``run`` (sync, and the ``stream=True`` generator) and its memory steps: model output,
  tool calls, observations, errors and token usage, with ``managed_agents`` attributed
  to their own names.
- :func:`wrap_pipeline` / :class:`HaystackTracer` record a Haystack ``Pipeline``:
  every component run is a step (generators as chat steps with model / usage from
  ``meta``, retrievers as retrieval tool steps, ``ToolInvoker`` as tool steps).

None of these double-record: a chat or tool step that a wrapped client
(:func:`crashlab.wrap`), :func:`crashlab.instrument` or ``@crashlab.tool`` already
recorded while the framework call ran is not recorded a second time.
"""

from __future__ import annotations

import functools
import inspect
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from typing import Any

from ..capture._steps import begin_step
from ..capture.normalize import to_jsonable, tool_call_part, tool_result_part
from ..context import agent_scope, current, current_agent
from ..models import utcnow
from .frameworks import (
    _agent_name,
    _as_args,
    _attr,
    _call_input,
    _last_idx,
    _live_steps,
    _maybe_record,
    _new_ops,
    _split_system,
    _text_of,
    _usage_attrs,
    _usage_payload,
    _warn,
)

_API_LI = "llama_index"
_API_SMOL = "smolagents"
_API_HAY = "haystack"
_RETRIEVAL_TOOL = "search_docs"  # crashlab.rag's default tool name
_MAX_TEXT = 20_000

# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _reraise_control(prefix: str, exc: BaseException) -> None:
    """Warn about adapter bugs, but let crashlab's own stop signals through."""
    from ..checks.engine import CircuitBreakerError

    names = {c.__name__ for c in type(exc).__mro__}
    if isinstance(exc, CircuitBreakerError) or "GuardViolation" in names:
        raise exc
    _warn(prefix, exc)


def _mark() -> dict[str, Any] | None:
    ctx = current()
    if ctx is None:
        return None
    return {"started_at": utcnow(), "t0": time.perf_counter(), "last_idx": _last_idx(ctx)}


def _stamp(step: Any, mark: dict[str, Any] | None) -> None:
    if step is None or not mark:
        return
    if mark.get("started_at") is not None:
        step.started_at = mark["started_at"]
    if mark.get("t0") is not None:
        step.t0 = mark["t0"]


def _from_epoch(start: Any, duration: Any) -> dict[str, Any]:
    mark: dict[str, Any] = {}
    try:
        if start is not None:
            mark["started_at"] = datetime.fromtimestamp(float(start), tz=timezone.utc)
        if duration is not None:
            mark["t0"] = time.perf_counter() - float(duration)
    except (TypeError, ValueError, OverflowError, OSError):
        return {}
    return mark


def _enum_text(value: Any) -> str:
    inner = getattr(value, "value", value)
    return str(inner) if inner is not None else ""


_ROLES = {
    "human": "user",
    "ai": "assistant",
    "chatbot": "assistant",
    "model": "assistant",
    "function": "tool",
    "tool-response": "tool",
    "tool_response": "tool",
    "tool-call": "assistant",
    "tool_call": "assistant",
    "developer": "system",
}


def _role(message: Any) -> str:
    raw = _attr(message, "role", "type", default="user")
    role = _enum_text(raw).strip().lower() or "user"
    return _ROLES.get(role, role)


def _msg_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    text = _attr(message, "text")
    if isinstance(text, str):
        return text
    return _text_of(message)


def _tool_calls_of(message: Any) -> list[tuple[Any, str, Any]]:
    """``(id, name, arguments)`` for every tool call a message carries."""
    found: list[Any] = []
    for key in ("tool_calls",):
        value = _attr(message, key)
        if isinstance(value, (list, tuple)):
            found.extend(value)
    extra = _attr(message, "additional_kwargs")
    if isinstance(extra, dict) and isinstance(extra.get("tool_calls"), (list, tuple)):
        found.extend(extra["tool_calls"])
    blocks = _attr(message, "blocks")
    if isinstance(blocks, (list, tuple)):
        found.extend(
            b for b in blocks if _attr(b, "tool_name") and _attr(b, "tool_kwargs") is not None
        )
    out: list[tuple[Any, str, Any]] = []
    for tc in found:
        fn = _attr(tc, "function")
        name = _attr(tc, "tool_name", "name") or _attr(fn, "name")
        args = _attr(tc, "arguments", "tool_kwargs", "args")
        if args is None:
            args = _attr(fn, "arguments")
        if name:
            out.append(
                (_attr(tc, "id", "tool_call_id"), str(name), args if args is not None else {})
            )
    return out


def _chat_message(message: Any) -> dict[str, Any]:
    role = _role(message)
    text = _msg_text(message)
    calls = _tool_calls_of(message)
    parts: list[dict[str, Any]] = []
    if role == "tool":
        call_id = _attr(message, "tool_call_id") or (
            _attr(_attr(message, "additional_kwargs"), "tool_call_id")
        )
        if call_id:
            return {"role": "tool", "parts": [tool_result_part(call_id, text)]}
    if text:
        parts.append({"type": "text", "content": text})
    parts.extend(tool_call_part(cid, name, args) for cid, name, args in calls)
    if not parts:
        parts.append({"type": "text", "content": ""})
    return {"role": role, "parts": parts}


def _chat_messages(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        return [{"role": "user", "parts": [{"type": "text", "content": value}]}]
    if isinstance(value, (list, tuple)):
        return [_chat_message(m) for m in value]
    return [_chat_message(value)]


def _doc_of(obj: Any) -> dict[str, Any]:
    """LlamaIndex ``NodeWithScore`` / Haystack ``Document`` / dict -> ``{id, text, score,
    metadata}`` (the shape :mod:`crashlab.rag` reads)."""
    node = _attr(obj, "node") or obj
    if isinstance(node, str):
        return {"id": None, "text": node[:_MAX_TEXT]}
    text: Any = None
    getter = getattr(node, "get_content", None)
    if callable(getter):
        with suppress(Exception):
            text = getter()
    if not isinstance(text, str):
        text = _attr(node, "text", "content", "page_content")
    doc: dict[str, Any] = {
        "id": to_jsonable(_attr(node, "node_id", "id_", "id", "doc_id")),
        "text": (text if isinstance(text, str) else _text_of(text))[:_MAX_TEXT],
    }
    score = _attr(obj, "score")
    if score is None:
        score = _attr(node, "score")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        doc["score"] = float(score)
    meta = _attr(node, "metadata", "meta")
    meta = dict(meta) if isinstance(meta, dict) else {}
    ref = _attr(node, "ref_doc_id")
    if isinstance(ref, str) and ref and "ref_doc_id" not in meta:
        meta["ref_doc_id"] = ref  # the source document a LlamaIndex chunk came from
    if meta:
        doc["metadata"] = to_jsonable(meta)
    return doc


def _docs_of(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, (list, tuple)):
        return []
    return [_doc_of(i) for i in items]


def _query_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    q = _attr(value, "query_str", "query", "text")
    return q if isinstance(q, str) else _text_of(value)


def _scoped_attrs(api: str, attrs: dict[str, Any]) -> dict[str, Any]:
    out = {"crashlab.api": api, **attrs}
    scoped = current_agent()
    if scoped:
        out.setdefault("gen_ai.agent.name", scoped)
    return out


def _other_live(ctx: Any, last_idx: int, op: str, api: str, name: str | None = None) -> bool:
    """A ``op`` step recorded since ``last_idx`` by something other than this adapter
    (a wrapped client, ``instrument()``, ``@crashlab.tool``): do not record it twice."""
    return any(
        s.attributes.get("crashlab.api") != api and (name is None or s.name == name)
        for s in _new_ops(ctx, last_idx, op)
    )


def _emit_chat(
    api: str,
    mark: dict[str, Any] | None,
    *,
    model: Any,
    messages: list[dict[str, Any]],
    reply: dict[str, Any] | None,
    usage: Any = None,
    error: BaseException | None = None,
    attrs: dict[str, Any] | None = None,
) -> None:
    ctx = current()
    if ctx is None:
        return
    if mark and _other_live(ctx, mark["last_idx"], "chat", api):
        return
    name = _agent_name(_enum_text(model) if model is not None else None) or "llm"
    system, rest = _split_system(messages)
    request: dict[str, Any] = {"messages": rest, "params": {"model": name}}
    if system:
        request["system"] = system
    step = begin_step(
        "chat",
        name,
        request,
        attributes=_scoped_attrs(
            api, {"gen_ai.request.model": name, **_usage_attrs(usage), **(attrs or {})}
        ),
        hash_input={"model": name, "messages": rest, "system": system},
    )
    _stamp(step, mark)
    if step is None:
        return
    if error is not None:
        step.fail(error)
        return
    reply = reply or {"role": "assistant", "parts": [{"type": "text", "content": ""}]}
    calls = [
        {"id": p.get("id"), "type": "function", "name": p["name"], "arguments": p["arguments"]}
        for p in reply["parts"]
        if p.get("type") == "tool_call"
    ]
    payload: dict[str, Any] = {
        "messages": [reply],
        "tool_calls": calls,
        "finish_reason": "tool_calls" if calls else "stop",
        "model": name,
    }
    usage_payload = _usage_payload(usage)
    if usage_payload:
        payload["usage"] = usage_payload
    step.finish(payload)


def _emit_tool(
    api: str,
    mark: dict[str, Any] | None,
    *,
    name: str,
    args: Any,
    result: Any = None,
    error: BaseException | None = None,
    attrs: dict[str, Any] | None = None,
    is_retrieval: bool = False,
) -> None:
    ctx = current()
    if ctx is None:
        return
    if mark and _other_live(ctx, mark["last_idx"], "execute_tool", api, name):
        return
    arguments = _as_args(args)
    step = begin_step(
        "execute_tool",
        name,
        {"args": arguments},
        attributes=_scoped_attrs(
            api,
            {
                "gen_ai.tool.name": name,
                "gen_ai.tool.type": "datastore" if is_retrieval else "function",
                **(attrs or {}),
            },
        ),
        hash_input={"tool": name, "args": arguments},
    )
    _stamp(step, mark)
    if step is None:
        return
    if error is not None:
        step.fail(error, is_tool_error=True)
        return
    step.finish({"result": to_jsonable(result)})


def _emit_custom(api: str, mark: dict[str, Any] | None, *, name: str, request: Any,
                 response: Any = None, error: BaseException | None = None,
                 attrs: dict[str, Any] | None = None) -> None:  # fmt: skip
    step = begin_step(
        "custom",
        name,
        {"input": to_jsonable(request)},
        attributes=_scoped_attrs(api, attrs or {}),
    )
    _stamp(step, mark)
    if step is None:
        return
    if error is not None:
        step.fail(error)
        return
    step.finish({"output": to_jsonable(response)})


def _err(value: Any) -> BaseException:
    if isinstance(value, BaseException):
        return value
    return RuntimeError(str(value) if value else "error")


@contextmanager
def _scope(name: str | None) -> Iterator[None]:
    if name:
        with agent_scope(name):
            yield
    else:
        yield


# ---------------------------------------------------------------------------
# LlamaIndex
# ---------------------------------------------------------------------------


class LlamaIndexHandler:
    """Duck-typed LlamaIndex instrumentation event handler + span handler.

    Attach to a dispatcher (``handler.attach()`` does both)::

        h = crashlab.integrations.llama_index_handler()
        h.attach()                      # root dispatcher, event + span handler
        with crashlab.record("rag-bot") as run:
            run.output = str(query_engine.query("..."))

    Without an active :class:`crashlab.record` run it does nothing. ``retrieval_tool``
    is the tool name retrieval results are recorded under (default ``search_docs``, the
    :mod:`crashlab.rag` default).
    """

    def __init__(self, *, retrieval_tool: str = _RETRIEVAL_TOOL, trace_agents: bool = True) -> None:
        self.retrieval_tool = retrieval_tool
        self.trace_agents = trace_agents
        self._lock = threading.Lock()
        self._pending: dict[str, dict[str, Any]] = {}
        self._scopes: dict[str, Any] = {}
        self._tool_calls: list[dict[str, Any]] = []
        self._dispatcher: Any = None

    @classmethod
    def class_name(cls) -> str:
        return "CrashlabLlamaIndexHandler"

    # -- wiring ------------------------------------------------------------------
    def attach(self, dispatcher: Any = None) -> LlamaIndexHandler:
        """Add this handler to ``dispatcher`` (default: the root dispatcher). Imports
        ``llama_index.core.instrumentation`` only here, when LlamaIndex is installed."""
        if dispatcher is None:
            from llama_index.core.instrumentation import get_dispatcher

            dispatcher = get_dispatcher()
        dispatcher.add_event_handler(self)
        dispatcher.add_span_handler(self)
        self._dispatcher = dispatcher
        return self

    def detach(self) -> None:
        d, self._dispatcher = self._dispatcher, None
        if d is None:
            return
        for attr, items in (("event_handlers", self), ("span_handlers", self)):
            lst = getattr(d, attr, None)
            if isinstance(lst, list) and items in lst:
                lst.remove(items)

    def close(self) -> None:
        """Record tool calls that never saw a result, drop dangling scopes."""
        self._flush_tool_calls()
        with self._lock:
            self._pending.clear()
            scopes, self._scopes = self._scopes, {}
        for cm in scopes.values():
            with suppress(Exception):
                cm.__exit__(None, None, None)

    # -- event handler -----------------------------------------------------------
    def handle(self, event: Any, **kwargs: Any) -> None:
        kind = type(event).__name__
        try:
            self._handle(kind, event)
        except Exception as exc:
            _reraise_control(f"LlamaIndex handler ({kind})", exc)

    def _sid(self, event: Any) -> str:
        sid = getattr(event, "span_id", None)
        return str(sid) if sid else f"event-{id(event)}"

    def _handle(self, kind: str, event: Any) -> None:
        if current() is None:
            return
        sid = self._sid(event)
        if kind in ("LLMChatStartEvent", "LLMCompletionStartEvent"):
            self._flush_tool_calls()
            md = _attr(event, "model_dict")
            model = _attr(md, "model", "model_name") if isinstance(md, dict) else None
            prompt = _attr(event, "prompt")
            msgs = (
                _chat_messages(prompt)
                if kind == "LLMCompletionStartEvent"
                else _chat_messages(_attr(event, "messages"))
            )
            with self._lock:
                nested = kind == "LLMCompletionStartEvent" and any(
                    p.get("event") == "chat" for p in self._pending.values()
                )
                # a completion-only LLM wrapped as a chat model reports both: keep the chat
                self._pending[sid] = (
                    {"kind": "skip"}
                    if nested
                    else {
                        "kind": "chat",
                        "event": "chat" if kind == "LLMChatStartEvent" else "completion",
                        "mark": _mark(),
                        "model": model,
                        "msgs": msgs,
                    }
                )
        elif kind in ("LLMChatEndEvent", "LLMCompletionEndEvent"):
            with self._lock:
                p = self._pending.pop(sid, None)
            if p is not None and p["kind"] == "skip":
                return
            self._finish_chat(kind, event, p)
        elif kind == "RetrievalStartEvent":
            q = _query_text(_attr(event, "str_or_query_bundle"))
            with self._lock:
                self._pending[sid] = {"kind": "retrieval", "mark": _mark(), "query": q}
        elif kind == "RetrievalEndEvent":
            with self._lock:
                p = self._pending.pop(sid, None)
            q = p["query"] if p else _query_text(_attr(event, "str_or_query_bundle"))
            _emit_tool(
                _API_LI,
                p["mark"] if p else None,
                name=self.retrieval_tool,
                args={"query": q},
                result=_docs_of(_attr(event, "nodes")),
                is_retrieval=True,
            )
        elif kind == "AgentToolCallEvent":
            tool = _attr(event, "tool")
            name = _agent_name(_attr(tool, "name")) or "tool"
            with self._lock:
                self._tool_calls.append(
                    {"name": name, "args": _attr(event, "arguments"), "mark": _mark()}
                )
        elif kind in ("AgentRunStepEndEvent", "QueryEndEvent"):
            self._flush_tool_calls()
        elif kind in ("SpanDropEvent", "ExceptionEvent"):
            err = _attr(event, "exception") or _attr(event, "err_str")
            with self._lock:
                p = self._pending.pop(sid, None)
            if p is not None:
                self._fail(p, _err(err))
            elif kind == "ExceptionEvent":
                _emit_custom(
                    _API_LI, _mark(), name="llama_index.exception", request={},
                    error=_err(err),
                )  # fmt: skip

    def _fail(self, p: dict[str, Any], exc: BaseException) -> None:
        if p["kind"] == "skip":
            return
        if p["kind"] == "chat":
            _emit_chat(
                _API_LI, p["mark"], model=p["model"], messages=p["msgs"], reply=None, error=exc
            )
        elif p["kind"] == "retrieval":
            _emit_tool(
                _API_LI, p["mark"], name=self.retrieval_tool, args={"query": p["query"]},
                error=exc, is_retrieval=True,
            )  # fmt: skip
        elif p["kind"] == "tool":
            _emit_tool(_API_LI, p["mark"], name=p["name"], args=p["args"], error=exc)

    def _finish_chat(self, kind: str, event: Any, p: dict[str, Any] | None) -> None:
        response = _attr(event, "response")
        raw = _attr(response, "raw")
        if kind == "LLMChatEndEvent":
            message = _attr(response, "message")
            reply = _chat_message(message) if message is not None else None
            msgs = p["msgs"] if p else _chat_messages(_attr(event, "messages"))
        else:
            reply = {
                "role": "assistant",
                "parts": [{"type": "text", "content": str(_attr(response, "text", default=""))}],
            }
            msgs = p["msgs"] if p else _chat_messages(_attr(event, "prompt"))
        if reply is not None:
            reply["role"] = "assistant"
        usage = _attr(raw, "usage") or _attr(response, "additional_kwargs")
        model = (p or {}).get("model") or _attr(raw, "model")
        _emit_chat(
            _API_LI, p["mark"] if p else None, model=model, messages=msgs, reply=reply, usage=usage
        )

    def _flush_tool_calls(self) -> None:
        with self._lock:
            calls, self._tool_calls = self._tool_calls, []
        for c in calls:
            _emit_tool(
                _API_LI, c["mark"], name=c["name"], args=c["args"], result=None,
                attrs={"crashlab.llama_index.result_unavailable": True},
            )  # fmt: skip

    # -- span handler ------------------------------------------------------------
    def span_enter(
        self, id_: Any = None, bound_args: Any = None, instance: Any = None, parent_id: Any = None,
        tags: Any = None, **kwargs: Any,
    ) -> None:  # fmt: skip
        try:
            self._span_enter(str(id_), bound_args, instance)
        except Exception as exc:
            _reraise_control("LlamaIndex handler (span_enter)", exc)

    def span_exit(
        self, id_: Any = None, bound_args: Any = None, instance: Any = None, result: Any = None,
        **kwargs: Any,
    ) -> None:  # fmt: skip
        try:
            sid = str(id_)
            with self._lock:
                p = self._pending.pop(sid, None)
            self._close_scope(sid)
            if p is not None and p["kind"] == "tool" and current() is not None:
                self._finish_tool(p, result)
        except Exception as exc:
            _reraise_control("LlamaIndex handler (span_exit)", exc)

    def span_drop(
        self, id_: Any = None, bound_args: Any = None, instance: Any = None, err: Any = None,
        **kwargs: Any,
    ) -> None:  # fmt: skip
        try:
            sid = str(id_)
            with self._lock:
                p = self._pending.pop(sid, None)
            self._close_scope(sid)
            if p is not None and current() is not None:
                self._fail(p, _err(err))
        except Exception as exc:
            _reraise_control("LlamaIndex handler (span_drop)", exc)

    # BaseSpanHandler-style aliases
    new_span = span_enter

    def prepare_to_exit_span(self, id_: Any = None, bound_args: Any = None, instance: Any = None,
                             result: Any = None, **kwargs: Any) -> None:  # fmt: skip
        self.span_exit(id_, bound_args, instance, result)

    def prepare_to_drop_span(self, id_: Any = None, bound_args: Any = None, instance: Any = None,
                             err: Any = None, **kwargs: Any) -> None:  # fmt: skip
        self.span_drop(id_, bound_args, instance, err)

    def _span_enter(self, sid: str, bound_args: Any, instance: Any) -> None:
        method = sid.rsplit("-", 5)[0] if sid.count("-") >= 5 else sid.split("-")[0]
        meta = _attr(instance, "metadata")
        tool_name = _agent_name(_attr(meta, "name")) if meta is not None else None
        if tool_name and method.endswith((".call", ".acall")) and current() is not None:
            args = self._span_args(bound_args)
            with self._lock:
                stash = next((c for c in self._tool_calls if c["name"] == tool_name), None)
                if stash is not None:
                    self._tool_calls.remove(stash)
                self._pending[sid] = {
                    "kind": "tool",
                    "name": tool_name,
                    "args": args,
                    "mark": (stash or {}).get("mark") or _mark(),
                }
            return
        agent = _agent_name(_attr(instance, "name")) if instance is not None else None
        if self.trace_agents and agent and type(instance).__name__.endswith("Agent"):
            cm = agent_scope(agent)
            cm.__enter__()
            with self._lock:
                self._scopes[sid] = cm

    def _span_args(self, bound_args: Any) -> Any:
        arguments = _attr(bound_args, "arguments")
        if not isinstance(arguments, dict):
            return {}
        kw = arguments.get("kwargs")
        pos = arguments.get("args")
        if isinstance(kw, dict) and kw:
            return kw
        if isinstance(pos, (tuple, list)) and pos:
            return pos[0] if isinstance(pos[0], dict) else {"input": pos[0]}
        rest = {k: v for k, v in arguments.items() if k not in ("self", "args", "kwargs")}
        return rest

    def _close_scope(self, sid: str) -> None:
        with self._lock:
            cm = self._scopes.pop(sid, None)
        if cm is not None:
            with suppress(Exception):
                cm.__exit__(None, None, None)

    def _finish_tool(self, p: dict[str, Any], result: Any) -> None:
        content = _attr(result, "content")
        raw_out = _attr(result, "raw_output")
        value = raw_out if isinstance(raw_out, (str, int, float, bool, dict, list)) else content
        if value is None:
            value = content if content is not None else result
        if _attr(result, "is_error"):
            _emit_tool(
                _API_LI, p["mark"], name=p["name"], args=p["args"],
                error=RuntimeError(str(content or "tool error")),
            )  # fmt: skip
            return
        _emit_tool(_API_LI, p["mark"], name=p["name"], args=p["args"], result=value)


def llama_index_handler(
    *, retrieval_tool: str = _RETRIEVAL_TOOL, trace_agents: bool = True
) -> LlamaIndexHandler:
    """A :class:`LlamaIndexHandler`; ``.attach()`` it to the LlamaIndex root dispatcher
    (or pass it to ``dispatcher.add_event_handler`` / ``add_span_handler`` yourself)."""
    return LlamaIndexHandler(retrieval_tool=retrieval_tool, trace_agents=trace_agents)


def _response_output(result: Any) -> Any:
    if result is None:
        return None
    if hasattr(result, "response_gen") or type(result).__name__.startswith("Streaming"):
        return None  # str() would consume the stream
    text = _attr(result, "response")
    if isinstance(text, str):
        return text
    return to_jsonable(str(result))


def _source_nodes(result: Any) -> list[Any]:
    nodes = _attr(result, "source_nodes")
    return list(nodes) if isinstance(nodes, (list, tuple)) else []


def _wrap_engine_method(orig: Callable[..., Any], name: str, tool: str) -> Callable[..., Any]:
    def after(run: Any, result: Any, last_idx: int, mark: dict[str, Any] | None, args: Any) -> Any:
        ctx = current()
        if ctx is not None:
            nodes = _source_nodes(result)
            if nodes and not any(
                s.idx > last_idx and s.operation == "execute_tool" and s.name == tool
                for s in _live_steps(ctx)
            ):
                _emit_tool(_API_LI, None, name=tool, args=args, result=_docs_of(nodes),
                           is_retrieval=True)  # fmt: skip
        out = _response_output(result)
        if run is not None and out is not None:
            run.output = out
        return result

    def query_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        first = args[0] if args else next(iter(kwargs.values()), "")
        return {"query": _query_text(first)}

    if inspect.iscoroutinefunction(orig):

        @functools.wraps(orig)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            qa = query_args(args, kwargs)
            with _maybe_record(name, qa) as run:
                ctx = current()
                last = _last_idx(ctx) if ctx is not None else -1
                result = await orig(*args, **kwargs)
                return after(run, result, last, None, qa)

        async_wrapper.__crashlab_li_wrapped__ = True  # type: ignore[attr-defined]
        return async_wrapper

    @functools.wraps(orig)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        qa = query_args(args, kwargs)
        with _maybe_record(name, qa) as run:
            ctx = current()
            last = _last_idx(ctx) if ctx is not None else -1
            result = orig(*args, **kwargs)
            return after(run, result, last, None, qa)

    wrapper.__crashlab_li_wrapped__ = True  # type: ignore[attr-defined]
    return wrapper


def wrap_query_engine(
    engine: Any, *, name: str | None = None, retrieval_tool: str = _RETRIEVAL_TOOL
) -> Any:
    """Record a LlamaIndex query / chat engine's ``query`` / ``aquery`` / ``chat`` /
    ``achat`` calls: each runs inside :class:`crashlab.record` (unless a run is already
    active) with the response text as the run output.

    When no :class:`LlamaIndexHandler` recorded a retrieval, the response's
    ``source_nodes`` are recorded as the retrieval tool step, so ``crashlab.rag`` oracles
    work with just this wrapper (LLM calls still need the handler, ``crashlab.wrap`` or
    ``crashlab.instrument()``). Wrapping is idempotent; the engine is returned.
    """
    if engine is None:
        raise TypeError("wrap_query_engine(): engine is required")
    label = _agent_name(name, getattr(engine, "name", None)) or "llama-index"
    for method in ("query", "aquery", "chat", "achat"):
        orig = getattr(engine, method, None)
        if not callable(orig) or getattr(orig, "__crashlab_li_wrapped__", False):
            continue
        with suppress(Exception):
            setattr(engine, method, _wrap_engine_method(orig, label, retrieval_tool))
    with suppress(Exception):
        engine.__crashlab_li_name__ = label
    return engine


# ---------------------------------------------------------------------------
# smolagents
# ---------------------------------------------------------------------------

_SEEN = "__crashlab_recorded__"


def _smol_model_name(agent: Any) -> str:
    model = _attr(agent, "model")
    fallback = type(model).__name__ if model is not None else None
    return _agent_name(_attr(model, "model_id", "model_name", "model"), fallback) or "llm"


def _is_step(step: Any, *kinds: str) -> bool:
    return type(step).__name__ in kinds


def _step_error(step: Any) -> BaseException | None:
    err = _attr(step, "error")
    if err is None:
        return None
    if isinstance(err, BaseException):
        return err
    return RuntimeError(str(_attr(err, "message", default=err)))


def _smol_calls(step: Any) -> list[tuple[Any, str, Any]]:
    out: list[tuple[Any, str, Any]] = []
    declared = _attr(step, "tool_calls", default=None)
    if not declared:  # a failed tool call may not be copied onto the step
        declared = _attr(_attr(step, "model_output_message"), "tool_calls", default=None)
    for tc in declared or []:
        name = _attr(tc, "name")
        args = _attr(tc, "arguments", "args")
        if name is None:
            fn = _attr(tc, "function")
            name, args = _attr(fn, "name"), _attr(fn, "arguments")
        if name:
            out.append((_attr(tc, "id"), str(name), args if args is not None else {}))
    return out


def _smol_text(value: Any) -> str:
    return _text_of(value) if value is not None else ""


def _record_smol_step(agent: Any, step: Any, *, label: str | None = None) -> None:
    """Record one smolagents memory step (``ActionStep`` / ``PlanningStep``) once."""
    if current() is None or not _is_step(step, "ActionStep", "PlanningStep"):
        return
    ctx = current()
    seen = getattr(ctx, "_crashlab_smol_seen", None)
    if seen is None:
        seen = set()
        with suppress(Exception):
            ctx._crashlab_smol_seen = seen
    if getattr(step, _SEEN, False) or id(step) in seen:
        return
    seen.add(id(step))
    with suppress(Exception):
        setattr(step, _SEEN, True)
    cursors = getattr(ctx, "_crashlab_smol_cursor", None)
    if cursors is None:
        cursors = {}
        with suppress(Exception):
            ctx._crashlab_smol_cursor = cursors
    timing = _attr(step, "timing")
    mark_base = _from_epoch(
        _attr(timing, "start_time") or _attr(step, "start_time"),
        _attr(timing, "duration") or _attr(step, "duration"),
    )
    # steps recorded since the previous memory step belong to this one (live client / tool)
    mark = {"last_idx": cursors.get(id(agent), -1), **mark_base}
    scope = _agent_name(label, _attr(agent, "name"))
    with _scope(scope):
        out_msg = _attr(step, "model_output_message")
        calls = _smol_calls(step)
        text = _smol_text(_attr(out_msg, "content")) or _smol_text(_attr(step, "model_output"))
        usage = _attr(step, "token_usage") or _attr(out_msg, "token_usage")
        is_plan = _is_step(step, "PlanningStep")
        if out_msg is not None or text or calls:
            parts: list[dict[str, Any]] = []
            plan = _attr(step, "plan") if is_plan else None
            body = text or (plan if isinstance(plan, str) else "")
            if body:
                parts.append({"type": "text", "content": body})
            parts.extend(tool_call_part(cid, n, a) for cid, n, a in calls)
            if not parts:
                parts.append({"type": "text", "content": ""})
            reply = {"role": "assistant", "parts": parts}
            attrs: dict[str, Any] = {
                "crashlab.smolagents.step": "planning" if is_plan else "action"
            }
            n = _attr(step, "step_number")
            if isinstance(n, int):
                attrs["crashlab.smolagents.step_number"] = n
            _emit_chat(
                _API_SMOL, mark, model=_smol_model_name(agent),
                messages=_chat_messages(_attr(step, "model_input_messages")),
                reply=reply, usage=usage, attrs=attrs,
            )  # fmt: skip
        if is_plan:
            cursors[id(agent)] = _last_idx(ctx)
            return
        err = _step_error(step)
        observations = _attr(step, "observations")
        result = observations if observations is not None else _attr(step, "action_output")
        for i, (_, tname, targs) in enumerate(calls):
            last = i == len(calls) - 1
            _emit_tool(
                _API_SMOL, mark, name=tname, args=targs, result=to_jsonable(result),
                error=err if (err is not None and last) else None,
            )  # fmt: skip
        if err is not None and not calls:
            _emit_custom(_API_SMOL, mark, name="smolagents.error", request={}, error=err)
        cursors[id(agent)] = _last_idx(ctx)


def crashlab_step_callback(*, name: str | None = None) -> Callable[..., None]:
    """A smolagents step callback that records each finished memory step.

    Works as ``agent.step_callbacks`` entry in both smolagents styles: a plain list
    (``callbacks(step)``) and the newer registry (``callbacks(step, agent=agent)``)::

        agent = ToolCallingAgent(tools=[...], model=model,
                                 step_callbacks=[crashlab.integrations.crashlab_step_callback()])

    It is a no-op outside :class:`crashlab.record`, and :func:`wrap_smolagent` uses the
    same recorder, so a step is never recorded twice.
    """

    def callback(memory_step: Any, agent: Any = None, **kwargs: Any) -> None:
        try:
            _record_smol_step(agent, memory_step, label=name)
        except Exception as exc:
            _reraise_control("smolagents step callback", exc)

    callback.__crashlab_smol_callback__ = True  # type: ignore[attr-defined]
    return callback


def _install_smol_callback(agent: Any) -> None:
    registry = getattr(agent, "step_callbacks", None)
    if registry is None:
        return
    items = getattr(registry, "_callbacks", None)
    if isinstance(items, dict):
        existing = [c for cbs in items.values() for c in cbs]
    elif isinstance(registry, list):
        existing = list(registry)
    else:
        existing = []
    if any(getattr(c, "__crashlab_smol_callback__", False) for c in existing):
        return
    cb = crashlab_step_callback()
    if isinstance(registry, list):
        registry.append(cb)
        return
    register = getattr(registry, "register", None)
    mod = sys.modules.get("smolagents.memory")
    if callable(register) and mod is not None:
        for cls_name in ("ActionStep", "PlanningStep"):
            cls = getattr(mod, cls_name, None)
            if cls is not None:
                with suppress(Exception):
                    register(cls, cb)


def _memory_steps(agent: Any) -> list[Any]:
    steps = _attr(_attr(agent, "memory"), "steps")
    return list(steps) if isinstance(steps, (list, tuple)) else []


def _wants_stream(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    if "stream" in kwargs:
        return bool(kwargs["stream"])
    return len(args) > 1 and args[1] is True


def _set_cursor(agent: Any) -> None:
    ctx = current()
    if ctx is None:
        return
    cursors = getattr(ctx, "_crashlab_smol_cursor", None)
    if cursors is None:
        cursors = {}
        with suppress(Exception):
            ctx._crashlab_smol_cursor = cursors
    cursors[id(agent)] = _last_idx(ctx)


def _reconcile_smol(agent: Any, start: int, label: str | None) -> None:
    for step in _memory_steps(agent)[start:]:
        _record_smol_step(agent, step, label=label)


def _smol_output(result: Any) -> Any:
    out = _attr(result, "output") if type(result).__name__ == "RunResult" else result
    return to_jsonable(out)


@contextmanager
def _smol_depth() -> Iterator[bool]:
    """Yield True for the outermost wrapped smolagents run of the active record (a managed
    agent's nested ``run`` must not overwrite the run output with its own answer)."""
    ctx = current()
    if ctx is None:
        yield True
        return
    depth = getattr(ctx, "_crashlab_smol_depth", 0)
    with suppress(Exception):
        ctx._crashlab_smol_depth = depth + 1
    try:
        yield depth == 0
    finally:
        with suppress(Exception):
            ctx._crashlab_smol_depth = depth


def _wrap_smol_run(
    agent: Any, orig: Callable[..., Any], record_name: str, scope: str | None
) -> Any:
    @functools.wraps(orig)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        call_input = _call_input(orig, args, kwargs)
        reset = kwargs.get("reset", args[2] if len(args) > 2 else True)
        stream = _wants_stream(args, kwargs)
        if stream:
            inner = orig(*args, **kwargs)
            return _smol_stream(agent, inner, record_name, scope, call_input)
        with _maybe_record(record_name, call_input) as run, _scope(scope), _smol_depth() as top:
            start = 0 if reset else len(_memory_steps(agent))
            _set_cursor(agent)
            result = orig(*args, **kwargs)
            _reconcile_smol(agent, start, scope)
            if top and run is not None and result is not None and run.output is None:
                run.output = _smol_output(result)
            return result

    wrapper.__crashlab_smol_run__ = True  # type: ignore[attr-defined]
    return wrapper


def _smol_stream(agent: Any, inner: Any, name: str, scope: str | None, call_input: Any) -> Any:
    def gen() -> Iterator[Any]:
        with _maybe_record(name, call_input) as run, _scope(scope), _smol_depth() as top:
            _set_cursor(agent)
            for item in inner:
                kind = type(item).__name__
                if kind in ("ActionStep", "PlanningStep"):
                    _record_smol_step(agent, item, label=scope)
                elif kind == "FinalAnswerStep" and top and run is not None:
                    run.output = to_jsonable(_attr(item, "output"))
                yield item

    return gen()


def wrap_smolagent(agent: Any, *, name: str | None = None) -> Any:
    """Record a smolagents agent: ``agent.run(...)`` runs inside :class:`crashlab.record`
    (unless a run is already active), both plain and ``stream=True`` (a generator).

    Each memory step is recorded: the model output as a ``chat`` step (model, token usage,
    tool calls), every tool call as an ``execute_tool`` step with its observation, and
    tool / parsing errors as failing steps. ``managed_agents`` are wrapped recursively and
    their steps carry their own ``gen_ai.agent.name``. ``crashlab_step_callback`` is
    installed on ``agent.step_callbacks`` so steps are recorded as they finish. Idempotent;
    the agent is returned. smolagents is not imported.
    """
    if agent is None:
        raise TypeError("wrap_smolagent(): agent is required")
    label = _agent_name(name, getattr(agent, "name", None)) or "smolagents"
    if getattr(agent, "__crashlab_smol_wrapped__", False):
        return agent
    _install_smol_callback(agent)
    managed = getattr(agent, "managed_agents", None)
    subs = list(managed.values()) if isinstance(managed, dict) else list(managed or [])
    for sub in subs:
        target = sub if callable(getattr(sub, "run", None)) else getattr(sub, "agent", None)
        if target is not None and target is not agent:
            wrap_smolagent(
                target, name=_agent_name(getattr(sub, "name", None), getattr(target, "name", None))
            )
    orig = getattr(agent, "run", None)
    scope = _agent_name(getattr(agent, "name", None), name) if (
        getattr(agent, "name", None) or subs or name
    ) else None  # fmt: skip
    if callable(orig) and not getattr(orig, "__crashlab_smol_run__", False):
        with suppress(Exception):
            agent.run = _wrap_smol_run(agent, orig, label, scope)
    with suppress(Exception):
        agent.__crashlab_smol_wrapped__ = True
        agent.__crashlab_smol_name__ = label
    return agent


# ---------------------------------------------------------------------------
# Haystack
# ---------------------------------------------------------------------------


def _shrink(value: Any, _depth: int = 0) -> Any:
    """JSON-able, bounded copy: long strings cut, float vectors summarised."""
    value = to_jsonable(value)
    if isinstance(value, str):
        return value if len(value) <= 2000 else value[:2000] + f"... [{len(value)} chars]"
    if isinstance(value, list):
        if len(value) > 8 and all(isinstance(v, (int, float)) for v in value):
            return f"[{len(value)} numbers]"
        items = [_shrink(v, _depth + 1) for v in value[:50]]
        if len(value) > 50:
            items.append(f"... [{len(value)} items]")
        return items
    if isinstance(value, dict) and _depth < 6:
        return {str(k): _shrink(v, _depth + 1) for k, v in list(value.items())[:50]}
    return value


def _hay_kind(cls_name: str, outputs: Any, inputs: Any) -> str:
    if cls_name == "ToolInvoker":
        return "tools"
    out_keys = set(outputs) if isinstance(outputs, dict) else set()
    if "Retriever" in cls_name:
        return "retriever"
    if (
        cls_name.endswith("Generator")
        or cls_name == "Agent"
        or ("replies" in out_keys and ("prompt" in (inputs or {}) or "messages" in (inputs or {})))
    ):
        return "generator"
    if "documents" in out_keys:
        return "documents"
    return "other"


def _tool_result_items(outputs: Any) -> list[Any]:
    msgs = _attr(outputs, "tool_messages", default=None) if isinstance(outputs, dict) else None
    out: list[Any] = []
    for m in msgs or []:
        results = _attr(m, "tool_call_results", default=None)
        out.extend(results if isinstance(results, (list, tuple)) else [])
    return out


def _record_component(
    comp: str,
    cls_name: str,
    inputs: Any,
    outputs: Any,
    error: BaseException | None,
    mark: dict[str, Any] | None,
    *,
    retrieval_tool: str,
    retrievers: set[str] | None,
) -> None:
    ctx = current()
    if ctx is None:
        return
    inputs = inputs if isinstance(inputs, dict) else {}
    kind = _hay_kind(cls_name, outputs, inputs)
    if retrievers is not None and comp in retrievers:
        kind = "retriever"
    attrs = {"crashlab.haystack.component": comp, "crashlab.haystack.type": cls_name}
    if kind == "generator":
        _record_generator(comp, inputs, outputs, error, mark, attrs)
    elif kind == "retriever":
        query = inputs.get("query", inputs.get("query_embedding"))
        docs = _docs_of(outputs.get("documents") if isinstance(outputs, dict) else None)
        _emit_tool(
            _API_HAY, mark, name=retrieval_tool,
            args={"query": query if isinstance(query, str) else _shrink(query), "component": comp},
            result=docs, error=error, attrs=attrs, is_retrieval=True,
        )  # fmt: skip
    elif kind == "tools":
        results = _tool_result_items(outputs)
        if error is not None or not results:
            _emit_tool(_API_HAY, mark, name=comp, args=_shrink(inputs), result=_shrink(outputs),
                       error=error, attrs=attrs)  # fmt: skip
            return
        for r in results:
            origin = _attr(r, "origin")
            is_err = bool(_attr(r, "error"))
            result = _attr(r, "result")
            _emit_tool(
                _API_HAY, mark, name=str(_attr(origin, "tool_name", default=comp)),
                args=_attr(origin, "arguments", default={}), result=_shrink(result),
                error=RuntimeError(_text_of(result) or "tool error") if is_err else None,
                attrs={**attrs, "gen_ai.tool.call.id": _attr(origin, "id")},
            )  # fmt: skip
    elif kind == "documents":
        out_docs = _docs_of(outputs.get("documents") if isinstance(outputs, dict) else None)
        _emit_tool(_API_HAY, mark, name=comp, args=_shrink(_summarize_inputs(inputs)),
                   result=out_docs, error=error, attrs=attrs)  # fmt: skip
    else:
        _emit_custom(
            _API_HAY, mark, name=comp, request=_shrink(_summarize_inputs(inputs)),
            response=_shrink(outputs), error=error, attrs=attrs,
        )  # fmt: skip


def _summarize_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {k: _shrink(v) for k, v in inputs.items()}


def _record_generator(
    comp: str, inputs: dict[str, Any], outputs: Any, error: BaseException | None,
    mark: dict[str, Any] | None, attrs: dict[str, Any],
) -> None:  # fmt: skip
    meta_list = outputs.get("meta") if isinstance(outputs, dict) else None
    meta = meta_list[0] if isinstance(meta_list, list) and meta_list else {}
    meta = meta if isinstance(meta, dict) else {}
    replies = outputs.get("replies") if isinstance(outputs, dict) else None
    first = replies[0] if isinstance(replies, list) and replies else None
    messages = (
        _chat_messages(inputs.get("messages"))
        if "messages" in inputs
        else _chat_messages(inputs.get("prompt"))
    )
    reply = _chat_message(first) if first is not None else None
    if reply is not None:
        reply["role"] = "assistant"
    model = meta.get("model") or inputs.get("model") or comp
    _emit_chat(
        _API_HAY, mark, model=model, messages=messages, reply=reply,
        usage=meta.get("usage"), error=error, attrs=attrs,
    )  # fmt: skip


def _retrievers_of(value: Any) -> set[str] | None:
    return {str(v) for v in value} if value else None


def _wrap_component_method(
    inst: Any,
    comp: str,
    orig: Callable[..., Any],
    *,
    retrieval_tool: str,
    retrievers: set[str] | None,
) -> Callable[..., Any]:
    cls_name = type(inst).__name__
    is_agent = cls_name == "Agent"

    def run_inputs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        if kwargs or not args:
            return dict(kwargs)
        return _call_input(orig, args, kwargs)

    def record(mark: Any, inputs: Any, outputs: Any, error: BaseException | None) -> None:
        if current() is None:
            return
        try:
            _record_component(comp, cls_name, inputs, outputs, error, mark,
                              retrieval_tool=retrieval_tool, retrievers=retrievers)  # fmt: skip
        except Exception as exc:
            _reraise_control(f"Haystack component {comp!r}", exc)

    if inspect.iscoroutinefunction(orig):

        @functools.wraps(orig)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            mark, inputs = _mark(), run_inputs(args, kwargs)
            with _scope(comp if is_agent else None):
                try:
                    outputs = await orig(*args, **kwargs)
                except BaseException as exc:
                    record(mark, inputs, None, exc)
                    raise
                record(mark, inputs, outputs, None)
                return outputs

        return async_wrapper

    @functools.wraps(orig)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        mark, inputs = _mark(), run_inputs(args, kwargs)
        with _scope(comp if is_agent else None):
            try:
                outputs = orig(*args, **kwargs)
            except BaseException as exc:
                record(mark, inputs, None, exc)
                raise
            record(mark, inputs, outputs, None)
            return outputs

    return wrapper


def _pipeline_components(pipeline: Any) -> list[tuple[str, Any]]:
    walk = getattr(pipeline, "walk", None)
    if callable(walk):
        with suppress(Exception):
            return [(str(n), c) for n, c in walk()]
    comps = getattr(pipeline, "components", None)
    if isinstance(comps, dict):
        return [(str(n), c) for n, c in comps.items()]
    return []


def _pipeline_output(result: Any) -> Any:
    if isinstance(result, dict):
        for out in result.values():
            replies = out.get("replies") if isinstance(out, dict) else None
            if isinstance(replies, list) and replies:
                text = _msg_text(replies[0])
                if text:
                    return text
    return _shrink(result)


def wrap_pipeline(
    pipeline: Any,
    *,
    name: str | None = None,
    retrieval_tool: str = _RETRIEVAL_TOOL,
    retrievers: list[str] | None = None,
) -> Any:
    """Record a Haystack ``Pipeline`` / ``AsyncPipeline``: ``run`` (and ``run_async``)
    execute inside :class:`crashlab.record` unless a run is already active, and every
    component run becomes a step.

    Generators (``*Generator``, ``Agent``) are ``chat`` steps with the model and token
    usage from ``meta``; ``*Retriever`` components are recorded as the ``retrieval_tool``
    (default ``search_docs``) with ``[{id, text, score, metadata}]`` so
    :mod:`crashlab.rag` works (name other components with ``retrievers=[...]``);
    ``ToolInvoker`` results are tool steps (errors as failing tool steps); other components
    are ``custom`` steps with bounded inputs / outputs. Wrapping is idempotent; the
    pipeline is returned. ``haystack`` is not imported.
    """
    if pipeline is None:
        raise TypeError("wrap_pipeline(): pipeline is required")
    meta = getattr(pipeline, "metadata", None)
    meta_name = meta.get("name") if isinstance(meta, dict) else None
    label = _agent_name(name, getattr(pipeline, "name", None), meta_name) or "haystack"
    if getattr(pipeline, "__crashlab_haystack_wrapped__", False):
        return pipeline
    rset = _retrievers_of(retrievers)
    for comp, inst in _pipeline_components(pipeline):
        for method in ("run", "run_async"):
            orig = getattr(inst, method, None)
            if not callable(orig) or getattr(orig, "__crashlab_hay_component__", False):
                continue
            wrapped = _wrap_component_method(
                inst, comp, orig, retrieval_tool=retrieval_tool, retrievers=rset
            )
            wrapped.__crashlab_hay_component__ = True  # type: ignore[attr-defined]
            with suppress(Exception):
                setattr(inst, method, wrapped)
    for method in ("run", "run_async"):
        orig = getattr(pipeline, method, None)
        if not callable(orig):
            continue
        with suppress(Exception):
            setattr(pipeline, method, _wrap_pipeline_method(orig, label))
    with suppress(Exception):
        pipeline.__crashlab_haystack_wrapped__ = True
        pipeline.__crashlab_haystack_name__ = label
    return pipeline


def _pipeline_input(orig: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    bound = _call_input(orig, args, kwargs)
    return {k: _shrink(v) for k, v in bound.items()}


def _wrap_pipeline_method(orig: Callable[..., Any], name: str) -> Callable[..., Any]:
    if inspect.iscoroutinefunction(orig):

        @functools.wraps(orig)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with _maybe_record(name, _pipeline_input(orig, args, kwargs)) as run:
                result = await orig(*args, **kwargs)
                if run is not None and run.output is None:
                    run.output = _pipeline_output(result)
                return result

        return async_wrapper

    @functools.wraps(orig)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with _maybe_record(name, _pipeline_input(orig, args, kwargs)) as run:
            result = orig(*args, **kwargs)
            if run is not None and run.output is None:
                run.output = _pipeline_output(result)
            return result

    return wrapper


class _HaystackSpan:
    """Haystack ``tracing.Span`` interface: collects tags (content tags too)."""

    def __init__(self, operation: str, tags: dict[str, Any] | None = None) -> None:
        self.operation = operation
        self.tags: dict[str, Any] = dict(tags or {})

    def set_tag(self, key: str, value: Any) -> None:
        self.tags[key] = value

    def set_tags(self, tags: dict[str, Any]) -> None:
        self.tags.update(tags)

    def set_content_tag(self, key: str, value: Any) -> None:
        self.tags[key] = value

    def raw_span(self) -> Any:
        return self

    def get_correlation_data_for_logs(self) -> dict[str, Any]:
        return {}


class HaystackTracer:
    """Duck-typed Haystack tracer: ``haystack.tracing.enable_tracing(HaystackTracer())``.

    ``haystack.component.run`` spans become steps (the same mapping as
    :func:`wrap_pipeline`), and a ``haystack.pipeline.run`` span starts a
    :class:`crashlab.record` run when none is active. Components already recorded by
    :func:`wrap_pipeline` are not recorded twice. Haystack is not imported.
    """

    def __init__(self, *, name: str = "haystack", retrieval_tool: str = _RETRIEVAL_TOOL,
                 retrievers: list[str] | None = None) -> None:  # fmt: skip
        self.name = name
        self.retrieval_tool = retrieval_tool
        self.retrievers = _retrievers_of(retrievers)
        self.is_content_tracing_enabled = True
        self._stack: list[_HaystackSpan] = []

    def current_span(self) -> _HaystackSpan | None:
        return self._stack[-1] if self._stack else None

    @contextmanager
    def trace(
        self, operation_name: str, tags: dict[str, Any] | None = None, parent_span: Any = None
    ) -> Iterator[_HaystackSpan]:
        span = _HaystackSpan(operation_name, tags)
        mark = _mark()
        self._stack.append(span)
        error: BaseException | None = None
        record_cm = None
        run = None
        try:
            if operation_name == "haystack.pipeline.run" and current() is None:
                data = _shrink(span.tags.get("haystack.pipeline.input_data"))
                record_cm = _maybe_record(self.name, {"data": data})
                run = record_cm.__enter__()
            yield span
        except BaseException as exc:
            error = exc
            raise
        finally:
            if self._stack and self._stack[-1] is span:
                self._stack.pop()
            if operation_name == "haystack.component.run":
                self._record(span, error, mark)
            if record_cm is not None:
                if run is not None and error is None and run.output is None:
                    out = span.tags.get("haystack.pipeline.output_data")
                    if out is not None:
                        run.output = _pipeline_output(out)
                record_cm.__exit__(*sys.exc_info())

    def _record(self, span: _HaystackSpan, error: BaseException | None, mark: Any) -> None:
        ctx = current()
        if ctx is None:
            return
        comp = str(span.tags.get("haystack.component.name") or "component")
        if mark and any(
            s.attributes.get("crashlab.haystack.component") == comp
            for s in _live_steps(ctx)
            if s.idx > mark["last_idx"]
        ):
            return  # wrap_pipeline already recorded this component run
        try:
            _record_component(
                comp,
                str(span.tags.get("haystack.component.type") or comp),
                span.tags.get("haystack.component.input"),
                span.tags.get("haystack.component.output"),
                error,
                mark,
                retrieval_tool=self.retrieval_tool,
                retrievers=self.retrievers,
            )
        except Exception as exc:
            _reraise_control(f"Haystack tracer ({comp})", exc)


__all__ = [
    "HaystackTracer",
    "LlamaIndexHandler",
    "crashlab_step_callback",
    "llama_index_handler",
    "wrap_pipeline",
    "wrap_query_engine",
    "wrap_smolagent",
]
