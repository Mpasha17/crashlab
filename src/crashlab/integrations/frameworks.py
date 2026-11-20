"""First-class adapters for OpenAI Agents SDK, PydanticAI and CrewAI.

Zero hard dependency: this module never imports ``agents``, ``pydantic_ai``,
``crewai`` or provider SDKs. Every hook is duck-typed, so the classes and helpers
work (and tests can drive them) when the optional packages are not installed.
``import crashlab.integrations.frameworks`` is therefore safe in a core-only
environment.

- :class:`OpenAIAgentsTracingProcessor` / :func:`crashlab_tracing_processor`
  implement the OpenAI Agents SDK ``TracingProcessor`` interface
  (``on_trace_start``, ``on_trace_end``, ``on_span_start``, ``on_span_end``,
  ``shutdown``, ``force_flush``). Agent, generation, function/tool and handoff
  spans are recorded as crashlab steps with ``gen_ai.agent.name`` attribution.
- :class:`PydanticAIAdapter` / :func:`wrap_pydantic_ai_agent` wrap a
  PydanticAI-like ``Agent`` (``run``, ``run_sync``, ``run_stream``) so each call
  runs inside :class:`crashlab.record`, with tool and usage capture.
- :class:`CrewAICallback` / :func:`wrap_crew` hook CrewAI ``step_callback`` /
  ``task_callback`` and wrap ``crew.kickoff(...)`` inside ``crashlab.record``,
  attributing steps to each agent's role.
"""

from __future__ import annotations

import functools
import inspect
import json
import threading
import time
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from typing import Any

from ..capture._steps import begin_step
from ..capture.agent import record
from ..capture.normalize import to_jsonable
from ..context import agent_scope, current, current_agent
from ..models import utcnow

_API_AGENTS = "openai.agents"
_API_PAI = "pydantic_ai"
_API_CREW = "crewai"


def _warn(prefix: str, exc: BaseException) -> None:
    warnings.warn(
        f"crashlab: {prefix} failed: {type(exc).__name__}: {exc}",
        RuntimeWarning,
        stacklevel=3,
    )


def _live_steps(ctx: Any) -> list[Any]:
    return ctx.steps if getattr(ctx, "steps", None) is not None else ctx.run.steps


def _last_idx(ctx: Any) -> int:
    steps = _live_steps(ctx)
    return steps[-1].idx if steps else -1


def _new_ops(ctx: Any, last_idx: int, operation: str) -> list[Any]:
    return [s for s in _live_steps(ctx) if s.idx > last_idx and s.operation == operation]


def _as_args(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            return {"input": value}
        if isinstance(parsed, dict):
            return {str(k): to_jsonable(v) for k, v in parsed.items()}
        return {"input": to_jsonable(parsed)}
    return {"input": to_jsonable(value)}


def _usage_attrs(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if callable(usage):
        try:
            usage = usage()
        except Exception:
            return {}
    raw: dict[str, Any]
    if isinstance(usage, dict):
        raw = usage
    else:
        raw = {}
        for key in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "prompt_tokens",
            "completion_tokens",
            "request_tokens",
            "response_tokens",
        ):
            if hasattr(usage, key):
                raw[key] = getattr(usage, key)
    inp = raw.get("input_tokens", raw.get("prompt_tokens", raw.get("request_tokens")))
    out = raw.get("output_tokens", raw.get("completion_tokens", raw.get("response_tokens")))
    total = raw.get("total_tokens")
    attrs: dict[str, Any] = {}
    try:
        if inp is not None:
            attrs["gen_ai.usage.input_tokens"] = int(inp)
        if out is not None:
            attrs["gen_ai.usage.output_tokens"] = int(out)
        if total is not None:
            attrs["crashlab.usage.total_tokens"] = int(total)
    except (TypeError, ValueError):
        return {}
    return attrs


def _usage_payload(usage: Any) -> dict[str, Any] | None:
    attrs = _usage_attrs(usage)
    if not attrs:
        if isinstance(usage, dict) and usage:
            return to_jsonable(usage)
        return None
    payload = {
        "input_tokens": attrs.get("gen_ai.usage.input_tokens"),
        "output_tokens": attrs.get("gen_ai.usage.output_tokens"),
        "total_tokens": attrs.get("crashlab.usage.total_tokens"),
    }
    return {k: v for k, v in payload.items() if v is not None}


def _text_of(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content", value.get("text", value.get("output", "")))
        return _text_of(content)
    if isinstance(value, list):
        return "".join(_text_of(item) for item in value)
    content = getattr(value, "content", None)
    if content is not None and content is not value:
        return _text_of(content)
    text = getattr(value, "text", None)
    if isinstance(text, str):
        return text
    return str(value)


def _role_of(message: Any) -> str:
    if isinstance(message, dict):
        typ = message.get("role") or message.get("type") or message.get("kind") or "user"
    else:
        typ = (
            getattr(message, "role", None)
            or getattr(message, "type", None)
            or getattr(message, "kind", None)
            or "user"
        )
    typ = str(typ)
    return {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "developer": "system",
        "tool": "tool",
        "model": "assistant",
        "model_request": "user",
        "model_response": "assistant",
        "request": "user",
        "response": "assistant",
    }.get(typ, typ)


def _as_message(message: Any) -> dict[str, Any]:
    if isinstance(message, str):
        return {"role": "user", "parts": [{"type": "text", "content": message}]}
    role = _role_of(message)
    body = _text_of(message)
    return {"role": role, "parts": [{"type": "text", "content": body}]}


def _split_system(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    rest: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "system":
            texts = [p.get("content", "") for p in msg.get("parts") or [] if isinstance(p, dict)]
            system_parts.append("".join(str(t) for t in texts))
        else:
            rest.append(msg)
    system = "\n\n".join(p for p in system_parts if p) or None
    return system, rest


def _messages_of(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_as_message(value)]
    if isinstance(value, dict) and ("role" in value or "content" in value or "parts" in value):
        return [_as_message(value)]
    if isinstance(value, list):
        out: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, list):
                out.extend(_messages_of(item))
            else:
                out.append(_as_message(item))
        return out
    return [_as_message(value)]


def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        for name in names:
            if name in obj and obj[name] is not None:
                return obj[name]
        return default
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def _span_data(span: Any) -> Any:
    data = getattr(span, "span_data", None)
    if data is not None:
        return data
    export = getattr(span, "export", None)
    if callable(export):
        try:
            dumped = export() or {}
        except Exception:
            dumped = {}
        if isinstance(dumped, dict):
            return dumped.get("span_data", dumped)
    return span


def _span_type(data: Any) -> str:
    typ = _attr(data, "type")
    if typ:
        return str(typ)
    export = getattr(data, "export", None)
    if callable(export):
        try:
            dumped = export() or {}
        except Exception:
            dumped = {}
        if isinstance(dumped, dict) and dumped.get("type"):
            return str(dumped["type"])
    return ""


def _agent_name(*candidates: Any) -> str | None:
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _call_input(fn: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        from ..capture._util import bind_arguments

        return bind_arguments(fn, args, kwargs)
    except Exception:
        return {"args": to_jsonable(list(args)), "kwargs": to_jsonable(kwargs)}


@contextmanager
def _maybe_record(name: str, input: dict[str, Any] | None = None) -> Iterator[Any]:
    """Enter ``crashlab.record`` when no run is active; otherwise yield the current run."""
    ctx = current()
    if ctx is not None:
        yield ctx.run
        return
    with record(name, input=input or {}) as run:
        yield run


def _apply_output(run: Any, result: Any) -> None:
    if run is None or result is None:
        return
    output = _result_output(result)
    if output is not None:
        run.output = output


def _result_output(result: Any) -> Any:
    if result is None:
        return None
    for key in ("output", "data", "final_output", "raw"):
        if isinstance(result, dict) and key in result:
            return to_jsonable(result[key])
        if hasattr(result, key):
            value = getattr(result, key)
            if value is not None and not callable(value):
                return to_jsonable(value)
    return to_jsonable(result)


# ---------------------------------------------------------------------------
# OpenAI Agents SDK
# ---------------------------------------------------------------------------


class OpenAIAgentsTracingProcessor:
    """Duck-typed OpenAI Agents SDK ``TracingProcessor``.

    Records agent, generation, function and handoff spans into the active
    crashlab run (or starts one on ``on_trace_start`` when ``auto_record`` is
    true). Works without the ``agents`` package installed.
    """

    def __init__(self, *, name: str | None = None, auto_record: bool = True) -> None:
        self.name = name
        self.auto_record = auto_record
        self._lock = threading.Lock()
        self._record: record | None = None
        self._pending: dict[str, dict[str, Any]] = {}
        self._scopes: dict[str, Any] = {}

    def on_trace_start(self, trace: Any) -> None:
        try:
            self._on_trace_start(trace)
        except Exception as exc:
            _warn("OpenAI Agents tracing processor on_trace_start", exc)

    def on_trace_end(self, trace: Any) -> None:
        try:
            self._on_trace_end(trace)
        except Exception as exc:
            _warn("OpenAI Agents tracing processor on_trace_end", exc)

    def on_span_start(self, span: Any) -> None:
        try:
            self._on_span_start(span)
        except Exception as exc:
            _warn("OpenAI Agents tracing processor on_span_start", exc)

    def on_span_end(self, span: Any) -> None:
        try:
            self._on_span_end(span)
        except Exception as exc:
            _warn("OpenAI Agents tracing processor on_span_end", exc)

    def shutdown(self) -> None:
        try:
            self._close_record(None)
        except Exception as exc:
            _warn("OpenAI Agents tracing processor shutdown", exc)

    def force_flush(self) -> None:
        return None

    def _span_key(self, span: Any) -> str:
        sid = getattr(span, "span_id", None)
        return str(sid) if sid is not None else f"span-{id(span)}"

    def _on_trace_start(self, trace: Any) -> None:
        if current() is not None or not self.auto_record:
            return
        name = self.name or _agent_name(getattr(trace, "name", None)) or "openai-agents"
        rec = record(name, input={"trace_id": getattr(trace, "trace_id", None)})
        rec.__enter__()
        with self._lock:
            self._record = rec

    def _on_trace_end(self, trace: Any) -> None:
        self._close_record(trace)

    def _close_record(self, trace: Any) -> None:
        with self._lock:
            rec = self._record
            self._record = None
            scopes = list(self._scopes.items())
            self._scopes.clear()
            self._pending.clear()
        for _, cm in reversed(scopes):
            with suppress(Exception):
                cm.__exit__(None, None, None)
        if rec is None:
            return
        try:
            if trace is not None and rec.ctx is not None:
                output = _attr(trace, "output", "final_output")
                if output is not None:
                    rec.ctx.run.output = to_jsonable(output)
            rec.__exit__(None, None, None)
        except Exception as exc:
            _warn("OpenAI Agents tracing processor closing record", exc)

    def _open_scope(self, key: str, name: str) -> None:
        if key in self._scopes:
            return
        cm = agent_scope(name)
        cm.__enter__()
        self._scopes[key] = cm

    def _close_scope(self, key: str) -> None:
        cm = self._scopes.pop(key, None)
        if cm is not None:
            cm.__exit__(None, None, None)

    def _on_span_start(self, span: Any) -> None:
        data = _span_data(span)
        typ = _span_type(data)
        key = self._span_key(span)
        if typ == "agent":
            agent = _agent_name(_attr(data, "name"), self.name)
            if agent:
                with self._lock:
                    self._open_scope(key, agent)
        ctx = current()
        if ctx is None:
            return
        with self._lock:
            self._pending[key] = {
                "type": typ,
                "started_at": utcnow(),
                "t0": time.perf_counter(),
                "last_idx": _last_idx(ctx),
            }

    def _on_span_end(self, span: Any) -> None:
        data = _span_data(span)
        typ = _span_type(data)
        key = self._span_key(span)
        with self._lock:
            pending = self._pending.pop(key, None)
        ctx = current()
        try:
            if ctx is not None:
                last_idx = pending["last_idx"] if pending is not None else _last_idx(ctx)
                started_at = pending["started_at"] if pending is not None else None
                t0 = pending["t0"] if pending is not None else None
                if typ == "agent":
                    self._record_agent(span, data, last_idx, started_at, t0)
                elif typ in {"generation", "response"}:
                    self._record_generation(span, data, typ, last_idx, started_at, t0)
                elif typ == "function":
                    self._record_function(span, data, last_idx, started_at, t0)
                elif typ == "handoff":
                    self._record_handoff(span, data, last_idx, started_at, t0)
        finally:
            if typ == "agent":
                with self._lock:
                    self._close_scope(key)

    def _stamp(self, step: Any, started_at: Any, t0: float | None) -> None:
        if step is None:
            return
        if started_at is not None:
            step.started_at = started_at
        if t0 is not None:
            step.t0 = t0

    def _span_error(self, span: Any) -> Any:
        err = getattr(span, "error", None)
        if not err:
            return None
        if isinstance(err, dict):
            message = err.get("message") or err.get("data") or "span error"
        else:
            message = getattr(err, "message", None) or str(err)
        return RuntimeError(str(message))

    def _record_agent(
        self, span: Any, data: Any, last_idx: int, started_at: Any, t0: float | None
    ) -> None:
        ctx = current()
        if ctx is None or _new_ops(ctx, last_idx, "invoke_agent"):
            return
        name = _agent_name(_attr(data, "name"), self.name) or "agent"
        request = {
            "handoffs": to_jsonable(_attr(data, "handoffs")),
            "tools": to_jsonable(_attr(data, "tools")),
            "output_type": to_jsonable(_attr(data, "output_type")),
        }
        step = begin_step(
            "invoke_agent",
            name,
            request,
            attributes={
                "gen_ai.agent.name": name,
                "crashlab.api": _API_AGENTS,
            },
            hash_input={"agent": name, "request": request},
        )
        self._stamp(step, started_at, t0)
        if step is None:
            return
        exc = self._span_error(span)
        if exc is not None:
            step.fail(exc)
            return
        step.finish({"result": to_jsonable(_attr(data, "output", "metadata", default=name))})

    def _record_generation(
        self,
        span: Any,
        data: Any,
        typ: str,
        last_idx: int,
        started_at: Any,
        t0: float | None,
    ) -> None:
        ctx = current()
        if ctx is None or _new_ops(ctx, last_idx, "chat"):
            return
        model = (
            _agent_name(
                _attr(data, "model"),
                _attr(_attr(data, "response"), "model"),
                typ,
            )
            or "llm"
        )
        raw_input = _attr(data, "input")
        messages = _messages_of(raw_input)
        system, rest = _split_system(messages)
        params = to_jsonable(_attr(data, "model_config") or {}) or {}
        if not isinstance(params, dict):
            params = {"model_config": params}
        params.setdefault("model", model)
        request: dict[str, Any] = {"messages": rest, "params": params}
        if system:
            request["system"] = system
        usage = _attr(data, "usage") or _attr(_attr(data, "response"), "usage")
        attrs = {
            "gen_ai.request.model": model,
            "crashlab.api": _API_AGENTS,
            **_usage_attrs(usage),
        }
        scoped = current_agent()
        if scoped:
            attrs.setdefault("gen_ai.agent.name", scoped)
        step = begin_step(
            "chat",
            model,
            request,
            attributes=attrs,
            hash_input={"model": model, "messages": rest, "system": system},
        )
        self._stamp(step, started_at, t0)
        if step is None:
            return
        exc = self._span_error(span)
        if exc is not None:
            step.fail(exc)
            return
        raw_output = _attr(data, "output")
        if raw_output is None:
            response_obj = _attr(data, "response")
            raw_output = _attr(response_obj, "output", "output_text", "content") or response_obj
        out_messages = _messages_of(raw_output) or [
            {"role": "assistant", "parts": [{"type": "text", "content": _text_of(raw_output)}]}
        ]
        payload: dict[str, Any] = {"messages": out_messages, "model": model}
        usage_payload = _usage_payload(usage)
        if usage_payload:
            payload["usage"] = usage_payload
        step.finish(payload)

    def _record_function(
        self, span: Any, data: Any, last_idx: int, started_at: Any, t0: float | None
    ) -> None:
        ctx = current()
        if ctx is None:
            return
        name = _agent_name(_attr(data, "name"), "tool") or "tool"
        if _new_ops(ctx, last_idx, "execute_tool") or _already_tool(ctx, name):
            return
        args = _as_args(_attr(data, "input"))
        attrs: dict[str, Any] = {
            "gen_ai.tool.name": name,
            "gen_ai.tool.type": "function",
            "crashlab.api": _API_AGENTS,
        }
        scoped = current_agent()
        if scoped:
            attrs.setdefault("gen_ai.agent.name", scoped)
        mcp = _attr(data, "mcp_data")
        if mcp is not None:
            attrs["crashlab.mcp"] = to_jsonable(mcp)
        step = begin_step(
            "execute_tool",
            name,
            {"args": args},
            attributes=attrs,
            hash_input={"tool": name, "args": args},
        )
        self._stamp(step, started_at, t0)
        if step is None:
            return
        exc = self._span_error(span)
        if exc is not None:
            step.fail(
                exc, is_tool_error=True, response={"result": to_jsonable(_attr(data, "output"))}
            )
            return
        step.finish({"result": to_jsonable(_attr(data, "output"))})

    def _record_handoff(
        self, span: Any, data: Any, last_idx: int, started_at: Any, t0: float | None
    ) -> None:
        ctx = current()
        from_agent = _agent_name(_attr(data, "from_agent")) or "?"
        to_agent = _agent_name(_attr(data, "to_agent")) or "?"
        label = f"{from_agent} -> {to_agent}"
        if ctx is None:
            return
        if any(
            s.operation == "custom" and s.name == label and s.idx > last_idx
            for s in _live_steps(ctx)
        ):
            return
        attrs: dict[str, Any] = {
            "crashlab.api": _API_AGENTS,
            "crashlab.handoff.from": from_agent,
            "crashlab.handoff.to": to_agent,
        }
        scoped = current_agent() or (from_agent if from_agent != "?" else None)
        if scoped:
            attrs["gen_ai.agent.name"] = scoped
        request = {"from_agent": from_agent, "to_agent": to_agent}
        step = begin_step(
            "custom",
            label,
            request,
            attributes=attrs,
            hash_input=request,
        )
        self._stamp(step, started_at, t0)
        if step is None:
            return
        exc = self._span_error(span)
        if exc is not None:
            step.fail(exc)
            return
        step.finish({"result": request})


def _already_tool(ctx: Any, name: str) -> bool:
    steps = _live_steps(ctx)
    if not steps:
        return False
    last = steps[-1]
    return last.operation == "execute_tool" and last.name == name


def crashlab_tracing_processor(
    *, name: str | None = None, auto_record: bool = True
) -> OpenAIAgentsTracingProcessor:
    """Return a tracing processor for ``agents.add_trace_processor(...)``.

    ``name`` is used when the processor starts its own ``crashlab.record`` (when
    ``auto_record`` is true and no run is already active) and as a fallback
    agent name. The processor is duck-typed: it does not require ``agents``.
    """
    return OpenAIAgentsTracingProcessor(name=name, auto_record=auto_record)


# ---------------------------------------------------------------------------
# PydanticAI
# ---------------------------------------------------------------------------


_TOOL_FN_ATTRS = ("function", "fn", "callback", "call", "_function", "func")


def _iter_function_tools(agent: Any) -> list[tuple[Any, Any, str]]:
    found: list[tuple[Any, Any, str]] = []
    containers = []
    for attr in ("_function_tools", "function_tools", "_tools", "tools"):
        value = getattr(agent, attr, None)
        if value:
            containers.append(value)
    seen: set[int] = set()
    for container in containers:
        items: list[Any]
        if isinstance(container, dict):
            items = list(container.items())
        elif isinstance(container, (list, tuple)):
            items = [(None, tool) for tool in container]
        else:
            continue
        for key, tool in items:
            ident = id(tool)
            if ident in seen:
                continue
            seen.add(ident)
            name = (
                _agent_name(
                    key if isinstance(key, str) else None,
                    _attr(tool, "name", "__name__"),
                )
                or "tool"
            )
            found.append((container, tool, name))
    return found


def _tool_callable(tool: Any) -> tuple[str, Any] | None:
    for attr in _TOOL_FN_ATTRS:
        fn = getattr(tool, attr, None)
        if callable(fn):
            return attr, fn
    if callable(tool):
        return "", tool
    return None


def _tool_args(
    fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    bound = _call_input(fn, args, kwargs)
    if set(bound) <= {"args", "kwargs"}:
        raw: dict[str, Any] = {}
        positional = bound.get("args")
        if isinstance(positional, dict):
            raw.update(positional)
        elif positional not in (None, [], ()):
            raw["input"] = to_jsonable(positional)
        extra = bound.get("kwargs")
        if isinstance(extra, dict):
            raw.update(extra)
        return raw
    return {k: v for k, v in bound.items() if k not in {"self", "cls"}}


def _wrap_callable_tool(fn: Callable[..., Any], name: str) -> Callable[..., Any]:
    if getattr(fn, "__crashlab_tool__", False) or getattr(fn, "__crashlab_framework_tool__", False):
        return fn

    def _start(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        raw_args = _tool_args(fn, args, kwargs)
        return begin_step(
            "execute_tool",
            name,
            {"args": raw_args if isinstance(raw_args, dict) else _as_args(raw_args)},
            attributes={
                "gen_ai.tool.name": name,
                "gen_ai.tool.type": "function",
                "crashlab.api": _API_PAI,
            },
            hash_input={"tool": name, "args": raw_args},
        )

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            pending = _start(args, kwargs)
            try:
                result = await fn(*args, **kwargs)
            except BaseException as exc:
                if pending is not None:
                    pending.fail(exc, is_tool_error=True)
                raise
            if pending is not None:
                pending.finish({"result": to_jsonable(result)})
            return result

        async_wrapper.__crashlab_framework_tool__ = True  # type: ignore[attr-defined]
        return async_wrapper

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        pending = _start(args, kwargs)
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:
            if pending is not None:
                pending.fail(exc, is_tool_error=True)
            raise
        if pending is not None:
            pending.finish({"result": to_jsonable(result)})
        return result

    wrapper.__crashlab_framework_tool__ = True  # type: ignore[attr-defined]
    return wrapper


def _install_tool_wrappers(agent: Any) -> None:
    for _container, tool, name in _iter_function_tools(agent):
        found = _tool_callable(tool)
        if found is None:
            continue
        attr, fn = found
        wrapped = _wrap_callable_tool(fn, name)
        if wrapped is fn:
            continue
        if attr:
            with suppress(Exception):
                setattr(tool, attr, wrapped)
        elif callable(tool):
            continue


def _wrap_model_request(model: Any) -> None:
    if model is None or getattr(model, "__crashlab_framework_model__", False):
        return
    orig = getattr(model, "request", None)
    if not callable(orig):
        return
    model_name = _agent_name(_attr(model, "model_name", "model", "name"), "pydantic-ai") or "llm"

    def _chat_step(messages: Any) -> Any:
        normalized = _messages_of(messages)
        system, rest = _split_system(normalized)
        request: dict[str, Any] = {"messages": rest, "params": {"model": model_name}}
        if system:
            request["system"] = system
        return begin_step(
            "chat",
            model_name,
            request,
            attributes={"gen_ai.request.model": model_name, "crashlab.api": _API_PAI},
            hash_input={"model": model_name, "messages": rest, "system": system},
        )

    def _finish_chat(pending: Any, result: Any, exc: BaseException | None) -> None:
        if pending is None:
            return
        if exc is not None:
            pending.fail(exc)
            return
        text = _text_of(result)
        usage = _attr(result, "usage")
        attrs = _usage_attrs(usage)
        pending.finish(
            {
                "messages": [{"role": "assistant", "parts": [{"type": "text", "content": text}]}],
                "model": model_name,
                **({"usage": _usage_payload(usage)} if _usage_payload(usage) else {}),
            },
            attributes=attrs or None,
        )

    if inspect.iscoroutinefunction(orig):

        @functools.wraps(orig)
        async def arequest(*args: Any, **kwargs: Any) -> Any:
            messages = args[0] if args else kwargs.get("messages")
            pending = _chat_step(messages)
            try:
                result = await orig(*args, **kwargs)
            except BaseException as exc:
                _finish_chat(pending, None, exc)
                raise
            _finish_chat(pending, result, None)
            return result

        try:
            model.request = arequest  # type: ignore[method-assign]
            model.__crashlab_framework_model__ = True
        except Exception:
            return
        return

    @functools.wraps(orig)
    def request(*args: Any, **kwargs: Any) -> Any:
        messages = args[0] if args else kwargs.get("messages")
        pending = _chat_step(messages)
        try:
            result = orig(*args, **kwargs)
        except BaseException as exc:
            _finish_chat(pending, None, exc)
            raise
        _finish_chat(pending, result, None)
        return result

    try:
        model.request = request  # type: ignore[method-assign]
        model.__crashlab_framework_model__ = True
    except Exception:
        return


def _message_parts(message: Any) -> list[Any]:
    parts = _attr(message, "parts")
    if isinstance(parts, list):
        return parts
    return [message]


def _synthesize_from_result(result: Any, last_idx: int) -> None:
    """If the run captured no steps, rebuild chat/tool calls from result messages."""
    ctx = current()
    if ctx is None or _last_idx(ctx) > last_idx:
        return
    messages = None
    for key in ("all_messages", "new_messages", "messages"):
        value = getattr(result, key, None) if result is not None else None
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        if value:
            messages = value
            break
    if not messages:
        return
    pending_calls: list[tuple[str, Any]] = []
    for message in messages:
        kind = str(_attr(message, "kind", "type", default="") or "").lower()
        parts = _message_parts(message)
        is_response = kind in {"model_response", "response", "assistant", "model"} or any(
            str(_attr(p, "part_kind", "kind", "type", default="") or "").lower()
            in {"tool-call", "tool_call", "text"}
            and _attr(p, "tool_name", "tool")
            for p in parts
        )
        texts: list[str] = []
        tool_calls: list[tuple[str, Any]] = []
        tool_returns: list[tuple[str, Any]] = []
        for part in parts:
            pkind = str(_attr(part, "part_kind", "kind", "type", default="") or "").lower()
            tname = _attr(part, "tool_name", "tool", "name")
            if pkind in {"tool-call", "tool_call"} or (
                tname and pkind in {"tool-call-part", ""} and _attr(part, "args") is not None
            ):
                tool_calls.append((str(tname or "tool"), _attr(part, "args", "arguments") or {}))
            elif pkind in {"tool-return", "tool_return", "tool-call-response", "tool_result"} or (
                tname and _attr(part, "content", "result") is not None and pkind
            ):
                tool_returns.append(
                    (str(tname or "tool"), _attr(part, "content", "result", "output"))
                )
            else:
                text = _text_of(_attr(part, "content", "text") or part)
                if text:
                    texts.append(text)
        if tool_returns:
            for tname, output in tool_returns:
                step = begin_step(
                    "execute_tool",
                    tname,
                    {"args": pending_calls.pop(0)[1] if pending_calls else {}},
                    attributes={
                        "gen_ai.tool.name": tname,
                        "gen_ai.tool.type": "function",
                        "crashlab.api": _API_PAI,
                    },
                )
                if step is not None:
                    step.finish({"result": to_jsonable(output)})
        if is_response or tool_calls or (texts and kind in {"model_response", "response", "ai"}):
            body = "".join(texts)
            step = begin_step(
                "chat",
                "pydantic-ai",
                {
                    "messages": [{"role": "user", "parts": [{"type": "text", "content": ""}]}],
                    "params": {},
                },
                attributes={"crashlab.api": _API_PAI, **_usage_attrs(_attr(result, "usage"))},
            )
            if step is not None:
                payload: dict[str, Any] = {
                    "messages": [
                        {"role": "assistant", "parts": [{"type": "text", "content": body}]}
                    ],
                }
                step.finish(payload)
            pending_calls.extend(tool_calls)


def _after_pai_result(run: Any, result: Any, last_idx: int) -> Any:
    _synthesize_from_result(result, last_idx)
    _apply_output(run, result)
    # if usage exists but no chat step carried it, stamp the last chat step
    ctx = current()
    usage = _attr(result, "usage") if result is not None else None
    attrs = _usage_attrs(usage)
    if ctx is not None and attrs:
        chats = [s for s in _live_steps(ctx) if s.idx > last_idx and s.operation == "chat"]
        if chats and "gen_ai.usage.input_tokens" not in (chats[-1].attributes or {}):
            chats[-1].attributes.update(attrs)
            payload = _usage_payload(usage)
            if payload and isinstance(chats[-1].response, dict):
                chats[-1].response.setdefault("usage", payload)
    return result


def _wrap_pai_method(orig: Callable[..., Any], name: str, method: str) -> Callable[..., Any]:
    if method in {"run_stream", "run_stream_sync"}:

        @functools.wraps(orig)
        def stream_wrapper(*args: Any, **kwargs: Any) -> Any:
            return _wrap_stream_result(orig(*args, **kwargs), name, _call_input(orig, args, kwargs))

        return stream_wrapper

    if inspect.iscoroutinefunction(orig):

        @functools.wraps(orig)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with _maybe_record(name, _call_input(orig, args, kwargs)) as run:
                last_idx = _last_idx(current()) if current() is not None else -1
                result = await orig(*args, **kwargs)
                return _after_pai_result(run, result, last_idx)

        return async_wrapper

    if inspect.isasyncgenfunction(orig):

        @functools.wraps(orig)
        async def agen_wrapper(*args: Any, **kwargs: Any) -> Any:
            with _maybe_record(name, _call_input(orig, args, kwargs)) as run:
                last_idx = _last_idx(current()) if current() is not None else -1
                items = []
                async for item in orig(*args, **kwargs):
                    items.append(item)
                    yield item
                _after_pai_result(run, items[-1] if items else None, last_idx)

        return agen_wrapper

    @functools.wraps(orig)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with _maybe_record(name, _call_input(orig, args, kwargs)) as run:
            last_idx = _last_idx(current()) if current() is not None else -1
            result = orig(*args, **kwargs)
            return _after_pai_result(run, result, last_idx)

    return wrapper


def _wrap_stream_result(inner: Any, name: str, input: dict[str, Any]) -> Any:
    if inspect.isasyncgen(inner):

        async def gen() -> Any:
            ctx = current()
            last_idx = _last_idx(ctx) if ctx is not None else -1
            with _maybe_record(name, input) as run:
                item = None
                async for item in inner:
                    yield item
                _after_pai_result(run, item, last_idx)

        return gen()

    if hasattr(inner, "__aenter__"):

        class _AsyncCM:
            async def __aenter__(self) -> Any:
                self._cm = _maybe_record(name, input)
                self._run = self._cm.__enter__()
                self._last_idx = _last_idx(current()) if current() is not None else -1
                self._result = await inner.__aenter__()
                return self._result

            async def __aexit__(self, *exc: Any) -> Any:
                try:
                    _after_pai_result(self._run, getattr(self, "_result", None), self._last_idx)
                    inner_suppress = await inner.__aexit__(*exc)
                finally:
                    rec_suppress = self._cm.__exit__(*exc)
                return bool(inner_suppress) or bool(rec_suppress)

        return _AsyncCM()

    if hasattr(inner, "__enter__"):

        class _SyncCM:
            def __enter__(self) -> Any:
                self._cm = _maybe_record(name, input)
                self._run = self._cm.__enter__()
                self._last_idx = _last_idx(current()) if current() is not None else -1
                self._result = inner.__enter__()
                return self._result

            def __exit__(self, *exc: Any) -> Any:
                try:
                    _after_pai_result(self._run, getattr(self, "_result", None), self._last_idx)
                    inner_suppress = inner.__exit__(*exc)
                finally:
                    rec_suppress = self._cm.__exit__(*exc)
                return bool(inner_suppress) or bool(rec_suppress)

        return _SyncCM()
    return inner


class PydanticAIAdapter:
    """Wrap a PydanticAI-like ``Agent`` so runs are recorded by crashlab.

    The adapter is purely duck-typed: the object needs ``run``, ``run_sync``
    and/or ``run_stream`` (whatever is present is wrapped). Function tools on
    ``agent._function_tools`` (or ``tools``) and a model ``request`` method are
    wrapped in place when they exist. ``wrap()`` is idempotent.
    """

    def __init__(self, agent: Any | None = None, *, name: str | None = None) -> None:
        self.agent = agent
        self.name = name

    def wrap(self, agent: Any | None = None, *, name: str | None = None) -> Any:
        target = self.agent if agent is None else agent
        if target is None:
            raise TypeError("PydanticAIAdapter.wrap(): agent is required")
        label = name if name is not None else self.name
        return wrap_pydantic_ai_agent(target, name=label)


def wrap_pydantic_ai_agent(agent: Any, *, name: str | None = None) -> Any:
    """Return ``agent`` with ``run`` / ``run_sync`` / ``run_stream`` recorded.

    Each method runs inside :class:`~crashlab.capture.agent.record` when no run
    is already active (nested ``record()`` is not supported). Tool calls and
    model requests are captured when the agent exposes ``_function_tools`` or a
    model with ``request``. Wrapping is idempotent.
    """
    if agent is None:
        raise TypeError("wrap_pydantic_ai_agent(): agent is required")
    label = _agent_name(name, getattr(agent, "name", None)) or "pydantic-ai"
    if getattr(agent, "__crashlab_pydantic_wrapped__", False):
        return agent
    _install_tool_wrappers(agent)
    _wrap_model_request(getattr(agent, "model", None))
    for method in ("run", "run_sync", "run_stream", "run_stream_sync"):
        orig = getattr(agent, method, None)
        if not callable(orig):
            continue
        with suppress(Exception):
            setattr(agent, method, _wrap_pai_method(orig, label, method))
    with suppress(Exception):
        agent.__crashlab_pydantic_wrapped__ = True
        agent.__crashlab_pydantic_name__ = label
    return agent


# ---------------------------------------------------------------------------
# CrewAI
# ---------------------------------------------------------------------------


def _parse_tool_input(value: Any) -> dict[str, Any]:
    return _as_args(value)


def _output_role(output: Any, fallback: str | None = None) -> str | None:
    return _agent_name(
        current_agent(),
        _attr(output, "agent", "role", "agent_role"),
        fallback,
    )


def _looks_like_action(output: Any) -> bool:
    return _attr(output, "tool") not in (None, "") and (
        hasattr(output, "tool") or (isinstance(output, dict) and "tool" in output)
    )


def _looks_like_finish(output: Any) -> bool:
    return _attr(output, "output") is not None and not _looks_like_action(output)


class CrewAICallback:
    """CrewAI ``step_callback`` / ``task_callback`` hooks that record into crashlab.

    Pass ``callback.step_callback`` and ``callback.task_callback`` to a ``Crew``
    (or use :func:`wrap_crew`, which installs them and wraps ``kickoff``). Tool
    actions are ``execute_tool`` steps; task outputs are custom steps labelled
    with the agent's role. Already-recorded tool calls (for example from
    ``@crashlab.tool``) are not duplicated.
    """

    def step_callback(self, output: Any) -> None:
        try:
            self._on_step(output)
        except Exception as exc:
            _warn("CrewAI step_callback", exc)

    def task_callback(self, output: Any) -> None:
        try:
            self._on_task(output)
        except Exception as exc:
            _warn("CrewAI task_callback", exc)

    def bind_agent(
        self, role: str, existing: Callable[..., Any] | None = None
    ) -> Callable[[Any], Any]:
        """A ``step_callback`` that attributes every step to ``role``."""

        def bound(output: Any) -> Any:
            with agent_scope(role):
                self.step_callback(output)
            if existing is not None and existing is not bound:
                return existing(output)
            return None

        bound.__crashlab_crew_bound_role__ = role  # type: ignore[attr-defined]
        return bound

    def _on_step(self, output: Any) -> None:
        ctx = current()
        if ctx is None or output is None:
            return
        role = _output_role(output)
        if _looks_like_action(output):
            name = _agent_name(_attr(output, "tool"), "tool") or "tool"
            if _already_tool(ctx, name):
                return
            args = _parse_tool_input(_attr(output, "tool_input", "input", "args"))
            attrs: dict[str, Any] = {
                "gen_ai.tool.name": name,
                "gen_ai.tool.type": "function",
                "crashlab.api": _API_CREW,
            }
            if role:
                attrs["gen_ai.agent.name"] = role
            step = begin_step(
                "execute_tool",
                name,
                {"args": args},
                attributes=attrs,
                hash_input={"tool": name, "args": args},
            )
            if step is None:
                return
            err = _attr(output, "error")
            result = _attr(output, "result", "output")
            if err:
                exc = err if isinstance(err, BaseException) else RuntimeError(str(err))
                step.fail(exc, is_tool_error=True, response={"result": to_jsonable(result)})
                return
            step.finish({"result": to_jsonable(result)})
            return
        if _looks_like_finish(output):
            label = role or "finish"
            attrs = {"crashlab.api": _API_CREW}
            if role:
                attrs["gen_ai.agent.name"] = role
            step = begin_step(
                "custom",
                label,
                {"thought": to_jsonable(_attr(output, "thought"))},
                attributes=attrs,
                hash_input={"agent": label, "output": to_jsonable(_attr(output, "output"))},
            )
            if step is not None:
                step.finish({"result": to_jsonable(_attr(output, "output", "text"))})

    def _on_task(self, output: Any) -> None:
        ctx = current()
        if ctx is None or output is None:
            return
        role = _output_role(output)
        label = _agent_name(_attr(output, "name", "description"), role, "task") or "task"
        attrs: dict[str, Any] = {"crashlab.api": _API_CREW, "crashlab.crew.task": label}
        if role:
            attrs["gen_ai.agent.name"] = role
        step = begin_step(
            "custom",
            label,
            {
                "description": to_jsonable(_attr(output, "description")),
                "expected_output": to_jsonable(_attr(output, "expected_output")),
            },
            attributes=attrs,
            hash_input={"task": label, "agent": role},
        )
        if step is not None:
            step.finish(
                {"result": to_jsonable(_attr(output, "raw", "output", "pydantic", "json_dict"))}
            )


def _chain_callback(existing: Any, hook: Callable[..., Any]) -> Callable[..., Any]:
    if existing is None or existing is hook:
        return hook

    def chained(*args: Any, **kwargs: Any) -> Any:
        hook(*args, **kwargs)
        return existing(*args, **kwargs)

    chained.__crashlab_chained_callback__ = True  # type: ignore[attr-defined]
    return chained


def _wrap_agent_execution(agent: Any, role: str) -> None:
    for method in ("execute_task", "execute", "kickoff"):
        orig = getattr(agent, method, None)
        if not callable(orig) or getattr(orig, "__crashlab_crew_scoped__", False):
            continue
        if inspect.iscoroutinefunction(orig):

            @functools.wraps(orig)
            async def async_wrapper(
                *args: Any, _orig: Any = orig, _role: str = role, **kwargs: Any
            ) -> Any:
                with agent_scope(_role):
                    return await _orig(*args, **kwargs)

            async_wrapper.__crashlab_crew_scoped__ = True  # type: ignore[attr-defined]
            try:
                setattr(agent, method, async_wrapper)
            except Exception:
                continue
            continue

        @functools.wraps(orig)
        def wrapper(*args: Any, _orig: Any = orig, _role: str = role, **kwargs: Any) -> Any:
            with agent_scope(_role):
                return _orig(*args, **kwargs)

        wrapper.__crashlab_crew_scoped__ = True  # type: ignore[attr-defined]
        try:
            setattr(agent, method, wrapper)
        except Exception:
            continue


def _wrap_kickoff(name: str, orig: Callable[..., Any]) -> Callable[..., Any]:
    if inspect.iscoroutinefunction(orig):

        @functools.wraps(orig)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with _maybe_record(name, _call_input(orig, args, kwargs)) as run:
                result = await orig(*args, **kwargs)
                _apply_output(run, result)
                return result

        async_wrapper.__crashlab_crew_kickoff__ = True  # type: ignore[attr-defined]
        return async_wrapper

    @functools.wraps(orig)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with _maybe_record(name, _call_input(orig, args, kwargs)) as run:
            result = orig(*args, **kwargs)
            _apply_output(run, result)
            return result

    wrapper.__crashlab_crew_kickoff__ = True  # type: ignore[attr-defined]
    return wrapper


def wrap_crew(crew: Any, *, name: str | None = None) -> Any:
    """Install crashlab hooks on a CrewAI-like ``crew`` and wrap ``kickoff``.

    Sets per-agent ``step_callback`` hooks (attributed to each agent's ``role``),
    a crew-level ``task_callback``, and wraps ``kickoff`` / ``kickoff_async``
    (and other ``kickoff*`` methods) so the execution runs inside
    :class:`~crashlab.capture.agent.record` when no run is already active.
    Wrapping is idempotent. The ``crewai`` package is not required.
    """
    if crew is None:
        raise TypeError("wrap_crew(): crew is required")
    label = _agent_name(name, getattr(crew, "name", None)) or "crewai"
    callback = getattr(crew, "__crashlab_crew_callback__", None)
    if not isinstance(callback, CrewAICallback):
        callback = CrewAICallback()
        with suppress(Exception):
            crew.__crashlab_crew_callback__ = callback
    already = bool(getattr(crew, "__crashlab_crew_wrapped__", False))
    for agent in getattr(crew, "agents", None) or []:
        role = _agent_name(getattr(agent, "role", None), getattr(agent, "name", None))
        if not role:
            continue
        existing = getattr(agent, "step_callback", None)
        if not getattr(existing, "__crashlab_crew_bound_role__", None):
            with suppress(Exception):
                agent.step_callback = callback.bind_agent(role, existing)
        _wrap_agent_execution(agent, role)
    if not already:
        with suppress(Exception):
            crew.step_callback = _chain_callback(
                getattr(crew, "step_callback", None), callback.step_callback
            )
        with suppress(Exception):
            crew.task_callback = _chain_callback(
                getattr(crew, "task_callback", None), callback.task_callback
            )
        for method in (
            "kickoff",
            "kickoff_async",
            "akickoff",
            "kickoff_for_each",
            "kickoff_for_each_async",
        ):
            orig = getattr(crew, method, None)
            if not callable(orig) or getattr(orig, "__crashlab_crew_kickoff__", False):
                continue
            with suppress(Exception):
                setattr(crew, method, _wrap_kickoff(label, orig))
        with suppress(Exception):
            crew.__crashlab_crew_wrapped__ = True
            crew.__crashlab_crew_name__ = label
    return crew


__all__ = [
    "CrewAICallback",
    "OpenAIAgentsTracingProcessor",
    "PydanticAIAdapter",
    "crashlab_tracing_processor",
    "wrap_crew",
    "wrap_pydantic_ai_agent",
]
