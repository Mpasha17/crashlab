"""OpenTelemetry GenAI (OTLP/JSON) -> crashlab runs: :func:`crashlab.otel.to_otlp` in reverse."""

from __future__ import annotations

import re
from typing import Any

from ..models import Run
from ._core import (
    IngestError,
    Node,
    assemble,
    chat_request,
    chat_response,
    make_error,
    maybe_json,
    norm_messages,
    raw_finish,
    text_of,
    to_float,
    to_int,
    tool_args,
)
from ._spans import FlatSpan, flatten, group_by_trace, otlp_spans

_CHAT_OPS = {"chat", "text_completion", "generate_content", "completion", "responses"}
_AGENT_OPS = {"invoke_agent", "create_agent", "invoke_workflow", "workflow", "agent"}
_PARAMS = {
    "temperature": "gen_ai.request.temperature",
    "top_p": "gen_ai.request.top_p",
    "top_k": "gen_ai.request.top_k",
    "max_tokens": "gen_ai.request.max_tokens",
    "seed": "gen_ai.request.seed",
    "frequency_penalty": "gen_ai.request.frequency_penalty",
    "presence_penalty": "gen_ai.request.presence_penalty",
}
_MSG_EVENTS = {
    "gen_ai.system.message": "system",
    "gen_ai.user.message": "user",
    "gen_ai.assistant.message": "assistant",
    "gen_ai.tool.message": "tool",
}
_TOOL_ARGS = (
    "gen_ai.tool.call.arguments",
    "gen_ai.tool.arguments",
    "tool.arguments",
    "traceloop.entity.input",
    "input.value",
)
_TOOL_RESULT = (
    "gen_ai.tool.call.result",
    "gen_ai.tool.result",
    "tool.result",
    "traceloop.entity.output",
    "output.value",
)


def _first(attrs: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        if attrs.get(k) is not None:
            return attrs[k]
    return None


def operation_of(sp: FlatSpan) -> str:
    """``chat`` | ``tool`` | ``embeddings`` | ``agent`` | ``skip`` for a GenAI span."""
    a = sp.attrs
    op = str(a.get("gen_ai.operation.name") or "").lower()
    if op in _CHAT_OPS:
        return "chat"
    if op == "execute_tool":
        return "tool"
    if op in ("embeddings", "embedding"):
        return "embeddings"
    if op in _AGENT_OPS:
        return "agent"
    if op == "retrieval":
        return "tool"
    kind = str(a.get("traceloop.span.kind") or "").lower()
    if a.get("gen_ai.tool.name") or kind == "tool":
        return "tool"
    if kind in ("agent", "workflow"):
        return "agent"
    if (
        a.get("gen_ai.request.model")
        or a.get("gen_ai.response.model")
        or a.get("gen_ai.usage.input_tokens") is not None
        or a.get("gen_ai.usage.prompt_tokens") is not None
        or any(k.startswith(("gen_ai.prompt.", "gen_ai.completion.")) for k in a)
    ):
        return "chat"
    if not sp.parent_id:
        return "skip"  # a root span of any kind still names the run
    return "skip"


def _indexed(attrs: dict[str, Any], prefix: str) -> list[dict[str, Any]]:
    """``gen_ai.prompt.0.role`` ... -> ``[{"role": ...}]`` (nested ``tool_calls.M.*`` too)."""
    rows: dict[int, dict[str, Any]] = {}
    for key, val in attrs.items():
        m = re.fullmatch(re.escape(prefix) + r"\.(\d+)\.(.+)", key)
        if m:
            rows.setdefault(int(m.group(1)), {})[m.group(2)] = val
    out = []
    for i in sorted(rows):
        row = rows[i]
        msg: dict[str, Any] = {"role": row.get("role") or "user", "content": row.get("content")}
        calls: dict[int, dict[str, Any]] = {}
        for k, v in row.items():
            m = re.fullmatch(r"tool_calls\.(\d+)\.(\w+)", k)
            if m:
                calls.setdefault(int(m.group(1)), {})[m.group(2)] = v
        if calls:
            msg["tool_calls"] = [
                {
                    "id": c.get("id"),
                    "type": "function",
                    "function": {"name": c.get("name"), "arguments": c.get("arguments")},
                }
                for _, c in sorted(calls.items())
            ]
        if row.get("tool_call_id"):
            msg["tool_call_id"] = row["tool_call_id"]
            msg["role"] = "tool"
        if row.get("finish_reason"):
            msg["finish_reason"] = row["finish_reason"]
        out.append(msg)
    return out


def _event_messages(sp: FlatSpan) -> tuple[list[Any], list[Any]]:
    ins: list[Any] = []
    outs: list[Any] = []
    for ev in sp.events:
        name = ev.get("name")
        at = ev.get("attrs") or {}
        if name in _MSG_EVENTS:
            body = maybe_json(at.get("body") if at.get("body") is not None else at.get("content"))
            msg = dict(body) if isinstance(body, dict) else {"content": body}
            msg.setdefault("role", _MSG_EVENTS[name])
            if name == "gen_ai.tool.message" and at.get("id"):
                msg.setdefault("tool_call_id", at["id"])
            ins.append(msg)
        elif name == "gen_ai.choice":
            body = maybe_json(
                at.get("message") if at.get("message") is not None else at.get("body")
            )
            msg = dict(body.get("message", body)) if isinstance(body, dict) else {"content": body}
            msg.setdefault("role", "assistant")
            if at.get("finish_reason") and "finish_reason" not in msg:
                msg["finish_reason"] = at["finish_reason"]
            outs.append(msg)
    return ins, outs


def _merge_event_attrs(sp: FlatSpan) -> dict[str, Any]:
    """Span attributes plus the payload attributes the newer conventions put on the
    ``gen_ai.client.inference.operation.details`` event."""
    attrs = dict(sp.attrs)
    for ev in sp.events:
        for key, val in (ev.get("attrs") or {}).items():
            if key.startswith(
                (
                    "gen_ai.input.",
                    "gen_ai.output.",
                    "gen_ai.system_instructions",
                    "gen_ai.tool.definitions",
                )
            ):
                attrs.setdefault(key, val)
    return attrs


def _error(sp: FlatSpan, *, tool: bool, crashlab_made: bool) -> Any:
    exc = sp.exception()
    if exc is not None:
        info = make_error(
            exc.get("exception.type") or sp.attrs.get("error.type"),
            exc.get("exception.message") or sp.status_message,
            is_tool_error=bool(exc.get("crashlab.is_tool_error")) or tool,
            traceback=exc.get("exception.stacktrace"),
        )
        return info
    if sp.status_code == "ERROR" and not crashlab_made:
        return make_error(
            sp.attrs.get("error.type") or "SpanError", sp.status_message, is_tool_error=tool
        )
    return None


def _usage(a: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    inp = to_int(a.get("gen_ai.usage.input_tokens", a.get("gen_ai.usage.prompt_tokens")))
    out = to_int(a.get("gen_ai.usage.output_tokens", a.get("gen_ai.usage.completion_tokens")))
    total = to_int(a.get("gen_ai.usage.total_tokens", a.get("llm.usage.total_tokens")))
    return inp, out, total


def genai_node(sp: FlatSpan) -> Node:
    kind = operation_of(sp)
    a = _merge_event_attrs(sp)
    crashlab_made = "crashlab.step" in sp.attrs or "crashlab.run_id" in sp.attrs
    node = Node(
        id=sp.span_id or f"span-{id(sp)}",
        parent=sp.parent_id,
        kind=kind,
        name=sp.name,
        start=sp.start,
        end=sp.end,
    )
    if kind == "chat":
        _fill_chat(node, sp, a)
        node.error = _error(sp, tool=False, crashlab_made=crashlab_made)
    elif kind == "tool":
        name = str(a.get("gen_ai.tool.name") or re.sub(r"^(execute_tool|tool)\s+", "", sp.name))
        node.name = name or "tool"
        args = tool_args(_first(a, _TOOL_ARGS))
        if isinstance(args, dict) and set(args) <= {"args", "kwargs"} and "kwargs" in args:
            args = args["kwargs"] if isinstance(args["kwargs"], dict) else {}
        node.request = {"args": args}
        res = _first(a, _TOOL_RESULT)
        node.error = _error(sp, tool=True, crashlab_made=crashlab_made)
        node.response = None if node.error else {"result": maybe_json(res)}
        node.attrs = _keep(
            a,
            (
                "gen_ai.tool.call.id",
                "gen_ai.tool.name",
                "gen_ai.tool.description",
                "gen_ai.agent.name",
            ),
        )
        node.attrs["gen_ai.tool.name"] = node.name
    elif kind == "embeddings":
        model = a.get("gen_ai.request.model") or a.get("gen_ai.response.model")
        node.name = str(model or "unknown-model")
        node.request = {"params": {"model": model} if model else {}}
        inp, _, total = _usage(a)
        node.response = {"usage": {"input_tokens": inp, "total_tokens": total or inp}}
        node.error = _error(sp, tool=False, crashlab_made=crashlab_made)
        node.attrs = _keep(
            a,
            (
                "gen_ai.system",
                "gen_ai.provider.name",
                "gen_ai.request.model",
                "gen_ai.usage.input_tokens",
            ),
        )
    elif kind in ("agent", "skip"):
        node.name = str(
            a.get("gen_ai.agent.name")
            or re.sub(r"^(invoke_agent|create_agent|invoke_workflow)\s+", "", sp.name)
            or "run"
        )
        msgs_in = norm_messages(a.get("gen_ai.input.messages")) or norm_messages(
            _indexed(a, "gen_ai.prompt") or _event_messages(sp)[0]
        )
        msgs_out = norm_messages(a.get("gen_ai.output.messages")) or norm_messages(
            _indexed(a, "gen_ai.completion") or _event_messages(sp)[1]
        )
        last_user = next((m for m in reversed(msgs_in) if m["role"] == "user"), None)
        if last_user:
            node.input = _msg_text(last_user)
        elif a.get("input.value") is not None:
            node.input = maybe_json(a["input.value"])
        last_asst = next((m for m in reversed(msgs_out) if m["role"] == "assistant"), None)
        if last_asst:
            node.output = _msg_text(last_asst)
        elif a.get("output.value") is not None:
            node.output = maybe_json(a["output.value"])
        node.error = _error(sp, tool=False, crashlab_made=crashlab_made)
        if (
            node.error is None
            and sp.status_code == "ERROR"
            and not crashlab_made
            and sp.exception() is None
        ):
            node.error = make_error("SpanError", sp.status_message)
    return node


def _msg_text(msg: dict[str, Any]) -> str:
    return "".join(
        p.get("content") or "" for p in msg.get("parts") or [] if p.get("type") == "text"
    )


def _keep(a: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {k: a[k] for k in keys if a.get(k) is not None}


def _fill_chat(node: Node, sp: FlatSpan, a: dict[str, Any]) -> None:
    ev_in, ev_out = _event_messages(sp)
    raw_in = maybe_json(a.get("gen_ai.input.messages")) or _indexed(a, "gen_ai.prompt") or ev_in
    raw_out = (
        maybe_json(a.get("gen_ai.output.messages")) or _indexed(a, "gen_ai.completion") or ev_out
    )
    msgs_in, msgs_out = norm_messages(raw_in), norm_messages(raw_out)
    model = a.get("gen_ai.request.model") or a.get("gen_ai.response.model")
    params: dict[str, Any] = {}
    for key, attr in _PARAMS.items():
        v = a.get(attr)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            params[key] = v
    if a.get("gen_ai.request.stop_sequences"):
        params["stop"] = maybe_json(a["gen_ai.request.stop_sequences"])
    tools = maybe_json(a.get("gen_ai.tool.definitions"))
    if not tools:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": t.get("name"),
                    "description": t.get("description"),
                    "parameters": maybe_json(t.get("parameters")),
                },
            }
            for t in _indexed_plain(a, "llm.request.functions")
        ]
    node.request = chat_request(
        msgs_in,
        model=str(model) if model else None,
        tools=tools if isinstance(tools, list) else None,
        system=a.get("gen_ai.system_instructions"),
        params=params,
    )
    node.name = str(model) if model else "unknown-model"
    inp, out, total = _usage(a)
    finish = maybe_json(a.get("gen_ai.response.finish_reasons"))
    finish = (finish[0] if isinstance(finish, list) and finish else finish) or raw_finish(raw_out)
    node.response = chat_response(
        msgs_out,
        model=str(a.get("gen_ai.response.model") or model or "") or None,
        finish=finish,
        inp=inp,
        out=out,
        total=total,
        response_id=a.get("gen_ai.response.id"),
    )
    attrs = _keep(
        a,
        (
            "gen_ai.system",
            "gen_ai.provider.name",
            "gen_ai.request.model",
            "gen_ai.response.model",
            "gen_ai.response.id",
            "gen_ai.agent.name",
            "gen_ai.tool.call.id",
            *_PARAMS.values(),
        ),
    )
    if inp is not None:
        attrs["gen_ai.usage.input_tokens"] = inp
    if out is not None:
        attrs["gen_ai.usage.output_tokens"] = out
    if total is not None and total != (inp or 0) + (out or 0):
        attrs["crashlab.usage.total_tokens"] = total
    cost = to_float(a.get("crashlab.cost_usd"))
    if cost is not None:
        attrs["crashlab.usage.cost_usd"] = cost
    attrs.pop("gen_ai.tool.call.id", None)
    node.attrs = attrs


def _indexed_plain(attrs: dict[str, Any], prefix: str) -> list[dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for key, val in attrs.items():
        m = re.fullmatch(re.escape(prefix) + r"\.(\d+)\.(\w+)", key)
        if m:
            rows.setdefault(int(m.group(1)), {})[m.group(2)] = val
    return [rows[i] for i in sorted(rows)]


def build_runs(spans: list[FlatSpan], fmt: str) -> list[Run]:
    from .openinference import oi_node

    runs: list[Run] = []
    for trace_id, group in group_by_trace(spans).items():
        nodes = []
        for sp in group:
            sp.attrs = flatten(sp.attrs)
            use_oi = "openinference.span.kind" in sp.attrs
            nodes.append(oi_node(sp) if use_oi else genai_node(sp))
        if not any(n.kind in ("chat", "tool", "embeddings") for n in nodes):
            continue  # a trace with no GenAI calls is not an agent run
        runs.append(assemble(nodes, fmt=fmt, trace_id=trace_id))
    if not runs:
        raise IngestError(
            "no GenAI spans found: expected spans with gen_ai.* (or openinference.*) attributes"
        )
    return runs


def from_otlp(data: Any) -> list[Run]:
    """Runs from an OTLP/JSON export: one run per trace id.

    Reads the OpenTelemetry GenAI semantic conventions (``gen_ai.operation.name`` chat /
    execute_tool / invoke_agent, ``gen_ai.request.model``, ``gen_ai.usage.*``,
    ``gen_ai.input.messages`` / ``gen_ai.output.messages``, tool name / arguments /
    result), the older ``gen_ai.prompt.N.*`` / ``gen_ai.completion.N.*`` attributes and
    ``gen_ai.*.message`` events, and spans written by ``crashlab export --format otel``.
    Spans carrying ``openinference.span.kind`` are read as OpenInference.
    """
    from . import load_json_input

    return build_runs(otlp_spans(load_json_input(data)), "otel")


__all__ = ["from_otlp", "genai_node", "operation_of", "text_of"]
