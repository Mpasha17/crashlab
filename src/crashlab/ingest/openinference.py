"""OpenInference / Arize Phoenix spans -> crashlab runs."""

from __future__ import annotations

import re
from typing import Any

from ..models import Run
from ._core import (
    IngestError,
    Node,
    chat_request,
    chat_response,
    docs_result,
    make_error,
    maybe_json,
    norm_messages,
    parse_time,
    to_int,
    tool_args,
)
from ._spans import FlatSpan, decode_attrs, flatten, group_by_trace, otlp_spans

_PARAMS = (
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "seed",
    "frequency_penalty",
    "presence_penalty",
)
_AGENT_KINDS = {"AGENT"}
_STEP_KINDS = {"LLM": "chat", "TOOL": "tool", "RETRIEVER": "tool", "EMBEDDING": "embeddings"}


def _rows(a: dict[str, Any], prefix: str) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for key, val in a.items():
        m = re.fullmatch(re.escape(prefix) + r"\.(\d+)\.(.+)", key)
        if m:
            rows.setdefault(int(m.group(1)), {})[m.group(2)] = val
    return rows


def _messages(a: dict[str, Any], prefix: str) -> list[dict[str, Any]]:
    """``llm.input_messages.N.message.*`` -> OpenAI-style message dicts."""
    out = []
    for _, row in sorted(_rows(a, prefix).items()):
        role = row.get("message.role") or "user"
        content = row.get("message.content")
        if content is None:  # multi-part ``message.contents.J.message_content.text``
            texts = [
                v
                for _, p in sorted(_rows(row, "message.contents").items())
                for k, v in p.items()
                if k == "message_content.text"
            ]
            content = "".join(str(t) for t in texts) if texts else None
        msg: dict[str, Any] = {"role": role, "content": content}
        calls = []
        for _, tc in sorted(_rows(row, "message.tool_calls").items()):
            calls.append(
                {
                    "id": tc.get("tool_call.id"),
                    "type": "function",
                    "function": {
                        "name": tc.get("tool_call.function.name"),
                        "arguments": tc.get("tool_call.function.arguments"),
                    },
                }
            )
        if calls:
            msg["tool_calls"] = calls
        if row.get("message.tool_call_id"):
            msg["tool_call_id"] = row["message.tool_call_id"]
            msg["role"] = "tool"
        if row.get("message.name"):
            msg["name"] = row["message.name"]
        out.append(msg)
    return out


def _error(sp: FlatSpan, *, tool: bool) -> Any:
    exc = sp.exception()
    if exc is not None:
        return make_error(
            exc.get("exception.type"),
            exc.get("exception.message") or sp.status_message,
            is_tool_error=tool,
            traceback=exc.get("exception.stacktrace"),
        )
    if sp.status_code == "ERROR":
        return make_error("SpanError", sp.status_message, is_tool_error=tool)
    return None


def oi_node(sp: FlatSpan) -> Node:
    a = sp.attrs
    span_kind = str(a.get("openinference.span.kind") or "").upper()
    kind = _STEP_KINDS.get(span_kind) or ("agent" if span_kind in _AGENT_KINDS else "skip")
    node = Node(
        id=sp.span_id or f"span-{id(sp)}",
        parent=sp.parent_id,
        kind=kind,
        name=sp.name,
        start=sp.start,
        end=sp.end,
    )
    if kind == "chat":
        model = a.get("llm.model_name") or a.get("llm.response.model")
        msgs_in = norm_messages(_messages(a, "llm.input_messages"))
        if not msgs_in and a.get("input.value") is not None:
            msgs_in = norm_messages(maybe_json(a["input.value"]))
        msgs_out = norm_messages(_messages(a, "llm.output_messages"))
        if not msgs_out and a.get("output.value") is not None:
            ov = maybe_json(a["output.value"])
            if isinstance(ov, dict) and isinstance(ov.get("choices"), list):
                msgs_out = norm_messages(
                    [c.get("message") for c in ov["choices"] if isinstance(c, dict)]
                )
            else:
                msgs_out = norm_messages(
                    [{"role": "assistant", "content": ov if isinstance(ov, str) else str(ov)}]
                )
        invocation = maybe_json(a.get("llm.invocation_parameters"))
        params = {
            k: invocation[k]
            for k in _PARAMS
            if isinstance(invocation, dict) and invocation.get(k) is not None
        }
        tools = [
            maybe_json(t.get("tool.json_schema"))
            for _, t in sorted(_rows(a, "llm.tools").items())
            if t.get("tool.json_schema")
        ]
        node.request = chat_request(
            msgs_in,
            model=str(model) if model else None,
            tools=[t for t in tools if isinstance(t, dict)],
            params=params,
        )
        node.name = str(model) if model else "unknown-model"
        inp = to_int(a.get("llm.token_count.prompt"))
        out = to_int(a.get("llm.token_count.completion"))
        total = to_int(a.get("llm.token_count.total"))
        node.response = chat_response(
            msgs_out, model=str(model) if model else None, inp=inp, out=out, total=total
        )
        attrs: dict[str, Any] = {}
        if model:
            attrs["gen_ai.request.model"] = str(model)
            attrs["gen_ai.response.model"] = str(model)
        prov = a.get("llm.provider") or a.get("llm.system")
        if prov:
            attrs["gen_ai.provider.name"] = str(prov)
        if inp is not None:
            attrs["gen_ai.usage.input_tokens"] = inp
        if out is not None:
            attrs["gen_ai.usage.output_tokens"] = out
        if total is not None and total != (inp or 0) + (out or 0):
            attrs["crashlab.usage.total_tokens"] = total
        for k, v in params.items():
            if k in _PARAMS and isinstance(v, (int, float)) and not isinstance(v, bool):
                attrs[f"gen_ai.request.{k}"] = v
        node.attrs = attrs
        node.error = _error(sp, tool=False)
    elif kind == "tool":
        if span_kind == "RETRIEVER":
            docs = [d for _, d in sorted(_rows(a, "retrieval.documents").items())]
            node.request = {"args": {"query": text_or_none(a.get("input.value"))}}
            node.error = _error(sp, tool=True)
            node.response = (
                None
                if node.error
                else {
                    "result": docs_result(
                        [{k.removeprefix("document."): v for k, v in d.items()} for d in docs]
                    )
                }
            )
            node.name = sp.name or "retriever"
        else:
            node.name = str(
                a.get("tool.name") or a.get("tool_call.function.name") or sp.name or "tool"
            )
            raw = a.get("tool_call.function.arguments", a.get("input.value"))
            node.request = {"args": tool_args(raw)}
            node.error = _error(sp, tool=True)
            res = a.get("output.value")
            node.response = None if node.error else {"result": maybe_json(res)}
            if a.get("tool_call.id"):
                node.attrs["gen_ai.tool.call.id"] = str(a["tool_call.id"])
        node.attrs["gen_ai.tool.name"] = node.name
    elif kind == "embeddings":
        model = a.get("embedding.model_name")
        node.name = str(model or "unknown-model")
        node.request = {"params": {"model": model} if model else {}}
        node.response = {"usage": {"input_tokens": to_int(a.get("llm.token_count.prompt"))}}
        node.error = _error(sp, tool=False)
    else:
        node.input = maybe_json(a.get("input.value")) if a.get("input.value") is not None else None
        node.output = (
            maybe_json(a.get("output.value")) if a.get("output.value") is not None else None
        )
        node.error = _error(sp, tool=False)
        if kind == "agent":
            node.name = str(a.get("agent.name") or sp.name or "agent")
    return node


def text_or_none(v: Any) -> Any:
    v = maybe_json(v)
    return v if v not in ("", None) else None


def _phoenix_span(rec: dict[str, Any]) -> FlatSpan:
    """One Phoenix span: nested (``{"context": {...}, "attributes": {...}}``) or a
    DataFrame ``to_json(orient="records")`` row (``context.span_id``, ``attributes.llm...``)."""
    ctx = rec.get("context") if isinstance(rec.get("context"), dict) else {}
    attrs: dict[str, Any] = {}
    if isinstance(rec.get("attributes"), dict):
        attrs.update(flatten(rec["attributes"]))
    for key, val in rec.items():
        if key.startswith("attributes."):
            attrs[key[len("attributes.") :]] = val
    attrs = {k: (v if not hasattr(v, "tolist") else v.tolist()) for k, v in attrs.items()}
    # flattened rows may hold whole lists of dicts under one key
    attrs = flatten(attrs)
    if "openinference.span.kind" not in attrs:
        sk = rec.get("span_kind") or attrs.get("span_kind")
        if sk:
            attrs["openinference.span.kind"] = sk
    code = str(rec.get("status_code") or "UNSET").upper()
    events = []
    for ev in rec.get("events") or []:
        if isinstance(ev, dict):
            events.append({"name": ev.get("name"), "attrs": decode_attrs(ev.get("attributes"))})
    return FlatSpan(
        trace_id=str(
            ctx.get("trace_id") or rec.get("context.trace_id") or rec.get("trace_id") or ""
        ),
        span_id=str(ctx.get("span_id") or rec.get("context.span_id") or rec.get("span_id") or ""),
        parent_id=(str(rec.get("parent_id") or rec.get("parent_span_id") or "") or None),
        name=str(rec.get("name") or ""),
        start=parse_time(rec.get("start_time")),
        end=parse_time(rec.get("end_time")),
        attrs=attrs,
        status_code={"OK": "OK", "ERROR": "ERROR"}.get(code, "UNSET"),
        status_message=rec.get("status_message"),
        events=events,
    )


def phoenix_spans(data: Any) -> list[FlatSpan]:
    if isinstance(data, dict):
        for key in ("spans", "data"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data]
    if not isinstance(data, list):
        raise IngestError("expected a list of spans")
    out = []
    for rec in data:
        if not isinstance(rec, dict):
            raise IngestError("a span must be a JSON object")
        out.append(_phoenix_span(rec))
    return out


def from_openinference(data: Any) -> list[Run]:
    """Runs from OpenInference spans: an OTLP/JSON export whose spans carry
    ``openinference.span.kind``, or Arize Phoenix span records (nested JSON, or a
    DataFrame ``to_json(orient="records")``). One run per trace id.

    Reads ``llm.input_messages.N.message.*`` / ``llm.output_messages``,
    ``llm.token_count.*``, ``llm.model_name``, ``llm.invocation_parameters``, ``tool.name``,
    ``input.value`` / ``output.value`` and ``retrieval.documents.N.document.*``."""
    from . import load_json_input
    from .otlp import build_runs

    data = load_json_input(data)
    probe = data[0] if isinstance(data, list) and data else data
    if isinstance(probe, dict) and ("resourceSpans" in probe or "resource_spans" in probe):
        return build_runs(otlp_spans(data), "openinference")
    spans = phoenix_spans(data)
    if not any("openinference.span.kind" in sp.attrs for sp in spans):
        raise IngestError("no OpenInference spans found (attribute openinference.span.kind)")
    for sp in spans:
        sp.attrs = flatten(sp.attrs)
    return build_runs_oi(spans)


def build_runs_oi(spans: list[FlatSpan]) -> list[Run]:
    from ._core import assemble

    runs = []
    for trace_id, group in group_by_trace(spans).items():
        nodes = [oi_node(sp) for sp in group]
        if any(n.kind in ("chat", "tool", "embeddings") for n in nodes):
            runs.append(assemble(nodes, fmt="openinference", trace_id=trace_id))
    if not runs:
        raise IngestError("no LLM or tool spans found")
    return runs


__all__ = ["from_openinference", "oi_node"]
