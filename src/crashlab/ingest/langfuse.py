"""Langfuse trace exports -> crashlab runs."""

from __future__ import annotations

from typing import Any

from ..models import Run
from ._core import (
    IngestError,
    Node,
    assemble,
    chat_request,
    chat_response,
    docs_result,
    make_error,
    maybe_json,
    norm_messages,
    parse_time,
    text_of,
    to_float,
    to_int,
    tool_args,
)

_PARAMS = (
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "seed",
    "frequency_penalty",
    "presence_penalty",
)


def traces_of(data: Any) -> list[dict[str, Any]]:
    """The trace objects in an export: one trace, a list, or ``{"data": [...]}`` (the
    public API's list response)."""
    if isinstance(data, dict) and isinstance(data.get("data"), list) and "observations" not in data:
        data = data["data"]
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or not data:
        raise IngestError("expected a Langfuse trace object, a list of traces or {'data': [...]}")
    for t in data:
        if not isinstance(t, dict) or not isinstance(t.get("observations"), list):
            raise IngestError(
                "a Langfuse trace needs an 'observations' list (export traces with "
                "their observations, e.g. GET /api/public/traces/{id})"
            )
        if any(not isinstance(o, dict) for o in t["observations"]):
            raise IngestError("the trace lists observation ids only; export the full observations")
    return data


def _tokens(o: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    u = o.get("usageDetails") if isinstance(o.get("usageDetails"), dict) else None
    u = u or (o.get("usage") if isinstance(o.get("usage"), dict) else {}) or {}
    inp = to_int(
        u.get("input", u.get("promptTokens", u.get("prompt_tokens", o.get("promptTokens"))))
    )
    out = to_int(
        u.get(
            "output",
            u.get("completionTokens", u.get("completion_tokens", o.get("completionTokens"))),
        )
    )
    total = to_int(
        u.get("total", u.get("totalTokens", u.get("total_tokens", o.get("totalTokens"))))
    )
    return inp, out, total


def _output_messages(out: Any) -> list[dict[str, Any]]:
    out = maybe_json(out)
    if isinstance(out, dict) and isinstance(out.get("choices"), list):
        return norm_messages([c.get("message") for c in out["choices"] if isinstance(c, dict)])
    if isinstance(out, dict) and "content" in out or isinstance(out, dict) and "tool_calls" in out:
        msg = dict(out)
        ak = msg.get("additional_kwargs")
        if isinstance(ak, dict) and ak.get("tool_calls") and not msg.get("tool_calls"):
            msg["tool_calls"] = ak["tool_calls"]
        msg.setdefault("role", "assistant")
        return norm_messages([msg])
    if isinstance(out, list) and out and all(isinstance(m, dict) and "role" in m for m in out):
        return norm_messages(out)
    if out is None:
        return []
    return norm_messages([{"role": "assistant", "content": text_of(out)}])


def _input_messages(inp: Any) -> tuple[list[dict[str, Any]], Any]:
    inp = maybe_json(inp)
    tools = None
    if isinstance(inp, dict):
        tools = inp.get("tools") or inp.get("functions")
        inp = inp.get("messages", inp.get("input", inp.get("prompt", inp)))
    if isinstance(inp, list) and len(inp) == 1 and isinstance(inp[0], list):
        inp = inp[0]
    if isinstance(inp, list) and all(isinstance(m, dict) for m in inp):
        return norm_messages(inp), tools
    if inp is None:
        return [], tools
    return norm_messages([{"role": "user", "content": text_of(inp)}]), tools


def _error(o: dict[str, Any], *, tool: bool) -> Any:
    if str(o.get("level") or "").upper() != "ERROR":
        return None
    meta = o.get("metadata") if isinstance(o.get("metadata"), dict) else {}
    return make_error(
        meta.get("error_type") or meta.get("exception_type") or "Error",
        o.get("statusMessage") or "observation level ERROR",
        is_tool_error=tool,
    )


def _node(o: dict[str, Any], parent: str, has_children: bool) -> Node | None:
    typ = str(o.get("type") or "SPAN").upper()
    node = Node(
        id=str(o.get("id")),
        parent=str(o["parentObservationId"]) if o.get("parentObservationId") else parent,
        kind="skip",
        name=str(o.get("name") or typ.lower()),
        start=parse_time(o.get("startTime")),
        end=parse_time(o.get("endTime") or o.get("completionStartTime")),
    )
    if typ == "GENERATION":
        node.kind = "chat"
        msgs_in, tools = _input_messages(o.get("input"))
        mp = o.get("modelParameters") if isinstance(o.get("modelParameters"), dict) else {}
        params = {
            k: mp[k]
            for k in _PARAMS
            if isinstance(mp.get(k), (int, float)) and not isinstance(mp.get(k), bool)
        }
        model = o.get("model")
        node.request = chat_request(
            msgs_in,
            model=str(model) if model else None,
            tools=tools if isinstance(tools, list) else None,
            params=params,
        )
        node.name = str(model) if model else "unknown-model"
        inp, out, total = _tokens(o)
        msgs_out = _output_messages(o.get("output"))
        node.response = chat_response(
            msgs_out, model=str(model) if model else None, inp=inp, out=out, total=total
        )
        attrs: dict[str, Any] = {f"gen_ai.request.{k}": v for k, v in params.items()}
        if model:
            attrs["gen_ai.request.model"] = attrs["gen_ai.response.model"] = str(model)
        if inp is not None:
            attrs["gen_ai.usage.input_tokens"] = inp
        if out is not None:
            attrs["gen_ai.usage.output_tokens"] = out
        if total is not None and total != (inp or 0) + (out or 0):
            attrs["crashlab.usage.total_tokens"] = total
        cost = to_float(o.get("calculatedTotalCost", o.get("totalPrice")))
        if cost is not None:
            attrs["crashlab.usage.cost_usd"] = cost
        node.attrs = attrs
        node.error = _error(o, tool=False)
        return node
    if typ == "EMBEDDING":
        node.kind = "embeddings"
        model = o.get("model")
        node.name = str(model or "unknown-model")
        node.request = {"params": {"model": model} if model else {}}
        inp, _, total = _tokens(o)
        node.response = {"usage": {"input_tokens": inp, "total_tokens": total or inp}}
        node.error = _error(o, tool=False)
        return node
    is_tool = typ in ("TOOL", "RETRIEVER") or (
        typ == "SPAN"
        and not has_children
        and (
            o.get("input") is not None
            or o.get("output") is not None
            or str(o.get("level") or "").upper() == "ERROR"
        )
    )
    if is_tool:
        node.kind = "tool"
        node.error = _error(o, tool=True)
        if typ == "RETRIEVER":
            node.request = {"args": tool_args(o.get("input"))}
            docs = maybe_json(o.get("output"))
            docs = docs.get("documents", docs) if isinstance(docs, dict) else docs
            node.response = (
                None
                if node.error
                else {"result": docs_result(docs if isinstance(docs, list) else [])}
            )
        else:
            node.request = {"args": tool_args(o.get("input"))}
            node.response = None if node.error else {"result": maybe_json(o.get("output"))}
        node.attrs = {"gen_ai.tool.name": node.name}
        return node
    if typ == "AGENT":
        node.kind = "agent"
    return node if typ != "EVENT" else None


def from_langfuse(data: Any) -> list[Run]:
    """Runs from Langfuse trace exports (the trace JSON with its ``observations``): one
    run per trace.

    ``GENERATION`` observations become LLM calls (``model``, ``modelParameters``,
    ``input`` / ``output`` messages, ``usage`` / ``usageDetails``); ``TOOL`` and
    ``RETRIEVER`` observations, and leaf ``SPAN`` s that have an input or output, become
    tool calls; ``AGENT`` observations label their descendants; other spans only group.
    ``level: "ERROR"`` marks a failed call. Parents come from ``parentObservationId``."""
    from . import load_json_input

    runs = []
    for trace in traces_of(load_json_input(data)):
        tid = str(trace.get("id") or trace.get("traceId") or "")
        if not tid:
            raise IngestError("a Langfuse trace has no id")
        root_id = f"trace:{tid}"
        obs = trace["observations"]
        parents = {str(o.get("parentObservationId")) for o in obs if o.get("parentObservationId")}
        nodes = [
            Node(
                id=root_id,
                parent=None,
                kind="agent",
                name=str(trace.get("name") or "langfuse-trace"),
                start=parse_time(trace.get("timestamp")),
                input=_payload(trace.get("input")),
                output=_payload(trace.get("output")),
            )
        ]
        for o in obs:
            node = _node(o, root_id, str(o.get("id")) in parents)
            if node is not None:
                nodes.append(node)
        meta = {
            "session_id": trace.get("sessionId"),
            "release": trace.get("release"),
            "name": trace.get("name"),
        }
        runs.append(assemble(nodes, fmt="langfuse", trace_id=tid, extra_meta=meta))
    return runs


def _payload(v: Any) -> Any:
    v = maybe_json(v)
    if isinstance(v, dict) and len(v) == 1 and isinstance(next(iter(v.values())), str):
        return next(iter(v.values()))
    if isinstance(v, list):
        return (
            text_of(v)
            if v and not all(isinstance(m, dict) and "role" in m for m in v)
            else _last_user(v)
        )
    return v


def _last_user(msgs: list[Any]) -> Any:
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("role") == "user":
            return text_of(m.get("content"))
    return None


__all__ = ["from_langfuse"]
