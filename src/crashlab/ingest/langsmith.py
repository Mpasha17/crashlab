"""LangSmith run exports -> crashlab runs."""

from __future__ import annotations

import ast
import re
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
    to_int,
)

_ROLES = {
    "humanmessage": "user",
    "humanmessagechunk": "user",
    "aimessage": "assistant",
    "aimessagechunk": "assistant",
    "systemmessage": "system",
    "toolmessage": "tool",
    "functionmessage": "tool",
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
}
_PARAMS = (
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "seed",
    "frequency_penalty",
    "presence_penalty",
)


def _lc_message(m: Any) -> dict[str, Any] | None:
    """A LangChain-serialized (``{"lc":1,"id":[..."HumanMessage"],"kwargs":{...}}``) or
    plain message as an OpenAI-style dict."""
    m = maybe_json(m)
    if isinstance(m, str):
        return {"role": "user", "content": m}
    if not isinstance(m, dict):
        return None
    if m.get("lc") and isinstance(m.get("kwargs"), dict):
        cls = str((m.get("id") or ["message"])[-1]).lower()
        data = dict(m["kwargs"])
        role = _ROLES.get(cls) or data.get("role") or "user"
        data["role"] = role
    else:
        data = dict(m.get("data") or m) if isinstance(m.get("data"), dict) else dict(m)
        raw = str(m.get("type") or m.get("role") or "user").lower()
        data["role"] = _ROLES.get(raw, raw)
    ak = data.get("additional_kwargs")
    if isinstance(ak, dict) and ak.get("tool_calls") and not data.get("tool_calls"):
        data["tool_calls"] = ak["tool_calls"]
    for k in (
        "type",
        "additional_kwargs",
        "response_metadata",
        "usage_metadata",
        "id",
        "example",
        "invalid_tool_calls",
    ):
        if k != "id" or data["role"] != "tool":
            data.pop(k, None)
    if data["role"] == "tool":
        data["tool_call_id"] = data.get("tool_call_id") or data.pop("id", None)
    return data


def _messages(raw: Any) -> list[dict[str, Any]]:
    raw = maybe_json(raw)
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list):
        raw = raw[0]
    if not isinstance(raw, list):
        raw = [raw] if raw is not None else []
    return norm_messages([m for m in map(_lc_message, raw) if m is not None])


def _inputs_messages(inputs: Any) -> list[dict[str, Any]]:
    if isinstance(inputs, dict):
        if inputs.get("messages") is not None:
            return _messages(inputs["messages"])
        if inputs.get("prompts"):
            return norm_messages(
                [{"role": "user", "content": text_of(p)} for p in inputs["prompts"]]
            )
        for k in ("input", "prompt", "question", "query"):
            if inputs.get(k) is not None:
                return norm_messages([{"role": "user", "content": text_of(inputs[k])}])
        return []
    return norm_messages([{"role": "user", "content": text_of(inputs)}]) if inputs else []


def _outputs(outputs: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """``(messages, info)``; info has finish_reason / model / usage when present."""
    info: dict[str, Any] = {}
    if not isinstance(outputs, dict):
        return norm_messages(
            [{"role": "assistant", "content": text_of(outputs)}]
        ) if outputs else [], info
    gens = outputs.get("generations")
    if isinstance(gens, list) and gens:
        first = gens[0][0] if isinstance(gens[0], list) and gens[0] else gens[0]
        if isinstance(first, dict):
            gi = first.get("generation_info") or {}
            info["finish"] = gi.get("finish_reason")
            msg = _lc_message(first.get("message")) if first.get("message") else None
            if msg is None:
                msg = {"role": "assistant", "content": first.get("text")}
            _usage_from(first.get("message"), info)
            msg["role"] = "assistant" if msg.get("role") in ("user", None) else msg["role"]
            lo = outputs.get("llm_output") or {}
            tu = lo.get("token_usage") or {}
            info.setdefault("inp", to_int(tu.get("prompt_tokens")))
            info.setdefault("out", to_int(tu.get("completion_tokens")))
            info.setdefault("total", to_int(tu.get("total_tokens")))
            info["model"] = lo.get("model_name") or lo.get("model")
            return norm_messages([msg]), {k: v for k, v in info.items() if v is not None}
    if isinstance(outputs.get("choices"), list):
        u = outputs.get("usage") or {}
        info = {
            "inp": to_int(u.get("prompt_tokens")),
            "out": to_int(u.get("completion_tokens")),
            "total": to_int(u.get("total_tokens")),
            "model": outputs.get("model"),
            "finish": (outputs["choices"][0] or {}).get("finish_reason")
            if outputs["choices"]
            else None,
        }
        msgs = norm_messages([c.get("message") for c in outputs["choices"] if isinstance(c, dict)])
        return msgs, {k: v for k, v in info.items() if v is not None}
    msg = _lc_message(outputs["message"]) if outputs.get("message") else None
    if msg is None and ("content" in outputs or "tool_calls" in outputs):
        msg = _lc_message({**outputs, "role": outputs.get("role", "assistant")})
    if msg is None:
        for k in ("output", "text", "answer", "content"):
            if outputs.get(k) is not None:
                msg = {"role": "assistant", "content": text_of(outputs[k])}
                break
    if msg is not None:
        _usage_from(outputs.get("message") or outputs, info)
        msg["role"] = "assistant" if msg.get("role") in ("user", None) else msg["role"]
        return norm_messages([msg]), {k: v for k, v in info.items() if v is not None}
    return [], info


def _usage_from(message: Any, info: dict[str, Any]) -> None:
    message = maybe_json(message)
    if not isinstance(message, dict):
        return
    kw = message.get("kwargs") if isinstance(message.get("kwargs"), dict) else message
    um = kw.get("usage_metadata") if isinstance(kw.get("usage_metadata"), dict) else None
    if um:
        info["inp"], info["out"], info["total"] = (
            to_int(um.get("input_tokens")),
            to_int(um.get("output_tokens")),
            to_int(um.get("total_tokens")),
        )
    rm = kw.get("response_metadata") if isinstance(kw.get("response_metadata"), dict) else {}
    if rm.get("finish_reason"):
        info["finish"] = rm["finish_reason"]
    if rm.get("model_name") or rm.get("model"):
        info["model"] = rm.get("model_name") or rm.get("model")


def _error(run: dict[str, Any], *, tool: bool) -> Any:
    err = run.get("error")
    if not err:
        return None
    text = str(err)
    m = re.match(
        r"\s*([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Timeout))\b[:(]?\s*(.*)", text, re.S
    )
    typ, msg = (m.group(1), (m.group(2) or text).strip().rstrip(")")) if m else ("RunError", text)
    return make_error(
        typ, msg or text, is_tool_error=tool, traceback=text if "Traceback" in text else None
    )


def _tool_args(inputs: Any) -> dict[str, Any]:
    inputs = maybe_json(inputs)
    if isinstance(inputs, dict):
        if set(inputs) == {"input"}:
            inner = inputs["input"]
            if isinstance(inner, str):
                parsed = _parse_literal(inner)
                return parsed if isinstance(parsed, dict) else {"input": inner}
            return inner if isinstance(inner, dict) else {"input": inner}
        return {k: v for k, v in inputs.items() if k not in ("config", "callbacks", "run_manager")}
    if isinstance(inputs, str):
        parsed = _parse_literal(inputs)
        return parsed if isinstance(parsed, dict) else {"input": inputs}
    return {}


def _parse_literal(text: str) -> Any:
    t = text.strip()
    if not t.startswith("{"):
        return None
    try:
        import json

        return json.loads(t)
    except ValueError:
        try:
            return ast.literal_eval(t)
        except (ValueError, SyntaxError):
            return None


def _tool_result(outputs: Any) -> Any:
    outputs = maybe_json(outputs)
    if isinstance(outputs, dict) and set(outputs) <= {"output"}:
        outputs = maybe_json(outputs.get("output"))
    if isinstance(outputs, dict) and outputs.get("lc") and isinstance(outputs.get("kwargs"), dict):
        return maybe_json(outputs["kwargs"].get("content"))
    return outputs


def _summ(payload: Any) -> Any:
    payload = maybe_json(payload)
    if isinstance(payload, dict):
        for k in ("output", "answer", "result", "input", "question", "query", "text"):
            if isinstance(payload.get(k), str) and len(payload) <= 2:
                return payload[k]
        msgs = payload.get("messages")
        if isinstance(msgs, list) and msgs:
            last = _lc_message(msgs[-1])
            return text_of(last.get("content")) if last else payload
    return payload if payload not in ({}, None) else None


def _walk(
    run: dict[str, Any],
    parent: str | None,
    trace: str | None,
    out: list[tuple[dict[str, Any], str | None, str | None]],
) -> None:
    trace = str(run["trace_id"]) if run.get("trace_id") else trace
    out.append((run, parent, trace))
    for child in run.get("child_runs") or []:
        if isinstance(child, dict):
            _walk(child, str(run.get("id")), trace or str(run.get("id")), out)


def _node(run: dict[str, Any], parent: str | None) -> Node:
    rtype = str(run.get("run_type") or "chain").lower()
    extra = run.get("extra") if isinstance(run.get("extra"), dict) else {}
    inv = extra.get("invocation_params") if isinstance(extra.get("invocation_params"), dict) else {}
    meta = extra.get("metadata") if isinstance(extra.get("metadata"), dict) else {}
    node = Node(
        id=str(run.get("id")),
        parent=parent or (str(run["parent_run_id"]) if run.get("parent_run_id") else None),
        kind="skip",
        name=str(run.get("name") or rtype),
        start=parse_time(run.get("start_time")),
        end=parse_time(run.get("end_time")),
    )
    inputs, outputs = maybe_json(run.get("inputs")), maybe_json(run.get("outputs"))
    if rtype == "llm":
        node.kind = "chat"
        msgs_out, info = _outputs(outputs)
        model = (
            inv.get("model")
            or inv.get("model_name")
            or inv.get("model_id")
            or meta.get("ls_model_name")
            or info.get("model")
        )
        params = {
            k: inv[k]
            for k in _PARAMS
            if isinstance(inv.get(k), (int, float)) and not isinstance(inv[k], bool)
        }
        tools = inv.get("tools") or inv.get("functions")
        node.request = chat_request(
            _inputs_messages(inputs),
            model=str(model) if model else None,
            tools=tools if isinstance(tools, list) else None,
            params=params,
        )
        node.name = str(model) if model else "unknown-model"
        inp = (
            to_int(run.get("prompt_tokens"))
            if run.get("prompt_tokens") is not None
            else info.get("inp")
        )
        out = (
            to_int(run.get("completion_tokens"))
            if run.get("completion_tokens") is not None
            else info.get("out")
        )
        total = (
            to_int(run.get("total_tokens"))
            if run.get("total_tokens") is not None
            else info.get("total")
        )
        node.response = chat_response(
            msgs_out,
            model=str(model) if model else None,
            finish=info.get("finish"),
            inp=inp,
            out=out,
            total=total,
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
        node.attrs = attrs
        node.error = _error(run, tool=False)
    elif rtype == "tool":
        node.kind = "tool"
        node.request = {"args": _tool_args(inputs)}
        node.error = _error(run, tool=True)
        node.response = None if node.error else {"result": _tool_result(outputs)}
        node.attrs = {"gen_ai.tool.name": node.name}
        call_id = meta.get("tool_call_id") or (
            inputs.get("tool_call_id") if isinstance(inputs, dict) else None
        )
        if not call_id and isinstance(outputs, dict) and isinstance(outputs.get("kwargs"), dict):
            call_id = outputs["kwargs"].get("tool_call_id")
        if not call_id and isinstance(outputs, dict) and isinstance(outputs.get("output"), dict):
            kw = outputs["output"].get("kwargs")
            call_id = kw.get("tool_call_id") if isinstance(kw, dict) else None
        if call_id:
            node.attrs["gen_ai.tool.call.id"] = str(call_id)
    elif rtype == "retriever":
        node.kind = "tool"
        q = inputs.get("query", inputs.get("input")) if isinstance(inputs, dict) else inputs
        node.request = {"args": {"query": q}}
        node.error = _error(run, tool=True)
        docs = outputs.get("documents") if isinstance(outputs, dict) else outputs
        node.response = (
            None if node.error else {"result": docs_result(docs if isinstance(docs, list) else [])}
        )
        node.attrs = {"gen_ai.tool.name": node.name}
    elif rtype == "embedding":
        node.kind = "embeddings"
        model = inv.get("model") or inv.get("model_name")
        node.name = str(model or "unknown-model")
        node.request = {"params": {"model": model} if model else {}}
        node.response = {"usage": {"input_tokens": to_int(run.get("prompt_tokens"))}}
        node.error = _error(run, tool=False)
    else:
        node.input, node.output = _summ(inputs), _summ(outputs)
        node.error = _error(run, tool=False)
    return node


def from_langsmith(data: Any) -> list[Run]:
    """Runs from LangSmith run exports: a run tree (nested ``child_runs``), a list of
    trees, or a flat list of runs linked by ``parent_run_id`` / ``trace_id``. One crashlab
    run per trace.

    ``run_type`` ``llm`` becomes an LLM call (LangChain-serialized or OpenAI-style
    messages, ``generations`` / ``choices`` outputs, token counts), ``tool`` a tool call,
    ``retriever`` a retrieval tool call whose documents (``id`` / ``text`` / ``metadata``)
    ``crashlab.rag`` reads; chains only group. ``error`` marks a failed call."""
    from . import load_json_input

    data = load_json_input(data)
    if isinstance(data, dict) and isinstance(data.get("runs"), list):
        data = data["runs"]
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or not data:
        raise IngestError("expected a LangSmith run object or a list of runs")
    flat: list[tuple[dict[str, Any], str | None, str | None]] = []
    for r in data:
        if not isinstance(r, dict) or "run_type" not in r and "id" not in r:
            raise IngestError("a LangSmith run needs 'id' and 'run_type'")
        _walk(r, str(r["parent_run_id"]) if r.get("parent_run_id") else None, None, flat)
    parent_of = {str(r.get("id")): p for r, p, _ in flat}
    trace_of = {str(r.get("id")): t for r, _, t in flat if t}

    def trace_for(run_id: str) -> str:
        seen: set[str] = set()
        cur = run_id
        while cur not in seen:
            seen.add(cur)
            if trace_of.get(cur):
                return trace_of[cur]
            up = parent_of.get(cur)
            if not up or up not in parent_of:
                return cur
            cur = up
        return run_id

    groups: dict[str, list[Node]] = {}
    for run, parent, _ in flat:
        groups.setdefault(trace_for(str(run.get("id"))), []).append(_node(run, parent))
    runs = []
    for tid, nodes in groups.items():
        ids = {n.id for n in nodes}
        for n in nodes:
            if n.parent not in ids:
                n.parent = None
        if any(n.kind in ("chat", "tool", "embeddings") for n in nodes):
            runs.append(assemble(nodes, fmt="langsmith", trace_id=tid))
    if not runs:
        raise IngestError("no llm / tool / retriever runs found in the LangSmith export")
    return runs


__all__ = ["from_langsmith"]
