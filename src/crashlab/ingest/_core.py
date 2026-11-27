"""Shared pieces of the trace importers: parsing helpers and the run assembler.

Every importer turns its format into a flat list of :class:`Node` (one per span /
observation / run) and calls :func:`assemble`, which orders them, links parents, hashes
requests the way live capture does (so an imported run can be replayed) and builds the
:class:`~crashlab.models.Run`.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from .._ids import _encode
from .._isotime import parse_iso
from ..capture import normalize as N
from ..models import ErrorInfo, Run, Step, Usage
from ..store.redact import get_redactor

_HEX16 = re.compile(r"^[0-9a-f]{16}$")


class IngestError(ValueError):
    """The input is not a trace this importer understands."""


# --- small parsers ---------------------------------------------------------------------


def maybe_json(value: Any) -> Any:
    """A JSON string parsed; anything else (or invalid JSON) unchanged."""
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", "replace")
    if isinstance(value, str) and value.strip()[:1] in ("{", "[", '"'):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def parse_time(value: Any) -> datetime | None:
    """ISO-8601 text, or epoch seconds / ms / us / ns (int, float or digit string)."""
    if value is None or value == "" or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        if isinstance(value, str) and not re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
            ts = parse_iso(value)
            return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        num = float(value)
    except (TypeError, ValueError):
        return None
    mag = abs(num)
    if mag >= 1e17:  # ns
        num /= 1e9
    elif mag >= 1e14:  # us
        num /= 1e6
    elif mag >= 1e11:  # ms
        num /= 1e3
    try:
        return datetime.fromtimestamp(num, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def to_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def text_of(value: Any) -> str:
    """Best-effort plain text of a message content / payload."""
    value = maybe_json(value)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(text_of(v) for v in value)
    if isinstance(value, dict):
        for key in ("text", "content", "output", "answer", "result", "value"):
            if isinstance(value.get(key), (str, list)):
                return text_of(value[key])
    return json.dumps(value, ensure_ascii=False, default=str)


def hex16(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def span_hex(value: Any, fallback: str) -> str:
    if isinstance(value, str) and _HEX16.fullmatch(value.lower()):
        return value.lower()
    return hex16(str(value) if value else fallback)


def stable_run_id(trace_key: str, start: datetime | None) -> str:
    """A ULID-shaped id derived from the trace id, so importing a file twice finds the
    run it already made."""
    ms = int((start or datetime(2000, 1, 1, tzinfo=timezone.utc)).timestamp() * 1000)
    digest = int.from_bytes(hashlib.sha256(trace_key.encode("utf-8")).digest()[:10], "big")
    return _encode(max(ms, 0), 10) + _encode(digest, 16)


# --- messages --------------------------------------------------------------------------


def norm_messages(raw: Any) -> list[dict[str, Any]]:
    """OpenAI-style, Anthropic-style or OTel ``parts`` messages -> crashlab messages."""
    raw = maybe_json(raw)
    if raw is None:
        return []
    if isinstance(raw, (str, dict)):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for m in raw:
        m = maybe_json(m)
        if isinstance(m, str):
            out.extend(N.normalize_messages([{"role": "user", "content": m}]))
        elif isinstance(m, dict) and isinstance(m.get("parts"), list):
            parts = []
            for p in m["parts"]:
                if isinstance(p, dict) and p.get("type") == "tool_call":
                    parts.append(N.tool_call_part(p.get("id"), p.get("name"), p.get("arguments")))
                elif isinstance(p, dict) and p.get("type") == "tool_call_response":
                    parts.append(N.tool_result_part(p.get("id"), p.get("response")))
                elif isinstance(p, dict) and p.get("type") == "text":
                    parts.append({"type": "text", "content": text_of(p.get("content"))})
                else:
                    np_ = N.normalize_part(p)
                    if np_ is not None:
                        parts.append(np_)
            msg: dict[str, Any] = {"role": str(m.get("role") or "user"), "parts": parts}
            if m.get("name"):
                msg["name"] = m["name"]
            out.append(msg)
        elif isinstance(m, dict):
            out.extend(N.normalize_messages([_openai_like(m)]))
    return out


def raw_finish(raw: Any) -> Any:
    """The first ``finish_reason`` on a list of raw messages (OTel puts it on each output
    message); crashlab keeps it on the response, not the message."""
    raw = maybe_json(raw)
    for m in [raw] if isinstance(raw, dict) else raw if isinstance(raw, list) else []:
        m = maybe_json(m)
        if isinstance(m, dict) and m.get("finish_reason"):
            return m["finish_reason"]
    return None


def _openai_like(m: dict[str, Any]) -> dict[str, Any]:
    """Tolerate role aliases (``human`` / ``ai`` / ``assistant``) and ``tool_calls`` whose
    items are flat (``{"id","name","arguments"|"args"}``) instead of OpenAI's nested form."""
    m = dict(m)
    role = str(m.get("role") or m.get("type") or "user").lower()
    m["role"] = {"human": "user", "ai": "assistant", "model": "assistant", "function": "tool"}.get(
        role, role
    )
    m.pop("type", None)
    calls = []
    for tc in m.get("tool_calls") or []:
        tc = maybe_json(tc)
        if not isinstance(tc, dict):
            continue
        if isinstance(tc.get("function"), dict):
            calls.append(tc)
        else:
            calls.append(
                {
                    "id": tc.get("id"),
                    "function": {
                        "name": tc.get("name"),
                        "arguments": tc.get("arguments", tc.get("args")),
                    },
                }
            )
    if "tool_calls" in m:
        m["tool_calls"] = calls
    return m


def tool_calls_of(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for msg in messages:
        if msg.get("role") == "assistant":
            out.extend(N._tool_calls_from_parts(msg.get("parts") or []))
    return out


_FINISH = {
    "tool_call": "tool_calls",
    "function_call": "tool_calls",
    "end_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "STOP": "stop",
}


def chat_response(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    finish: Any = None,
    inp: int | None = None,
    out: int | None = None,
    total: int | None = None,
    response_id: str | None = None,
) -> dict[str, Any]:
    calls = tool_calls_of(messages)
    reason = _FINISH.get(str(finish), finish) if finish else ("tool_calls" if calls else "stop")
    return {
        "messages": messages,
        "tool_calls": calls,
        "finish_reason": reason,
        "finish_reasons": [reason],
        "model": model,
        "id": response_id,
        "usage": N._usage(inp, out, total),
    }


def chat_request(
    messages: list[dict[str, Any]],
    *,
    model: str | None,
    tools: Any = None,
    system: Any = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sys_text, rest = N.split_leading_system(messages)
    if system:
        extra = system_instructions(system)
        sys_text = "\n\n".join(t for t in (extra, sys_text) if t) or None
    try:
        norm_tools = N.normalize_tools(tools) if tools else []
    except Exception:
        norm_tools = []
    p = {k: v for k, v in (params or {}).items() if v is not None}
    if model:
        p["model"] = model
    return {"system": sys_text, "messages": rest, "tools": norm_tools, "params": p}


def system_instructions(value: Any) -> str | None:
    value = maybe_json(value)
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    if isinstance(value, dict):
        value = [value]
    if isinstance(value, list):
        texts = [text_of(v.get("content", v) if isinstance(v, dict) else v) for v in value]
        return "\n\n".join(t for t in texts if t) or None
    return None


def _provider_of(attrs: dict[str, Any], model: str | None) -> str:
    prov = str(attrs.get("gen_ai.provider.name") or attrs.get("gen_ai.system") or "").lower()
    if "anthropic" in prov or (not prov and str(model or "").lower().startswith("claude")):
        return "anthropic"
    return "openai"


def synth_raw_response(response: dict[str, Any], provider: str) -> dict[str, Any]:
    """A provider-shaped response (OpenAI ``ChatCompletion`` / Anthropic ``Message``) built
    from the normalized one: what replay hands back to the code that made the call."""
    texts, calls = [], response.get("tool_calls") or []
    for m in response.get("messages") or []:
        texts.extend(
            p.get("content") or "" for p in m.get("parts") or [] if p.get("type") == "text"
        )
    text = "".join(texts)
    usage = response.get("usage") or {}
    inp, out = usage.get("input_tokens"), usage.get("output_tokens")
    reason = response.get("finish_reason")
    if provider == "anthropic":
        content: list[dict[str, Any]] = ([{"type": "text", "text": text}] if text else []) + [
            {
                "type": "tool_use",
                "id": c.get("id"),
                "name": c.get("name"),
                "input": c.get("arguments"),
            }
            for c in calls
        ]
        stop = {"tool_calls": "tool_use", "stop": "end_turn", "length": "max_tokens"}.get(
            reason, reason
        )
        return {
            "id": response.get("id") or "msg_imported",
            "type": "message",
            "role": "assistant",
            "model": response.get("model") or "unknown",
            "content": content,
            "stop_reason": stop,
            "stop_sequence": None,
            "usage": {"input_tokens": inp or 0, "output_tokens": out or 0},
        }
    message: dict[str, Any] = {"role": "assistant", "content": text or (None if calls else "")}
    if calls:
        message["tool_calls"] = [
            {
                "id": c.get("id"),
                "type": "function",
                "function": {
                    "name": c.get("name"),
                    "arguments": json.dumps(c.get("arguments"), ensure_ascii=False),
                },
            }
            for c in calls
        ]
    return {
        "id": response.get("id") or "chatcmpl-imported",
        "object": "chat.completion",
        "created": 0,
        "model": response.get("model") or "unknown",
        "choices": [{"index": 0, "message": message, "finish_reason": reason or "stop"}],
        "usage": {
            "prompt_tokens": inp or 0,
            "completion_tokens": out or 0,
            "total_tokens": usage.get("total_tokens") or (inp or 0) + (out or 0),
        },
    }


def tool_args(raw: Any) -> Any:
    args = N.parse_arguments(maybe_json(raw) if not isinstance(raw, str) else raw)
    return args if isinstance(args, dict) else ({"input": args} if args not in ("", None) else {})


def docs_result(documents: Any) -> list[dict[str, Any]]:
    """Retrieved documents as ``[{id, text, score, metadata}]``: what ``crashlab.rag``
    reads from a retrieval tool step."""
    out = []
    for i, d in enumerate(documents or []):
        d = maybe_json(d)
        if isinstance(d, str):
            d = {"text": d}
        if not isinstance(d, dict):
            continue
        doc = d.get("document") if isinstance(d.get("document"), dict) else d
        meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
        meta = meta or (
            doc.get("document.metadata") if isinstance(doc.get("document.metadata"), dict) else {}
        )
        text = None
        for key in ("page_content", "content", "text", "document.content"):
            if isinstance(doc.get(key), str):
                text = doc[key]
                break
        doc_id = doc.get("id", doc.get("document.id"))
        if doc_id is None:
            for key in ("id", "doc_id", "document_id", "source"):
                if meta.get(key) is not None:
                    doc_id = meta[key]
                    break
        entry: dict[str, Any] = {
            "id": doc_id if doc_id is not None else f"doc-{i + 1}",
            "text": text or "",
        }
        score = doc.get("score", doc.get("document.score", d.get("score")))
        if score is not None:
            entry["score"] = score
        if meta:
            entry["metadata"] = meta
        out.append(entry)
    return out


def make_error(
    typ: Any, message: Any, *, is_tool_error: bool = False, traceback: Any = None
) -> ErrorInfo:
    return ErrorInfo(
        type=str(typ or "Error"),
        message=str(message or typ or "error"),
        is_tool_error=is_tool_error,
        traceback=str(traceback) if traceback else None,
    )


# --- nodes and assembly ------------------------------------------------------------------


@dataclass
class Node:
    """One span / observation / run, already mapped to crashlab terms.

    ``kind``: ``chat`` | ``tool`` | ``embeddings`` (these become steps), ``agent`` (a
    grouping span: names the agent for its descendants, or is the run itself when it is
    the root), ``skip`` (ignored; its children are re-parented).
    """

    id: str
    parent: str | None
    kind: str
    name: str
    start: datetime | None = None
    end: datetime | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    request: dict[str, Any] = field(default_factory=dict)
    response: dict[str, Any] | None = None
    error: ErrorInfo | None = None
    input: Any = None  # run-level payloads (agent / root nodes)
    output: Any = None
    order: int = 0


_OP = {"chat": "chat", "tool": "execute_tool", "embeddings": "embeddings"}


def _depth(node: Node, nodes: dict[str, Node]) -> int:
    d, seen, cur = 0, {node.id}, node
    while cur.parent in nodes and cur.parent not in seen:
        seen.add(cur.parent)  # type: ignore[arg-type]
        cur = nodes[cur.parent]  # type: ignore[index]
        d += 1
    return d


def assemble(
    nodes: list[Node],
    *,
    fmt: str,
    trace_id: str,
    name: str | None = None,
    extra_meta: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Run:
    """Build one :class:`Run` from the nodes of a single trace."""
    if not nodes:
        raise IngestError("the trace has no spans / observations")
    for i, n in enumerate(nodes):
        n.order = i
    by_id: dict[str, Node] = {}
    for n in nodes:
        if n.id in by_id:  # duplicate ids: keep the first
            continue
        by_id[n.id] = n
    nodes = [n for n in nodes if by_id.get(n.id) is n]
    for n in nodes:
        if n.parent == n.id or (n.parent is not None and n.parent not in by_id):
            n.parent = None
    roots = sorted((n for n in nodes if n.parent is None), key=_sort_key)
    run_node = next((r for r in roots if r.kind == "agent"), None) or next(
        (r for r in roots if r.kind == "skip"), None
    )

    def ancestors(n: Node) -> list[Node]:
        out, seen, cur = [], {n.id}, n
        while cur.parent is not None and cur.parent in by_id and cur.parent not in seen:
            seen.add(cur.parent)
            cur = by_id[cur.parent]
            out.append(cur)
        return out

    candidates = [n for n in nodes if n.kind in _OP]
    timed = all(n.start for n in candidates)
    emit = sorted(
        candidates,
        key=lambda n: (
            (n.start.timestamp() if timed and n.start else 0.0),
            _depth(n, by_id),
            n.order,
        ),
    )
    if not emit:
        raise IngestError("the trace has no LLM or tool calls (nothing to import)")
    used: set[str] = set()
    span_ids: dict[str, str] = {}
    for n in nodes:
        sid = span_hex(n.id, f"{trace_id}:{n.order}")
        while sid in used:
            sid = hex16(sid + "+")
        used.add(sid)
        span_ids[n.id] = sid
    root_span = span_ids[run_node.id] if run_node is not None else None

    idx_of = {n.id: i for i, n in enumerate(emit)}
    # tool calls without an id: pair them with the earlier chat call that requested them
    claimed: set[str] = set()
    for n in emit:
        if n.kind == "chat":
            continue
        if n.kind == "tool" and not n.attrs.get("gen_ai.tool.call.id"):
            for c in reversed(
                [e for e in emit if e.kind == "chat" and idx_of[e.id] < idx_of[n.id]]
            ):
                hit = next(
                    (
                        tc
                        for tc in (c.response or {}).get("tool_calls") or []
                        if tc.get("name") == n.name and tc.get("id") and tc["id"] not in claimed
                    ),
                    None,
                )
                if hit:
                    n.attrs["gen_ai.tool.call.id"] = hit["id"]
                    claimed.add(hit["id"])
                    break
        elif n.attrs.get("gen_ai.tool.call.id"):
            claimed.add(str(n.attrs["gen_ai.tool.call.id"]))

    run_id = stable_run_id(
        f"{fmt}:{trace_id}", min((n.start for n in nodes if n.start), default=None)
    )
    starts = [n.start for n in nodes if n.start]
    ends = [n.end for n in nodes if n.end]
    started = min(starts) if starts else datetime.now(timezone.utc)
    steps: list[Step] = []
    redactor = get_redactor()
    for i, n in enumerate(emit):
        chain = ancestors(n)
        agent = next(
            (a.name for a in chain if a.kind == "agent" and a is not run_node and a.name), None
        )
        attrs = {
            "gen_ai.operation.name": {
                "chat": "chat",
                "tool": "execute_tool",
                "embeddings": "embeddings",
            }[n.kind]
        }
        attrs.update({k: v for k, v in n.attrs.items() if v is not None})
        if agent and "gen_ai.agent.name" not in attrs:
            attrs["gen_ai.agent.name"] = agent
        parent_node = next((a for a in chain if a.kind in _OP), None)
        parent_span = span_ids[parent_node.id] if parent_node is not None else root_span
        if n.kind == "tool":
            hash_in: Any = {"tool": n.name, "args": n.request.get("args")}
        else:
            hash_in = n.request
        raw_response = None
        if n.kind == "chat" and n.response is not None:
            provider = _provider_of(attrs, n.request.get("params", {}).get("model"))
            raw_response = synth_raw_response(n.response, provider)
            attrs.setdefault(
                "crashlab.api",
                "anthropic.messages" if provider == "anthropic" else "openai.chat.completions",
            )
        st = n.start or started
        dur = ((n.end - n.start).total_seconds() * 1000) if n.start and n.end else 0.0
        steps.append(
            Step(
                run_id=run_id,
                idx=i,
                span_id=span_ids[n.id],
                parent_span_id=parent_span,
                operation=_OP[n.kind],  # type: ignore[arg-type]
                name=n.name,
                attributes=attrs,
                request=n.request,
                response=n.response,
                raw_response=raw_response,
                error=n.error,
                request_hash=N.request_hash(redactor.redact(hash_in)),
                started_at=st,
                duration_ms=round(max(dur, 0.0), 3),
                source="live",
            )
        )
    ended = (
        max(ends)
        if ends
        else started + timedelta(milliseconds=max((s.duration_ms for s in steps), default=0.0))
    )
    chats = [s for s in steps if s.operation == "chat"]
    usage = Usage(
        steps=len(steps),
        llm_calls=len(chats),
        tool_calls=sum(s.operation == "execute_tool" for s in steps),
        errors=sum(s.error is not None for s in steps),
        wall_time_ms=round(max((ended - started).total_seconds() * 1000, 0.0), 3),
    )
    for s in chats:
        u = (s.response or {}).get("usage") or {}
        usage.input_tokens += to_int(u.get("input_tokens")) or 0
        usage.output_tokens += to_int(u.get("output_tokens")) or 0
        usage.total_tokens += to_int(u.get("total_tokens")) or 0
    run_in = run_node.input if run_node is not None else None
    run_out = run_node.output if run_node is not None else None
    if run_in is None:
        first = next(
            (m for s in chats for m in s.request.get("messages") or [] if m.get("role") == "user"),
            None,
        )
        run_in = (
            text_of([p.get("content") for p in first["parts"] if p.get("type") == "text"])
            if first
            else None
        )
    if run_out is None:
        for s in reversed(chats):
            msgs = (s.response or {}).get("messages") or []
            txt = "".join(
                p.get("content") or ""
                for m in msgs
                for p in m.get("parts") or []
                if p.get("type") == "text"
            )
            if txt:
                run_out = txt
                break
    if isinstance(run_in, dict):
        run_input = run_in
    else:
        run_input = {"task": run_in} if run_in not in (None, "") else {}
    run_error = run_node.error if run_node is not None else None
    if run_node is None and len(roots) == 1 and roots[0].kind in _OP:
        run_error = roots[0].error
    run_name = (
        name
        or (run_node.name if run_node is not None else None)
        or (roots[0].name if roots else None)
        or emit[0].name
    )
    meta = {"source": fmt, "trace_id": trace_id}
    meta.update({k: v for k, v in (extra_meta or {}).items() if v is not None})
    return Run(
        run_id=run_id,
        name=str(run_name),
        input=run_input,
        output=run_out,
        status="error" if run_error is not None else "ok",
        error=run_error,
        started_at=started,
        ended_at=ended,
        env={"metadata": meta},
        tags=["imported", f"import:{fmt}", *(tags or [])],
        usage=usage,
        span_id=root_span,
        steps=steps,
    )


def _sort_key(n: Node) -> tuple[float, int]:
    return (n.start.timestamp() if n.start else float("inf"), n.order)
