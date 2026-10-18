"""Provider-native request/response -> crashlab's common shape, plus request hashing.

Common shapes::

    request  = {"system": str | None, "messages": [...], "tools": [...], "params": {...}}
    response = {"messages": [...], "tool_calls": [{"id","type","name","arguments"}],
                "finish_reason": str | None, "model": str | None, "id": str | None,
                "usage": {"input_tokens","output_tokens","total_tokens"}}

``request_hash`` is the sha256 of the canonical JSON (sorted keys, compact separators)
of the *redacted* normalized request, so it can be recomputed from a cassette.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import enum
import hashlib
import json
from typing import Any

_SENTINEL_TYPES = {"NotGiven", "Omit", "_NotGiven", "NoneType_"}
# kwargs that never change model behaviour and are excluded from the normalized request
_TRANSPORT_KWARGS = {"extra_headers", "extra_query", "timeout"}


def is_not_given(obj: Any) -> bool:
    return type(obj).__name__ in _SENTINEL_TYPES


def to_jsonable(obj: Any, _depth: int = 0) -> Any:
    """Best-effort conversion of SDK objects / arbitrary values into JSON-able data."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else str(obj)
    if _depth > 50:
        return repr(obj)
    if isinstance(obj, enum.Enum):
        return to_jsonable(obj.value, _depth + 1)
    if isinstance(obj, (_dt.datetime, _dt.date, _dt.time)):
        return obj.isoformat()
    if isinstance(obj, (bytes, bytearray)):
        return f"<{len(obj)} bytes>"
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, _depth + 1) for k, v in obj.items() if not is_not_given(v)}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v, _depth + 1) for v in obj]
    if is_not_given(obj):
        return None
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return to_jsonable(dump(mode="json"), _depth + 1)
        except TypeError:
            return to_jsonable(dump(), _depth + 1)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj), _depth + 1)
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            return to_jsonable(to_dict(), _depth + 1)
        except Exception:
            pass
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return {
            k: to_jsonable(v, _depth + 1)
            for k, v in vars(obj).items()
            if not k.startswith("_") and not callable(v) and not is_not_given(v)
        }
    return repr(obj)


def canonical_json(obj: Any) -> str:
    return json.dumps(
        to_jsonable(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )


def request_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


# --- OpenAI chat.completions -----------------------------------------------------------


def _content_text(content: Any) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text", "")))
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    return str(content)


def normalize_openai_chat_request(kwargs: dict[str, Any]) -> dict[str, Any]:
    data = {k: to_jsonable(v) for k, v in kwargs.items() if not is_not_given(v)}
    messages = list(data.pop("messages", None) or [])
    tools = data.pop("tools", None) or []
    for k in _TRANSPORT_KWARGS:
        data.pop(k, None)

    system_parts: list[str] = []
    while (
        messages
        and isinstance(messages[0], dict)
        and messages[0].get("role")
        in (
            "system",
            "developer",
        )
    ):
        text = _content_text(messages.pop(0).get("content"))
        if text:
            system_parts.append(text)

    return {
        "system": "\n\n".join(system_parts) if system_parts else None,
        "messages": messages,
        "tools": tools,
        "params": data,
    }


def _normalize_tool_call(tc: dict[str, Any]) -> dict[str, Any]:
    fn = tc.get("function") or {}
    return {
        "id": tc.get("id"),
        "type": tc.get("type") or "function",
        "name": fn.get("name") if fn else tc.get("name"),
        "arguments": fn.get("arguments") if fn else tc.get("arguments"),
    }


def normalize_openai_chat_response(resp: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(normalized, raw)`` for a ChatCompletion-like object."""
    raw = to_jsonable(resp)
    if not isinstance(raw, dict):
        raw = {"value": raw}
    choices = raw.get("choices") or []
    messages: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    finish_reasons: list[str | None] = []
    for choice in choices:
        msg = dict(choice.get("message") or {})
        calls = [_normalize_tool_call(tc) for tc in (msg.get("tool_calls") or [])]
        out: dict[str, Any] = {
            "role": msg.get("role") or "assistant",
            "content": msg.get("content"),
        }
        if calls:
            out["tool_calls"] = calls
        if msg.get("refusal"):
            out["refusal"] = msg["refusal"]
        messages.append(out)
        finish_reasons.append(choice.get("finish_reason"))
        if not tool_calls:  # tool calls of the first choice drive the loop
            tool_calls = calls
    usage_raw = raw.get("usage") or {}
    inp = usage_raw.get("prompt_tokens", usage_raw.get("input_tokens"))
    outp = usage_raw.get("completion_tokens", usage_raw.get("output_tokens"))
    usage = {
        "input_tokens": inp,
        "output_tokens": outp,
        "total_tokens": usage_raw.get("total_tokens")
        if usage_raw.get("total_tokens") is not None
        else ((inp or 0) + (outp or 0) if (inp is not None or outp is not None) else None),
    }
    normalized = {
        "messages": messages,
        "tool_calls": tool_calls,
        "finish_reason": finish_reasons[0] if finish_reasons else None,
        "finish_reasons": finish_reasons,
        "model": raw.get("model"),
        "id": raw.get("id"),
        "usage": usage,
    }
    return normalized, raw
