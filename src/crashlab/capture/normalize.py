"""Provider-native request/response -> one common shape, plus request hashing.

All providers (OpenAI chat.completions, OpenAI responses, Anthropic messages, LiteLLM)
map into the same message / part model, loosely following the OpenTelemetry GenAI
``gen_ai.input.messages`` schema::

    request  = {"system": str | None, "messages": [Message], "tools": [Tool], "params": {...}}
    response = {"messages": [Message], "tool_calls": [ToolCall], "finish_reason": str | None,
                "finish_reasons": [...], "provider_finish_reason": str | None,
                "model": str | None, "id": str | None, "usage": {...}}

    Message  = {"role": "system" | "user" | "assistant" | "tool", "parts": [Part], "name"?}
    Part     = {"type": "text", "content": str}
             | {"type": "tool_call", "id", "name", "arguments": <parsed JSON, or raw str>}
             | {"type": "tool_call_response", "id", "response": str | [Part], "is_error"?}
             | {"type": "image", "source": "url", "url"} | {"type": "image", "source":
               "base64", "media_type", "placeholder"}   (image bytes are never stored)
             | {"type": "reasoning", "content"} | {"type": "refusal", "content"}
             | {"type": "file" | "audio", "placeholder", ...} | {"type": <other>, "data"}
    ToolCall = {"id", "type": "function", "name", "arguments"}
    Tool     = {"type": "function", "name", "description"?, "parameters"?} | {"type": <server
               tool type>, "name"?, "config"}

Provider quirks that are normalized away (so equivalent requests hash the same): dict key
order, string vs. list-of-text-parts content, whitespace inside JSON tool arguments,
Anthropic ``tool_result`` blocks inside user messages (become ``tool`` messages),
``cache_control`` markers, stop vs. stop_sequences, max_tokens aliases and tool_choice
spellings. Transport-only kwargs (headers, timeouts, stream flags, API keys) are dropped.

``request_hash`` is the sha256 of the canonical JSON (sorted keys, compact separators) of
the *redacted* normalized request, so it can be recomputed from a cassette.
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import datetime as _dt
import enum
import hashlib
import json
import re
from typing import Any

_SENTINEL_TYPES = {"NotGiven", "Omit", "_NotGiven"}

# kwargs that never change model behaviour; excluded from the normalized request
TRANSPORT_KWARGS = frozenset(
    {
        "extra_headers",
        "extra_query",
        "timeout",
        "stream",
        "stream_options",
        # LiteLLM routing / credentials / testing knobs
        "api_key",
        "api_base",
        "base_url",
        "api_version",
        "organization",
        "custom_llm_provider",
        "num_retries",
        "max_retries",
        "mock_response",
        "mock_tool_calls",
        "logger_fn",
        "metadata",
    }
)

_DATA_URL = re.compile(r"^data:(?P<mt>[\w.+-]+/[\w.+-]+)?(?:;[^,]*)?;base64,(?P<data>.*)$", re.S)
_BASE64ISH = re.compile(r"^[A-Za-z0-9+/=\s_-]+$")
BINARY_PLACEHOLDER_MIN = 1024  # strings longer than this that look like base64 are elided


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
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return binary_placeholder(bytes(obj))
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
        except Exception:
            pass
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


# --- binary payloads -----------------------------------------------------------------


def binary_placeholder(data: bytes | str, media_type: str | None = None) -> str:
    raw = data.encode("ascii", "ignore") if isinstance(data, str) else data
    if isinstance(data, str):
        try:
            size = len(base64.b64decode(data, validate=False))
        except (binascii.Error, ValueError):
            size = len(raw)
    else:
        size = len(raw)
    digest = hashlib.sha256(raw).hexdigest()[:16]
    return f"<{media_type or 'binary'} {size} bytes sha256:{digest}>"


def strip_binary(obj: Any) -> Any:
    """Replace inline base64 blobs / data: URLs in a JSON-like structure with placeholders."""
    if isinstance(obj, str):
        if obj.startswith("data:"):
            m = _DATA_URL.match(obj)
            if m:
                return binary_placeholder(m.group("data"), m.group("mt"))
        if len(obj) > BINARY_PLACEHOLDER_MIN and " " not in obj[:200] and _BASE64ISH.match(obj):
            return binary_placeholder(obj)
        return obj
    if isinstance(obj, dict):
        return {k: strip_binary(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [strip_binary(v) for v in obj]
    return obj


# --- parts ---------------------------------------------------------------------------


def parse_arguments(args: Any) -> Any:
    """Tool-call arguments as parsed JSON; invalid JSON is kept as the raw string."""
    if args is None:
        return {}
    if isinstance(args, str):
        if not args.strip():
            return {}
        try:
            return json.loads(args)
        except ValueError:
            return args
    return to_jsonable(args)


def _drop_none(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def _image_from_url(url: str | None, detail: Any = None) -> dict[str, Any]:
    if url and url.startswith("data:"):
        m = _DATA_URL.match(url)
        mt = m.group("mt") if m else None
        data = m.group("data") if m else url
        return _drop_none(
            {
                "type": "image",
                "source": "base64",
                "media_type": mt,
                "placeholder": binary_placeholder(data, mt or "image"),
                "detail": detail,
            }
        )
    return _drop_none({"type": "image", "source": "url", "url": url, "detail": detail})


def tool_call_part(call_id: Any, name: Any, arguments: Any) -> dict[str, Any]:
    return {
        "type": "tool_call",
        "id": call_id,
        "name": name,
        "arguments": parse_arguments(arguments),
    }


def normalize_part(block: Any) -> dict[str, Any] | None:
    """One content block/part (any provider) -> common part. None for empty text."""
    block = to_jsonable(block)
    if isinstance(block, str):
        return {"type": "text", "content": block} if block else None
    if not isinstance(block, dict):
        return {"type": "unknown", "data": block}
    t = block.get("type")
    if t in ("text", "input_text", "output_text"):
        text = block.get("text")
        if text is None:
            text = block.get("content")
        return {"type": "text", "content": text} if text else None
    if t == "refusal":
        return {"type": "refusal", "content": block.get("refusal") or block.get("content")}
    if t == "image_url":
        img = block.get("image_url")
        img = img if isinstance(img, dict) else {"url": img}
        return _image_from_url(img.get("url"), img.get("detail"))
    if t == "input_image":
        if block.get("image_url"):
            return _image_from_url(block["image_url"], block.get("detail"))
        return _drop_none({"type": "image", "source": "file", "file_id": block.get("file_id")})
    if t == "image":
        src = block.get("source") or {}
        if src.get("type") == "base64":
            mt = src.get("media_type")
            return _drop_none(
                {
                    "type": "image",
                    "source": "base64",
                    "media_type": mt,
                    "placeholder": binary_placeholder(src.get("data") or "", mt or "image"),
                }
            )
        if src.get("type") == "url":
            return {"type": "image", "source": "url", "url": src.get("url")}
        return _drop_none(
            {"type": "image", "source": src.get("type"), "file_id": src.get("file_id")}
        )
    if t == "input_audio":
        audio = block.get("input_audio") or {}
        fmt = audio.get("format")
        return _drop_none(
            {
                "type": "audio",
                "format": fmt,
                "placeholder": binary_placeholder(audio.get("data") or "", f"audio/{fmt or '?'}"),
            }
        )
    if t in ("file", "input_file", "document"):
        inner = block.get("file") if isinstance(block.get("file"), dict) else block
        src = inner.get("source") if isinstance(inner.get("source"), dict) else {}
        data = inner.get("file_data") or src.get("data")
        out = {
            "type": "file",
            "filename": inner.get("filename") or block.get("title"),
            "file_id": inner.get("file_id"),
            "url": inner.get("file_url") or src.get("url"),
            "media_type": src.get("media_type"),
        }
        if isinstance(data, str) and data:
            out["placeholder"] = (
                strip_binary(data) if data.startswith("data:") else binary_placeholder(data)
            )
        return _drop_none(out)
    if t == "thinking":
        return {"type": "reasoning", "content": block.get("thinking")}
    if t == "redacted_thinking":
        return {"type": "reasoning", "content": "[redacted_thinking]"}
    if t in ("tool_use", "server_tool_use"):
        part = tool_call_part(block.get("id"), block.get("name"), block.get("input"))
        if t == "server_tool_use":
            part["server"] = True
        return part
    if t == "tool_result":
        return tool_result_part(
            block.get("tool_use_id"), block.get("content"), block.get("is_error")
        )
    data = {k: v for k, v in block.items() if k not in ("type", "cache_control")}
    return {"type": t or "unknown", "data": strip_binary(data)}


def content_parts(content: Any) -> list[dict[str, Any]]:
    content = to_jsonable(content)
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "content": content}] if content else []
    if isinstance(content, dict):
        content = [content]
    parts = [normalize_part(b) for b in content]
    return [p for p in parts if p is not None]


def _tool_response_value(content: Any) -> Any:
    """Tool result content: plain string when it is all text, else a list of parts."""
    content = to_jsonable(content)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content
    parts = content_parts(content)
    if all(p["type"] == "text" for p in parts):
        return "".join(p["content"] for p in parts)
    return parts


def tool_result_part(call_id: Any, content: Any, is_error: Any = None) -> dict[str, Any]:
    part: dict[str, Any] = {
        "type": "tool_call_response",
        "id": call_id,
        "response": _tool_response_value(content),
    }
    if is_error:
        part["is_error"] = True
    return part


# --- messages ------------------------------------------------------------------------


def _message(role: str, parts: list[dict[str, Any]], name: Any = None) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": role, "parts": parts}
    if name:
        msg["name"] = name
    return msg


def normalize_messages(messages: Any) -> list[dict[str, Any]]:
    """Messages from OpenAI chat, Anthropic, LiteLLM or OpenAI Responses input items."""
    if messages is None:
        return []
    if isinstance(messages, str):  # Responses API: input="..."
        return [_message("user", content_parts(messages))]
    out: list[dict[str, Any]] = []
    for raw in messages:
        m = to_jsonable(raw)
        if not isinstance(m, dict):
            out.append(_message("user", content_parts(m)))
            continue
        item_type = m.get("type")
        role = m.get("role")

        # OpenAI Responses API input items
        if item_type == "function_call":
            part = tool_call_part(
                m.get("call_id") or m.get("id"), m.get("name"), m.get("arguments")
            )
            if out and out[-1]["role"] == "assistant":
                out[-1]["parts"].append(part)
            else:
                out.append(_message("assistant", [part]))
            continue
        if item_type == "function_call_output":
            out.append(_message("tool", [tool_result_part(m.get("call_id"), m.get("output"))]))
            continue
        if item_type == "reasoning":
            summary = "".join(
                s.get("text", "") for s in m.get("summary") or [] if isinstance(s, dict)
            )
            part = {"type": "reasoning", "content": summary}
            if out and out[-1]["role"] == "assistant":
                out[-1]["parts"].append(part)
            else:
                out.append(_message("assistant", [part]))
            continue

        if role == "tool":  # OpenAI chat / LiteLLM tool result
            out.append(
                _message("tool", [tool_result_part(m.get("tool_call_id"), m.get("content"))])
            )
            continue
        if role == "function":  # legacy OpenAI function result
            out.append(
                _message(
                    "tool", [tool_result_part(m.get("name"), m.get("content"))], name=m.get("name")
                )
            )
            continue

        role = "system" if role == "developer" else (role or "user")
        content = m.get("content")
        blocks = content if isinstance(content, list) else None
        parts: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        if blocks is not None:
            for b in blocks:
                p = normalize_part(b)
                if p is None:
                    continue
                (tool_results if p["type"] == "tool_call_response" else parts).append(p)
        else:
            parts = content_parts(content)
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            parts.append(tool_call_part(tc.get("id"), fn.get("name"), fn.get("arguments")))
        if m.get("function_call"):
            fc = m["function_call"]
            parts.append(tool_call_part(None, fc.get("name"), fc.get("arguments")))
        if m.get("refusal"):
            parts.append({"type": "refusal", "content": m["refusal"]})
        # Anthropic: tool_result blocks in a user turn become separate tool messages
        for tr in tool_results:
            out.append(_message("tool", [tr]))
        if parts or not tool_results:
            out.append(_message(role, parts, m.get("name")))
    return out


def system_text(system: Any) -> str | None:
    """Anthropic ``system`` (str or text blocks) / Responses ``instructions`` -> str."""
    if system is None:
        return None
    parts = content_parts(system)
    texts = [p["content"] for p in parts if p["type"] == "text"]
    return "\n\n".join(texts) if texts else None


def split_leading_system(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    texts: list[str] = []
    i = 0
    while i < len(messages) and messages[i]["role"] == "system":
        texts.extend(p["content"] for p in messages[i]["parts"] if p["type"] == "text")
        i += 1
    return ("\n\n".join(texts) if texts else None), messages[i:]


# --- tools / params ------------------------------------------------------------------


def normalize_tools(tools: Any) -> list[dict[str, Any]]:
    out = []
    for raw in to_jsonable(tools) or []:
        if not isinstance(raw, dict):
            continue
        t = raw.get("type")
        if t == "function" and isinstance(raw.get("function"), dict):
            fn = raw["function"]
            out.append(
                _drop_none(
                    {
                        "type": "function",
                        "name": fn.get("name"),
                        "description": fn.get("description"),
                        "parameters": fn.get("parameters"),
                        "strict": fn.get("strict"),
                    }
                )
            )
        elif t == "function" or ("input_schema" in raw and t in (None, "custom")):
            out.append(
                _drop_none(
                    {
                        "type": "function",
                        "name": raw.get("name"),
                        "description": raw.get("description"),
                        "parameters": raw.get("parameters", raw.get("input_schema")),
                        "strict": raw.get("strict"),
                    }
                )
            )
        else:
            config = {k: v for k, v in raw.items() if k not in ("type", "name", "cache_control")}
            out.append(_drop_none({"type": t, "name": raw.get("name"), "config": config or None}))
    return out


def normalize_tool_choice(choice: Any) -> Any:
    choice = to_jsonable(choice)
    if choice is None:
        return None
    if isinstance(choice, str):
        return {"type": {"any": "required"}.get(choice, choice)}
    if isinstance(choice, dict):
        t = choice.get("type")
        if t == "function":
            name = (choice.get("function") or {}).get("name") or choice.get("name")
            return {"type": "tool", "name": name}
        if t == "tool":
            return {"type": "tool", "name": choice.get("name")}
        if t in ("auto", "none", "any", "required"):
            return {"type": {"any": "required"}.get(t, t)}
    return choice


def normalize_params(data: dict[str, Any], renames: dict[str, str] | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for k, v in data.items():
        if k in TRANSPORT_KWARGS or v is None:
            continue
        params[(renames or {}).get(k, k)] = v
    if "stop" in params and isinstance(params["stop"], str):
        params["stop"] = [params["stop"]]
    if "tool_choice" in params:
        tc = to_jsonable(params["tool_choice"])
        if isinstance(tc, dict) and tc.get("disable_parallel_tool_use") is not None:
            params.setdefault("parallel_tool_calls", not tc["disable_parallel_tool_use"])
        params["tool_choice"] = normalize_tool_choice(tc)
    return params


def _kwargs_jsonable(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {k: to_jsonable(v) for k, v in kwargs.items() if not is_not_given(v) and v is not None}


# --- requests ------------------------------------------------------------------------


def normalize_openai_chat_request(kwargs: dict[str, Any]) -> dict[str, Any]:
    """OpenAI chat.completions (and LiteLLM completion) kwargs -> common request."""
    data = _kwargs_jsonable(kwargs)
    messages = normalize_messages(data.pop("messages", None) or [])
    tools = normalize_tools(data.pop("tools", None) or [])
    if "functions" in data:  # legacy function calling
        tools += normalize_tools(
            [{"type": "function", "function": f} for f in data.pop("functions")]
        )
    system, messages = split_leading_system(messages)
    return {
        "system": system,
        "messages": messages,
        "tools": tools,
        "params": normalize_params(data, {"max_completion_tokens": "max_tokens"}),
    }


def normalize_anthropic_request(kwargs: dict[str, Any]) -> dict[str, Any]:
    data = _kwargs_jsonable(kwargs)
    system = system_text(data.pop("system", None))
    messages = normalize_messages(data.pop("messages", None) or [])
    tools = normalize_tools(data.pop("tools", None) or [])
    return {
        "system": system,
        "messages": messages,
        "tools": tools,
        "params": normalize_params(data, {"stop_sequences": "stop"}),
    }


def normalize_openai_responses_request(kwargs: dict[str, Any]) -> dict[str, Any]:
    data = _kwargs_jsonable(kwargs)
    messages = normalize_messages(data.pop("input", None))
    instructions = system_text(data.pop("instructions", None))
    lead, messages = split_leading_system(messages)
    system = "\n\n".join(s for s in (instructions, lead) if s) or None
    tools = normalize_tools(data.pop("tools", None) or [])
    return {
        "system": system,
        "messages": messages,
        "tools": tools,
        "params": normalize_params(data, {"max_output_tokens": "max_tokens"}),
    }


# --- responses -----------------------------------------------------------------------

_OPENAI_FINISH = {"function_call": "tool_calls"}
_ANTHROPIC_FINISH = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "refusal": "content_filter",
    "pause_turn": "pause",
}


def _usage(inp: Any, out: Any, total: Any = None, **extra: Any) -> dict[str, Any]:
    usage = {
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": total
        if total is not None
        else ((inp or 0) + (out or 0) if (inp is not None or out is not None) else None),
    }
    usage.update({k: v for k, v in extra.items() if v})
    return usage


def _tool_calls_from_parts(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"id": p["id"], "type": "function", "name": p["name"], "arguments": p["arguments"]}
        for p in parts
        if p["type"] == "tool_call"
    ]


def _as_dict(resp: Any) -> dict[str, Any]:
    raw = to_jsonable(resp)
    return raw if isinstance(raw, dict) else {"value": raw}


def normalize_openai_chat_response(resp: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(normalized, raw)`` for a ChatCompletion-like object (OpenAI / LiteLLM)."""
    raw = _as_dict(resp)
    messages: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    provider_reasons: list[Any] = []
    for choice in raw.get("choices") or []:
        msg = dict(choice.get("message") or {})
        msg["role"] = msg.get("role") or "assistant"
        msg.pop("tool_call_id", None)
        norm = normalize_messages([msg])
        message = norm[0] if norm else _message("assistant", [])
        messages.append(message)
        provider_reasons.append(choice.get("finish_reason"))
        if not tool_calls:  # tool calls of the first choice drive the loop
            tool_calls = _tool_calls_from_parts(message["parts"])
    u = raw.get("usage") or {}
    details_in = u.get("prompt_tokens_details") or {}
    details_out = u.get("completion_tokens_details") or {}
    usage = _usage(
        u.get("prompt_tokens", u.get("input_tokens")),
        u.get("completion_tokens", u.get("output_tokens")),
        u.get("total_tokens"),
        cache_read_input_tokens=details_in.get("cached_tokens"),
        reasoning_tokens=details_out.get("reasoning_tokens"),
    )
    reasons = [_OPENAI_FINISH.get(r, r) for r in provider_reasons]
    normalized = {
        "messages": messages,
        "tool_calls": tool_calls,
        "finish_reason": reasons[0] if reasons else None,
        "finish_reasons": reasons,
        "provider_finish_reason": provider_reasons[0] if provider_reasons else None,
        "model": raw.get("model"),
        "id": raw.get("id"),
        "usage": usage,
    }
    return normalized, raw


def normalize_anthropic_response(resp: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = _as_dict(resp)
    parts = content_parts(raw.get("content") or [])
    message = _message("assistant", parts)
    stop = raw.get("stop_reason")
    reason = _ANTHROPIC_FINISH.get(stop, stop) if stop else None
    u = raw.get("usage") or {}
    cache_read = u.get("cache_read_input_tokens") or 0
    cache_write = u.get("cache_creation_input_tokens") or 0
    inp = u.get("input_tokens")
    usage = _usage(
        (inp or 0) + cache_read + cache_write if inp is not None else None,
        u.get("output_tokens"),
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
    )
    normalized = {
        "messages": [message],
        "tool_calls": _tool_calls_from_parts(parts),
        "finish_reason": reason,
        "finish_reasons": [reason] if reason else [],
        "provider_finish_reason": stop,
        "model": raw.get("model"),
        "id": raw.get("id"),
        "usage": usage,
    }
    return normalized, raw


def normalize_openai_responses_response(resp: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = _as_dict(resp)
    parts: list[dict[str, Any]] = []
    for item in raw.get("output") or []:
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        if t == "message":
            parts.extend(content_parts(item.get("content") or []))
        elif t == "function_call":
            parts.append(
                tool_call_part(item.get("call_id"), item.get("name"), item.get("arguments"))
            )
        elif t == "reasoning":
            summary = "".join(s.get("text", "") for s in item.get("summary") or [])
            if summary:
                parts.append({"type": "reasoning", "content": summary})
        else:
            data = {k: v for k, v in item.items() if k != "type"}
            parts.append({"type": t or "unknown", "data": strip_binary(data)})
    tool_calls = _tool_calls_from_parts(parts)
    status = raw.get("status")
    incomplete = (raw.get("incomplete_details") or {}).get("reason")
    if status == "incomplete":
        reason = "length" if incomplete == "max_output_tokens" else (incomplete or "incomplete")
    elif status == "failed":
        reason = "error"
    elif tool_calls:
        reason = "tool_calls"
    elif status in ("completed", None):
        reason = "stop"
    else:
        reason = status
    u = raw.get("usage") or {}
    usage = _usage(
        u.get("input_tokens"),
        u.get("output_tokens"),
        u.get("total_tokens"),
        cache_read_input_tokens=(u.get("input_tokens_details") or {}).get("cached_tokens"),
        reasoning_tokens=(u.get("output_tokens_details") or {}).get("reasoning_tokens"),
    )
    normalized = {
        "messages": [_message("assistant", parts)],
        "tool_calls": tool_calls,
        "finish_reason": reason,
        "finish_reasons": [reason] if reason else [],
        "provider_finish_reason": incomplete or status,
        "model": raw.get("model"),
        "id": raw.get("id"),
        "usage": usage,
    }
    return normalized, raw


def message_text(message: dict[str, Any]) -> str:
    """Concatenated text parts of a normalized message."""
    return "".join(p.get("content") or "" for p in message.get("parts", []) if p["type"] == "text")
