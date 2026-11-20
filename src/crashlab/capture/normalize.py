"""Provider-native request/response -> one common shape, plus request hashing.

All providers (OpenAI chat.completions, OpenAI responses, Anthropic messages, LiteLLM,
Google GenAI ``generate_content``) map into the same message / part model, loosely
following the OpenTelemetry GenAI ``gen_ai.input.messages`` schema::

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
import decimal
import enum
import hashlib
import json
import os
import re
import sys
import uuid
from pathlib import PurePath
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


def huge_int(n: int) -> bool:
    """An int too long for ``str()`` / ``json.dumps`` (``sys.set_int_max_str_digits``,
    4300 digits by default): stored as its hex string, which has no such limit."""
    bits = n.bit_length()
    if bits < 2000:  # ~600 digits: under the smallest limit Python allows (640)
        return False
    limit = getattr(sys, "get_int_max_str_digits", lambda: 0)()
    return bool(limit) and bits * 0.30103 >= limit - 2


def to_jsonable(obj: Any, _depth: int = 0) -> Any:
    """Best-effort conversion of SDK objects / arbitrary values into JSON-able data."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return obj if not huge_int(obj) else hex(obj)
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else str(obj)
    if _depth > 50:
        return repr(obj)
    if isinstance(obj, enum.Enum):
        return to_jsonable(obj.value, _depth + 1)
    if isinstance(obj, (_dt.datetime, _dt.date, _dt.time)):
        return obj.isoformat()
    if isinstance(obj, _dt.timedelta):
        return obj.total_seconds()
    if isinstance(obj, (decimal.Decimal, uuid.UUID)):
        return str(obj)
    if isinstance(obj, PurePath):
        return os.fspath(obj)
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return binary_placeholder(bytes(obj))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, _depth + 1) for k, v in obj.items() if not is_not_given(v)}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v, _depth + 1) for v in obj]
    if is_not_given(obj):
        return None
    if isinstance(obj, type):
        # a class passed as a value (``response_format=MyModel``): its schema, not an instance
        name = f"{obj.__module__}.{obj.__qualname__}"
        schema = getattr(obj, "model_json_schema", None)
        if callable(schema):
            try:
                return {"class": obj.__qualname__, "schema": to_jsonable(schema(), _depth + 1)}
            except Exception:
                pass
        return {"class": name}
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            # warnings=False: ParsedChatCompletion's generic ``parsed`` field warns on dump
            return to_jsonable(dump(mode="json", warnings=False), _depth + 1)
        except TypeError:
            return to_jsonable(dump(), _depth + 1)
        except Exception:
            pass
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # field by field, not dataclasses.asdict: asdict deep-copies every value and
        # raises on locks / clients / sockets held by framework context objects
        return {
            f.name: to_jsonable(getattr(obj, f.name, None), _depth + 1)
            for f in dataclasses.fields(obj)
        }
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
    return hashlib.sha256(canonical_json(obj).encode("utf-8", "surrogatepass")).hexdigest()


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


# Keys whose long base64 values are media payloads (image/audio/file bytes).
_MEDIA_KEYS = frozenset(
    {
        "data",
        "b64_json",
        "file_data",
        "bytes",
        "blob",
        "image",
        "audio",
        "inline_data",
        "inlineData",
    }
)
# Containers whose ``data`` is an opaque token the provider needs back verbatim.
_OPAQUE_TYPES = frozenset({"redacted_thinking"})


def strip_binary(obj: Any, _key: str | None = None) -> Any:
    """Replace inline media (base64 blobs, data: URLs) in a JSON-like structure with placeholders.

    Only ``data:`` URLs and long base64 values under media keys (``data``, ``b64_json``,
    ``file_data``...) are elided. Opaque provider tokens that must round-trip verbatim
    (Gemini ``thought_signature``, Anthropic ``signature`` / ``redacted_thinking.data``,
    OpenAI ``encrypted_content``) are kept, so a recording can still be replayed.
    """
    if isinstance(obj, str):
        if obj.startswith("data:"):
            m = _DATA_URL.match(obj)
            if m:
                return binary_placeholder(m.group("data"), m.group("mt"))
        if (
            _key in _MEDIA_KEYS
            and len(obj) > BINARY_PLACEHOLDER_MIN
            and " " not in obj[:200]
            and _BASE64ISH.match(obj)
        ):
            return binary_placeholder(obj)
        return obj
    if isinstance(obj, dict):
        if obj.get("type") in _OPAQUE_TYPES:
            return obj
        return {k: strip_binary(v, k if isinstance(k, str) else None) for k, v in obj.items()}
    if isinstance(obj, list):
        return [strip_binary(v, _key) for v in obj]
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
    # Google GenAI parts are often untyped: {text}, {function_call}, {function_response}
    if not t:
        text = block.get("text")
        if isinstance(text, str) and text:
            if block.get("thought"):
                return {"type": "reasoning", "content": text}
            return {"type": "text", "content": text}
        fc = block.get("function_call") or block.get("functionCall")
        if isinstance(fc, dict):
            args = fc.get("args") if fc.get("args") is not None else fc.get("arguments")
            return tool_call_part(fc.get("id") or block.get("id"), fc.get("name"), args)
        fr = block.get("function_response") or block.get("functionResponse")
        if isinstance(fr, dict):
            return tool_result_part(
                fr.get("id") or block.get("id"),
                fr.get("response") if fr.get("response") is not None else fr.get("parts"),
            )
        inline = block.get("inline_data") or block.get("inlineData")
        if isinstance(inline, dict):
            mt = inline.get("mime_type") or inline.get("mimeType")
            data = inline.get("data") or ""
            kind = "image" if str(mt or "").startswith("image") else "file"
            raw = data if isinstance(data, (str, bytes, bytearray)) else str(data)
            return _drop_none(
                {
                    "type": kind,
                    "source": "base64",
                    "media_type": mt,
                    "placeholder": binary_placeholder(raw, mt),
                }
            )
        file_data = block.get("file_data") or block.get("fileData")
        if isinstance(file_data, dict):
            return _drop_none(
                {
                    "type": "file",
                    "url": file_data.get("file_uri") or file_data.get("fileUri"),
                    "media_type": file_data.get("mime_type") or file_data.get("mimeType"),
                }
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
        if role == "assistant":
            reasoning = m.get("reasoning_content") or m.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                parts.insert(0, {"type": "reasoning", "content": reasoning})
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


def normalize_openai_embeddings_request(kwargs: dict[str, Any]) -> dict[str, Any]:
    """``embeddings.create`` kwargs -> common request (the inputs stay as given)."""
    data = _kwargs_jsonable(kwargs)
    inputs = data.pop("input", None)
    return {"input": inputs, "params": normalize_params(data, {})}


def normalize_openai_embeddings_response(resp: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """The vectors live only in the raw response (replay rebuilds from it); the normalized
    response keeps a summary that is readable in ``crashlab show`` and the report."""
    raw = to_jsonable(resp)
    raw = raw if isinstance(raw, dict) else {"data": raw}
    data = [d for d in raw.get("data") or [] if isinstance(d, dict)]
    usage = raw.get("usage") or {}
    norm = {
        "model": raw.get("model"),
        "embeddings": [
            {
                "index": d.get("index"),
                "dimensions": len(d["embedding"]) if isinstance(d.get("embedding"), list) else None,
            }
            for d in data
        ],  # fmt: skip
        "usage": {
            "input_tokens": usage.get("prompt_tokens"),
            "total_tokens": usage.get("total_tokens"),
        },
    }
    return norm, raw


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
    reasons = [
        # Gemini's OpenAI-compatible stream reports "stop" on a tool-calling turn.
        "tool_calls"
        if r == "stop" and any(p["type"] == "tool_call" for p in msg["parts"])
        else _OPENAI_FINISH.get(r, r)
        for r, msg in zip(provider_reasons, messages, strict=True)
    ]
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


# --- Google GenAI (google.genai / google-genai SDK) -----------------------------------

_GENAI_FINISH = {
    "STOP": "stop",
    "END_TURN": "stop",
    "MAX_TOKENS": "length",
    "MAX_OUTPUT_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
    "IMAGE_SAFETY": "content_filter",
    "LANGUAGE": "stop",
    "MALFORMED_FUNCTION_CALL": "error",
    "OTHER": "other",
    "FINISH_REASON_UNSPECIFIED": None,
}


def _as_map(value: Any) -> dict[str, Any]:
    value = to_jsonable(value)
    return value if isinstance(value, dict) else {}


def _pick(data: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in data and data[name] is not None:
            return data[name]
    return default


def _role(role: Any) -> str:
    r = str(role or "user").lower()
    if r in ("model", "assistant"):
        return "assistant"
    if r in ("function", "tool"):
        return "tool"
    if r in ("system", "developer"):
        return "system"
    return "user"


def _contents_as_messages(contents: Any) -> list[Any]:
    """Google ``contents`` (string, Part, Content, or a list of those) -> message dicts."""
    contents = to_jsonable(contents)
    if contents is None or contents == []:
        return []
    if isinstance(contents, str):
        return [{"role": "user", "content": contents}]
    if isinstance(contents, dict):
        contents = [contents]
    if not isinstance(contents, list):
        return [{"role": "user", "content": contents}]
    out: list[Any] = []
    for item in contents:
        if item is None:
            continue
        if isinstance(item, str):
            out.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            out.append({"role": "user", "content": item})
            continue
        # A Content has parts and/or a role; a bare Part has text / function_call / ...
        if "parts" in item or "role" in item:
            role = _role(item.get("role"))
            parts = item.get("parts")
            out.append({"role": role, "content": parts if parts is not None else item.get("text")})
        else:
            out.append({"role": "user", "content": [item]})
    return out


def _genai_tools(config: dict[str, Any], extra: Any = None) -> list[dict[str, Any]]:
    raw_tools = []
    if extra:
        raw_tools.extend(to_jsonable(extra) or [])
    tools = _pick(config, "tools", default=None)
    if tools:
        raw_tools.extend(tools if isinstance(tools, list) else [tools])
    out: list[dict[str, Any]] = []
    for tool in raw_tools:
        tool = to_jsonable(tool)
        if not isinstance(tool, dict):
            continue
        decls = _pick(tool, "function_declarations", "functionDeclarations", default=None)
        if isinstance(decls, list):
            for d in decls:
                d = to_jsonable(d) or {}
                if not isinstance(d, dict):
                    continue
                out.append(
                    _drop_none(
                        {
                            "type": "function",
                            "name": d.get("name"),
                            "description": d.get("description"),
                            "parameters": _pick(
                                d, "parameters", "parameters_json_schema", "parametersJsonSchema"
                            ),
                        }
                    )
                )
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        if fn.get("name") or tool.get("type") == "function":
            out.append(
                _drop_none(
                    {
                        "type": "function",
                        "name": fn.get("name") or tool.get("name"),
                        "description": fn.get("description") or tool.get("description"),
                        "parameters": _pick(
                            fn, "parameters", "parameters_json_schema", "input_schema"
                        )
                        or _pick(tool, "parameters", "parameters_json_schema", "input_schema"),
                    }
                )
            )
            continue
        # built-in tools (google_search, code_execution, ...)
        config_keys = {
            k: v
            for k, v in tool.items()
            if k not in ("type", "name", "function_declarations", "functionDeclarations")
        }
        typ = tool.get("type") or next(iter(config_keys), None)
        out.append(
            _drop_none({"type": typ, "name": tool.get("name"), "config": config_keys or None})
        )
    return out


def normalize_google_genai_request(kwargs: dict[str, Any]) -> dict[str, Any]:
    """``generate_content`` / ``send_message`` kwargs (native or fork-patched) -> common request."""
    data = _kwargs_jsonable(kwargs)
    config = _as_map(data.pop("config", None))
    # Fork patches (family "chat") edit top-level messages / tools / system
    messages_kw = data.pop("messages", None)
    tools_kw = data.pop("tools", None)
    system_kw = data.pop("system", None)
    contents = data.pop("contents", None)
    if contents is not None:
        messages = normalize_messages(_contents_as_messages(contents))
    else:
        messages = normalize_messages(messages_kw or [])
    system = system_text(_pick(config, "system_instruction", "systemInstruction"))
    if system is None:
        system = system_text(system_kw)
    lead, messages = split_leading_system(messages)
    if lead:
        system = "\n\n".join(s for s in (system, lead) if s) or None
    tools = _genai_tools(config, tools_kw)
    # Lift sampling fields out of config so gen_ai.request.* attributes are set
    lifted = {}
    for src, dest in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("topP", "top_p"),
        ("top_k", "top_k"),
        ("topK", "top_k"),
        ("max_output_tokens", "max_tokens"),
        ("maxOutputTokens", "max_tokens"),
        ("seed", "seed"),
        ("stop_sequences", "stop"),
        ("stopSequences", "stop"),
        ("candidate_count", "n"),
        ("candidateCount", "n"),
    ):
        if src in config and dest not in lifted and dest not in data:
            lifted[dest] = config[src]
    # leftover config (safety, thinking, response schema, ...) stays under params.config
    keep = {
        k: v
        for k, v in config.items()
        if k
        not in {
            "system_instruction",
            "systemInstruction",
            "tools",
            "temperature",
            "top_p",
            "topP",
            "top_k",
            "topK",
            "max_output_tokens",
            "maxOutputTokens",
            "seed",
            "stop_sequences",
            "stopSequences",
            "candidate_count",
            "candidateCount",
        }
        and v is not None
    }
    data.update(lifted)
    if keep:
        data["config"] = keep
    if "model" in data and isinstance(data["model"], str) and data["model"].startswith("models/"):
        data["model"] = data["model"][7:]
    return {
        "system": system,
        "messages": messages,
        "tools": tools,
        "params": normalize_params(data, {"max_output_tokens": "max_tokens"}),
    }


def _candidates(raw: dict[str, Any]) -> list[dict[str, Any]]:
    cands = raw.get("candidates") or []
    return [c for c in cands if isinstance(c, dict)]


def _parts_from_raw(raw: dict[str, Any]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for cand in _candidates(raw):
        content = cand.get("content") or {}
        if isinstance(content, dict):
            for p in content.get("parts") or []:
                n = normalize_part(p)
                if n is not None:
                    parts.append(n)
    return parts


def _text_from_raw(resp: Any, raw: dict[str, Any]) -> str:
    text = getattr(resp, "text", None)
    if isinstance(text, str):
        return text
    if isinstance(raw.get("text"), str):
        return raw["text"]
    return "".join(p["content"] for p in _parts_from_raw(raw) if p["type"] == "text")


def _function_calls_from_raw(resp: Any, raw: dict[str, Any]) -> list[dict[str, Any]]:
    calls = getattr(resp, "function_calls", None)
    if calls is None:
        calls = raw.get("function_calls")
    out: list[dict[str, Any]] = []
    if calls:
        for fc in to_jsonable(calls) or []:
            if not isinstance(fc, dict):
                continue
            inner = fc.get("function_call") if isinstance(fc.get("function_call"), dict) else fc
            out.append(
                _drop_none(
                    {
                        "id": inner.get("id"),
                        "name": inner.get("name"),
                        "args": inner.get("args")
                        if inner.get("args") is not None
                        else inner.get("arguments"),
                    }
                )
            )
        if out:
            return out
    return [
        _drop_none({"id": p.get("id"), "name": p.get("name"), "args": p.get("arguments")})
        for p in _parts_from_raw(raw)
        if p["type"] == "tool_call"
    ]


def _provider_finish(raw: dict[str, Any]) -> str | None:
    for cand in _candidates(raw):
        reason = cand.get("finish_reason") or cand.get("finishReason")
        if reason is not None:
            return str(reason)
    for key in ("finish_reason", "finishReason"):
        if raw.get(key) is not None:
            return str(raw[key])
    return None


def _canon_finish(provider: str | None, tool_calls: list[Any]) -> str | None:
    if tool_calls:
        return "tool_calls"
    if not provider:
        return "stop"
    key = provider.split(".")[-1] if provider else provider
    mapped = _GENAI_FINISH.get(
        key, _GENAI_FINISH.get(key.upper() if key else "", key.lower() if key else None)
    )
    if mapped is not None or key in _GENAI_FINISH or (key or "").upper() in _GENAI_FINISH:
        return mapped
    return key.lower() if key else None


def _usage_from_metadata(meta: Any) -> dict[str, Any]:
    u = _as_map(meta)
    inp = _pick(u, "prompt_token_count", "promptTokenCount")
    out = _pick(u, "candidates_token_count", "candidatesTokenCount")
    total = _pick(u, "total_token_count", "totalTokenCount")
    cached = _pick(u, "cached_content_token_count", "cachedContentTokenCount")
    thinking = _pick(u, "thoughts_token_count", "thoughtsTokenCount")
    return _usage(
        inp,
        out,
        total,
        cache_read_input_tokens=cached,
        reasoning_tokens=thinking,
    )


def normalize_google_genai_response(resp: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """``GenerateContentResponse`` (or a stream assembly / ReplayObject) -> common + raw."""
    raw = _as_dict(resp)
    text = _text_from_raw(resp, raw)
    function_calls = _function_calls_from_raw(resp, raw)
    parts = _parts_from_raw(raw)
    if not parts and text:
        parts = [{"type": "text", "content": text}]
    if not any(p["type"] == "tool_call" for p in parts) and function_calls:
        for fc in function_calls:
            parts.append(tool_call_part(fc.get("id"), fc.get("name"), fc.get("args")))
    tool_calls = _tool_calls_from_parts(parts)
    provider_reason = _provider_finish(raw)
    reason = _canon_finish(provider_reason, tool_calls)
    usage_meta = raw.get("usage_metadata") or raw.get("usageMetadata")
    if usage_meta is None:
        usage_meta = getattr(resp, "usage_metadata", None)
    model = raw.get("model_version") or raw.get("modelVersion") or raw.get("model")
    if not model:
        model = getattr(resp, "model_version", None) or getattr(resp, "model", None)
    resp_id = raw.get("response_id") or raw.get("responseId") or raw.get("id")
    # Replay without google-genai: ReplayObject needs these computed properties as fields
    if "text" not in raw or raw.get("text") is None:
        raw["text"] = text
    if raw.get("function_calls") is None:
        raw["function_calls"] = function_calls
    if not raw.get("candidates") and (text or function_calls):
        raw_parts: list[dict[str, Any]] = []
        if text:
            raw_parts.append({"text": text})
        for fc in function_calls:
            raw_parts.append(
                {
                    "function_call": _drop_none(
                        {"id": fc.get("id"), "name": fc.get("name"), "args": fc.get("args")}
                    )
                }
            )
        raw["candidates"] = [
            {
                "content": {"role": "model", "parts": raw_parts},
                "finish_reason": provider_reason or ("STOP" if reason == "stop" else reason),
            }
        ]
    normalized = {
        "messages": [_message("assistant", parts)],
        "tool_calls": tool_calls,
        "finish_reason": reason,
        "finish_reasons": [reason] if reason else [],
        "provider_finish_reason": provider_reason,
        "model": model,
        "id": resp_id,
        "usage": _usage_from_metadata(usage_meta),
    }
    return normalized, raw
