"""Recording proxy for ``google.genai.Client`` (the ``google-genai`` SDK).

Intercepts ``client.models.generate_content`` / ``generate_content_stream``,
``client.chats.create`` (wrapping ``send_message`` / ``send_message_stream``), and the
async twins on ``client.aio``. ``google.genai`` is never imported: any object with the
same shape is wrapped, and replay rebuilds a lightweight object so ``response.text``,
``response.function_calls`` and ``candidates[0].content.parts`` work without the SDK.
"""

from __future__ import annotations

import inspect
from typing import Any

from ..faults import current_policies
from . import normalize as N
from ._chat import (
    ChatAdapter,
    ChatCall,
    _apply_llm_fault,
    _wrap_result,
    invoke,
    prepare_request,
)
from ._proxy import Proxy, WrappedClient, server_address
from .streams import Assembler

GENAI = ChatAdapter(
    provider="google_genai",
    api="google.genai.models",
    normalize_request=N.normalize_google_genai_request,
    normalize_response=N.normalize_google_genai_response,
    new_assembler=lambda: GenAIAssembler(),
)


# --- stream assembly -----------------------------------------------------------------


class GenAIAssembler(Assembler):
    """Assemble ``generate_content_stream`` chunks into one GenerateContentResponse-like dict."""

    def __init__(self) -> None:
        super().__init__()
        self._texts: list[str] = []
        self._cumulative = False
        self._function_calls: list[dict[str, Any]] = []
        self._usage: Any = None
        self._model: Any = None
        self._finish: Any = None
        self._prompt_feedback: Any = None
        self._response_id: Any = None
        self._last: dict[str, Any] = {}

    def _add(self, data: dict[str, Any]) -> None:
        self._last = data
        text = data.get("text")
        if not isinstance(text, str):
            text = N._text_from_raw(data, data)  # type: ignore[attr-defined]
        if isinstance(text, str) and text:
            if self._texts and text.startswith("".join(self._texts)):
                # cumulative snapshots (some SDK versions): keep the latest full text
                self._texts = [text]
                self._cumulative = True
            else:
                self._texts.append(text)
        calls = N._function_calls_from_raw(data, data)  # type: ignore[attr-defined]
        for fc in calls:
            if fc and fc not in self._function_calls:
                self._function_calls.append(fc)
        usage = data.get("usage_metadata") or data.get("usageMetadata")
        if usage:
            self._usage = usage
        model = data.get("model_version") or data.get("modelVersion") or data.get("model")
        if model:
            self._model = model
        finish = N._provider_finish(data)  # type: ignore[attr-defined]
        if finish:
            self._finish = finish
        if data.get("prompt_feedback") or data.get("promptFeedback"):
            self._prompt_feedback = data.get("prompt_feedback") or data.get("promptFeedback")
        rid = data.get("response_id") or data.get("responseId") or data.get("id")
        if rid:
            self._response_id = rid

    def result(self) -> dict[str, Any]:
        text = (
            "".join(self._texts)
            if not self._cumulative
            else (self._texts[-1] if self._texts else "")
        )
        raw_parts: list[dict[str, Any]] = []
        if text:
            raw_parts.append({"text": text})
        for fc in self._function_calls:
            raw_parts.append(
                {
                    "function_call": N._drop_none(
                        {"id": fc.get("id"), "name": fc.get("name"), "args": fc.get("args")}
                    )
                }
            )
        out: dict[str, Any] = {
            "text": text,
            "function_calls": list(self._function_calls),
            "candidates": [
                {
                    "content": {"role": "model", "parts": raw_parts or [{"text": ""}]},
                    "finish_reason": self._finish or "STOP",
                }
            ],
        }
        if self._usage is not None:
            out["usage_metadata"] = self._usage
        if self._model is not None:
            out["model_version"] = self._model
        if self._prompt_feedback is not None:
            out["prompt_feedback"] = self._prompt_feedback
        if self._response_id is not None:
            out["response_id"] = self._response_id
        return out


# --- mutable response (so crashlab.faults can edit a Google-shaped object) ------------


class _Part:
    """A writable content part: enough for Anthropic-style fault mutation and GenAI attrs."""

    def __init__(
        self,
        text: str | None = None,
        function_call: Any = None,
        type: str | None = None,
        **extra: Any,
    ) -> None:
        self.text = text
        self.function_call = function_call
        self.type = type
        self.id = extra.pop("id", None)
        self.name = extra.pop("name", None)
        self.input = extra.pop("input", None)
        for k, v in extra.items():
            setattr(self, k, v)

    def to_dict(self) -> dict[str, Any]:
        if self.function_call is not None:
            fc = N.to_jsonable(self.function_call)
            return {"function_call": fc if isinstance(fc, dict) else {"name": fc}}
        if self.type == "tool_use" or (self.name and self.input is not None and not self.text):
            return {
                "function_call": N._drop_none(
                    {"id": self.id, "name": self.name, "args": N.parse_arguments(self.input)}
                )
            }
        return N._drop_none({"text": self.text})


def _dump_part(part: Any) -> dict[str, Any]:
    if hasattr(part, "to_dict") and type(part).to_dict is not getattr(dict, "fromkeys", None):
        try:
            dumped = part.to_dict()
            if isinstance(dumped, dict) and dumped:
                return dumped
        except Exception:
            pass
    fc = getattr(part, "function_call", None)
    typ = getattr(part, "type", None)
    name = getattr(part, "name", None)
    inp = getattr(part, "input", None)
    text = getattr(part, "text", None)
    ident = getattr(part, "id", None)
    if typ == "tool_use" or (name and inp is not None and not text):
        return {
            "function_call": N._drop_none(
                {"id": ident, "name": name, "args": N.parse_arguments(inp)}
            )
        }
    if fc is not None:
        fc = N.to_jsonable(fc)
        if not isinstance(fc, dict) or not fc:
            fc = N._drop_none(
                {
                    "id": getattr(part.function_call, "id", None),
                    "name": getattr(part.function_call, "name", None),
                    "args": getattr(part.function_call, "args", None),
                }
            )
        return {"function_call": fc}
    if isinstance(text, str):
        return N._drop_none({"text": text})
    dumped = N.to_jsonable(part)
    return dumped if isinstance(dumped, dict) else {"text": str(part)}


def _parts_of(resp: Any) -> list[_Part]:

    parts: list[_Part] = []
    candidates = getattr(resp, "candidates", None)
    if not candidates and isinstance(resp, dict):
        candidates = resp.get("candidates")
    content = None
    if candidates:
        first = candidates[0]
        content = getattr(first, "content", None)
        if content is None and isinstance(first, dict):
            content = first.get("content")
    raw_parts = getattr(content, "parts", None) if content is not None else None
    if raw_parts is None and isinstance(content, dict):
        raw_parts = content.get("parts")
    for p in raw_parts or []:
        if isinstance(p, _Part):
            parts.append(p)
            continue
        if isinstance(p, dict):
            fc = p.get("function_call") or p.get("functionCall")
            parts.append(_Part(text=p.get("text"), function_call=fc))
            continue
        parts.append(
            _Part(
                text=getattr(p, "text", None),
                function_call=getattr(p, "function_call", None),
            )
        )
    if not parts:
        text = getattr(resp, "text", None)
        if isinstance(text, str) and text:
            parts.append(_Part(text=text))
        for fc in getattr(resp, "function_calls", None) or []:
            parts.append(_Part(function_call=fc))
    return parts


class MutableGenAIResponse:
    """Wrap a live GenerateContentResponse so ``faults.llm_*`` mutations stick.

    Existing fault helpers edit either OpenAI ``choices`` or Anthropic ``content`` /
    ``stop_reason``. This facade exposes a writable ``content`` list of parts and keeps
    ``.text``, ``.function_calls`` and ``.candidates`` in sync, including after a mutation.
    """

    def __init__(self, inner: Any = None, **data: Any) -> None:
        if inner is None and data:
            inner = data
        if isinstance(inner, MutableGenAIResponse):
            inner = inner.to_dict()
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_content_override", None)
        stop = None
        if isinstance(inner, dict):
            stop = N._provider_finish(inner)
        elif inner is not None:
            cands = getattr(inner, "candidates", None) or []
            if cands:
                first = cands[0]
                stop = getattr(first, "finish_reason", None) or (
                    first.get("finish_reason") if isinstance(first, dict) else None
                )
            stop = stop or getattr(inner, "finish_reason", None)
        object.__setattr__(self, "stop_reason", stop)

    def __getattr__(self, name: str) -> Any:
        inner = object.__getattribute__(self, "_inner")
        if isinstance(inner, dict):
            try:
                from ..replay.rebuild import ReplayObject

                return ReplayObject.wrap(inner[name])
            except (KeyError, TypeError):
                if name.startswith("__"):
                    raise AttributeError(name) from None
                return None
        return getattr(inner, name)

    @property
    def content(self) -> list[_Part]:
        override = object.__getattribute__(self, "_content_override")
        if override is not None:
            return override
        # cache so in-place mutations (llm_truncate) stick on later .text reads
        parts = _parts_of(object.__getattribute__(self, "_inner"))
        object.__setattr__(self, "_content_override", parts)
        return parts

    @content.setter
    def content(self, value: Any) -> None:
        object.__setattr__(self, "_content_override", list(value or []))

    @property
    def text(self) -> str:
        return "".join(
            p.text if isinstance(getattr(p, "text", None), str) else "" for p in self.content
        )

    @property
    def function_calls(self) -> list[Any]:
        from types import SimpleNamespace

        out: list[Any] = []
        for p in self.content:
            fc = getattr(p, "function_call", None)
            if fc is not None:
                if not isinstance(fc, dict) and not hasattr(fc, "__dict__"):
                    fc = SimpleNamespace(
                        id=getattr(fc, "id", None),
                        name=getattr(fc, "name", None),
                        args=getattr(fc, "args", None) or getattr(fc, "arguments", None),
                    )
                out.append(fc)
                continue
            if getattr(p, "type", None) == "tool_use" or (
                getattr(p, "name", None) and getattr(p, "input", None) is not None
            ):
                out.append(
                    SimpleNamespace(
                        id=getattr(p, "id", None),
                        name=p.name,
                        args=N.parse_arguments(p.input),
                    )
                )
        return out

    @property
    def candidates(self) -> list[Any]:
        from types import SimpleNamespace

        parts = []
        for p in self.content:
            dumped = p.to_dict() if hasattr(p, "to_dict") else N.to_jsonable(p)
            parts.append(SimpleNamespace(**dumped) if isinstance(dumped, dict) else dumped)
        finish = self.stop_reason or "STOP"
        return [
            SimpleNamespace(
                content=SimpleNamespace(role="model", parts=self.content),
                finish_reason=finish,
            )
        ]

    @property
    def usage_metadata(self) -> Any:
        inner = object.__getattribute__(self, "_inner")
        if isinstance(inner, dict):
            return inner.get("usage_metadata") or inner.get("usageMetadata")
        return getattr(inner, "usage_metadata", None)

    @property
    def model_version(self) -> Any:
        inner = object.__getattribute__(self, "_inner")
        if isinstance(inner, dict):
            return inner.get("model_version") or inner.get("modelVersion") or inner.get("model")
        return getattr(inner, "model_version", None) or getattr(inner, "model", None)

    def to_dict(self) -> dict[str, Any]:
        inner = object.__getattribute__(self, "_inner")
        raw = N.to_jsonable(inner) if not isinstance(inner, dict) else dict(inner)
        if not isinstance(raw, dict):
            raw = {}
        parts = [_dump_part(p) for p in self.content]
        raw["text"] = self.text
        raw["function_calls"] = [
            N._drop_none(
                {
                    "id": getattr(fc, "id", None) if not isinstance(fc, dict) else fc.get("id"),
                    "name": getattr(fc, "name", None)
                    if not isinstance(fc, dict)
                    else fc.get("name"),
                    "args": (
                        getattr(fc, "args", None)
                        if not isinstance(fc, dict)
                        else (fc.get("args") if fc.get("args") is not None else fc.get("arguments"))
                    ),
                }
            )
            for fc in self.function_calls
        ]
        raw["candidates"] = [
            {
                "content": {"role": "model", "parts": parts},
                "finish_reason": self.stop_reason or "STOP",
            }
        ]
        if "usage_metadata" not in raw and self.usage_metadata is not None:
            raw["usage_metadata"] = N.to_jsonable(self.usage_metadata)
        if "model_version" not in raw and self.model_version is not None:
            raw["model_version"] = self.model_version
        return raw

    model_dump = to_dict

    @classmethod
    def model_validate(cls, data: Any) -> MutableGenAIResponse:
        return cls(data)


def _maybe_mutable(resp: Any) -> Any:
    if resp is None or not current_policies():
        return resp
    return MutableGenAIResponse(resp)


# --- request shaping -----------------------------------------------------------------


def _config_map(config: Any) -> dict[str, Any]:
    return N._as_map(config)


def _expand_kwargs(args: tuple, kwargs: dict[str, Any], *, bind_to: Any = None) -> dict[str, Any]:
    """Merge positional SDK args into a generate_content-shaped dict, plus messages for forks."""
    data = dict(kwargs)
    if args and bind_to is not None:
        try:
            bound = inspect.signature(bind_to).bind_partial(*args, **kwargs)
            data = dict(bound.arguments)
            data.update(data.pop("kwargs", {}) or {})
        except (TypeError, ValueError):
            if "model" not in data and len(args) >= 1:
                data["model"] = args[0]
            if "contents" not in data and len(args) >= 2:
                data["contents"] = args[1]
    if "messages" not in data:
        contents = data.get("contents")
        if contents is not None:
            data["messages"] = N._contents_as_messages(contents)
    config = _config_map(data.get("config"))
    if "tools" not in data and config:
        tools = N._genai_tools(config)
        if tools:
            # OpenAI-shaped, so RequestEditor (family "chat") can patch descriptions
            data["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.get("name"),
                        "description": t.get("description"),
                        "parameters": t.get("parameters"),
                    },
                }
                if t.get("type") == "function"
                else t
                for t in tools
            ]
    sys_inst = N._pick(config, "system_instruction", "systemInstruction") if config else None
    if sys_inst is not None and "system" not in data:
        data["system"] = sys_inst
    return data


def _to_native(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Expanded / patched kwargs -> ``generate_content(model=, contents=, config=)``."""
    native: dict[str, Any] = {}
    if "model" in kwargs and kwargs["model"] is not None:
        native["model"] = kwargs["model"]
    config = _config_map(kwargs.get("config"))
    if "system" in kwargs and kwargs["system"] is not None:
        config["system_instruction"] = kwargs["system"]
    if kwargs.get("temperature") is not None:
        config["temperature"] = kwargs["temperature"]
    if kwargs.get("max_tokens") is not None:
        config["max_output_tokens"] = kwargs["max_tokens"]
    if kwargs.get("max_output_tokens") is not None:
        config["max_output_tokens"] = kwargs["max_output_tokens"]
    if kwargs.get("top_p") is not None:
        config["top_p"] = kwargs["top_p"]
    if kwargs.get("top_k") is not None:
        config["top_k"] = kwargs["top_k"]
    if kwargs.get("stop"):
        config["stop_sequences"] = kwargs["stop"]
    if kwargs.get("tools") and "function_declarations" not in str(config.get("tools", "")):
        decls = []
        for t in kwargs["tools"]:
            t = N.to_jsonable(t) or {}
            if not isinstance(t, dict):
                continue
            fn = t.get("function") if isinstance(t.get("function"), dict) else t
            if fn.get("name"):
                decls.append(
                    N._drop_none(
                        {
                            "name": fn.get("name"),
                            "description": fn.get("description"),
                            "parameters": fn.get("parameters") or fn.get("input_schema"),
                        }
                    )
                )
        if decls:
            config["tools"] = [{"function_declarations": decls}]
    if "contents" in kwargs and "messages" not in kwargs:
        native["contents"] = kwargs["contents"]
    elif kwargs.get("messages") is not None:
        native["contents"] = _messages_to_contents(kwargs["messages"], kwargs.get("system"))
    elif "contents" in kwargs:
        native["contents"] = kwargs["contents"]
    if config:
        native["config"] = config
    # pass through any other native keys the caller actually used
    for key in ("config",):
        if key in kwargs and key not in native and kwargs[key] is not None:
            native[key] = kwargs[key]
    return native


def _messages_to_contents(messages: Any, system: Any = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in N.to_jsonable(messages) or []:
        if not isinstance(msg, dict):
            out.append({"role": "user", "parts": [{"text": str(msg)}]})
            continue
        role = msg.get("role") or "user"
        if role in ("system", "developer"):
            continue  # carried as config.system_instruction
        g_role = "model" if role == "assistant" else ("user" if role != "tool" else "user")
        parts: list[dict[str, Any]] = []
        content = msg.get("content")
        if isinstance(content, str) and content:
            parts.append({"text": content})
        elif isinstance(content, list):
            for p in content:
                if isinstance(p, str):
                    if p:
                        parts.append({"text": p})
                elif isinstance(p, dict):
                    if p.get("type") == "text" and p.get("content"):
                        parts.append({"text": p["content"]})
                    elif p.get("type") == "tool_call":
                        parts.append(
                            {
                                "function_call": N._drop_none(
                                    {
                                        "id": p.get("id"),
                                        "name": p.get("name"),
                                        "args": p.get("arguments"),
                                    }
                                )
                            }
                        )
                    elif p.get("type") == "tool_call_response":
                        parts.append(
                            {
                                "function_response": N._drop_none(
                                    {"id": p.get("id"), "response": p.get("response")}
                                )
                            }
                        )
                    elif "text" in p or "function_call" in p or "function_response" in p:
                        parts.append(p)
        if role == "tool" and not any("function_response" in p for p in parts):
            parts = [
                {
                    "function_response": N._drop_none(
                        {
                            "id": msg.get("tool_call_id") or msg.get("id"),
                            "name": msg.get("name"),
                            "response": content,
                        }
                    )
                }
            ]
        out.append({"role": g_role, "parts": parts or [{"text": ""}]})
    return out


# --- history helpers -----------------------------------------------------------------


def _chat_history(chat: Any) -> list[Any]:
    getter = getattr(chat, "get_history", None)
    if callable(getter):
        try:
            hist = getter()
            if hist is not None:
                return list(hist)
        except TypeError:
            try:
                hist = getter(curated=False)
                if hist is not None:
                    return list(hist)
            except Exception:
                pass
        except Exception:
            pass
    for attr in ("_comprehensive_history", "history", "_history", "_curated_history"):
        val = getattr(chat, attr, None)
        if val:
            return list(val)
    return []


def _as_user_content(message: Any) -> dict[str, Any]:
    message = N.to_jsonable(message)
    if isinstance(message, str):
        return {"role": "user", "parts": [{"text": message}]}
    if isinstance(message, dict):
        if "parts" in message or "role" in message:
            msg = dict(message)
            msg.setdefault("role", "user")
            return msg
        return {"role": "user", "parts": [message]}
    if isinstance(message, list):
        # a list of parts or of contents: treat as one user turn of parts when untyped
        if message and isinstance(message[0], dict) and "role" in message[0]:
            # already a history-shaped list; the last item is the new message
            last = message[-1]
            last = dict(last)
            last.setdefault("role", "user")
            return last
        return {"role": "user", "parts": message}
    return {"role": "user", "parts": [{"text": str(message)}]}


def _content_from_response(resp: Any) -> dict[str, Any]:
    raw = N.to_jsonable(resp)
    raw = raw if isinstance(raw, dict) else {}
    cands = raw.get("candidates") or []
    if cands and isinstance(cands[0], dict):
        content = cands[0].get("content")
        if isinstance(content, dict) and content.get("parts") is not None:
            content = dict(content)
            content.setdefault("role", "model")
            return content
    parts: list[dict[str, Any]] = []
    text = raw.get("text") or getattr(resp, "text", None)
    if isinstance(text, str) and text:
        parts.append({"text": text})
    for fc in raw.get("function_calls") or getattr(resp, "function_calls", None) or []:
        fc = N.to_jsonable(fc)
        if isinstance(fc, dict):
            parts.append({"function_call": fc})
    return {"role": "model", "parts": parts or [{"text": ""}]}


# --- method wrappers -----------------------------------------------------------------


def _is_async_fn(fn: Any) -> bool:
    if inspect.iscoroutinefunction(fn) or inspect.isasyncgenfunction(fn):
        return True
    name = type(getattr(fn, "__self__", None)).__name__ if getattr(fn, "__self__", None) else ""
    return "Async" in name


def _wrap_generate(inner: Any, adapter: ChatAdapter, server: str | None, *, is_async: bool) -> Any:
    fn = inner.generate_content

    def generate_content(*args: Any, **kwargs: Any) -> Any:
        call_kwargs = _expand_kwargs(args, kwargs, bind_to=fn)

        def bound(*a: Any, **k: Any) -> Any:
            if k and ("contents" in k or "messages" in k or "config" in k or "model" in k):
                native = _to_native(k)
                return _maybe_mutable(fn(**native) if native else fn(*a, **k))
            return _maybe_mutable(fn(*a, **k))

        return invoke(adapter, bound, (), call_kwargs, server=server, is_async=is_async)

    generate_content.__doc__ = getattr(fn, "__doc__", None)
    return generate_content


def _synthesize_chunks(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Rebuild a short stream from the recorded final response."""
    text = raw.get("text")
    if not isinstance(text, str):
        text = N._text_from_raw(raw, raw)
    calls = raw.get("function_calls") or N._function_calls_from_raw(raw, raw)
    usage = raw.get("usage_metadata") or raw.get("usageMetadata")
    model = raw.get("model_version") or raw.get("modelVersion") or raw.get("model")
    finish = N._provider_finish(raw) or "STOP"
    chunks: list[dict[str, Any]] = []
    pieces = _cut_text(text or "")
    if not pieces and calls:
        pieces = [""]
    for i, piece in enumerate(pieces):
        last = i == len(pieces) - 1
        parts: list[dict[str, Any]] = []
        if piece:
            parts.append({"text": piece})
        if last:
            for fc in calls or []:
                fc = N.to_jsonable(fc)
                if isinstance(fc, dict):
                    parts.append({"function_call": fc})
        chunk: dict[str, Any] = {
            "text": piece,
            "function_calls": list(calls or []) if last else [],
            "candidates": [
                {
                    "content": {"role": "model", "parts": parts or [{"text": piece}]},
                    "finish_reason": finish if last else None,
                }
            ],
        }
        if last and usage is not None:
            chunk["usage_metadata"] = usage
        if model is not None:
            chunk["model_version"] = model
        chunks.append(chunk)
    return chunks or [
        {
            "text": "",
            "function_calls": list(calls or []),
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": ""}]}, "finish_reason": finish}
            ],
            **({"usage_metadata": usage} if usage is not None else {}),
        }
    ]


def _cut_text(text: str, size: int = 24) -> list[str]:
    if not text:
        return []
    if len(text) <= size:
        return [text]
    return [text[i : i + size] for i in range(0, len(text), size)]


def _replay_stream(call: ChatCall, src: Any, *, is_async_gen: bool) -> Any:
    from ..replay import rebuild as R
    from ..replay import synth

    call.pending.finish_from(src)  # type: ignore[union-attr]
    error = R.rebuild_exception(src.error) if src.error is not None else None
    raw = src.raw_response if isinstance(src.raw_response, dict) else {}
    chunk_type = src.attributes.get("crashlab.stream.chunk_type")
    items = [R.rebuild(c, chunk_type) for c in _synthesize_chunks(raw or {})]
    if error is not None and not items:
        if is_async_gen:

            async def _empty_fail() -> Any:
                raise error

            return _empty_fail()

        def _fail() -> Any:
            raise error
            yield  # pragma: no cover

        return _fail()
    stream = (
        synth.AsyncReplayStream(items, error) if is_async_gen else synth.ReplayStream(items, error)
    )
    return stream


def _wrap_stream(inner: Any, adapter: ChatAdapter, server: str | None, *, is_async: bool) -> Any:
    fn = inner.generate_content_stream
    is_async_gen = inspect.isasyncgenfunction(fn) or is_async

    def generate_content_stream(*args: Any, **kwargs: Any) -> Any:
        call_kwargs = _expand_kwargs(args, kwargs, bind_to=fn)
        patched, idx, applied = prepare_request(adapter, call_kwargs)
        if applied:
            args, kwargs, call_kwargs = (), _to_native(patched), patched
        call = ChatCall(adapter, call_kwargs, server=server, stream=True, idx=idx, patches=applied)
        src = call.replay_source()
        if src is not None:
            return _replay_stream(call, src, is_async_gen=is_async_gen)
        _apply_llm_fault(call)
        try:
            if applied:
                result = fn(**kwargs) if kwargs else fn(*args, **kwargs)
            else:
                result = fn(*args, **kwargs)
        except BaseException as exc:
            call.fail(exc)
            raise
        if inspect.isawaitable(result) and not inspect.isasyncgen(result):

            async def _awaited() -> Any:
                try:
                    resp = await result
                except BaseException as exc:
                    call.fail(exc)
                    raise
                return _wrap_result(call, resp, is_async=True)

            return _awaited()
        return _wrap_result(call, result, is_async=is_async_gen)

    generate_content_stream.__doc__ = getattr(fn, "__doc__", None)
    return generate_content_stream


def _wrap_models(models: Any, adapter: ChatAdapter, server: str | None, *, is_async: bool) -> Proxy:
    kw: dict[str, Any] = {}
    if callable(getattr(models, "generate_content", None)):
        kw["generate_content"] = _wrap_generate(models, adapter, server, is_async=is_async)
    if callable(getattr(models, "generate_content_stream", None)):
        kw["generate_content_stream"] = _wrap_stream(models, adapter, server, is_async=is_async)
    return Proxy(models, **kw)


# --- chats ---------------------------------------------------------------------------


class WrappedChat(Proxy):
    """A ``chats.create`` session whose send_message* calls are recorded."""

    def __init__(
        self,
        chat: Any,
        adapter: ChatAdapter,
        server: str | None,
        *,
        is_async: bool,
        model: Any = None,
        config: Any = None,
        history: list[Any] | None = None,
    ) -> None:
        start = history if history is not None else _chat_history(chat)
        # Proxy.__setattr__ delegates to inner; set our fields with object.__setattr__
        object.__setattr__(self, "_adapter", adapter)
        object.__setattr__(self, "_server", server)
        object.__setattr__(self, "_is_async", is_async)
        object.__setattr__(
            self, "_model", model or getattr(chat, "_model", None) or getattr(chat, "model", None)
        )
        object.__setattr__(
            self, "_config", config if config is not None else getattr(chat, "_config", None)
        )
        object.__setattr__(self, "_history", list(start))
        # Bind methods after the Proxy is initialized so `self` is usable
        object.__setattr__(self, "_inner", chat)
        overrides: dict[str, Any] = {}
        if callable(getattr(chat, "send_message", None)):
            overrides["send_message"] = self._wrap_send(chat, stream=False)
        if callable(getattr(chat, "send_message_stream", None)):
            overrides["send_message_stream"] = self._wrap_send(chat, stream=True)
        if callable(getattr(chat, "get_history", None)):

            def get_history(*args: Any, **kwargs: Any) -> list[Any]:
                return list(self._history)

            get_history.__doc__ = getattr(chat.get_history, "__doc__", None)
            overrides["get_history"] = get_history
        super().__init__(chat, **overrides)

    def _commit(self, user_content: dict[str, Any], resp: Any) -> None:
        self._history.append(user_content)
        self._history.append(_content_from_response(resp))

    def _send_kwargs(self, message: Any, config: Any) -> dict[str, Any]:
        user = _as_user_content(message)
        contents = list(self._history) + [user]
        cfg = config if config is not None else self._config
        return _expand_kwargs(
            (),
            {"model": self._model, "contents": contents, "config": cfg},
        )

    def _wrap_send(self, chat: Any, *, stream: bool) -> Any:
        inner = chat.send_message_stream if stream else chat.send_message
        is_async = self._is_async or _is_async_fn(inner)
        is_async_gen = stream and (inspect.isasyncgenfunction(inner) or is_async)
        adapter = self._adapter
        server = self._server

        def send(*args: Any, **kwargs: Any) -> Any:
            message = args[0] if args else kwargs.get("message", kwargs.get("contents"))
            config = kwargs.get("config")
            user = _as_user_content(message)
            call_kwargs = self._send_kwargs(message, config)
            if stream:
                patched, idx, applied = prepare_request(adapter, call_kwargs)
                if applied:
                    args, kwargs, call_kwargs = (), _to_native(patched), patched
                call = ChatCall(
                    adapter, call_kwargs, server=server, stream=True, idx=idx, patches=applied
                )
                src = call.replay_source()
                if src is not None:
                    result = _replay_stream(call, src, is_async_gen=is_async_gen)
                    return _track_stream(
                        result, lambda resp: self._commit(user, resp), is_async_gen
                    )
                _apply_llm_fault(call)
                try:
                    result = inner(*args, **kwargs) if not applied else inner(**kwargs)
                except BaseException as exc:
                    call.fail(exc)
                    raise

                def finish(stream_obj: Any) -> Any:
                    wrapped = _wrap_result(call, stream_obj, is_async=is_async_gen)
                    return _track_stream(
                        wrapped, lambda resp: self._commit(user, resp), is_async_gen
                    )

                if inspect.isawaitable(result) and not inspect.isasyncgen(result):

                    async def _awaited() -> Any:
                        try:
                            resp = await result
                        except BaseException as exc:
                            call.fail(exc)
                            raise
                        return finish(resp)

                    return _awaited()
                return finish(result)

            def bound(*a: Any, **k: Any) -> Any:
                if k and ("contents" in k or "messages" in k or "model" in k):
                    # a fork patched the request: still deliver through send_message when
                    # we can, otherwise fall back (the live SDK just needs the new message)
                    return _maybe_mutable(inner(*args, **kwargs))
                return _maybe_mutable(inner(*a, **k))

            out = invoke(adapter, bound, (), call_kwargs, server=server, is_async=is_async)
            if is_async:

                async def _done() -> Any:
                    resp = await out
                    self._commit(user, resp)
                    return resp

                return _done()
            self._commit(user, out)
            return out

        send.__doc__ = getattr(inner, "__doc__", None)
        return send


def _track_stream(stream: Any, commit: Any, is_async: bool) -> Any:
    """Update chat history once a stream is fully consumed."""

    if is_async or (hasattr(stream, "__aiter__") and not hasattr(stream, "__iter__")):

        class _AsyncTrack:
            def __aiter__(self) -> Any:
                return self._iter()

            async def _iter(self) -> Any:
                chunks: list[Any] = []
                async for chunk in stream:
                    chunks.append(chunk)
                    yield chunk
                if chunks:
                    commit(_assembled(chunks))

            def __getattr__(self, name: str) -> Any:
                return getattr(stream, name)

        return _AsyncTrack()

    class _Track:
        def __iter__(self) -> Any:
            chunks: list[Any] = []
            for chunk in stream:
                chunks.append(chunk)
                yield chunk
            if chunks:
                commit(_assembled(chunks))

        def __getattr__(self, name: str) -> Any:
            return getattr(stream, name)

    return _Track()


def _assembled(chunks: list[Any]) -> dict[str, Any]:
    asm = GenAIAssembler()
    for chunk in chunks:
        asm.add(chunk)
    return asm.result()


def _wrap_chats(chats: Any, adapter: ChatAdapter, server: str | None, *, is_async: bool) -> Proxy:
    inner_create = getattr(chats, "create", None)
    if not callable(inner_create):
        return Proxy(chats)

    def create(*args: Any, **kwargs: Any) -> Any:
        chat = inner_create(*args, **kwargs)
        if inspect.isawaitable(chat):

            async def _after() -> Any:
                real = await chat
                return WrappedChat(
                    real,
                    adapter,
                    server,
                    is_async=True,
                    model=kwargs.get("model", args[0] if args else None),
                    config=kwargs.get("config"),
                    history=kwargs.get("history"),
                )

            return _after()
        return WrappedChat(
            chat,
            adapter,
            server,
            is_async=is_async,
            model=kwargs.get("model", args[0] if args else None),
            config=kwargs.get("config"),
            history=kwargs.get("history"),
        )

    create.__doc__ = getattr(inner_create, "__doc__", None)
    return Proxy(chats, create=create)


# --- client --------------------------------------------------------------------------


def _genai_server(client: Any) -> str | None:
    addr = server_address(client)
    if addr:
        return addr
    for obj in (client, getattr(client, "_api_client", None), getattr(client, "_client", None)):
        if obj is None:
            continue
        addr = server_address(obj)
        if addr:
            return addr
        for attr in ("_base_url", "base_url"):
            raw = getattr(obj, attr, None)
            if raw:
                addr = server_address(type("T", (), {"base_url": raw})())
                if addr:
                    return addr
        opts = getattr(obj, "_http_options", None) or getattr(obj, "http_options", None)
        raw = getattr(opts, "base_url", None) if opts is not None else None
        if raw:
            addr = server_address(type("T", (), {"base_url": raw})())
            if addr:
                return addr
    if getattr(client, "vertexai", False) or getattr(client, "_vertexai", False):
        return "aiplatform.googleapis.com"
    # AI Studio is the SDK default; set it so pricing resolves without a base_url
    return "generativelanguage.googleapis.com"


class WrappedGenAI(WrappedClient):
    """Thin proxy around ``google.genai.Client`` (or a duck-typed stand-in)."""

    def __init__(self, client: Any, *, agent: str | None = None) -> None:
        import dataclasses

        server = _genai_server(client)
        adapter = GENAI if agent is None else dataclasses.replace(GENAI, agent=agent)
        overrides: dict[str, Any] = {"_crashlab_agent": agent}
        models = getattr(client, "models", None)
        if models is not None:
            is_async = _is_async_fn(getattr(models, "generate_content", None))
            overrides["models"] = _wrap_models(models, adapter, server, is_async=is_async)
        chats = getattr(client, "chats", None)
        if chats is not None:
            overrides["chats"] = _wrap_chats(chats, adapter, server, is_async=False)
        aio = getattr(client, "aio", None)
        if aio is not None:
            aio_models = getattr(aio, "models", None)
            aio_chats = getattr(aio, "chats", None)
            aio_kw: dict[str, Any] = {}
            if aio_models is not None:
                aio_kw["models"] = _wrap_models(aio_models, adapter, server, is_async=True)
            if aio_chats is not None:
                aio_kw["chats"] = _wrap_chats(aio_chats, adapter, server, is_async=True)
            overrides["aio"] = Proxy(aio, **aio_kw)
        super().__init__(client, **overrides)

    def _rewrap(self, client: Any) -> WrappedGenAI:
        return WrappedGenAI(client, agent=self._crashlab_agent)


def looks_like_genai(client: Any) -> bool:
    """True for ``google.genai.Client`` / ``AsyncClient`` or any duck-typed equivalent."""
    models = getattr(client, "models", None)
    if callable(getattr(models, "generate_content", None)):
        return True
    aio_models = getattr(getattr(client, "aio", None), "models", None)
    if callable(getattr(aio_models, "generate_content", None)):
        return True
    mod = type(client).__module__ or ""
    if mod == "google.genai" or mod.startswith("google.genai."):
        return type(client).__name__ in {"Client", "AsyncClient"}
    return False


def wrap_genai(client: Any, *, agent: str | None = None) -> WrappedGenAI:
    if not looks_like_genai(client):
        raise TypeError(f"{type(client).__name__} has no models.generate_content method")
    return WrappedGenAI(client, agent=agent)
