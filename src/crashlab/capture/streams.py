"""Stream assemblers and pass-through stream wrappers.

The wrappers hand every chunk/event to the caller unchanged and feed a copy to an
assembler; when the stream ends (exhausted, closed, errored or garbage-collected) the
assembled final response is recorded as the step's response.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .normalize import to_jsonable

if TYPE_CHECKING:
    from ._chat import ChatCall


# --- assemblers ----------------------------------------------------------------------


class Assembler:
    """Collects stream chunks and builds a provider-native final response dict."""

    def __init__(self) -> None:
        self.chunks = 0
        self.chunk_type: str | None = None  # class of the first chunk, for replay

    def add(self, chunk: Any) -> None:
        self.chunks += 1
        if self.chunk_type is None:
            cls = type(chunk)
            self.chunk_type = f"{cls.__module__}.{cls.__qualname__}"
        data = to_jsonable(chunk)
        if isinstance(data, dict):
            self._add(data)

    def _add(self, data: dict[str, Any]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def result(self) -> dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


class OpenAIChatAssembler(Assembler):
    """ChatCompletionChunk stream (OpenAI, LiteLLM) -> ChatCompletion-shaped dict."""

    def __init__(self) -> None:
        super().__init__()
        self.meta: dict[str, Any] = {}
        self.choices: dict[int, dict[str, Any]] = {}
        self.usage: dict[str, Any] | None = None

    def _add(self, data: dict[str, Any]) -> None:
        for key in ("id", "model", "created", "system_fingerprint", "service_tier"):
            if data.get(key) is not None:
                self.meta[key] = data[key]
        if data.get("usage"):
            self.usage = data["usage"]
        for ch in data.get("choices") or []:
            idx = ch.get("index") or 0
            st = self.choices.setdefault(
                idx,
                {
                    "role": None,
                    "content": [],
                    "refusal": [],
                    "reasoning": [],
                    "reasoning_key": None,
                    "tool_calls": {},
                    "finish": None,
                },
            )
            delta = ch.get("delta") or {}
            if delta.get("role"):
                st["role"] = delta["role"]
            if delta.get("content"):
                st["content"].append(delta["content"])
            if delta.get("refusal"):
                st["refusal"].append(delta["refusal"])
            for rkey in ("reasoning_content", "reasoning"):
                if isinstance(delta.get(rkey), str) and delta[rkey]:
                    st["reasoning"].append(delta[rkey])
                    st["reasoning_key"] = st["reasoning_key"] or rkey
                    break
            for tc in delta.get("tool_calls") or []:
                tidx = tc.get("index") or 0
                call = st["tool_calls"].setdefault(
                    tidx, {"id": None, "type": "function", "name": [], "arguments": []}
                )
                if tc.get("id"):
                    call["id"] = tc["id"]
                for k, v in tc.items():  # provider extras, e.g. Gemini extra_content
                    if k not in ("index", "id", "type", "function") and v is not None:
                        call.setdefault("extra", {})[k] = v
                fn = tc.get("function") or {}
                if fn.get("name"):
                    call["name"].append(fn["name"])
                if fn.get("arguments"):
                    call["arguments"].append(fn["arguments"])
            if ch.get("finish_reason"):
                st["finish"] = ch["finish_reason"]

    def result(self) -> dict[str, Any]:
        choices = []
        for idx in sorted(self.choices):
            st = self.choices[idx]
            msg: dict[str, Any] = {
                "role": st["role"] or "assistant",
                "content": "".join(st["content"]) or None,
            }
            if st["refusal"]:
                msg["refusal"] = "".join(st["refusal"])
            if st["reasoning"]:
                msg[st["reasoning_key"]] = "".join(st["reasoning"])
            if st["tool_calls"]:
                msg["tool_calls"] = [
                    {
                        "id": c["id"],
                        "type": c["type"],
                        "function": {
                            "name": "".join(c["name"]),
                            "arguments": "".join(c["arguments"]),
                        },
                        **c.get("extra", {}),
                    }
                    for _, c in sorted(st["tool_calls"].items())
                ]
            choices.append({"index": idx, "message": msg, "finish_reason": st["finish"]})
        out = {"object": "chat.completion", **self.meta, "choices": choices}
        if self.usage:
            out["usage"] = self.usage
        return out


class AnthropicAssembler(Assembler):
    """Raw Anthropic message stream events -> Message-shaped dict."""

    def __init__(self) -> None:
        super().__init__()
        self.message: dict[str, Any] = {"type": "message", "role": "assistant", "content": []}
        self.blocks: dict[int, dict[str, Any]] = {}
        self.json_parts: dict[int, list[str]] = {}

    def _add(self, data: dict[str, Any]) -> None:
        t = data.get("type")
        if t == "message_start":
            msg = dict(data.get("message") or {})
            msg.pop("content", None)
            self.message.update(msg)
        elif t == "content_block_start":
            self.blocks[data.get("index", len(self.blocks))] = dict(data.get("content_block") or {})
        elif t == "content_block_delta":
            idx = data.get("index", 0)
            block = self.blocks.setdefault(idx, {"type": "text", "text": ""})
            delta = data.get("delta") or {}
            dt = delta.get("type")
            if dt == "text_delta":
                block["text"] = (block.get("text") or "") + delta.get("text", "")
            elif dt == "input_json_delta":
                self.json_parts.setdefault(idx, []).append(delta.get("partial_json", ""))
            elif dt == "thinking_delta":
                block["thinking"] = (block.get("thinking") or "") + delta.get("thinking", "")
            elif dt == "signature_delta":
                block["signature"] = delta.get("signature")
            elif dt == "citations_delta":
                block.setdefault("citations", []).append(delta.get("citation"))
        elif t == "message_delta":
            delta = data.get("delta") or {}
            for k in ("stop_reason", "stop_sequence"):
                if k in delta:
                    self.message[k] = delta[k]
            if data.get("usage"):
                usage = dict(self.message.get("usage") or {})
                usage.update({k: v for k, v in data["usage"].items() if v is not None})
                self.message["usage"] = usage

    def result(self) -> dict[str, Any]:
        content = []
        for idx in sorted(self.blocks):
            block = dict(self.blocks[idx])
            if idx in self.json_parts:
                block["input"] = "".join(self.json_parts[idx])  # parsed by normalize
            content.append(block)
        return {**self.message, "content": content}


class ResponsesAssembler(Assembler):
    """OpenAI Responses API stream events -> the final Response dict."""

    def __init__(self) -> None:
        super().__init__()
        self.response: dict[str, Any] = {}
        self.items: dict[int, dict[str, Any]] = {}

    def _add(self, data: dict[str, Any]) -> None:
        if isinstance(data.get("response"), dict):  # created / in_progress / completed / ...
            self.response = data["response"]
        if data.get("type") == "response.output_item.done" and isinstance(data.get("item"), dict):
            self.items[data.get("output_index", len(self.items))] = data["item"]

    def result(self) -> dict[str, Any]:
        out = dict(self.response)
        if not out.get("output") and self.items:
            out["output"] = [self.items[i] for i in sorted(self.items)]
        return out


# --- wrappers ------------------------------------------------------------------------


class _StreamBase:
    def __init__(self, inner: Any, call: ChatCall, assembler: Assembler) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_call", call)
        object.__setattr__(self, "_asm", assembler)
        object.__setattr__(self, "_finished", False)

    def _finish(self, *, error: BaseException | None = None, complete: bool = False) -> None:
        if self._finished:
            return
        object.__setattr__(self, "_finished", True)
        self._call.finish_stream(self._asm, error=error, complete=complete)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __del__(self) -> None:  # best effort: a stream abandoned mid-way is still recorded
        try:
            if not self._finished:
                self._finish(complete=False)
        except Exception:
            pass


class RecordingStream(_StreamBase):
    """Sync pass-through stream (OpenAI ``Stream``, Anthropic ``Stream``, LiteLLM)."""

    def __init__(self, inner: Any, call: ChatCall, assembler: Assembler) -> None:
        super().__init__(inner, call, assembler)
        object.__setattr__(self, "_it", None)

    def __iter__(self) -> RecordingStream:
        return self

    def __next__(self) -> Any:
        if self._it is None:
            object.__setattr__(self, "_it", iter(self._inner))
        try:
            chunk = next(self._it)
        except StopIteration:
            self._finish(complete=True)
            raise
        except BaseException as exc:
            self._finish(error=exc)
            raise
        self._asm.add(chunk)
        return chunk

    def __enter__(self) -> RecordingStream:
        enter = getattr(self._inner, "__enter__", None)
        if enter is not None:
            enter()
        return self

    def __exit__(self, exc_type, exc, tb) -> Any:
        exit_ = getattr(self._inner, "__exit__", None)
        try:
            return exit_(exc_type, exc, tb) if exit_ is not None else None
        finally:
            self._finish(error=exc)

    def close(self) -> None:
        try:
            close = getattr(self._inner, "close", None)
            if close is not None:
                close()
        finally:
            self._finish()


class AsyncRecordingStream(_StreamBase):
    """Async pass-through stream (OpenAI ``AsyncStream``, Anthropic, LiteLLM)."""

    def __init__(self, inner: Any, call: ChatCall, assembler: Assembler) -> None:
        super().__init__(inner, call, assembler)
        object.__setattr__(self, "_it", None)

    def __aiter__(self) -> AsyncRecordingStream:
        return self

    async def __anext__(self) -> Any:
        if self._it is None:
            object.__setattr__(self, "_it", self._inner.__aiter__())
        try:
            chunk = await self._it.__anext__()
        except StopAsyncIteration:
            self._finish(complete=True)
            raise
        except BaseException as exc:
            self._finish(error=exc)
            raise
        self._asm.add(chunk)
        return chunk

    async def __aenter__(self) -> AsyncRecordingStream:
        enter = getattr(self._inner, "__aenter__", None)
        if enter is not None:
            await enter()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> Any:
        exit_ = getattr(self._inner, "__aexit__", None)
        try:
            return await exit_(exc_type, exc, tb) if exit_ is not None else None
        finally:
            self._finish(error=exc)

    async def close(self) -> None:
        try:
            close = getattr(self._inner, "close", None)
            if close is not None:
                res = close()
                if hasattr(res, "__await__"):
                    await res
        finally:
            self._finish()


def _snapshot(stream: Any) -> Any:
    getter = getattr(stream, "_current_snapshot_or_none", None)
    try:
        if callable(getter):
            return getter()
        return getattr(stream, "current_message_snapshot", None)
    except Exception:
        return None


class _SnapshotAssembler(Assembler):
    """Uses the SDK's own accumulated snapshot (Anthropic ``messages.stream()``)."""

    def __init__(self, snapshot: Any) -> None:
        super().__init__()
        self.snapshot = to_jsonable(snapshot) if snapshot is not None else {}

    def result(self) -> dict[str, Any]:
        return self.snapshot if isinstance(self.snapshot, dict) else {}


class RecordingMessageStreamManager:
    """Wraps Anthropic ``client.messages.stream(...)`` (sync context manager).

    The user gets the SDK's own ``MessageStream`` unchanged; on exit crashlab records the
    SDK-accumulated message snapshot.
    """

    def __init__(self, inner: Any, call: ChatCall) -> None:
        self._inner = inner
        self._call = call
        self._stream: Any = None

    def __enter__(self) -> Any:
        try:
            self._stream = self._inner.__enter__()
        except BaseException as exc:
            self._call.fail(exc)
            raise
        return self._stream

    def __exit__(self, exc_type, exc, tb) -> Any:
        snap = _snapshot(self._stream)
        try:
            return self._inner.__exit__(exc_type, exc, tb)
        finally:
            complete = bool(snap is not None and getattr(snap, "stop_reason", None))
            self._call.finish_stream(_SnapshotAssembler(snap), error=exc, complete=complete)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class AsyncRecordingMessageStreamManager(RecordingMessageStreamManager):
    async def __aenter__(self) -> Any:
        try:
            self._stream = await self._inner.__aenter__()
        except BaseException as exc:
            self._call.fail(exc)
            raise
        return self._stream

    async def __aexit__(self, exc_type, exc, tb) -> Any:
        snap = _snapshot(self._stream)
        try:
            return await self._inner.__aexit__(exc_type, exc, tb)
        finally:
            complete = bool(snap is not None and getattr(snap, "stop_reason", None))
            self._call.finish_stream(_SnapshotAssembler(snap), error=exc, complete=complete)
