"""Stream assemblers and pass-through stream wrappers.

The wrappers hand every chunk/event to the caller unchanged and feed a copy to an
assembler; when the stream ends (exhausted, closed, errored or garbage-collected) the
assembled final response is recorded as the step's response.
"""

from __future__ import annotations

import weakref
from collections.abc import Iterator
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
        # some providers (Gemini's OpenAI endpoint) send tool-call deltas without an
        # index; replay must send them the same way, or a client that chokes on it
        # (smolagents) passes in replay and the recorded failure is not reproduced
        self.tool_index_missing = False
        self.plan: list[list[list[Any]]] = []
        self.plan_ok = True

    def _plan(self, data: dict[str, Any]) -> None:
        choices = data.get("choices") or []
        if not choices:
            self.plan.append([["u"]])
            return
        if len(choices) > 1 or (choices[0].get("index") or 0) != 0:
            self.plan_ok = False
            return
        ch = choices[0]
        delta = ch.get("delta") or {}
        ops: list[list[Any]] = []
        if isinstance(delta.get("content"), str):
            ops.append(["c", len(delta["content"])])
        for rkey in ("reasoning_content", "reasoning"):
            if isinstance(delta.get(rkey), str) and delta[rkey]:
                ops.append(["r", len(delta[rkey])])
                break
        if delta.get("tool_calls"):
            # per call piece: [index, has id, has type, name length, arguments length]
            # (None when the key was absent), so argument fragments replay as streamed
            pieces = []
            for tc in delta["tool_calls"]:
                fn = tc.get("function") or {}
                name, args = fn.get("name"), fn.get("arguments")
                pieces.append([tc.get("index") or 0, int(bool(tc.get("id"))),
                               int(bool(tc.get("type"))),
                               len(name) if isinstance(name, str) else None,
                               len(args) if isinstance(args, str) else None])  # fmt: skip
            ops.append(["t", pieces])
        if delta.get("refusal"):
            self.plan_ok = False
        if ch.get("finish_reason"):
            ops.append(["f"])
        self.plan.append(ops)

    def _add(self, data: dict[str, Any]) -> None:
        for key in ("id", "model", "created", "system_fingerprint", "service_tier"):
            if data.get(key) is not None:
                self.meta[key] = data[key]
        if data.get("usage"):
            self.usage = data["usage"]
        if self.plan_ok:
            self._plan(data)
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
                if tc.get("index") is None:
                    self.tool_index_missing = True
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

    @property
    def stream_plan(self) -> list[list[list[Any]]] | None:
        """The shape of every chunk (``["c", n]`` content piece, ``["r", n]`` reasoning
        piece, ``["t", pieces]`` tool-call delta, ``["f"]`` finish, ``["u"]`` usage-only chunk), so
        replay re-chunks the stream as it arrived: a generator agent that yields chunk by
        chunk reproduces. None for multi-choice streams."""
        return self.plan if self.plan_ok and len(self.plan) > 1 else None


class AnthropicAssembler(Assembler):
    """Raw Anthropic message stream events -> Message-shaped dict."""

    def __init__(self) -> None:
        super().__init__()
        self.message: dict[str, Any] = {"type": "message", "role": "assistant", "content": []}
        self.blocks: dict[int, dict[str, Any]] = {}
        self.json_parts: dict[int, list[str]] = {}
        self.pieces: dict[int, list[int]] = {}  # streamed delta lengths per block
        self.usage_events: dict[str, Any] = {}  # usage as message_start / _delta sent it

    def _add(self, data: dict[str, Any]) -> None:
        t = data.get("type")
        if t == "message_start":
            msg = dict(data.get("message") or {})
            msg.pop("content", None)
            if isinstance(msg.get("usage"), dict):
                self.usage_events["start"] = dict(msg["usage"])
            self.message.update(msg)
        elif t == "content_block_start":
            self.blocks[data.get("index", len(self.blocks))] = dict(data.get("content_block") or {})
        elif t == "content_block_delta":
            idx = data.get("index", 0)
            block = self.blocks.setdefault(idx, {"type": "text", "text": ""})
            delta = data.get("delta") or {}
            dt = delta.get("type")
            field = {"text_delta": "text", "input_json_delta": "partial_json",
                     "thinking_delta": "thinking"}.get(dt or "")  # fmt: skip
            if field and isinstance(delta.get(field), str):
                self.pieces.setdefault(idx, []).append(len(delta[field]))
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
                self.usage_events["delta"] = dict(data["usage"])
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

    @property
    def stream_plan(self) -> dict[str, Any] | None:
        """Delta lengths per content block, so replay re-chunks text / thinking / tool
        input as streamed (``text_stream`` consumers see the same pieces), and the usage
        ``message_start`` / ``message_delta`` carried."""
        blocks = {str(i): n for i, n in self.pieces.items() if len(n) > 1}
        plan: dict[str, Any] = {"blocks": blocks} if blocks else {}
        if "start" in self.usage_events and "delta" in self.usage_events:
            plan["usage"] = self.usage_events  # running token counts per event
        return plan or None


class ResponsesAssembler(Assembler):
    """OpenAI Responses API stream events -> the final Response dict.

    A stream closed before ``response.completed`` still keeps what arrived: items that
    were started (``output_item.added``) get the text / function-call argument deltas
    seen so far, marked ``status="incomplete"``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.response: dict[str, Any] = {}
        self.items: dict[int, dict[str, Any]] = {}
        self.partial: dict[int, dict[str, Any]] = {}  # output_index -> item in progress
        self.texts: dict[tuple[int, int], str] = {}  # (output_index, content_index) -> text
        self.args: dict[int, str] = {}  # output_index -> function call arguments so far
        self.pieces: dict[str, list[int]] = {}  # "oi:ci" / "oi:args" -> delta lengths
        # every event as [type, output_index, content_index, delta length]: replay sends
        # the same event sequence (content_part.added, output_text.done ...)
        self.events: list[list[Any]] = []

    def _note(self, data: dict[str, Any]) -> None:
        kind = data.get("type")
        if not isinstance(kind, str) or len(self.events) > 100_000:
            return
        oi, ci = data.get("output_index"), data.get("content_index", data.get("summary_index"))
        delta = data.get("delta")
        self.events.append([kind.removeprefix("response."), oi if isinstance(oi, int) else None,
                            ci if isinstance(ci, int) else None,
                            len(delta) if isinstance(delta, str) else None])  # fmt: skip

    def _add(self, data: dict[str, Any]) -> None:
        self._note(data)
        kind = data.get("type")
        if isinstance(data.get("response"), dict):  # created / in_progress / completed / ...
            self.response = data["response"]
        oi = data.get("output_index", 0)
        if not isinstance(oi, int):
            oi = 0
        if kind == "response.output_item.done" and isinstance(data.get("item"), dict):
            self.items[data.get("output_index", len(self.items))] = data["item"]
        elif kind == "response.output_item.added" and isinstance(data.get("item"), dict):
            self.partial[oi] = data["item"]
        elif kind == "response.output_text.delta" and isinstance(data.get("delta"), str):
            ci = data.get("content_index", 0) if isinstance(data.get("content_index"), int) else 0
            self.texts[(oi, ci)] = self.texts.get((oi, ci), "") + data["delta"]
            self.pieces.setdefault(f"{oi}:{ci}", []).append(len(data["delta"]))
            if oi not in self.partial:
                self.partial[oi] = {"type": "message", "id": data.get("item_id") or "",
                                    "role": "assistant", "content": []}  # fmt: skip
        elif kind == "response.function_call_arguments.delta" and isinstance(
            data.get("delta"), str
        ):
            self.args[oi] = self.args.get(oi, "") + data["delta"]
            self.pieces.setdefault(f"{oi}:args", []).append(len(data["delta"]))

    def _partial_items(self) -> dict[int, dict[str, Any]]:
        out: dict[int, dict[str, Any]] = {}
        for oi, base in self.partial.items():
            if oi in self.items:
                continue
            item = {**base, "status": "incomplete"}
            if item.get("type") == "message":
                parts = [dict(p) for p in item.get("content") or [] if isinstance(p, dict)]
                for (o, ci), text in sorted(self.texts.items()):
                    if o != oi:
                        continue
                    while len(parts) <= ci:
                        parts.append({"type": "output_text", "text": "", "annotations": []})
                    parts[ci] = {**parts[ci], "type": "output_text", "text": text}
                item["content"] = parts
            elif item.get("type") == "function_call" and oi in self.args:
                item["arguments"] = self.args[oi]
            out[oi] = item
        return out

    def result(self) -> dict[str, Any]:
        out = dict(self.response)
        if not out.get("output"):
            items = {**self._partial_items(), **self.items}
            if items:
                out["output"] = [items[i] for i in sorted(items)]
        return out

    @property
    def stream_plan(self) -> dict[str, Any] | None:
        """Delta lengths per text part / function call, for replay re-chunking, and the
        event sequence (``events``) when the stream completed."""
        parts = {k: n for k, n in self.pieces.items() if len(n) > 1}
        plan: dict[str, Any] = {"parts": parts} if parts else {}
        if self.events and self.events[-1][0] in ("completed", "incomplete", "failed"):
            plan["events"] = self.events
        return plan or None


# --- wrappers ------------------------------------------------------------------------


class _StreamBase:
    def __init__(self, inner: Any, call: ChatCall, assembler: Assembler) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_call", call)
        object.__setattr__(self, "_asm", assembler)
        object.__setattr__(self, "_finished", False)
        ctx = getattr(getattr(call, "pending", None), "ctx", None)
        if ctx is not None:
            ctx.open_streams.append(weakref.ref(self))

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

    def __init__(self, snapshot: Any, observer: _DeltaObserver | None = None) -> None:
        super().__init__()
        self.snapshot = to_jsonable(snapshot) if snapshot is not None else {}
        self.observer = observer

    def result(self) -> dict[str, Any]:
        return self.snapshot if isinstance(self.snapshot, dict) else {}

    @property
    def stream_plan(self) -> dict[str, Any] | None:
        return self.observer.plan() if self.observer is not None else None


_DELTA_FIELDS = {"text_delta": "text", "input_json_delta": "partial_json",
                 "thinking_delta": "thinking"}  # fmt: skip


class _DeltaObserver:
    """Sits between ``MessageStream`` and its raw SSE stream and notes each delta's
    length per content block (the chunk plan replay re-cuts the text with)."""

    def __init__(self, raw: Any) -> None:
        self._raw = raw
        self.pieces: dict[int, list[int]] = {}
        self.usage_events: dict[str, Any] = {}

    def _note(self, event: Any) -> None:
        etype = getattr(event, "type", None)
        if etype in ("message_start", "message_delta"):
            src = getattr(event, "message", None) if etype == "message_start" else event
            usage = to_jsonable(getattr(src, "usage", None))
            if isinstance(usage, dict):
                self.usage_events["start" if etype == "message_start" else "delta"] = usage
            return
        if etype != "content_block_delta":
            return
        delta = getattr(event, "delta", None)
        value = getattr(delta, _DELTA_FIELDS.get(getattr(delta, "type", None) or "", ""), None)
        if isinstance(value, str):
            self.pieces.setdefault(getattr(event, "index", 0) or 0, []).append(len(value))

    def __iter__(self) -> Iterator[Any]:
        for event in self._raw:
            self._note(event)
            yield event

    async def __aiter__(self) -> Any:
        async for event in self._raw:
            self._note(event)
            yield event

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)

    def plan(self) -> dict[str, Any] | None:
        blocks = {str(i): n for i, n in self.pieces.items() if len(n) > 1}
        plan: dict[str, Any] = {"blocks": blocks} if blocks else {}
        if "start" in self.usage_events and "delta" in self.usage_events:
            plan["usage"] = self.usage_events
        return plan or None


def _observe(stream: Any) -> _DeltaObserver | None:
    raw = getattr(stream, "_raw_stream", None)
    if raw is None:
        return None
    try:
        observer = _DeltaObserver(raw)
        stream._raw_stream = observer
    except Exception:  # SDK internals moved: record without a plan
        return None
    return observer


class RecordingMessageStreamManager:
    """Wraps Anthropic ``client.messages.stream(...)`` (sync context manager).

    The user gets the SDK's own ``MessageStream`` unchanged; on exit crashlab records the
    SDK-accumulated message snapshot.
    """

    def __init__(self, inner: Any, call: ChatCall) -> None:
        self._inner = inner
        self._call = call
        self._stream: Any = None
        self._observer: _DeltaObserver | None = None

    def __enter__(self) -> Any:
        try:
            self._stream = self._inner.__enter__()
        except BaseException as exc:
            self._call.fail(exc)
            raise
        self._observer = _observe(self._stream)
        return self._stream

    def __exit__(self, exc_type, exc, tb) -> Any:
        snap = _snapshot(self._stream)
        try:
            return self._inner.__exit__(exc_type, exc, tb)
        finally:
            complete = bool(snap is not None and getattr(snap, "stop_reason", None))
            self._call.finish_stream(
                _SnapshotAssembler(snap, self._observer), error=exc, complete=complete
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class AsyncRecordingMessageStreamManager(RecordingMessageStreamManager):
    async def __aenter__(self) -> Any:
        try:
            self._stream = await self._inner.__aenter__()
        except BaseException as exc:
            self._call.fail(exc)
            raise
        self._observer = _observe(self._stream)
        return self._stream

    async def __aexit__(self, exc_type, exc, tb) -> Any:
        snap = _snapshot(self._stream)
        try:
            return await self._inner.__aexit__(exc_type, exc, tb)
        finally:
            complete = bool(snap is not None and getattr(snap, "stop_reason", None))
            self._call.finish_stream(
                _SnapshotAssembler(snap, self._observer), error=exc, complete=complete
            )
