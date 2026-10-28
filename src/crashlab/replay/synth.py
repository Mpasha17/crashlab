"""Synthetic streams for replay.

Cassettes store the *assembled* response of a streamed call, so a replayed ``stream=True``
call gets a synthetic stream rebuilt from it: one chunk per content piece / tool call
instead of the original token-by-token chunks. Fed back through crashlab's assemblers it
yields the same final response. (Original chunk boundaries are not preserved.)
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

from .rebuild import rebuild, rebuild_chunk

# --- chunk synthesis -----------------------------------------------------------------


def _planned_chunks(
    raw: dict[str, Any], plan: Any, chunk: Callable[..., dict[str, Any]], tool_index: bool
) -> list[dict[str, Any]] | None:
    """Re-chunk a one-choice completion along the recorded chunk plan; None when the plan
    does not fit the text (e.g. redacted content) and the default chunking is used."""
    choices = raw.get("choices") or []
    if not isinstance(plan, list) or len(choices) != 1:
        return None
    ch = choices[0]
    msg = ch.get("message") or {}
    text = msg.get("content") or ""
    rkey = next((k for k in ("reasoning_content", "reasoning") if msg.get(k)), "reasoning")
    reasoning = msg.get(rkey) or ""
    try:
        ops = [op for entry in plan for op in entry]
        if sum(op[1] for op in ops if op[0] == "c") != len(text):
            return None
        if sum(op[1] for op in ops if op[0] == "r") != len(reasoning):
            return None
    except (TypeError, IndexError):
        return None
    out: list[dict[str, Any]] = []
    at_c = at_r = 0
    tools_sent = role_sent = False
    for entry in plan:
        if entry == [["u"]]:
            if raw.get("usage"):
                out.append(chunk(None, usage=raw["usage"]))
            continue
        delta: dict[str, Any] = {}
        if not role_sent:
            delta["role"] = msg.get("role") or "assistant"
            role_sent = True
        finish = None
        for op in entry:
            if op[0] == "c":
                delta["content"] = text[at_c : at_c + op[1]]
                at_c += op[1]
            elif op[0] == "r":
                delta[rkey] = reasoning[at_r : at_r + op[1]]
                at_r += op[1]
            elif op[0] == "t" and not tools_sent:
                tools_sent = True
                calls = []
                for i, tc in enumerate(msg.get("tool_calls") or []):
                    call = {k: v for k, v in tc.items() if k != "function"}
                    if tool_index:
                        call["index"] = i
                    call["function"] = dict(tc.get("function") or {})
                    calls.append(call)
                if calls:
                    delta["tool_calls"] = calls
            elif op[0] == "f":
                finish = ch.get("finish_reason")
        out.append(chunk({"index": ch.get("index") or 0, "delta": delta, "finish_reason": finish}))
    if msg.get("tool_calls") and not tools_sent:
        return None
    return out


def openai_chat_chunks(
    raw: dict[str, Any],
    *,
    tool_index: bool = True,
    plan: list[Any] | None = None,
) -> list[dict[str, Any]]:
    base = {"object": "chat.completion.chunk"}
    for key in ("id", "created", "model", "system_fingerprint", "service_tier"):
        if raw.get(key) is not None:
            base[key] = raw[key]
    base.setdefault("created", 0)
    base.setdefault("id", "replay")
    base.setdefault("model", "")

    def chunk(choice: dict[str, Any] | None, **extra: Any) -> dict[str, Any]:
        return {**base, "choices": [choice] if choice else [], **extra}

    planned = _planned_chunks(raw, plan, chunk, tool_index) if plan else None
    if planned is not None:
        return planned
    out: list[dict[str, Any]] = []
    for ch in raw.get("choices") or []:
        idx = ch.get("index") or 0
        msg = ch.get("message") or {}
        delta: dict[str, Any] = {"role": msg.get("role") or "assistant"}
        for key in ("reasoning_content", "reasoning"):
            if isinstance(msg.get(key), str) and msg[key]:
                delta[key] = msg[key]
        if msg.get("content"):
            delta["content"] = msg["content"]
        if msg.get("refusal"):
            delta["refusal"] = msg["refusal"]
        out.append(chunk({"index": idx, "delta": delta, "finish_reason": None}))
        for i, tc in enumerate(msg.get("tool_calls") or []):
            call = {k: v for k, v in tc.items() if k not in ("function",)}
            if tool_index:  # else: as recorded, the provider sent no index
                call["index"] = i
            call["function"] = dict(tc.get("function") or {})
            out.append(
                chunk({"index": idx, "delta": {"tool_calls": [call]}, "finish_reason": None})
            )
        out.append(chunk({"index": idx, "delta": {}, "finish_reason": ch.get("finish_reason")}))
    if raw.get("usage"):
        out.append(chunk(None, usage=raw["usage"]))
    return out


def _block_start(i: int, block: dict[str, Any]) -> dict[str, Any]:
    return {"type": "content_block_start", "index": i, "content_block": block}


def _block_delta(i: int, **delta: Any) -> dict[str, Any]:
    return {"type": "content_block_delta", "index": i, "delta": delta}


def _cut(text: str, lengths: Any) -> list[str]:
    """``text`` cut at the recorded delta lengths; whole when they do not add up."""
    if not isinstance(lengths, list) or not all(isinstance(n, int) for n in lengths):
        return [text]
    if sum(lengths) != len(text):
        return [text]
    out, at = [], 0
    for n in lengths:
        out.append(text[at : at + n])
        at += n
    return out


def anthropic_events(raw: dict[str, Any], plan: Any = None) -> list[dict[str, Any]]:
    blocks_plan = plan.get("blocks") if isinstance(plan, dict) else None
    blocks_plan = blocks_plan if isinstance(blocks_plan, dict) else {}
    usage = dict(raw.get("usage") or {})
    skip = ("content", "stop_reason", "stop_sequence")
    start_msg = {k: v for k, v in raw.items() if k not in skip}
    start_msg.update({"content": [], "stop_reason": None, "stop_sequence": None})
    start_msg.setdefault("type", "message")
    start_msg.setdefault("role", "assistant")
    start_msg["usage"] = {**usage, "output_tokens": usage.get("output_tokens", 0)}
    events: list[dict[str, Any]] = [{"type": "message_start", "message": start_msg}]
    for i, block in enumerate(raw.get("content") or []):
        t = block.get("type")
        if t == "text":
            events.append(_block_start(i, {**block, "text": ""}))
            for piece in _cut(block.get("text", ""), blocks_plan.get(str(i))):
                events.append(_block_delta(i, type="text_delta", text=piece))
        elif t in ("tool_use", "server_tool_use"):
            inp = block.get("input")
            partial = inp if isinstance(inp, str) else json.dumps(inp or {})
            events.append(_block_start(i, {**block, "input": {}}))
            for piece in _cut(partial, blocks_plan.get(str(i))):
                events.append(_block_delta(i, type="input_json_delta", partial_json=piece))
        elif t == "thinking":
            events.append(_block_start(i, {"type": "thinking", "thinking": "", "signature": ""}))
            for piece in _cut(block.get("thinking", ""), blocks_plan.get(str(i))):
                events.append(_block_delta(i, type="thinking_delta", thinking=piece))
            if block.get("signature"):
                events.append(_block_delta(i, type="signature_delta", signature=block["signature"]))
        else:  # redacted_thinking, web search results ...: sent whole
            events.append(_block_start(i, block))
        events.append({"type": "content_block_stop", "index": i})
    stop = {"stop_reason": raw.get("stop_reason"), "stop_sequence": raw.get("stop_sequence")}
    out_usage = {k: v for k, v in usage.items() if k != "output_tokens"}
    out_usage["output_tokens"] = usage.get("output_tokens", 0)
    events.append({"type": "message_delta", "delta": stop, "usage": out_usage})
    events.append({"type": "message_stop"})
    return events


def responses_events(raw: dict[str, Any], plan: Any = None) -> list[dict[str, Any]]:
    parts_plan = plan.get("parts") if isinstance(plan, dict) else None
    parts_plan = parts_plan if isinstance(parts_plan, dict) else {}
    seq = iter(range(1_000_000))
    created = {**raw, "status": "in_progress", "output": []}
    events: list[dict[str, Any]] = [
        {"type": "response.created", "sequence_number": next(seq), "response": created}
    ]
    for oi, item in enumerate(raw.get("output") or []):
        events.append({"type": "response.output_item.added", "sequence_number": next(seq),
                       "output_index": oi, "item": item})  # fmt: skip
        if item.get("type") == "message":
            for ci, part in enumerate(item.get("content") or []):
                if part.get("type") == "output_text":
                    for piece in _cut(part.get("text", ""), parts_plan.get(f"{oi}:{ci}")):
                        events.append({
                            "type": "response.output_text.delta", "sequence_number": next(seq),
                            "item_id": item.get("id", ""), "output_index": oi,
                            "content_index": ci, "delta": piece, "logprobs": [],
                        })  # fmt: skip
        elif item.get("type") == "function_call" and f"{oi}:args" in parts_plan:
            args = item.get("arguments") if isinstance(item.get("arguments"), str) else ""
            for piece in _cut(args, parts_plan[f"{oi}:args"]):
                events.append({
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": next(seq), "item_id": item.get("id", ""),
                    "output_index": oi, "delta": piece,
                })  # fmt: skip
        events.append({"type": "response.output_item.done", "sequence_number": next(seq),
                       "output_index": oi, "item": item})  # fmt: skip
    done = "response.incomplete" if raw.get("status") == "incomplete" else "response.completed"
    events.append({"type": done, "sequence_number": next(seq), "response": raw})
    return events


SYNTHESIZERS: dict[str, Callable[..., list[dict[str, Any]]]] = {
    "openai.chat.completions": openai_chat_chunks,
    "litellm.completion": openai_chat_chunks,
    "anthropic.messages": anthropic_events,
    "openai.responses": responses_events,
}


def synthesize(
    api: str,
    raw: dict[str, Any] | None,
    chunk_type: str | None,
    *,
    tool_index: bool = True,
    plan: Any = None,
) -> list[Any]:
    if not raw:
        return []
    make = SYNTHESIZERS.get(api, openai_chat_chunks)
    if make is openai_chat_chunks:
        chunks = make(raw, tool_index=tool_index, plan=plan)
    elif make in (anthropic_events, responses_events):
        chunks = make(raw, plan=plan)
    else:
        chunks = make(raw)
    return [rebuild_chunk(c, chunk_type) for c in chunks]


# --- stream objects ------------------------------------------------------------------


class _ReplayHTTPResponse:
    """Stand-in for the stream's ``httpx.Response``: the SDK stream helpers
    (``chat.completions.stream()``, ``responses.stream()``) close it on exit."""

    status_code = 200
    is_closed = False

    def __init__(self) -> None:
        self.headers: dict[str, str] = {}

    def close(self) -> None:
        self.is_closed = True

    async def aclose(self) -> None:
        self.is_closed = True


class ReplayStream:
    """A finished stream served from a cassette (sync). Raises ``error`` at the end, if any."""

    def __init__(self, items: list[Any], error: BaseException | None = None) -> None:
        self._items = list(items)
        self._error = error
        self._it: Iterator[Any] = iter(self._items)
        self.response = _ReplayHTTPResponse()  # no real HTTP response in replay

    def __iter__(self) -> ReplayStream:
        return self

    def __next__(self) -> Any:
        try:
            return next(self._it)
        except StopIteration:
            if self._error is not None:
                err, self._error = self._error, None
                raise err from None
            raise

    def __enter__(self) -> ReplayStream:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self._it = iter(())
        self._error = None


class AsyncReplayStream(ReplayStream):
    def __aiter__(self) -> AsyncReplayStream:
        return self

    async def __anext__(self) -> Any:
        try:
            return self.__next__()
        except StopIteration:
            raise StopAsyncIteration from None

    async def __aenter__(self) -> AsyncReplayStream:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def aclose(self) -> None:
        await self.close()

    async def close(self) -> None:  # type: ignore[override]  # SDK AsyncStream.close is async
        self._it = iter(())
        self._error = None


class _FallbackMessageStream:
    """Minimal stand-in for ``anthropic.lib.streaming.MessageStream`` (no SDK installed)."""

    def __init__(self, stream: ReplayStream, final: Any) -> None:
        self._stream = stream
        self._final = final

    def __iter__(self):
        return iter(self._stream)

    @property
    def text_stream(self) -> Iterator[str]:
        for ev in self._stream:
            delta = getattr(ev, "delta", None)
            if getattr(ev, "type", None) == "content_block_delta" and getattr(delta, "text", None):
                yield delta.text

    def until_done(self) -> None:
        for _ in self._stream:
            pass

    def get_final_message(self) -> Any:
        self.until_done()
        return self._final

    def get_final_text(self) -> str:
        msg = self.get_final_message()
        return "".join(getattr(b, "text", "") or "" for b in (msg.content or []))

    def close(self) -> None:
        self._stream.close()


def _message_stream(stream: ReplayStream, raw: dict[str, Any] | None, is_async: bool) -> Any:
    try:
        from anthropic import NOT_GIVEN
        from anthropic.lib.streaming import AsyncMessageStream, MessageStream

        cls = AsyncMessageStream if is_async else MessageStream
        return cls(stream, output_format=NOT_GIVEN)  # type: ignore[arg-type]
    except Exception:
        final = rebuild(raw or {}, "anthropic.types.Message")
        return _FallbackMessageStream(stream, final)


class ReplayMessageStreamManager:
    """Replay of Anthropic ``client.messages.stream(...)``: yields the SDK's own
    ``MessageStream`` over a synthetic event stream (``text_stream``,
    ``get_final_message()`` ... work as live)."""

    def __init__(self, events: list[Any], raw: dict[str, Any] | None, error: Any = None):
        self._events = events
        self._raw = raw
        self._error = error

    def __enter__(self) -> Any:
        return _message_stream(ReplayStream(self._events, self._error), self._raw, False)

    def __exit__(self, *exc: Any) -> None:
        return None


class AsyncReplayMessageStreamManager(ReplayMessageStreamManager):
    async def __aenter__(self) -> Any:
        return _message_stream(AsyncReplayStream(self._events, self._error), self._raw, True)

    async def __aexit__(self, *exc: Any) -> None:
        return None
