"""A fake OpenAI-like client: same object shapes as openai's ChatCompletion, no network."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel


class Function(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: Function


class Message(BaseModel):
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletion(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = 1760700000
    model: str
    choices: list[Choice]
    usage: CompletionUsage | None = None


def tool_call_response(n: int, name: str, args: dict[str, Any], model: str = "fake-model"):
    return ChatCompletion(
        id=f"chatcmpl-{n}",
        model=model,
        choices=[
            Choice(
                message=Message(
                    tool_calls=[
                        ToolCall(
                            id=f"call_{n}", function=Function(name=name, arguments=json.dumps(args))
                        )
                    ]
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=CompletionUsage(prompt_tokens=100 + n, completion_tokens=10, total_tokens=110 + n),
    )


def text_response(n: int, text: str, model: str = "fake-model"):
    return ChatCompletion(
        id=f"chatcmpl-{n}",
        model=model,
        choices=[Choice(message=Message(content=text), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=200, completion_tokens=20, total_tokens=220),
    )


class _Completions:
    def __init__(self, owner: FakeOpenAI) -> None:
        self.owner = owner

    def create(self, **kwargs: Any) -> Any:
        self.owner.calls.append(kwargs)
        item = self.owner.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        if kwargs.get("stream"):
            return item if isinstance(item, FakeStream) else FakeStream(item)
        return item


class _AsyncCompletions(_Completions):
    async def create(self, **kwargs: Any) -> Any:  # type: ignore[override]
        item = super().create(**kwargs)
        if kwargs.get("stream"):
            return AsyncFakeStream(item.items, item.fail_after)
        return item


class _Chat:
    def __init__(self, completions: Any) -> None:
        self.completions = completions


class FakeOpenAI:
    base_url = "http://fake.local/v1"

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []
        self.chat = _Chat(_Completions(self))
        self.responses = _Responses(self)


class AsyncFakeOpenAI(FakeOpenAI):
    def __init__(self, script: list[Any]) -> None:
        super().__init__(script)
        self.chat = _Chat(_AsyncCompletions(self))


class FakeAPIError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def refund_script(amount: float = 49.0) -> list[Any]:
    return [
        tool_call_response(1, "get_order", {"order_id": "A17"}),
        tool_call_response(2, "issue_refund", {"order_id": "A17", "amount": amount}),
        text_response(3, "Your refund of $49.00 for order A17 has been issued."),
    ]


# --- OpenAI streaming ----------------------------------------------------------------


class _Chunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = 1760700000
    model: str
    choices: list[dict]
    usage: dict | None = None


def chunks_for_text(text: str, n: int = 1, model: str = "fake-model", usage: bool = True):
    pieces = [text[i : i + 4] for i in range(0, len(text), 4)]
    out = [
        _Chunk(
            id=f"chatcmpl-{n}",
            model=model,
            choices=[{"index": 0, "delta": {"role": "assistant", "content": ""}}],
        )
    ]
    out += [
        _Chunk(id=f"chatcmpl-{n}", model=model, choices=[{"index": 0, "delta": {"content": p}}])
        for p in pieces
    ]
    out.append(
        _Chunk(
            id=f"chatcmpl-{n}",
            model=model,
            choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
        )
    )
    if usage:
        out.append(
            _Chunk(
                id=f"chatcmpl-{n}",
                model=model,
                choices=[],
                usage={"prompt_tokens": 30, "completion_tokens": 7, "total_tokens": 37},
            )
        )
    return out


def chunks_for_tool_call(n: int, name: str, args: dict, model: str = "fake-model"):
    arg_text = json.dumps(args)
    half = len(arg_text) // 2
    tc0 = {
        "index": 0,
        "id": f"call_{n}",
        "type": "function",
        "function": {"name": name, "arguments": ""},
    }
    return [
        _Chunk(
            id=f"c{n}",
            model=model,
            choices=[{"index": 0, "delta": {"role": "assistant", "tool_calls": [tc0]}}],
        ),
        _Chunk(
            id=f"c{n}",
            model=model,
            choices=[
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": arg_text[:half]}}]
                    },
                }
            ],
        ),
        _Chunk(
            id=f"c{n}",
            model=model,
            choices=[
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": arg_text[half:]}}]
                    },
                }
            ],
        ),
        _Chunk(
            id=f"c{n}",
            model=model,
            choices=[{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        ),
    ]


class FakeStream:
    """Mimics openai.Stream / anthropic.Stream: iterable, context manager, close()."""

    def __init__(self, items: list, fail_after: int | None = None) -> None:
        self.items = list(items)
        self.fail_after = fail_after
        self.closed = False

    def __iter__(self):
        for i, item in enumerate(self.items):
            if self.fail_after is not None and i == self.fail_after:
                raise ConnectionError("stream dropped")
            yield item

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.closed = True


class AsyncFakeStream(FakeStream):
    def __aiter__(self):
        return self._agen()

    async def _agen(self):
        for item in self.__iter__():
            yield item

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.close()


# --- OpenAI responses API ------------------------------------------------------------


def responses_tool_call(n: int, name: str, args: dict, model: str = "fake-model") -> dict:
    return {
        "id": f"resp_{n}",
        "object": "response",
        "model": model,
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "id": f"fc_{n}",
                "call_id": f"call_{n}",
                "name": name,
                "arguments": json.dumps(args),
                "status": "completed",
            }
        ],
        "usage": {"input_tokens": 40, "output_tokens": 8, "total_tokens": 48},
    }


def responses_text(n: int, text: str, model: str = "fake-model") -> dict:
    return {
        "id": f"resp_{n}",
        "object": "response",
        "model": model,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": f"msg_{n}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 60, "output_tokens": 12, "total_tokens": 72},
    }


class _Responses:
    def __init__(self, owner: FakeOpenAI) -> None:
        self.owner = owner

    def create(self, **kwargs):
        self.owner.calls.append(kwargs)
        item = self.owner.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


# --- Anthropic -----------------------------------------------------------------------


class ATextBlock(BaseModel):
    type: str = "text"
    text: str


class AToolUseBlock(BaseModel):
    type: str = "tool_use"
    id: str
    name: str
    input: dict


class AUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None


class AMessage(BaseModel):
    id: str
    type: str = "message"
    role: str = "assistant"
    model: str
    content: list[ATextBlock | AToolUseBlock]
    stop_reason: str | None
    stop_sequence: str | None = None
    usage: AUsage


def a_tool_use(n: int, name: str, args: dict, model: str = "claude-fake") -> AMessage:
    return AMessage(
        id=f"msg_{n}",
        model=model,
        content=[
            ATextBlock(text="Let me check."),
            AToolUseBlock(id=f"call_{n}", name=name, input=args),
        ],
        stop_reason="tool_use",
        usage=AUsage(input_tokens=100 + n, output_tokens=10, cache_read_input_tokens=5),
    )


def a_text(n: int, text: str, model: str = "claude-fake") -> AMessage:
    return AMessage(
        id=f"msg_{n}",
        model=model,
        content=[ATextBlock(text=text)],
        stop_reason="end_turn",
        usage=AUsage(input_tokens=200, output_tokens=20),
    )


def a_events_for(msg: AMessage) -> list[dict]:
    """Raw stream events (as dicts) that assemble into ``msg``."""
    events: list[dict] = [
        {
            "type": "message_start",
            "message": {
                **msg.model_dump(),
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": msg.usage.input_tokens, "output_tokens": 1},
            },
        },
    ]
    for i, block in enumerate(msg.content):
        if block.type == "text":
            events.append(
                {
                    "type": "content_block_start",
                    "index": i,
                    "content_block": {"type": "text", "text": ""},
                }
            )
            for j in range(0, len(block.text), 5):
                events.append(
                    {
                        "type": "content_block_delta",
                        "index": i,
                        "delta": {"type": "text_delta", "text": block.text[j : j + 5]},
                    }
                )
        else:
            events.append(
                {
                    "type": "content_block_start",
                    "index": i,
                    "content_block": {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": {},
                    },
                }
            )
            js = json.dumps(block.input)
            for j in range(0, len(js), 6):
                events.append(
                    {
                        "type": "content_block_delta",
                        "index": i,
                        "delta": {"type": "input_json_delta", "partial_json": js[j : j + 6]},
                    }
                )
        events.append({"type": "content_block_stop", "index": i})
    events.append(
        {
            "type": "message_delta",
            "delta": {"stop_reason": msg.stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": msg.usage.output_tokens},
        }
    )
    events.append({"type": "message_stop"})
    return events


class FakeMessageStream:
    """Mimics anthropic MessageStream: iterate events; SDK-side accumulated snapshot."""

    def __init__(self, msg: AMessage) -> None:
        self.msg = msg
        self.events = a_events_for(msg)
        self._consumed = 0

    def __iter__(self):
        for e in self.events:
            self._consumed += 1
            yield e

    @property
    def text_stream(self):
        for e in self:
            if e["type"] == "content_block_delta" and e["delta"]["type"] == "text_delta":
                yield e["delta"]["text"]

    def _current_snapshot_or_none(self):
        if self._consumed == 0:
            return None
        if self._consumed >= len(self.events):
            return self.msg
        return self.msg.model_copy(update={"stop_reason": None})

    def get_final_message(self):
        for _ in self:
            pass
        return self.msg


class FakeStreamManager:
    def __init__(self, msg: AMessage) -> None:
        self.stream = FakeMessageStream(msg)
        self.exited = False

    def __enter__(self):
        return self.stream

    def __exit__(self, *exc):
        self.exited = True


class AsyncFakeMessageStream(FakeMessageStream):
    def __aiter__(self):
        return self._agen()

    async def _agen(self):
        for e in self.__iter__():
            yield e

    async def get_final_message(self):  # type: ignore[override]
        async for _ in self:
            pass
        return self.msg


class AsyncFakeStreamManager:
    def __init__(self, msg: AMessage) -> None:
        self.stream = AsyncFakeMessageStream(msg)

    async def __aenter__(self):
        return self.stream

    async def __aexit__(self, *exc):
        return None


class _AMessages:
    def __init__(self, owner: FakeAnthropic) -> None:
        self.owner = owner

    def _next(self, kwargs):
        self.owner.calls.append(kwargs)
        item = self.owner.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def create(self, **kwargs):
        item = self._next(kwargs)
        if kwargs.get("stream"):
            return FakeStream(a_events_for(item) if isinstance(item, AMessage) else item)
        return item

    def stream(self, **kwargs):
        return FakeStreamManager(self._next(kwargs))


class _AsyncAMessages(_AMessages):
    async def create(self, **kwargs):  # type: ignore[override]
        item = self._next(kwargs)
        if kwargs.get("stream"):
            return AsyncFakeStream(a_events_for(item))
        return item

    def stream(self, **kwargs):  # returns an async context manager, like the SDK
        return AsyncFakeStreamManager(self._next(kwargs))


class FakeAnthropic:
    base_url = "https://api.anthropic.fake"

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.calls: list[dict] = []
        self.messages = _AMessages(self)


class AsyncFakeAnthropic(FakeAnthropic):
    def __init__(self, script: list) -> None:
        super().__init__(script)
        self.messages = _AsyncAMessages(self)


def anthropic_refund_script(amount: float = 49.0) -> list:
    return [
        a_tool_use(1, "get_order", {"order_id": "A17"}),
        a_tool_use(2, "issue_refund", {"order_id": "A17", "amount": amount}),
        a_text(3, "Your refund of $49.00 for order A17 has been issued."),
    ]
