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
        return item


class _AsyncCompletions(_Completions):
    async def create(self, **kwargs: Any) -> Any:  # type: ignore[override]
        return super().create(**kwargs)


class _Chat:
    def __init__(self, completions: Any) -> None:
        self.completions = completions


class FakeOpenAI:
    base_url = "http://fake.local/v1"

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []
        self.chat = _Chat(_Completions(self))


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
