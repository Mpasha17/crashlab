"""Anthropic ``messages.with_raw_response.create`` must record."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import crashlab
from crashlab.capture.anthropic import WrappedAnthropic


def _message(**kwargs: Any) -> Any:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        model=kwargs.get("model", "m"),
        id="msg-x",
        role="assistant",
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )


class _Messages:
    def __init__(self) -> None:
        self.calls = 0

        def raw_create(**kwargs: Any) -> Any:
            self.calls += 1
            parsed = _message(**kwargs)

            class _Raw:
                status_code = 200
                headers = {}

                def parse(self):
                    return parsed

            return _Raw()

        self.with_raw_response = SimpleNamespace(create=raw_create)

    def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        return _message(**kwargs)

    def stream(self, **kwargs: Any) -> Any:
        raise NotImplementedError


class _Client:
    def __init__(self) -> None:
        self.messages = _Messages()
        self.base_url = "https://api.anthropic.com"


def test_anthropic_with_raw_response_create_records(crashlab_dir):
    inner = _Client()
    wrapped = WrappedAnthropic(inner)

    @crashlab.agent(name="anthropic-raw")
    def run() -> str:
        resp = wrapped.messages.with_raw_response.create(
            model="m",
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
        )
        parsed = resp.parse()
        return parsed.content[0].text

    assert run() == "ok"
    rec = crashlab.load("last")
    assert len(rec.steps) == 1
    assert rec.steps[0].attributes.get("crashlab.api") == "anthropic.messages"
    assert rec.steps[0].response is not None
    assert inner.messages.calls == 1


class _AsyncMessages:
    def __init__(self) -> None:
        self.calls = 0

        async def raw_create(**kwargs: Any) -> Any:
            self.calls += 1
            parsed = _message(**kwargs)

            class _Raw:
                status_code = 200
                headers = {}

                def parse(self):
                    return parsed

            return _Raw()

        self.with_raw_response = SimpleNamespace(create=raw_create)

    async def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        return _message(**kwargs)

    def stream(self, **kwargs: Any) -> Any:
        raise NotImplementedError


class _AsyncClient:
    def __init__(self) -> None:
        self.messages = _AsyncMessages()
        self.base_url = "https://api.anthropic.com"


@pytest.mark.asyncio
async def test_async_anthropic_with_raw_response_records(crashlab_dir):
    inner = _AsyncClient()
    wrapped = WrappedAnthropic(inner)

    @crashlab.agent(name="anthropic-raw-async")
    async def run() -> str:
        resp = await wrapped.messages.with_raw_response.create(
            model="m",
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
        )
        return resp.parse().content[0].text

    assert await run() == "ok"
    rec = crashlab.load("last")
    assert len(rec.steps) == 1
    assert rec.steps[0].attributes.get("crashlab.api") == "anthropic.messages"
    assert inner.messages.calls == 1


def test_anthropic_with_options_rewrap_keeps_raw_recording(crashlab_dir):
    class _Inner:
        def __init__(self):
            self.messages = _Messages()
            self.base_url = "https://api.anthropic.com"

        def with_options(self, **kwargs):
            other = _Inner()
            other._opts = kwargs
            return other

    wrapped = WrappedAnthropic(_Inner())
    child = wrapped.with_options(timeout=10)

    @crashlab.agent(name="anthropic-raw-opts")
    def run() -> str:
        resp = child.messages.with_raw_response.create(
            model="m",
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
        )
        return resp.parse().content[0].text

    assert run() == "ok"
    assert crashlab.load("last").steps[0].attributes.get("crashlab.api") == "anthropic.messages"
