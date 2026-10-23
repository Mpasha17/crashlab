"""``with_raw_response.create`` must record (CrewAI's call path)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import crashlab
from crashlab.capture.openai import WrappedOpenAI


def _completion(**kwargs: Any) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        model=kwargs.get("model", "m"),
        id="chatcmpl-x",
    )


class _Completions:
    def __init__(self) -> None:
        self.calls = 0

        def raw_create(**kwargs: Any) -> Any:
            self.calls += 1
            parsed = _completion(**kwargs)

            class _Raw:
                status_code = 200
                headers = {}

                def parse(self):
                    return parsed

            return _Raw()

        self.with_raw_response = SimpleNamespace(create=raw_create)

    def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        return _completion(**kwargs)


class _Client:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_Completions())
        self.base_url = "https://example.test/v1"


def test_with_raw_response_create_goes_through_recording(crashlab_dir):
    inner = _Client()
    wrapped = WrappedOpenAI(inner)

    @crashlab.agent(name="raw-path")
    def run() -> str:
        resp = wrapped.chat.completions.with_raw_response.create(
            model="m", messages=[{"role": "user", "content": "hi"}]
        )
        return resp.parse().choices[0].message.content

    assert run() == "ok"
    rec = crashlab.load("last")
    assert len(rec.steps) == 1 and rec.steps[0].operation == "chat"
    assert rec.steps[0].response and rec.steps[0].response.get("messages")
    assert inner.chat.completions.calls == 1
