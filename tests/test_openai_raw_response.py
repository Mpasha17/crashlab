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


class _Responses:
    def __init__(self) -> None:
        self.calls = 0

        def raw_create(**kwargs: Any) -> Any:
            self.calls += 1
            parsed = SimpleNamespace(
                output_text="ok",
                model=kwargs.get("model", "m"),
                id="resp-x",
                output=[],
                usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
            )

            class _Raw:
                status_code = 200
                headers = {}

                def parse(self):
                    return parsed

            return _Raw()

        self.with_raw_response = SimpleNamespace(create=raw_create)

    def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        return SimpleNamespace(output_text="ok", model="m", id="resp-x", output=[])


class _ClientWithResponses:
    def __init__(self) -> None:
        # minimal chat so looks_like_openai / wrap still works; responses is the focus
        self.chat = SimpleNamespace(completions=_Completions())
        self.responses = _Responses()
        self.base_url = "https://example.test/v1"


def test_responses_with_raw_response_create_records(crashlab_dir):
    inner = _ClientWithResponses()
    wrapped = WrappedOpenAI(inner)

    @crashlab.agent(name="raw-responses")
    def run() -> str:
        resp = wrapped.responses.with_raw_response.create(model="m", input="hi")
        return resp.parse().output_text

    assert run() == "ok"
    rec = crashlab.load("last")
    assert len(rec.steps) == 1
    step = rec.steps[0]
    assert step.attributes.get("crashlab.api") == "openai.responses"
    assert step.response is not None
    assert inner.responses.calls == 1
