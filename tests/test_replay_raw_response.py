"""Replay of ``with_raw_response.create`` (LangChain ``ChatOpenAI``, CrewAI) and of
provider responses that a strict ``model_validate`` rejects (Groq ``service_tier``)."""

from __future__ import annotations

import pytest

import crashlab
from tests._transport import Script
from tests.test_sdk_transport import _openai_client


def _groq_body(text: str) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "openai/gpt-oss-20b",
        "service_tier": "on_demand",  # not in openai's Literal: model_validate rejects it
        "x_groq": {"id": "req_1"},
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def test_raw_response_replays_as_a_raw_response_with_a_real_completion(crashlab_dir):
    openai = pytest.importorskip("openai")
    from openai.types.chat import ChatCompletion

    script = Script([_groq_body("hello")])
    live = crashlab.wrap(_openai_client(openai, script))
    seen: list = []

    def run(client) -> str:
        with crashlab.record("raw-replay") as r:
            raw = client.chat.completions.with_raw_response.create(
                model="openai/gpt-oss-20b", messages=[{"role": "user", "content": "hi"}]
            )
            headers = dict(raw.headers)  # LangChain reads these
            parsed = raw.parse()
            seen.append((type(parsed), headers is not None, raw.text))
            r.output = parsed.choices[0].message.content
        return r.output

    run(live)
    src = crashlab.load("last")
    rep = crashlab.replay(src, fn=lambda: run(crashlab.wrap(openai.OpenAI(api_key="x"))))
    assert rep.status == "ok" and rep.output == "hello"
    assert len(script.requests) == 1  # replay made no HTTP call
    parsed_type, has_headers, text = seen[-1]
    assert issubclass(parsed_type, ChatCompletion)  # not a ReplayObject
    assert has_headers and '"hello"' in text


def test_plain_create_rebuilds_a_real_completion_despite_provider_extras(crashlab_dir):
    openai = pytest.importorskip("openai")
    from openai.types.chat import ChatCompletion

    script = Script([_groq_body("hi there")])
    live = crashlab.wrap(_openai_client(openai, script))
    types: list = []

    def run(client) -> None:
        with crashlab.record("groq-extras") as r:
            resp = client.chat.completions.create(model="m", messages=[])
            types.append(type(resp))
            r.output = resp.choices[0].message.content

    run(live)
    rep = crashlab.replay(
        crashlab.load("last"), fn=lambda: run(crashlab.wrap(openai.OpenAI(api_key="x")))
    )
    assert rep.status == "ok" and rep.output == "hi there"
    assert issubclass(types[-1], ChatCompletion)
