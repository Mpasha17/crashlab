"""Replay through every capture path: OpenAI chat / Responses, Anthropic, LiteLLM;
plain and streamed; sync and async. Fake clients only (no network, no keys)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

import crashlab
from crashlab.capture import normalize as N
from crashlab.capture.streams import AnthropicAssembler, OpenAIChatAssembler
from crashlab.integrations import litellm as cl_litellm
from crashlab.replay import rebuild as R
from crashlab.replay import synth
from tests.fakes import (
    AsyncFakeAnthropic,
    AsyncFakeOpenAI,
    FakeAnthropic,
    FakeOpenAI,
    FakeStream,
    a_text,
    a_tool_use,
    chunks_for_text,
    chunks_for_tool_call,
    responses_text,
    responses_tool_call,
    text_response,
    tool_call_response,
)
from tests.test_litellm import _fake_litellm


def _roundtrip(make_client, body, *, is_async: bool = False):
    """Record ``body(client)`` once, then replay it with an empty client (any live call
    would fail). The replay itself asserts identical requests and output."""

    def main(empty: bool = False) -> None:
        client = make_client(empty)

        async def arun() -> Any:
            async with crashlab.record("prov") as run:
                run.output = await body(client)

        if is_async:
            asyncio.run(arun())
            return
        with crashlab.record("prov") as run:
            run.output = body(client)

    main()
    src = crashlab.load("last")
    rep = crashlab.replay(src, fn=lambda: main(empty=True))
    assert rep.status == "ok" and rep.output == src.output
    assert all(s.source == "cassette" for s in rep.steps)
    return src, rep


def _openai(script, cls=FakeOpenAI):
    return lambda empty: crashlab.wrap(cls([] if empty else list(script)))


# --- OpenAI chat ---------------------------------------------------------------------


def test_openai_plain_tool_call():
    def body(c):
        r = c.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
        tc = r.choices[0].message.tool_calls[0]
        return [type(r).__name__, tc.id, tc.function.name, tc.function.arguments,
                r.usage.total_tokens]  # fmt: skip

    src, _ = _roundtrip(_openai([tool_call_response(1, "get_order", {"order_id": "A17"})]), body)
    assert src.output[0] == "ChatCompletion"
    assert src.steps[0].attributes["crashlab.response.type"] == "tests.fakes.ChatCompletion"


def test_openai_stream_text_and_tool_call():
    def body(c):
        text = "".join(
            ch.choices[0]["delta"].get("content") or ""
            for ch in c.chat.completions.create(
                model="m", messages=[], stream=True, stream_options={"include_usage": True}
            )  # fmt: skip
            if ch.choices
        )
        with c.chat.completions.create(model="m", messages=[], stream=True) as s:
            parts = [
                tc["function"].get("arguments") or ""
                for ch in s
                for tc in (ch.choices[0]["delta"].get("tool_calls") or [] if ch.choices else [])
            ]
        return [text, json.loads("".join(parts))]

    script = [
        chunks_for_text("Your refund is done."),
        chunks_for_tool_call(2, "get_order", {"order_id": "A17"}),
    ]
    src, rep = _roundtrip(_openai(script), body)
    assert src.output == ["Your refund is done.", {"order_id": "A17"}]
    assert rep.steps[0].attributes["crashlab.stream"] is True
    assert rep.steps[0].response == src.steps[0].response
    assert rep.usage.total_tokens == src.usage.total_tokens > 0


def test_openai_stream_error_midway_replays_partial_then_raises():
    def body(c):
        got = []
        try:
            for ch in c.chat.completions.create(model="m", messages=[], stream=True):
                got.append(ch.choices[0]["delta"].get("content") or "" if ch.choices else "")
        except ConnectionError as exc:
            return ["".join(got), str(exc)]
        return ["".join(got), None]

    src, rep = _roundtrip(_openai([FakeStream(chunks_for_text("abcdefgh"), fail_after=2)]), body)
    assert src.output[1] == "stream dropped" and src.output[0]
    assert rep.steps[0].error.type == "ConnectionError"


def test_openai_async_plain_and_stream():
    async def body(c):
        r = await c.chat.completions.create(model="m", messages=[])
        stream = await c.chat.completions.create(model="m", messages=[], stream=True)
        text = "".join([ch.choices[0]["delta"].get("content") or "" async for ch in stream
                        if ch.choices])  # fmt: skip
        return [r.choices[0].message.content, text]

    script = [text_response(1, "hello async"), chunks_for_text("streamed async")]
    src, _ = _roundtrip(_openai(script, AsyncFakeOpenAI), body, is_async=True)
    assert src.output == ["hello async", "streamed async"]


# --- OpenAI Responses ----------------------------------------------------------------


def test_openai_responses_plain_and_stream():
    final = responses_text(2, "streamed via responses")
    events = [
        {"type": "response.created", "response": {**final, "status": "in_progress", "output": []}},
        {"type": "response.completed", "response": final},
    ]

    def make(empty):
        fake = FakeOpenAI([] if empty else [responses_tool_call(1, "get_order", {"a": 1}), events])
        inner = fake.responses.create
        fake.responses.create = lambda **kw: (
            FakeStream(inner(**kw)) if kw.get("stream") else inner(**kw)
        )
        return crashlab.wrap(fake)

    def body(c):
        r = c.responses.create(model="m", input="hi")
        evs = list(c.responses.create(model="m", input="again", stream=True))
        # synthetic streams keep the final response, not the live event boundaries
        return [r["output"][0]["name"], evs[-1]["type"], evs[-1]["response"]["id"]]

    src, rep = _roundtrip(make, body)
    assert src.output[0] == "get_order"
    assert src.output[1] == "response.completed"
    assert rep.steps[1].response["messages"][0]["parts"][0]["content"] == "streamed via responses"


# --- Anthropic -----------------------------------------------------------------------


def test_anthropic_plain_stream_and_helper():
    script = [a_tool_use(1, "get_order", {"order_id": "A17"}), a_text(2, "streamed"),
              a_text(3, "helper text")]  # fmt: skip

    def body(c):
        r = c.messages.create(model="claude-x", max_tokens=10, messages=[])
        evs = list(c.messages.create(model="claude-x", max_tokens=10, messages=[], stream=True))
        text = "".join(e["delta"]["text"] for e in evs if e["type"] == "content_block_delta")
        with c.messages.stream(model="claude-x", max_tokens=10, messages=[]) as s:
            final = s.get_final_message()
        return [r.content[-1].input, text, final.content[-1].text]

    src, rep = _roundtrip(lambda e: crashlab.wrap(FakeAnthropic([] if e else script)), body)
    assert src.output == [{"order_id": "A17"}, "streamed", "helper text"]
    assert rep.steps[0].response["finish_reason"] == "tool_calls"


def test_anthropic_async_plain():
    async def body(c):
        r = await c.messages.create(model="claude-x", max_tokens=10, messages=[])
        return r.content[0].text

    make = lambda e: crashlab.wrap(AsyncFakeAnthropic([] if e else [a_text(1, "hi async")]))  # noqa: E731
    src, _ = _roundtrip(make, body, is_async=True)
    assert src.output == "hi async"


# --- LiteLLM -------------------------------------------------------------------------


def test_litellm_completion_stream_and_async():
    mod = _fake_litellm()
    cl_litellm.patch(mod)

    async def body(_):
        r = mod.completion("groq/llama", [{"role": "user", "content": "x"}], tools=[{"t": 1}])
        text = "".join(ch.choices[0]["delta"].get("content") or ""
                       for ch in mod.completion(model="groq/llama", messages=[], stream=True)
                       if ch.choices)  # fmt: skip
        a = await mod.acompletion(model="groq/llama", messages=[])
        return [r.choices[0].message.tool_calls[0].function.name, text,
                a.choices[0].message.content]  # fmt: skip

    calls_before = len(mod.calls)
    src, rep = _roundtrip(lambda e: None, body, is_async=True)
    assert src.output == ["get_order", "lite stream", "lite async"]
    assert len(mod.calls) == calls_before + 2  # only the recording reached the fake module
    assert rep.steps[0].attributes["gen_ai.provider.name"] == "groq"


# --- rebuilding / synthesis units ----------------------------------------------------


def test_rebuild_falls_back_to_attribute_object():
    raw = tool_call_response(1, "get_order", {"order_id": "A17"}).model_dump()
    obj = R.rebuild(raw, "no.such.module.Type")
    assert isinstance(obj, R.ReplayObject)
    tc = obj.choices[0].message.tool_calls[0]
    assert tc.function.arguments == '{"order_id": "A17"}' and tc["id"] == "call_1"
    assert obj.choices[0].message.refusal is None  # unset optional field
    assert tc.model_dump() == raw["choices"][0]["message"]["tool_calls"][0]
    assert R.rebuild(raw, None) == raw


def test_rebuild_exception_keeps_type_name_message_and_status():
    from crashlab.models import ErrorInfo

    exc = R.rebuild_exception(ErrorInfo(type="KeyError", message="\"order 'Z' not found\""))
    assert isinstance(exc, KeyError) and type(exc).__name__ == "KeyError"
    assert str(exc) == "\"order 'Z' not found\""
    exc = R.rebuild_exception(
        ErrorInfo(type="tests.fakes.FakeAPIError", message="boom", status_code=429)
    )
    from tests.fakes import FakeAPIError

    assert isinstance(exc, FakeAPIError) and exc.status_code == 429
    exc = R.rebuild_exception(ErrorInfo(type="gone.module.WeirdError", message="w"))
    assert isinstance(exc, R.ReplayedException) and type(exc).__name__ == "WeirdError"


@pytest.mark.parametrize(
    "resp",
    [
        tool_call_response(1, "get_order", {"order_id": "A17"}).model_dump(),
        text_response(2, "hello").model_dump(),
        {
            "id": "r",
            "object": "chat.completion",
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "x",
                        "reasoning": "think",
                        "tool_calls": [
                            {
                                "id": "c",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                                "extra_content": {"g": 1},
                            }
                        ],
                    },
                }
            ],
        },
    ],  # fmt: skip
)
def test_openai_synthetic_chunks_reassemble_to_the_same_response(resp):
    asm = OpenAIChatAssembler()
    for chunk in synth.openai_chat_chunks(resp):
        asm.add(chunk)
    assert (
        N.normalize_openai_chat_response(asm.result())[0]
        == (N.normalize_openai_chat_response(resp)[0])
    )


def test_anthropic_synthetic_events_reassemble_to_the_same_message():
    msg = a_tool_use(1, "get_order", {"order_id": "A17"}).model_dump()
    msg["content"].insert(0, {"type": "thinking", "thinking": "hmm", "signature": "sig"})
    asm = AnthropicAssembler()
    for ev in synth.anthropic_events(msg):
        asm.add(ev)
    assert N.normalize_anthropic_response(asm.result())[0] == N.normalize_anthropic_response(msg)[0]
