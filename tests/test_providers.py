"""Anthropic, streaming (OpenAI + Anthropic), Responses API — with fake clients."""

from __future__ import annotations

import gc

import pytest
from examples.refund_bot import agent as refund

import crashlab
from crashlab.capture.normalize import request_hash
from tests.fakes import (
    AsyncFakeAnthropic,
    AsyncFakeOpenAI,
    FakeAnthropic,
    FakeAPIError,
    FakeOpenAI,
    FakeStream,
    a_events_for,
    a_text,
    a_tool_use,
    anthropic_refund_script,
    chunks_for_text,
    chunks_for_tool_call,
    refund_script,
    responses_text,
    responses_tool_call,
)


def _strip_ids(step):
    return step.request["messages"], step.request["system"], step.request["tools"]


# --- Anthropic -----------------------------------------------------------------------


def test_anthropic_refund_bot_run():
    fake = FakeAnthropic(anthropic_refund_script())
    answer = refund.run_agent("Refund my order A17", client=crashlab.wrap(fake), model="claude-x")
    assert answer.startswith("Your refund")
    run = crashlab.load("last")
    assert run.input["provider"] is None
    assert [(s.operation, s.name) for s in run.steps] == [
        ("chat", "claude-x"),
        ("execute_tool", "get_order"),
        ("chat", "claude-x"),
        ("execute_tool", "issue_refund"),
        ("chat", "claude-x"),
    ]
    chat0, tool1, chat2, _, chat4 = run.steps
    assert chat0.attributes["gen_ai.provider.name"] == "anthropic"
    assert chat0.attributes["crashlab.api"] == "anthropic.messages"
    assert chat0.attributes["server.address"] == "api.anthropic.fake"
    assert chat0.attributes["gen_ai.request.max_tokens"] == 1024
    assert chat0.request["system"] == refund.SYSTEM_PROMPT
    assert chat0.request["tools"][0]["name"] == "get_order"
    assert chat0.response["finish_reason"] == "tool_calls"
    assert chat0.response["provider_finish_reason"] == "tool_use"
    assert chat0.response["tool_calls"] == [
        {"id": "call_1", "type": "function", "name": "get_order", "arguments": {"order_id": "A17"}}
    ]
    assert chat0.response["messages"][0]["parts"][0] == {"type": "text", "content": "Let me check."}
    # input tokens include cache reads (101 + 5)
    assert chat0.attributes["gen_ai.usage.input_tokens"] == 106
    assert chat0.attributes["gen_ai.usage.cache_read.input_tokens"] == 5
    assert tool1.attributes["gen_ai.tool.call.id"] == "call_1"
    assert chat4.response["finish_reason"] == "stop"
    # tool_result blocks are normalized into tool messages
    last = chat2.request["messages"][-1]
    assert last["role"] == "tool" and last["parts"][0]["id"] == "call_1"
    assert run.usage.llm_calls == 3 and run.usage.tool_calls == 2


def test_same_agent_same_shape_on_openai_and_anthropic():
    refund.run_agent("Refund my order A17", client=crashlab.wrap(FakeOpenAI(refund_script())))
    o_run = crashlab.load("last")
    refund.reset_db()
    refund.run_agent(
        "Refund my order A17", client=crashlab.wrap(FakeAnthropic(anthropic_refund_script()))
    )
    a_run = crashlab.load("last")
    o_chats, a_chats = o_run.chat_steps(), a_run.chat_steps()
    # first request is identical in the common shape (system, messages, tools)
    assert _strip_ids(o_chats[0]) == _strip_ids(a_chats[0])
    # tool calls and the tool results the model saw are identical too
    for o, a in zip(o_chats, a_chats, strict=True):
        assert o.response["tool_calls"] == a.response["tool_calls"]
        assert o.response["finish_reason"] == a.response["finish_reason"]
    o_tool_msgs = [m for m in o_chats[2].request["messages"] if m["role"] == "tool"]
    a_tool_msgs = [m for m in a_chats[2].request["messages"] if m["role"] == "tool"]
    assert o_tool_msgs == a_tool_msgs
    assert [s.result for s in o_run.tool_steps()] == [s.result for s in a_run.tool_steps()]


def test_anthropic_error_recorded():
    with pytest.raises(FakeAPIError):
        refund.run_agent(
            "x", client=crashlab.wrap(FakeAnthropic([FakeAPIError("overloaded", 529)])), model="c"
        )
    run = crashlab.load("last")
    assert run.steps[0].error.status_code == 529 and run.status == "error"


async def test_async_anthropic_create():
    client = crashlab.wrap(AsyncFakeAnthropic([a_text(1, "hello there")]))
    async with crashlab.record("a-async"):
        msg = await client.messages.create(
            model="c", max_tokens=10, messages=[{"role": "user", "content": "hi"}]
        )
    assert msg.content[0].text == "hello there"
    step = crashlab.load("last").steps[0]
    assert step.response["messages"][0]["parts"] == [{"type": "text", "content": "hello there"}]
    assert step.response["usage"]["output_tokens"] == 20


# --- Anthropic streaming --------------------------------------------------------------


def test_anthropic_create_stream_sync():
    msg = a_tool_use(1, "get_order", {"order_id": "A17"})
    client = crashlab.wrap(FakeAnthropic([msg]))
    with crashlab.record("a-stream"):
        stream = client.messages.create(model="c", max_tokens=10, messages=[], stream=True)
        events = list(stream)
    assert events == a_events_for(msg)  # passed through unchanged
    step = crashlab.load("last").steps[0]
    assert step.attributes["crashlab.stream"] is True
    assert step.attributes["crashlab.stream.complete"] is True
    assert step.attributes["crashlab.stream.chunks"] == len(events)
    assert step.response["tool_calls"] == [
        {"id": "call_1", "type": "function", "name": "get_order", "arguments": {"order_id": "A17"}}
    ]
    assert step.response["messages"][0]["parts"][0] == {"type": "text", "content": "Let me check."}
    assert step.response["finish_reason"] == "tool_calls"
    assert step.response["usage"]["output_tokens"] == 10
    assert "stream" not in step.request["params"]
    # streamed and non-streamed responses normalize identically
    client2 = crashlab.wrap(FakeAnthropic([msg]))
    with crashlab.record("a-plain"):
        client2.messages.create(model="c", max_tokens=10, messages=[])
    plain = crashlab.load("last").steps[0]
    assert plain.response["messages"] == step.response["messages"]
    assert plain.request_hash == step.request_hash


async def test_anthropic_create_stream_async():
    msg = a_text(1, "streamed hello")
    client = crashlab.wrap(AsyncFakeAnthropic([msg]))
    async with crashlab.record("a-astream"):
        stream = await client.messages.create(model="c", max_tokens=10, messages=[], stream=True)
        n = 0
        async for _ in stream:
            n += 1
    step = crashlab.load("last").steps[0]
    assert step.attributes["crashlab.stream.chunks"] == n
    assert step.response["messages"][0]["parts"][0]["content"] == "streamed hello"


def test_anthropic_messages_stream_helper():
    msg = a_text(1, "hello from the stream helper")
    fake = FakeAnthropic([msg])
    client = crashlab.wrap(fake)
    with crashlab.record("a-helper"):
        with client.messages.stream(model="c", max_tokens=10, messages=[]) as stream:
            text = "".join(stream.text_stream)
    assert text == "hello from the stream helper"
    step = crashlab.load("last").steps[0]
    assert step.attributes["crashlab.stream.complete"] is True
    assert step.response["messages"][0]["parts"][0]["content"] == text
    assert step.response["finish_reason"] == "stop"


def test_anthropic_messages_stream_helper_partial():
    client = crashlab.wrap(FakeAnthropic([a_text(1, "a long answer that we abandon")]))
    with crashlab.record("a-partial"):
        with client.messages.stream(model="c", max_tokens=10, messages=[]) as stream:
            next(iter(stream))
    step = crashlab.load("last").steps[0]
    assert step.attributes["crashlab.stream.complete"] is False
    assert step.error is None


async def test_anthropic_messages_stream_helper_async():
    client = crashlab.wrap(AsyncFakeAnthropic([a_text(1, "async helper")]))
    async with crashlab.record("a-ahelper"):
        async with client.messages.stream(model="c", max_tokens=10, messages=[]) as stream:
            final = await stream.get_final_message()
    assert final.content[0].text == "async helper"
    step = crashlab.load("last").steps[0]
    assert step.response["messages"][0]["parts"][0]["content"] == "async helper"


# --- OpenAI streaming -----------------------------------------------------------------


def test_openai_stream_text_sync():
    chunks = chunks_for_text("Your refund has been issued.")
    client = crashlab.wrap(FakeOpenAI([chunks]))
    with crashlab.record("o-stream"):
        stream = client.chat.completions.create(model="m", messages=[], stream=True,
                                                stream_options={"include_usage": True})  # fmt: skip
        got = [c for c in stream]
    assert got == chunks  # same objects, unchanged
    step = crashlab.load("last").steps[0]
    assert step.response["messages"][0]["parts"] == [
        {"type": "text", "content": "Your refund has been issued."}
    ]
    assert step.response["finish_reason"] == "stop"
    assert step.attributes["gen_ai.usage.input_tokens"] == 30
    assert step.attributes["crashlab.stream.complete"] is True
    assert step.raw_response["object"] == "chat.completion"
    assert "stream_options" not in step.request["params"]
    assert crashlab.load("last").usage.total_tokens == 37


def test_openai_stream_tool_call_and_context_manager():
    client = crashlab.wrap(FakeOpenAI([chunks_for_tool_call(7, "get_order", {"order_id": "A17"})]))
    with crashlab.record("o-stream-tool"):
        with client.chat.completions.create(model="m", messages=[], stream=True) as stream:
            for _ in stream:
                pass
    step = crashlab.load("last").steps[0]
    assert step.response["tool_calls"] == [
        {"id": "call_7", "type": "function", "name": "get_order", "arguments": {"order_id": "A17"}}
    ]
    assert step.response["finish_reason"] == "tool_calls"


def test_openai_stream_error_midway_records_partial():
    stream_src = FakeStream(chunks_for_text("partial answer here"), fail_after=3)
    client = crashlab.wrap(FakeOpenAI([stream_src]))
    with pytest.raises(ConnectionError), crashlab.record("o-stream-err"):
        for _ in client.chat.completions.create(model="m", messages=[], stream=True):
            pass
    run = crashlab.load("last")
    step = run.steps[0]
    assert step.error.type == "ConnectionError"
    assert step.response["messages"][0]["parts"][0]["content"] == "partial "
    assert step.attributes["crashlab.stream.complete"] is False
    assert run.status == "error"


def test_openai_stream_closed_early_and_abandoned():
    client = crashlab.wrap(FakeOpenAI([chunks_for_text("abcdefghijkl"), chunks_for_text("zz")]))
    with crashlab.record("o-stream-close"):
        s = client.chat.completions.create(model="m", messages=[], stream=True)
        next(s)
        next(s)
        s.close()
        s2 = client.chat.completions.create(model="m", messages=[], stream=True)
        del s2  # never iterated, garbage-collected
        gc.collect()
    steps = crashlab.load("last").steps
    assert [s.attributes["crashlab.stream.complete"] for s in steps] == [False, False]
    assert steps[0].response["messages"][0]["parts"][0]["content"] == "abcd"
    assert steps[1].attributes["crashlab.stream.chunks"] == 0


async def test_openai_stream_async():
    client = crashlab.wrap(AsyncFakeOpenAI([chunks_for_text("async stream ok")]))
    async with crashlab.record("o-astream"):
        stream = await client.chat.completions.create(model="m", messages=[], stream=True)
        parts = [c.choices[0]["delta"].get("content") async for c in stream if c.choices]
    assert "".join(p or "" for p in parts) == "async stream ok"
    step = crashlab.load("last").steps[0]
    assert step.response["messages"][0]["parts"][0]["content"] == "async stream ok"
    assert step.attributes["crashlab.stream.complete"] is True


def test_stream_hash_equals_non_stream_hash():
    client = crashlab.wrap(FakeOpenAI([chunks_for_text("x"), refund_script()[2]]))
    msgs = [{"role": "user", "content": "hi"}]
    with crashlab.record("o-hash"):
        list(client.chat.completions.create(model="m", messages=msgs, stream=True))
        client.chat.completions.create(model="m", messages=msgs)
    s1, s2 = crashlab.load("last").steps
    assert s1.request_hash == s2.request_hash == request_hash(s1.request)


# --- OpenAI responses API -------------------------------------------------------------


def test_openai_responses_create():
    client = crashlab.wrap(FakeOpenAI([responses_tool_call(1, "get_order", {"order_id": "A17"}),
                                       responses_text(2, "Done.")]))  # fmt: skip
    with crashlab.record("o-responses"):
        client.responses.create(model="m", instructions="Be brief.", input="Refund A17")
        client.responses.create(
            model="m",
            input=[
                {"role": "user", "content": "Refund A17"},
                {"type": "function_call", "call_id": "call_1", "name": "get_order",
                 "arguments": '{"order_id": "A17"}'},
                {"type": "function_call_output", "call_id": "call_1", "output": "{}"},
            ],
        )  # fmt: skip
    s1, s2 = crashlab.load("last").steps
    assert s1.attributes["crashlab.api"] == "openai.responses"
    assert s1.request["system"] == "Be brief."
    user_text = {"type": "text", "content": "Refund A17"}
    assert s1.request["messages"] == [{"role": "user", "parts": [user_text]}]
    assert s1.response["tool_calls"][0]["name"] == "get_order"
    assert s1.response["finish_reason"] == "tool_calls"
    assert s1.attributes["gen_ai.usage.input_tokens"] == 40
    assert [m["role"] for m in s2.request["messages"]] == ["user", "assistant", "tool"]
    assert s2.response["messages"][0]["parts"] == [{"type": "text", "content": "Done."}]


def test_openai_responses_stream():
    final = responses_text(1, "streamed via responses")
    events = [
        {"type": "response.created", "response": {**final, "status": "in_progress", "output": []}},
        {"type": "response.output_text.delta", "delta": "streamed "},
        {"type": "response.completed", "response": final},
    ]
    client = crashlab.wrap(FakeOpenAI([events]))
    fake_create = client.unwrapped.responses.create

    def create(**kw):  # the fake responses.create returns the raw list; make it a stream
        return FakeStream(fake_create(**kw))

    client.unwrapped.responses.create = create
    client = crashlab.wrap(client.unwrapped)
    with crashlab.record("o-responses-stream"):
        assert len(list(client.responses.create(model="m", input="hi", stream=True))) == 3
    step = crashlab.load("last").steps[0]
    assert step.response["messages"][0]["parts"][0]["content"] == "streamed via responses"
    assert step.attributes["crashlab.stream.complete"] is True
