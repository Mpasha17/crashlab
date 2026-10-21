"""Real anthropic / openai SDK clients over a mocked HTTP transport (no network)."""

from __future__ import annotations

import json

import pytest
from examples.refund_bot import agent as refund

import crashlab
from tests._transport import Script, httpx_module, sse
from tests.fakes import a_events_for, a_text, a_tool_use, chunks_for_text, chunks_for_tool_call

ANTHROPIC_KEY = "sk-ant-api03-SECRETSECRETSECRETSECRET0123456789"
OPENAI_KEY = "sk-proj-SECRETSECRETSECRETSECRET0123456789"


def _no_secrets(store_dir):
    text = "".join(p.read_text() for p in store_dir.rglob("*") if p.suffix in (".json", ".jsonl"))
    assert "SECRETSECRET" not in text
    import sqlite3

    dump = "\n".join(sqlite3.connect(store_dir / "index.sqlite").iterdump())
    assert "SECRETSECRET" not in dump


# --- anthropic -----------------------------------------------------------------------


@pytest.fixture
def anthropic_mod():
    return pytest.importorskip("anthropic")


def _anthropic_client(mod, script: Script, *, is_async=False):
    httpx = httpx_module()
    transport = httpx.MockTransport(script.handler(httpx))
    if is_async:
        return mod.AsyncAnthropic(
            api_key=ANTHROPIC_KEY,
            base_url="http://anthropic.local",
            http_client=httpx.AsyncClient(transport=transport),
            max_retries=0,
        )
    return mod.Anthropic(
        api_key=ANTHROPIC_KEY,
        base_url="http://anthropic.local",
        http_client=httpx.Client(transport=transport),
        max_retries=0,
    )


def test_anthropic_sdk_refund_bot(anthropic_mod, crashlab_dir):
    script = Script([
        a_tool_use(1, "get_order", {"order_id": "A17"}).model_dump(),
        a_tool_use(2, "issue_refund", {"order_id": "A17", "amount": 49.0}).model_dump(),
        a_text(3, "Refunded $49.00.").model_dump(),
    ])  # fmt: skip
    client = crashlab.wrap(_anthropic_client(anthropic_mod, script))
    answer = refund.run_agent("Refund A17", client=client, model="claude-test")
    assert answer == "Refunded $49.00."
    headers = script.requests[0].headers  # anthropic<2: x-api-key; >=2: Authorization: Bearer
    sent = headers.get("x-api-key") or headers.get("authorization", "").removeprefix("Bearer ")
    assert sent == ANTHROPIC_KEY
    run = crashlab.load("last")
    assert [s.operation for s in run.steps] == ["chat", "execute_tool"] * 2 + ["chat"]
    chat0 = run.steps[0]
    assert chat0.attributes["gen_ai.provider.name"] == "anthropic"
    assert chat0.attributes["server.address"] == "anthropic.local"
    assert chat0.response["tool_calls"][0]["arguments"] == {"order_id": "A17"}
    assert run.env.get("anthropic") == anthropic_mod.__version__
    _no_secrets(crashlab_dir)


def test_anthropic_sdk_stream_create_and_helper(anthropic_mod, crashlab_dir):
    msg = a_tool_use(1, "get_order", {"order_id": "A17"})
    body = (sse(a_events_for(msg), named=True), "text/event-stream")
    text_body = (sse(a_events_for(a_text(2, "Hello, streamed!")), named=True), "text/event-stream")
    script = Script([body, text_body])
    client = crashlab.wrap(_anthropic_client(anthropic_mod, script))
    with crashlab.record("sdk-a-stream"):
        events = list(client.messages.create(model="c", max_tokens=20, messages=[], stream=True))
        with client.messages.stream(model="c", max_tokens=20, messages=[]) as stream:
            text = "".join(stream.text_stream)
            final = stream.get_final_message()
    assert [e.type for e in events][0] == "message_start"
    assert text == "Hello, streamed!" and final.content[0].text == text
    s1, s2 = crashlab.load("last").steps
    assert s1.response["tool_calls"][0] == {
        "id": "call_1", "type": "function", "name": "get_order", "arguments": {"order_id": "A17"}
    }  # fmt: skip
    assert s1.attributes["crashlab.stream.complete"] is True
    assert s2.response["messages"][0]["parts"] == [{"type": "text", "content": text}]
    assert s2.response["finish_reason"] == "stop"
    assert s2.attributes["crashlab.stream.complete"] is True
    assert s2.attributes["gen_ai.usage.output_tokens"] == 20
    _no_secrets(crashlab_dir)


async def test_anthropic_sdk_async(anthropic_mod, crashlab_dir):
    body = (sse(a_events_for(a_text(2, "async streamed")), named=True), "text/event-stream")
    script = Script([a_text(1, "async hi").model_dump(), body, 500])
    client = crashlab.wrap(_anthropic_client(anthropic_mod, script, is_async=True))
    with pytest.raises(anthropic_mod.InternalServerError):
        async with crashlab.record("sdk-a-async"):
            msg = await client.messages.create(model="c", max_tokens=5, messages=[])
            assert msg.content[0].text == "async hi"
            async with client.messages.stream(model="c", max_tokens=5, messages=[]) as stream:
                async for _ in stream:
                    pass
            await client.messages.create(model="c", max_tokens=5, messages=[])
    s1, s2, s3 = crashlab.load("last").steps
    assert s1.response["messages"][0]["parts"][0]["content"] == "async hi"
    assert s2.response["messages"][0]["parts"][0]["content"] == "async streamed"
    assert s3.error.status_code == 500
    _no_secrets(crashlab_dir)


# --- openai --------------------------------------------------------------------------


@pytest.fixture
def openai_mod():
    return pytest.importorskip("openai")


def _openai_client(mod, script: Script, *, is_async=False):
    httpx = httpx_module()
    transport = httpx.MockTransport(script.handler(httpx))
    kwargs = dict(api_key=OPENAI_KEY, base_url="http://localhost:11434/v1", max_retries=0)
    if is_async:
        return mod.AsyncOpenAI(http_client=httpx.AsyncClient(transport=transport), **kwargs)
    return mod.OpenAI(http_client=httpx.Client(transport=transport), **kwargs)


def _chunks(items):
    return (sse([c.model_dump() for c in items], named=False, done=True), "text/event-stream")


def test_openai_sdk_streaming(openai_mod, crashlab_dir):
    script = Script([_chunks(chunks_for_tool_call(1, "get_order", {"order_id": "A17"})),
                     _chunks(chunks_for_text("All done, refund issued."))])  # fmt: skip
    client = crashlab.wrap(_openai_client(openai_mod, script))
    with crashlab.record("sdk-o-stream"):
        s = client.chat.completions.create(model="m", messages=[], stream=True)
        chunks = list(s)
        with client.chat.completions.create(
            model="m", messages=[], stream=True, stream_options={"include_usage": True}
        ) as s2:
            text = "".join(c.choices[0].delta.content or "" for c in s2 if c.choices)
    assert type(chunks[0]).__name__ == "ChatCompletionChunk"
    assert text == "All done, refund issued."
    a, b = crashlab.load("last").steps
    assert a.response["tool_calls"][0]["arguments"] == {"order_id": "A17"}
    assert a.response["finish_reason"] == "tool_calls"
    assert b.response["messages"][0]["parts"][0]["content"] == text
    assert b.attributes["gen_ai.usage.output_tokens"] == 7
    assert json.loads(script.requests[1].content)["stream_options"] == {"include_usage": True}
    _no_secrets(crashlab_dir)


async def test_openai_sdk_streaming_async(openai_mod, crashlab_dir):
    script = Script([_chunks(chunks_for_text("async chunks"))])
    client = crashlab.wrap(_openai_client(openai_mod, script, is_async=True))
    async with crashlab.record("sdk-o-astream"):
        stream = await client.chat.completions.create(model="m", messages=[], stream=True)
        got = [c async for c in stream]
    assert len(got) == len(chunks_for_text("async chunks"))
    step = crashlab.load("last").steps[0]
    assert step.response["messages"][0]["parts"][0]["content"] == "async chunks"


def test_openai_sdk_responses(openai_mod, crashlab_dir):
    from tests.fakes import responses_text, responses_tool_call

    final = responses_text(2, "streamed response text")
    events = [
        {"type": "response.created", "sequence_number": 0,
         "response": {**final, "status": "in_progress", "output": []}},
        {"type": "response.output_text.delta", "sequence_number": 1, "item_id": "msg_2",
         "output_index": 0, "content_index": 0, "delta": "streamed response text"},
        {"type": "response.completed", "sequence_number": 2, "response": final},
    ]  # fmt: skip
    script = Script([responses_tool_call(1, "get_order", {"order_id": "A17"}),
                     (sse(events, named=True), "text/event-stream")])  # fmt: skip
    client = crashlab.wrap(_openai_client(openai_mod, script))
    with crashlab.record("sdk-o-responses"):
        r = client.responses.create(model="m", input="Refund A17", instructions="Be brief.")
        assert r.output[0].name == "get_order"
        events_seen = list(client.responses.create(model="m", input="again", stream=True))
    assert events_seen[-1].type == "response.completed"
    s1, s2 = crashlab.load("last").steps
    assert s1.attributes["crashlab.api"] == "openai.responses"
    assert s1.request["system"] == "Be brief."
    assert s1.response["tool_calls"][0]["id"] == "call_1"
    assert s2.response["messages"][0]["parts"][0]["content"] == "streamed response text"
    assert s2.attributes["crashlab.stream.complete"] is True
    _no_secrets(crashlab_dir)
