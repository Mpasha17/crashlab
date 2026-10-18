"""Replay with the real openai / anthropic SDKs: recorded over a mock HTTP transport,
replayed with a *real* client that has no API key and no transport (the network guard
would trip on any connection). Skipped when the SDK is not installed."""

from __future__ import annotations

import asyncio

import pytest
from examples.refund_bot import agent as refund

import crashlab
from tests._transport import Script
from tests.fakes import a_events_for, a_text, a_tool_use, chunks_for_text, chunks_for_tool_call
from tests.test_sdk_transport import _anthropic_client, _chunks, _openai_client


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch):
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    refund.reset_db()
    yield
    refund.reset_db()


def _replay_main(fn_record, fn_replay):
    fn_record()
    src = crashlab.load("last")
    rep = crashlab.replay(src, fn=fn_replay)
    assert rep.status == "ok" and rep.output == src.output
    assert [s.request_hash for s in rep.steps] == [s.request_hash for s in src.steps]
    return src, rep


def test_openai_sdk_refund_bot_replays_without_key(crashlab_dir):
    openai = pytest.importorskip("openai")
    from tests.fakes import refund_script

    script = Script([r.model_dump() for r in refund_script()])
    client = crashlab.wrap(_openai_client(openai, script))
    refund.run_agent("Refund A17", client=client, model="gpt-test", provider="openai", faults=[])
    src = crashlab.load("last")
    n_requests = len(script.requests)
    refund.reset_db()
    # no client override: refund_bot builds a real OpenAI() client (placeholder key)
    rep = crashlab.replay(src)
    assert rep.status == "ok" and rep.outcome == "pass" and rep.output == src.output
    assert len(script.requests) == n_requests and refund.REFUNDS == []
    assert rep.steps[0].attributes["crashlab.response.type"].startswith("openai.types.chat")


def test_openai_sdk_streams_sync_and_async(crashlab_dir):
    openai = pytest.importorskip("openai")

    def run(script_or_none, *, is_async=False):
        def make():
            if script_or_none is None:
                return crashlab.wrap(openai.AsyncOpenAI() if is_async else openai.OpenAI())
            return crashlab.wrap(_openai_client(openai, script_or_none, is_async=is_async))

        if is_async:

            async def go():
                client = make()
                async with crashlab.record("sdk-replay-astream") as r:
                    s = await client.chat.completions.create(model="m", messages=[], stream=True)
                    r.output = "".join([c.choices[0].delta.content or "" async for c in s
                                        if c.choices])  # fmt: skip

            asyncio.run(go())
            return
        client = make()
        with crashlab.record("sdk-replay-stream") as r:
            chunks = list(client.chat.completions.create(model="m", messages=[], stream=True))
            calls = [tc for c in chunks if c.choices for tc in c.choices[0].delta.tool_calls or []]
            opts = {"include_usage": True}
            with client.chat.completions.create(
                model="m", messages=[], stream=True, stream_options=opts
            ) as s2:
                text = "".join(c.choices[0].delta.content or "" for c in s2 if c.choices)
            r.output = [type(chunks[0]).__name__, calls[0].function.name,
                        "".join(tc.function.arguments or "" for tc in calls), text]  # fmt: skip

    script = Script([_chunks(chunks_for_tool_call(1, "get_order", {"order_id": "A17"})),
                     _chunks(chunks_for_text("All done."))])  # fmt: skip
    src, _ = _replay_main(lambda: run(script), lambda: run(None))
    assert src.output == ["ChatCompletionChunk", "get_order", '{"order_id": "A17"}', "All done."]
    ascript = Script([_chunks(chunks_for_text("async chunks"))])
    src, _ = _replay_main(lambda: run(ascript, is_async=True), lambda: run(None, is_async=True))
    assert src.output == "async chunks"


def test_anthropic_sdk_create_stream_and_helper(crashlab_dir):
    anthropic = pytest.importorskip("anthropic")
    from tests._transport import sse

    def events(msg):
        return (sse(a_events_for(msg), named=True), "text/event-stream")

    def run(script):
        client = crashlab.wrap(_anthropic_client(anthropic, script) if script else
                               anthropic.Anthropic())  # fmt: skip
        with crashlab.record("sdk-replay-anthropic") as r:
            m = client.messages.create(model="claude-t", max_tokens=50, messages=[])
            raw = list(client.messages.create(model="claude-t", max_tokens=50, messages=[],
                                              stream=True))  # fmt: skip
            with client.messages.stream(model="claude-t", max_tokens=50, messages=[]) as s:
                streamed = "".join(s.text_stream)
                final = s.get_final_message()
            r.output = [type(m).__name__, m.content[-1].input, raw[0].type,
                        streamed, final.content[0].text, final.stop_reason]  # fmt: skip

    script = Script([
        a_tool_use(1, "get_order", {"order_id": "A17"}).model_dump(),
        events(a_text(2, "raw stream")),
        events(a_text(3, "helper stream")),
    ])  # fmt: skip
    src, rep = _replay_main(lambda: run(script), lambda: run(None))
    assert src.output == ["Message", {"order_id": "A17"}, "message_start", "helper stream",
                          "helper stream", "end_turn"]  # fmt: skip
    assert rep.steps[2].response["messages"][0]["parts"][0]["content"] == "helper stream"
