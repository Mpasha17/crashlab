"""A replayed Anthropic stream is event-for-event the recorded one: thinking, signature,
text and tool-input deltas as streamed, and the running usage message_start /
message_delta carried (a generator agent yielding raw events reproduces)."""

from __future__ import annotations

import pytest

import crashlab

from ._transport import httpx_module, sse

anthropic = pytest.importorskip("anthropic")
httpx = httpx_module()

U = {
    "input_tokens": 5,
    "output_tokens": 1,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
}
EVENTS = [
    {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "c",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": U,
        },
    },
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "thinking", "thinking": "", "signature": ""},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "thinking_delta", "thinking": "Hmm, "},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "thinking_delta", "thinking": "weather."},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "signature_delta", "signature": "sig=="},
    },
    {"type": "content_block_stop", "index": 0},
    {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Let"}},
    {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "text_delta", "text": " me check."},
    },
    {"type": "content_block_stop", "index": 1},
    {
        "type": "content_block_start",
        "index": 2,
        "content_block": {"type": "tool_use", "id": "tu_1", "name": "w", "input": {}},
    },
    {
        "type": "content_block_delta",
        "index": 2,
        "delta": {"type": "input_json_delta", "partial_json": ""},
    },
    {
        "type": "content_block_delta",
        "index": 2,
        "delta": {"type": "input_json_delta", "partial_json": '{"c": '},
    },
    {
        "type": "content_block_delta",
        "index": 2,
        "delta": {"type": "input_json_delta", "partial_json": '"P"}'},
    },
    {"type": "content_block_stop", "index": 2},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use", "stop_sequence": None},
        "usage": {"output_tokens": 30},
    },
    {"type": "message_stop"},
]
STATE = {"live": True}


def _handle(request):
    assert STATE["live"], "replay hit the network"
    body = sse(EVENTS, named=True)
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)


def _client():
    http = httpx.Client(transport=httpx.MockTransport(_handle))
    return crashlab.wrap(anthropic.Anthropic(api_key="sk-x", base_url="http://t", http_client=http))


@crashlab.agent(name="anth-raw-events")
def raw_events():
    stream = _client().messages.create(
        model="c", max_tokens=100, messages=[{"role": "user", "content": "hi"}], stream=True
    )
    for ev in stream:
        yield ev.model_dump()


@crashlab.agent(name="anth-helper-usage")
def helper_usage():
    with _client().messages.stream(
        model="c", max_tokens=100, messages=[{"role": "user", "content": "hi"}]
    ) as stream:
        for ev in stream:
            if ev.type in ("message_start", "message_delta"):
                usage = ev.message.usage if ev.type == "message_start" else ev.usage
                yield [ev.type, usage.output_tokens, usage.input_tokens]


@pytest.mark.parametrize("agent", [raw_events, helper_usage], ids=["create", "stream-helper"])
def test_replayed_anthropic_stream_matches_event_for_event(crashlab_dir, agent):
    live = list(agent())
    assert len(live) >= 2
    STATE["live"] = False
    try:
        replayed = crashlab.replay(crashlab.load("last"))
    finally:
        STATE["live"] = True
    assert replayed.status == "ok"
    assert replayed.output == live
