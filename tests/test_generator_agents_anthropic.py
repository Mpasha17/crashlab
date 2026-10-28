"""Anthropic streams replay with their recorded delta boundaries, so a generator agent
yielding ``text_stream`` pieces (or raw ``text_delta`` events) reproduces exactly."""

from __future__ import annotations

import asyncio

import pytest

import crashlab

from ._transport import httpx_module, sse

anthropic = pytest.importorskip("anthropic")
httpx = httpx_module()

PIECES = ["Hel", "lo ", "wör", "ld 🌍"]
START = {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-x",
         "content": [], "stop_reason": None, "stop_sequence": None,
         "usage": {"input_tokens": 3, "output_tokens": 1}}  # fmt: skip
EVENTS = [
    {"type": "message_start", "message": START},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    *[{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": p}}
      for p in PIECES],
    {"type": "content_block_stop", "index": 0},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
     "usage": {"output_tokens": 4}},
    {"type": "message_stop"},
]  # fmt: skip
LIVE = {"on": True}


def _handle(request):
    assert LIVE["on"], "replay hit the network"
    body = sse(EVENTS, named=True)
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)


async def _ahandle(request):
    return _handle(request)


def _client(is_async=False):
    kw = {"api_key": "sk-ant-x", "base_url": "http://anthropic.local", "max_retries": 0}
    if is_async:
        http = httpx.AsyncClient(transport=httpx.MockTransport(_ahandle))
        return crashlab.wrap(anthropic.AsyncAnthropic(http_client=http, **kw))
    http = httpx.Client(transport=httpx.MockTransport(_handle))
    return crashlab.wrap(anthropic.Anthropic(http_client=http, **kw))


MSGS = [{"role": "user", "content": "hi"}]


@crashlab.agent(name="text-stream")
def text_stream_agent():
    with _client().messages.stream(model="claude-x", max_tokens=9, messages=MSGS) as s:
        yield from s.text_stream


@crashlab.agent(name="raw-events")
def raw_events_agent():
    for ev in _client().messages.create(model="claude-x", max_tokens=9, messages=MSGS, stream=True):
        if ev.type == "content_block_delta":
            yield ev.delta.text


@crashlab.agent(name="async-text-stream")
async def async_text_stream_agent():
    async with _client(True).messages.stream(model="claude-x", max_tokens=9, messages=MSGS) as s:
        async for text in s.text_stream:
            yield text


def _check(run):
    assert run.output == PIECES and [s.operation for s in run.steps] == ["chat"]
    assert run.steps[0].response  # the snapshot is still recorded
    LIVE["on"] = False
    try:
        assert crashlab.replay(run).output == PIECES
    finally:
        LIVE["on"] = True


@pytest.mark.parametrize("agent", [text_stream_agent, raw_events_agent])
def test_sync_anthropic_stream_generator_replays_piece_for_piece(crashlab_dir, agent):
    assert list(agent()) == PIECES
    _check(crashlab.load("last"))


def test_async_text_stream_generator_replays_piece_for_piece(crashlab_dir):
    async def consume():
        return [t async for t in async_text_stream_agent()]

    assert asyncio.run(consume()) == PIECES
    _check(crashlab.load("last"))
