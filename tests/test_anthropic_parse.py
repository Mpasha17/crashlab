"""Anthropic ``messages.parse`` (structured outputs) must record and replay offline with
``.parsed_output`` rebuilt (it used to bypass the wrap: 0 steps, no warning)."""

from __future__ import annotations

import asyncio
import json

import pytest

import crashlab

from ._transport import httpx_module

anthropic = pytest.importorskip("anthropic")
pydantic = pytest.importorskip("pydantic")
if not hasattr(anthropic.Anthropic(api_key="x").messages, "parse"):
    pytest.skip("anthropic SDK without messages.parse", allow_module_level=True)
httpx = httpx_module()


class Weather(pydantic.BaseModel):
    city: str
    temp: int


ANSWER = {"city": "Paris", "temp": 21}
MESSAGE = {
    "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-x",
    "content": [{"type": "text", "text": json.dumps(ANSWER)}],
    "stop_reason": "end_turn", "stop_sequence": None,
    "usage": {"input_tokens": 5, "output_tokens": 5},
}  # fmt: skip
CALLS: list[str] = []
LIVE = {"on": True}


def _handle(request):
    CALLS.append(request.url.path)
    if not LIVE["on"]:
        raise AssertionError("replay hit the network")
    return httpx.Response(200, json=MESSAGE)


def _client(is_async: bool = False):
    transport = httpx.MockTransport(_handle)
    if is_async:
        return crashlab.wrap(anthropic.AsyncAnthropic(
            api_key="sk-ant-x", base_url="http://anthropic.local", max_retries=0,
            http_client=httpx.AsyncClient(transport=transport)))  # fmt: skip
    return crashlab.wrap(anthropic.Anthropic(
        api_key="sk-ant-x", base_url="http://anthropic.local", max_retries=0,
        http_client=httpx.Client(transport=transport)))  # fmt: skip


@crashlab.agent(name="anthropic-parse")
def parse_agent(q: str) -> dict:
    out = _client().messages.parse(
        model="claude-x", max_tokens=64, messages=[{"role": "user", "content": q}],
        output_format=Weather,
    )  # fmt: skip
    assert isinstance(out.parsed_output, Weather)
    return out.parsed_output.model_dump()


@crashlab.agent(name="anthropic-parse-async")
async def parse_async_agent(q: str) -> dict:
    out = await _client(True).messages.parse(
        model="claude-x", max_tokens=64, messages=[{"role": "user", "content": q}],
        output_format=Weather,
    )  # fmt: skip
    return out.parsed_output.model_dump()


def _check(run):
    assert [s.operation for s in run.steps] == ["chat"]
    assert run.steps[0].request["params"]["output_format"]["class"] == "Weather"
    LIVE["on"] = False
    try:
        replayed = crashlab.replay(run)
    finally:
        LIVE["on"] = True
    assert replayed.status == "ok" and replayed.output == ANSWER


def test_messages_parse_records_and_replays(crashlab_dir):
    assert parse_agent("weather?") == ANSWER
    _check(crashlab.load("last"))


def test_async_messages_parse_records_and_replays(crashlab_dir):
    assert asyncio.run(parse_async_agent("weather?")) == ANSWER
    _check(crashlab.load("last"))


def _sse_events() -> bytes:
    from ._transport import sse

    text = json.dumps(ANSWER)
    events = [
        {"type": "message_start", "message": {**MESSAGE, "content": [], "stop_reason": None}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
         "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    ]  # fmt: skip
    return sse(events, named=True)


@crashlab.agent(name="anthropic-beta")
def beta_agent(q: str) -> list:
    def handle(request):
        CALLS.append(request.url.path)
        assert LIVE["on"], "replay hit the network"
        if json.loads(request.content).get("stream"):
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=_sse_events()
            )
        return httpx.Response(200, json=MESSAGE)

    client = crashlab.wrap(anthropic.Anthropic(
        api_key="sk-ant-x", base_url="http://anthropic.local", max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handle))))  # fmt: skip
    beta = client.beta.messages
    msgs = [{"role": "user", "content": q}]
    created = beta.create(model="claude-x", max_tokens=64, messages=msgs)
    with beta.stream(model="claude-x", max_tokens=64, messages=msgs) as stream:
        streamed = stream.get_final_message()
    parsed = beta.parse(model="claude-x", max_tokens=64, messages=msgs, output_format=Weather)
    assert type(parsed).__name__.startswith("ParsedBetaMessage")
    assert type(created).__name__ == "BetaMessage"
    return [created.content[0].text, streamed.content[0].text, parsed.parsed_output.model_dump()]


def test_beta_messages_create_stream_parse_record_and_replay(crashlab_dir):
    if not callable(getattr(anthropic.Anthropic(api_key="x").beta.messages, "parse", None)):
        pytest.skip("anthropic SDK without beta.messages.parse")
    out = beta_agent("weather?")
    run = crashlab.load("last")
    assert [s.operation for s in run.steps] == ["chat"] * 3
    LIVE["on"] = False
    try:
        replayed = crashlab.replay(run)
    finally:
        LIVE["on"] = True
    assert replayed.status == "ok" and replayed.output == out


def test_generic_response_type_imports_unparametrized():
    from crashlab.replay.rebuild import import_path

    cls = import_path("anthropic.types.parsed_message.ParsedMessage[TypeVar]")
    assert cls is not None and cls.__name__ == "ParsedMessage"
