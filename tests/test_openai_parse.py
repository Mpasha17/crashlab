"""``chat.completions.parse`` / ``responses.parse`` (structured outputs) must record and
replay offline with ``.parsed`` rebuilt; a pydantic class as a request value must not crash."""

from __future__ import annotations

import json
import warnings

import pytest

import crashlab
from crashlab.capture.normalize import to_jsonable

openai = pytest.importorskip("openai")
pydantic = pytest.importorskip("pydantic")
try:  # openai>=3 ships on httpx2; older releases use httpx
    import httpx2 as httpx
except ImportError:
    httpx = pytest.importorskip("httpx")


class Weather(pydantic.BaseModel):
    city: str
    temp: int


ANSWER = {"city": "Paris", "temp": 21}
CHAT = {
    "id": "chatcmpl-1", "object": "chat.completion", "created": 1760700000, "model": "m",
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": json.dumps(ANSWER)}}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
}  # fmt: skip
RESP = {
    "id": "resp-1", "object": "response", "created_at": 1760700000, "model": "m",
    "status": "completed", "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
    "output": [{"type": "message", "id": "msg-1", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": json.dumps(ANSWER),
                             "annotations": []}]}],
    "usage": {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10,
              "input_tokens_details": {"cached_tokens": 0},
              "output_tokens_details": {"reasoning_tokens": 0}},
}  # fmt: skip

CALLS: list[str] = []
LIVE = {"on": True}


def _handle(request):
    CALLS.append(request.url.path)
    if not LIVE["on"]:
        raise AssertionError("replay hit the network")
    return httpx.Response(200, json=RESP if request.url.path.endswith("/responses") else CHAT)


def _client(is_async: bool = False):
    if is_async:
        http = httpx.AsyncClient(transport=httpx.MockTransport(_handle))
        return crashlab.wrap(openai.AsyncOpenAI(api_key="sk-x", base_url="http://t/v1",
                                                http_client=http))  # fmt: skip
    http = httpx.Client(transport=httpx.MockTransport(_handle))
    return crashlab.wrap(openai.OpenAI(api_key="sk-x", base_url="http://t/v1", http_client=http))


@crashlab.agent(name="parse-chat")
def chat_agent(q: str) -> dict:
    out = _client().chat.completions.parse(
        model="m", messages=[{"role": "user", "content": q}], response_format=Weather
    )
    parsed = out.choices[0].message.parsed
    assert isinstance(parsed, Weather)
    return parsed.model_dump()


@crashlab.agent(name="parse-responses")
def responses_agent(q: str) -> dict:
    out = _client().responses.parse(model="m", input=q, text_format=Weather)
    assert isinstance(out.output_parsed, Weather)
    return out.output_parsed.model_dump()


@crashlab.agent(name="parse-async")
async def async_agent(q: str) -> dict:
    out = await _client(True).chat.completions.parse(
        model="m", messages=[{"role": "user", "content": q}], response_format=Weather
    )
    return out.choices[0].message.parsed.model_dump()


def _record_then_replay(fn, *args):
    LIVE["on"] = True
    CALLS.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no pydantic serializer noise from recording
        assert fn(*args) == ANSWER
    run = crashlab.load("last")
    assert [s.operation for s in run.steps] == ["chat"]
    assert CALLS and len(CALLS) == 1
    LIVE["on"] = False
    try:
        replayed = crashlab.replay(run)
    finally:
        LIVE["on"] = True
    assert replayed.status == "ok" and replayed.output == ANSWER
    assert len(CALLS) == 1  # served from the cassette
    return run


def test_chat_completions_parse_records_and_replays(crashlab_dir):
    run = _record_then_replay(chat_agent, "weather?")
    fmt = run.steps[0].request["params"]["response_format"]
    assert fmt["class"] == "Weather" and "temp" in fmt["schema"]["properties"]


def test_responses_parse_records_and_replays(crashlab_dir):
    _record_then_replay(responses_agent, "weather?")


def test_async_chat_completions_parse_records_and_replays(crashlab_dir):
    import asyncio

    LIVE["on"] = True
    assert asyncio.run(async_agent("weather?")) == ANSWER
    run = crashlab.load("last")
    assert [s.operation for s in run.steps] == ["chat"]
    LIVE["on"] = False
    try:
        assert crashlab.replay(run).output == ANSWER
    finally:
        LIVE["on"] = True


def test_class_values_serialize_as_schema_not_crash():
    assert to_jsonable(Weather)["class"] == "Weather"
    assert to_jsonable(int) == {"class": "builtins.int"}
