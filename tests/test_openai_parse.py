"""``chat.completions.parse`` / ``responses.parse`` (structured outputs) and the
``.stream()`` helpers must record and replay offline with ``.parsed`` rebuilt; a pydantic
class as a request value must not crash; ``with_streaming_response`` warns it is unrecorded."""

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


def _sse(events: list[dict]) -> bytes:
    body = "".join(f"event: {e.get('type', 'message')}\ndata: {json.dumps(e)}\n\n" for e in events)
    return body.encode()


_TEXT = json.dumps(ANSWER)
CHAT_CHUNKS = [
    {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
     "choices": [{"index": 0, "delta": {"role": "assistant", "content": _TEXT[:9]},
                  "finish_reason": None}]},
    {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
     "choices": [{"index": 0, "delta": {"content": _TEXT[9:]}, "finish_reason": "stop"}]},
]  # fmt: skip
_PART = {"output_index": 0, "content_index": 0, "item_id": "msg-1"}
_MSG = RESP["output"][0]
RESP_EVENTS = [
    {"type": "response.created", "response": {**RESP, "status": "in_progress", "output": []}},
    {"type": "response.output_item.added", "output_index": 0,
     "item": {**_MSG, "status": "in_progress", "content": []}},
    {"type": "response.content_part.added", **_PART,
     "part": {"type": "output_text", "text": "", "annotations": []}},
    {"type": "response.output_text.delta", **_PART, "delta": _TEXT, "logprobs": []},
    {"type": "response.output_text.done", **_PART, "text": _TEXT, "logprobs": []},
    {"type": "response.content_part.done", **_PART,
     "part": {"type": "output_text", "text": _TEXT, "annotations": []}},
    {"type": "response.output_item.done", "output_index": 0, "item": _MSG},
    {"type": "response.completed", "response": RESP},
]  # fmt: skip
for _n, _e in enumerate(RESP_EVENTS):
    _e["sequence_number"] = _n


def _handle(request):
    CALLS.append(request.url.path)
    if not LIVE["on"]:
        raise AssertionError("replay hit the network")
    is_resp = request.url.path.endswith("/responses")
    if json.loads(request.content).get("stream"):
        body = _sse(RESP_EVENTS) if is_resp else _sse(CHAT_CHUNKS) + b"data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    return httpx.Response(200, json=RESP if is_resp else CHAT)


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


@crashlab.agent(name="stream-chat")
def stream_chat_agent(q: str) -> dict:
    with _client().chat.completions.stream(
        model="m", messages=[{"role": "user", "content": q}], response_format=Weather
    ) as stream:
        deltas = [e.delta for e in stream if e.type == "content.delta"]
        final = stream.get_final_completion()
    assert "".join(deltas) == _TEXT
    return final.choices[0].message.parsed.model_dump()


@crashlab.agent(name="stream-responses")
def stream_responses_agent(q: str) -> dict:
    with _client().responses.stream(model="m", input=q, text_format=Weather) as stream:
        for _ in stream:
            pass
        final = stream.get_final_response()
    return final.output_parsed.model_dump()


@crashlab.agent(name="stream-async")
async def stream_async_agent(q: str) -> dict:
    async with _client(True).chat.completions.stream(
        model="m", messages=[{"role": "user", "content": q}], response_format=Weather
    ) as stream:
        async for _ in stream:
            pass
        final = await stream.get_final_completion()
    return final.choices[0].message.parsed.model_dump()


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


def test_chat_stream_helper_records_and_replays(crashlab_dir):
    run = _record_then_replay(stream_chat_agent, "weather?")
    assert run.steps[0].response["messages"][0]["parts"][0]["content"] == _TEXT


def test_responses_stream_helper_records_and_replays(crashlab_dir):
    _record_then_replay(stream_responses_agent, "weather?")


def test_async_chat_stream_helper_records_and_replays(crashlab_dir):
    import asyncio

    LIVE["on"] = True
    assert asyncio.run(stream_async_agent("weather?")) == ANSWER
    run = crashlab.load("last")
    assert [s.operation for s in run.steps] == ["chat"]
    LIVE["on"] = False
    try:
        assert crashlab.replay(run).output == ANSWER
    finally:
        LIVE["on"] = True


def test_with_streaming_response_warns_it_is_not_recorded(crashlab_dir):
    from crashlab.capture import openai as cap

    cap._WARNED_STREAMING_RESPONSE.clear()
    client = _client()

    @crashlab.agent(name="raw-stream")
    def agent() -> str:
        with client.chat.completions.with_streaming_response.create(
            model="m", messages=[{"role": "user", "content": "hi"}]
        ) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]

    with pytest.warns(RuntimeWarning, match="with_streaming_response.create is not recorded"):
        assert agent() == _TEXT


@crashlab.agent(name="beta-parse")
def beta_parse_agent(q: str) -> dict:
    client = _client()
    out = client.beta.chat.completions.parse(
        model="m", messages=[{"role": "user", "content": q}], response_format=Weather
    )
    with client.beta.chat.completions.stream(
        model="m", messages=[{"role": "user", "content": q}], response_format=Weather
    ) as stream:
        for _ in stream:
            pass
        streamed = stream.get_final_completion().choices[0].message.parsed
    assert streamed == out.choices[0].message.parsed
    return out.choices[0].message.parsed.model_dump()


def test_beta_chat_completions_parse_and_stream_record(crashlab_dir):
    if getattr(getattr(openai.OpenAI(api_key="x"), "beta", None), "chat", None) is None:
        pytest.skip("openai SDK without client.beta.chat")
    LIVE["on"] = True
    CALLS.clear()
    assert beta_parse_agent("weather?") == ANSWER
    run = crashlab.load("last")
    assert [s.operation for s in run.steps] == ["chat", "chat"]
    LIVE["on"] = False
    try:
        assert crashlab.replay(run).output == ANSWER
    finally:
        LIVE["on"] = True
    assert len(CALLS) == 2
