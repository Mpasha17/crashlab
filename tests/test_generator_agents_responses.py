"""Responses-API streams replay with their recorded delta boundaries (text and function
call arguments), so generator agents over ``responses.create(stream=True)`` reproduce."""

from __future__ import annotations

import json

import pytest

import crashlab

from ._transport import httpx_module, sse

openai = pytest.importorskip("openai")
httpx = httpx_module()

TEXT = ["Hel", "lo ", "wörld"]
ARGS = ['{"ci', 'ty": "Par', 'is"}']
MSG = {"type": "message", "id": "msg-1", "role": "assistant", "status": "completed",
       "content": [{"type": "output_text", "text": "".join(TEXT), "annotations": []}]}  # fmt: skip
CALL = {"type": "function_call", "id": "fc-1", "call_id": "call-1", "name": "weather",
        "arguments": "".join(ARGS), "status": "completed"}  # fmt: skip
RESP = {"id": "resp-1", "object": "response", "created_at": 1, "model": "m",
        "status": "completed", "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        "output": [MSG, CALL]}  # fmt: skip
_PART = {"output_index": 0, "content_index": 0, "item_id": "msg-1"}
EVENTS = [
    {"type": "response.created", "response": {**RESP, "status": "in_progress", "output": []}},
    {"type": "response.output_item.added", "output_index": 0,
     "item": {**MSG, "status": "in_progress", "content": []}},
    *[{"type": "response.output_text.delta", **_PART, "delta": t, "logprobs": []} for t in TEXT],
    {"type": "response.output_item.done", "output_index": 0, "item": MSG},
    {"type": "response.output_item.added", "output_index": 1,
     "item": {**CALL, "arguments": "", "status": "in_progress"}},
    *[{"type": "response.function_call_arguments.delta", "output_index": 1, "item_id": "fc-1",
       "delta": a} for a in ARGS],
    {"type": "response.output_item.done", "output_index": 1, "item": CALL},
    {"type": "response.completed", "response": RESP},
]  # fmt: skip
for _n, _e in enumerate(EVENTS):
    _e["sequence_number"] = _n
LIVE = {"on": True}


def _handle(request):
    assert LIVE["on"], "replay hit the network"
    assert json.loads(request.content)["stream"] is True
    body = sse(EVENTS, named=True)
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)


@crashlab.agent(name="responses-gen")
def responses_gen():
    http = httpx.Client(transport=httpx.MockTransport(_handle))
    client = crashlab.wrap(openai.OpenAI(api_key="sk-x", base_url="http://t/v1", http_client=http))
    for ev in client.responses.create(model="m", input="weather?", stream=True):
        if ev.type in ("response.output_text.delta", "response.function_call_arguments.delta"):
            yield [ev.type.split(".")[1], ev.delta]


EXPECTED = [["output_text", t] for t in TEXT] + [["function_call_arguments", a] for a in ARGS]


def test_responses_stream_generator_replays_delta_for_delta(crashlab_dir):
    assert list(responses_gen()) == EXPECTED
    run = crashlab.load("last")
    assert [s.operation for s in run.steps] == ["chat"]
    LIVE["on"] = False
    try:
        assert crashlab.replay(run).output == EXPECTED
    finally:
        LIVE["on"] = True
