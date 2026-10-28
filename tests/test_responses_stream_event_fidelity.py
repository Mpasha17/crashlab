"""A replayed Responses-API stream is event-for-event the recorded one (content_part /
output_text.done / function_call_arguments.done / reasoning summary events included),
and streams with events the template does not rebuild still replay their deltas."""

from __future__ import annotations

import pytest

import crashlab

from ._transport import httpx_module, sse

openai = pytest.importorskip("openai")
httpx = httpx_module()

TEXT = ["Hel", "lo ", "wörld"]
ARGS = ['{"ci', 'ty": "Par', 'is"}']
PART = {"type": "output_text", "text": "".join(TEXT), "annotations": [], "logprobs": []}
MSG = {"type": "message", "id": "msg-1", "role": "assistant", "status": "completed",
       "content": [PART]}  # fmt: skip
CALL = {"type": "function_call", "id": "fc-1", "call_id": "call-1", "name": "weather",
        "arguments": "".join(ARGS), "status": "completed"}  # fmt: skip
SUMMARY = {"type": "summary_text", "text": "Think ing"}
REASONING = {"type": "reasoning", "id": "rs-1", "summary": [SUMMARY]}
USAGE = {"input_tokens": 3, "output_tokens": 9, "total_tokens": 12,
         "input_tokens_details": {"cached_tokens": 0},
         "output_tokens_details": {"reasoning_tokens": 4}}  # fmt: skip


def _resp(output):
    return {"id": "resp-1", "object": "response", "created_at": 1, "model": "m",
            "status": "completed", "parallel_tool_calls": True, "tool_choice": "auto",
            "tools": [], "output": output, "usage": USAGE}  # fmt: skip


def _events(reasoning=False, unknown=False):
    items = ([REASONING] if reasoning else []) + [MSG, CALL]
    m, c = (1, 2) if reasoning else (0, 1)
    resp = _resp(items)
    started = {**resp, "status": "in_progress", "output": [], "usage": None}
    p0 = {"output_index": m, "content_index": 0, "item_id": "msg-1"}
    r0 = {"output_index": 0, "summary_index": 0, "item_id": "rs-1"}
    ev = [{"type": "response.created", "response": started},
          {"type": "response.in_progress", "response": started}]  # fmt: skip
    if reasoning:
        ev += [
            {"type": "response.output_item.added", "output_index": 0,
             "item": {**REASONING, "summary": []}},
            {"type": "response.reasoning_summary_part.added", **r0,
             "part": {**SUMMARY, "text": ""}},
            {"type": "response.reasoning_summary_text.delta", **r0, "delta": "Think"},
            {"type": "response.reasoning_summary_text.delta", **r0, "delta": " ing"},
            {"type": "response.reasoning_summary_text.done", **r0, "text": SUMMARY["text"]},
            {"type": "response.reasoning_summary_part.done", **r0, "part": SUMMARY},
            {"type": "response.output_item.done", "output_index": 0, "item": REASONING},
        ]  # fmt: skip
    if unknown:
        ev.append({"type": "response.web_search_call.in_progress", "output_index": m,
                   "item_id": "ws-1"})  # fmt: skip
    ev += [
        {"type": "response.output_item.added", "output_index": m,
         "item": {**MSG, "status": "in_progress", "content": []}},
        {"type": "response.content_part.added", **p0,
         "part": {**PART, "text": "", "annotations": []}},
        *[{"type": "response.output_text.delta", **p0, "delta": t, "logprobs": []} for t in TEXT],
        {"type": "response.output_text.done", **p0, "text": "".join(TEXT), "logprobs": []},
        {"type": "response.content_part.done", **p0, "part": PART},
        {"type": "response.output_item.done", "output_index": m, "item": MSG},
        {"type": "response.output_item.added", "output_index": c,
         "item": {**CALL, "arguments": "", "status": "in_progress"}},
        *[{"type": "response.function_call_arguments.delta", "output_index": c,
           "item_id": "fc-1", "delta": a} for a in ARGS],
        {"type": "response.function_call_arguments.done", "output_index": c,
         "item_id": "fc-1", "arguments": "".join(ARGS), "name": "weather"},
        {"type": "response.output_item.done", "output_index": c, "item": CALL},
        {"type": "response.completed", "response": resp},
    ]  # fmt: skip
    for n, e in enumerate(ev):
        e["sequence_number"] = n
    return ev


STATE = {"live": True, "events": _events()}


def _handle(request):
    assert STATE["live"], "replay hit the network"
    body = sse(STATE["events"], named=True)
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)


def _client():
    http = httpx.Client(transport=httpx.MockTransport(_handle))
    return crashlab.wrap(openai.OpenAI(api_key="sk-x", base_url="http://t/v1", http_client=http))


@crashlab.agent(name="resp-raw-events")
def raw_events():
    for ev in _client().responses.create(model="m", input="hi", stream=True):
        yield ev.model_dump()


@crashlab.agent(name="resp-helper-events")
def helper_events():
    with _client().responses.stream(model="m", input="hi") as stream:
        for ev in stream:
            yield [ev.type, getattr(ev, "delta", None)]


@crashlab.agent(name="resp-text-deltas")
def text_deltas():
    for ev in _client().responses.create(model="m", input="hi", stream=True):
        if ev.type in ("response.output_text.delta", "response.function_call_arguments.delta"):
            yield [ev.type, ev.delta]


def _record_then_replay(agent, events):
    STATE.update(live=True, events=events)
    live = list(agent())
    STATE["live"] = False
    try:
        replayed = crashlab.replay(crashlab.load("last"))
    finally:
        STATE["live"] = True
    assert replayed.status == "ok"
    return live, replayed.output


@pytest.mark.parametrize("reasoning", [False, True], ids=["plain", "reasoning"])
@pytest.mark.parametrize("agent", [raw_events, helper_events], ids=["create", "stream-helper"])
def test_replayed_responses_stream_matches_event_for_event(crashlab_dir, agent, reasoning):
    events = _events(reasoning=reasoning)
    live, replayed = _record_then_replay(agent, events)
    assert len(live) == len(events)
    assert replayed == live


def test_unrebuilt_event_types_fall_back_to_synthesized_deltas(crashlab_dir):
    live, replayed = _record_then_replay(text_deltas, _events(unknown=True))
    assert [d for _, d in live] == TEXT + ARGS
    assert replayed == live
