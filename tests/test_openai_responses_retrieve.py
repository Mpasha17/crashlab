"""Background responses (``responses.create(background=True)`` then polling
``responses.retrieve``) are recorded, so replay stays offline and serves the same
polls; retrieve steps are 'custom' (not LLM calls)."""

from __future__ import annotations

import asyncio
import json

import pytest

import crashlab

from ._transport import httpx_module

openai = pytest.importorskip("openai")
httpx = httpx_module()

MSG = {"type": "message", "id": "msg-1", "role": "assistant", "status": "completed",
       "content": [{"type": "output_text", "text": "Deep answer", "annotations": []}]}  # fmt: skip


def _resp(status, output=()):
    out = {"id": "resp_bg1", "object": "response", "created_at": 1, "model": "m",
           "status": status, "parallel_tool_calls": True, "tool_choice": "auto",
           "tools": [], "output": list(output), "background": True}  # fmt: skip
    if status == "completed":
        out["usage"] = {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens_details": {"reasoning_tokens": 0}}  # fmt: skip
    return out


STATE = {"live": True, "polls": 0}


def _handle(request):
    assert STATE["live"], "replay hit the network"
    if request.method == "POST":
        assert json.loads(request.content)["background"] is True
        return httpx.Response(200, json=_resp("queued"))
    assert request.url.path.endswith("/responses/resp_bg1")
    STATE["polls"] += 1
    done = STATE["polls"] >= 2
    return httpx.Response(200, json=_resp("completed" if done else "in_progress",
                                          [MSG] if done else []))  # fmt: skip


async def _ahandle(request):
    return _handle(request)


@crashlab.agent(name="background")
def background() -> str:
    http = httpx.Client(transport=httpx.MockTransport(_handle))
    client = crashlab.wrap(openai.OpenAI(api_key="sk-x", base_url="http://t/v1", http_client=http))
    resp = client.responses.create(model="m", input="think hard", background=True)
    while resp.status in ("queued", "in_progress"):
        resp = client.responses.retrieve(resp.id)
    return resp.output_text


@crashlab.agent(name="abackground")
async def abackground() -> str:
    http = httpx.AsyncClient(transport=httpx.MockTransport(_ahandle))
    client = crashlab.wrap(
        openai.AsyncOpenAI(api_key="sk-x", base_url="http://t/v1", http_client=http)
    )
    resp = await client.responses.create(model="m", input="think hard", background=True)
    while resp.status in ("queued", "in_progress"):
        resp = await client.responses.retrieve(response_id=resp.id)
    return resp.output_text


@pytest.mark.parametrize("agent", [background, abackground], ids=["sync", "async"])
def test_background_polling_is_recorded_and_replays_offline(crashlab_dir, agent):
    STATE.update(live=True, polls=0)
    out = agent()
    if asyncio.iscoroutine(out):
        out = asyncio.run(out)
    assert out == "Deep answer"
    run = crashlab.load("last")
    assert [s.operation for s in run.steps] == ["chat", "custom", "custom"]
    assert run.steps[2].attributes["crashlab.response.type"].endswith("Response")
    STATE["live"] = False
    try:
        replayed = crashlab.replay(run)
    finally:
        STATE["live"] = True
    assert replayed.status == "ok" and replayed.output == "Deep answer"
