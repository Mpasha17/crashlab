"""Sub-agents running concurrently inside one run (an orchestrator fanning out with
asyncio.gather): every tool step must stay linked to the chat step that requested it.

Before the fix the run kept only the latest chat step's tool calls, so a slower
sub-agent's tool step lost its ``gen_ai.tool.call.id`` and ``describe_step`` /
``diagnose`` blamed the other sub-agent's chat step for it."""

from __future__ import annotations

import asyncio

import pytest

import crashlab
from crashlab.models import Run, Step
from tests.fakes import AsyncFakeOpenAI, text_response, tool_call_response

pytestmark = pytest.mark.requires_providers("pydantic")


@crashlab.tool()
async def search(q: str) -> str:
    await asyncio.sleep(0.05 if q == "a" else 0.0)
    return f"result for {q}"


async def _sub(client, q: str) -> str:
    r = await client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": q}], tools=[]
    )
    # sub-agent "a" is slower: the other sub-agent's chat step lands in between
    await asyncio.sleep(0.03 if q == "a" else 0.0)
    tc = r.choices[0].message.tool_calls[0]
    await search(q)
    r2 = await client.chat.completions.create(
        model="m",
        messages=[{"role": "user", "content": q},
                  {"role": "tool", "content": "x", "tool_call_id": tc.id}],
    )  # fmt: skip
    return r2.choices[0].message.content


@crashlab.agent(name="fanout")
def fanout() -> list:
    async def go():
        ca = crashlab.wrap(AsyncFakeOpenAI(
            [tool_call_response(1, "search", {"q": "a"}), text_response(3, "A done")]))  # fmt: skip
        cb = crashlab.wrap(AsyncFakeOpenAI(
            [tool_call_response(2, "search", {"q": "b"}), text_response(4, "B done")]))  # fmt: skip
        return await asyncio.gather(_sub(ca, "a"), _sub(cb, "b"))

    return asyncio.run(go())


def _requester(run: Run, tool: Step) -> int:
    call_id = tool.attributes.get("gen_ai.tool.call.id")
    return next(
        s.idx
        for s in run.steps
        if s.operation == "chat"
        and any(c.get("id") == call_id for c in (s.response or {}).get("tool_calls") or [])
    )


def test_every_concurrent_tool_step_keeps_its_call_id():
    assert fanout() == ["A done", "B done"]
    run = crashlab.load("last")
    tools = run.tool_calls("search")
    assert len(tools) == 2
    ids = {t.args["q"]: t.attributes.get("gen_ai.tool.call.id") for t in tools}
    assert ids == {"a": "call_1", "b": "call_2"}


def test_describe_step_names_the_requesting_chat_not_the_nearest():
    fanout()
    run = crashlab.load("last")
    for tool in run.tool_calls("search"):
        want = _requester(run, tool)
        assert run.caller_chat(tool.idx) == want
        assert f"called by chat step {want}" in run.describe_step(tool.idx)


def test_caller_chat_falls_back_to_nearest_chat_without_call_id():
    """0.0.3 cassettes / undecorated frameworks: no call id -> nearest preceding chat."""
    fanout()
    run = crashlab.load("last")
    tool = run.tool_calls("search")[0]
    tool.attributes.pop("gen_ai.tool.call.id")
    nearest = max(s.idx for s in run.steps if s.operation == "chat" and s.idx < tool.idx)
    assert run.caller_chat(tool.idx) == nearest


def test_reused_call_ids_link_to_latest_requesting_chat():
    """Providers that number calls per response (call_0 every turn) must link to the
    latest chat before the tool step that requested that id."""
    steps = [
        Step(run_id="r", idx=0, span_id="a", operation="chat", name="m", request_hash="h",
             response={"tool_calls": [{"id": "call_0", "name": "t", "arguments": {}}]}),
        Step(run_id="r", idx=1, span_id="b", operation="execute_tool", name="t",
             request_hash="h", attributes={"gen_ai.tool.call.id": "call_0"}),
        Step(run_id="r", idx=2, span_id="c", operation="chat", name="m", request_hash="h",
             response={"tool_calls": [{"id": "call_0", "name": "t", "arguments": {}}]}),
        Step(run_id="r", idx=3, span_id="d", operation="execute_tool", name="t",
             request_hash="h", attributes={"gen_ai.tool.call.id": "call_0"}),
    ]  # fmt: skip
    r = Run(run_id="r", name="x", steps=steps)
    assert r.caller_chat(1) == 0
    assert r.caller_chat(3) == 2


def test_concurrent_run_replays_offline():
    fanout()
    src = crashlab.load("last")
    crashlab.replay(src)
    rep = crashlab.load("last")
    assert rep.replay_of == src.run_id
    assert rep.status == "ok"
    ids = {t.args["q"]: t.attributes.get("gen_ai.tool.call.id") for t in rep.tool_calls("search")}
    assert ids == {"a": "call_1", "b": "call_2"}


def test_diagnose_heuristic_blames_the_requesting_chat():
    from crashlab.diagnose.heuristics import _caller_chat

    fanout()
    run = crashlab.load("last")
    for tool in run.tool_calls("search"):
        assert _caller_chat(run, tool.idx) == _requester(run, tool)
