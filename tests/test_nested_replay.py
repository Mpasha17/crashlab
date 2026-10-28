"""An agent used as a tool (agent -> tool -> sub-agent -> LLM): the tool is served from the
cassette in a replay, so the steps recorded inside it must not be expected again."""

from __future__ import annotations

import json

import crashlab
from tests.fakes import FakeOpenAI, text_response, tool_call_response

SCRIPT = [
    tool_call_response(1, "delegate", {"task": "look it up"}),  # orchestrator
    tool_call_response(2, "lookup", {"q": "x"}),  # sub-agent
    text_response(3, "sub done"),  # sub-agent
    text_response(4, "all done"),  # orchestrator
]
CLIENT: list = []


@crashlab.tool()
def lookup(q: str) -> str:
    return f"found {q}"


def _loop(prompt: str, tools: dict) -> str:
    msgs = [{"role": "user", "content": prompt}]
    while True:
        m = CLIENT[0].chat.completions.create(model="m", messages=list(msgs)).choices[0].message
        if not m.tool_calls:
            return m.content
        msgs.append({"role": "assistant", "tool_calls": [t.model_dump() for t in m.tool_calls]})
        for tc in m.tool_calls:
            out = tools[tc.function.name](**json.loads(tc.function.arguments))
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": out})


@crashlab.agent(name="sub")
def sub_agent(task: str) -> str:
    return _loop(task, {"lookup": lookup})


@crashlab.tool()
def delegate(task: str) -> str:
    return sub_agent(task)


@crashlab.agent(name="orchestrator")
def orchestrator(task: str) -> str:
    return _loop(task, {"delegate": delegate})


def test_replay_skips_steps_recorded_inside_a_served_tool():
    CLIENT[:] = [crashlab.wrap(FakeOpenAI(list(SCRIPT)))]
    assert orchestrator("plan") == "all done"
    run = crashlab.load("last")
    ops = [(s.operation, s.name) for s in run.steps]
    assert ops[1] == ("execute_tool", "delegate") and ("invoke_agent", "sub") in ops
    assert len(run.steps) == 7
    replayed = crashlab.replay(run)
    assert replayed.status == "ok" and replayed.output == "all done"
    # the sub-agent never ran again: only the orchestrator's own steps were re-made
    assert [s.idx for s in replayed.steps] == [0, 1, 6]
    assert crashlab.replay(replayed).output == "all done"


def test_fork_after_a_nested_agent_tool():
    CLIENT[:] = [crashlab.wrap(FakeOpenAI(list(SCRIPT)))]
    orchestrator("plan")
    run = crashlab.load("last")
    CLIENT[:] = [crashlab.wrap(FakeOpenAI([text_response(9, "forked answer")]))]
    branch = crashlab.fork(run, at_step=6, patch=crashlab.SystemPromptPatch(op="append", text="!"))
    assert branch.status == "ok" and branch.output == "forked answer"
