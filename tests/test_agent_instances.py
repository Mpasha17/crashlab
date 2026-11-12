"""Two concurrent sub-agents with the same name get told apart (``researcher#1`` /
``researcher#2``) from their conversations; sequential calls keep the plain name.

Live finding (OpenAI Agents SDK ``as_tool``, nov13): the manager ran the researcher agent
twice in parallel (python / rust); every step said ``researcher`` and nothing showed which
lookups belonged to which run of it."""

from __future__ import annotations

import json
from typing import Any

from typer.testing import CliRunner

import crashlab
from crashlab.store import Store

SYS = "Use lookup_release, then state the fact."


def _msgs(*items: tuple[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for role, content in items:
        if role == "call":
            out.append({"role": "assistant", "tool_calls": [{"id": content, "name": "lookup"}]})
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": content[0], "content": content[1]})
        else:
            out.append({"role": role, "content": content})
    return out


def _chat(idx: int, agent: str | None, messages: list[dict[str, Any]], *,
          call: str | None = None, text: str | None = None) -> crashlab.Step:  # fmt: skip
    resp: dict[str, Any] = {"messages": []}
    if call:
        resp["tool_calls"] = [{"id": call, "name": "lookup", "arguments": "{}"}]
    if text:
        resp["messages"] = [{"role": "assistant", "parts": [{"type": "text", "content": text}]}]
    return crashlab.Step(
        run_id="r", idx=idx, span_id=f"s{idx}", operation="chat", name="m", request_hash="h",
        request={"system": SYS, "messages": messages}, response=resp,
        attributes={"gen_ai.agent.name": agent} if agent else {},
    )  # fmt: skip


def _tool(idx: int, agent: str | None, call: str, q: str) -> crashlab.Step:
    attrs: dict[str, Any] = {"gen_ai.tool.call.id": call}
    if agent:
        attrs["gen_ai.agent.name"] = agent
    return crashlab.Step(
        run_id="r", idx=idx, span_id=f"t{idx}", operation="execute_tool", name="lookup",
        request_hash="h", request={"args": {"project": q}}, response={"result": q + " 1.0"},
        attributes=attrs,
    )  # fmt: skip


def _parallel(first_py: str = "python?", first_rs: str = "rust?") -> crashlab.Run:
    """Recorded shape of OpenAI Agents SDK as_tool x2 in parallel (run 01M78Q2V59)."""
    py0 = _msgs(("user", first_py))
    rs0 = _msgs(("user", first_rs))
    steps = [
        _chat(0, "manager", _msgs(("user", "python and rust?")), call="m1"),
        _chat(1, "researcher", py0, call="p1"),
        _chat(2, "researcher", rs0, call="r1"),
        _tool(3, "researcher", "p1", "python"),
        _chat(4, "researcher", py0 + _msgs(("call", "p1"), ("tool", ("p1", "python 1.0"))),
              text="python 1.0"),
        _tool(5, "researcher", "r1", "rust"),
        _chat(6, "researcher", rs0 + _msgs(("call", "r1"), ("tool", ("r1", "rust 1.0"))),
              text="rust 1.0"),
        _chat(7, "manager", _msgs(("user", "python and rust?")), text="both 1.0"),
    ]  # fmt: skip
    return crashlab.Run(run_id="r", name="t", steps=steps)


def test_parallel_same_name_sub_agents_are_numbered():
    run = _parallel()
    labels = run.agent_instances()
    assert [labels.get(i) for i in range(8)] == [
        "manager", "researcher#1", "researcher#2", "researcher#1",
        "researcher#1", "researcher#2", "researcher#2", "manager",
    ]  # fmt: skip
    assert run.agents == ["manager", "researcher"]  # names unchanged
    assert [s.idx for s in run.chat_steps(agent="researcher#2")] == [2, 6]
    assert [s.args["project"] for s in run.tool_calls(agent="researcher#1")] == ["python"]
    assert len(run.tool_calls(agent="researcher")) == 2  # the plain name still matches all


def test_identical_first_requests_are_split_by_tool_call_ids():
    run = _parallel("same question", "same question")
    labels = run.agent_instances()
    assert labels[1] != labels[2]
    assert labels[4] == labels[1] and labels[6] == labels[2]


def test_sequential_calls_and_multi_turn_agents_keep_the_plain_name():
    py0, rs0 = _msgs(("user", "python?")), _msgs(("user", "rust?"))
    steps = [
        _chat(0, "researcher", py0, call="p1"),
        _tool(1, "researcher", "p1", "python"),
        _chat(2, "researcher", py0 + _msgs(("call", "p1"), ("tool", ("p1", "x"))), text="x"),
        _chat(3, "researcher", rs0, call="r1"),  # called again later: not concurrent
        _tool(4, "researcher", "r1", "rust"),
        _chat(5, "researcher", rs0 + _msgs(("call", "r1"), ("tool", ("r1", "y"))), text="y"),
    ]
    run = crashlab.Run(run_id="r", name="t", steps=steps)
    assert set(run.agent_instances().values()) == {"researcher"}


def test_unlabelled_and_old_runs_have_no_instances():
    run = _parallel()
    for s in run.steps:
        s.attributes.pop("gen_ai.agent.name", None)
    assert run.agent_instances() == {}
    assert run.chat_steps(agent="researcher#1") == []


def _store(run: crashlab.Run) -> None:
    store = Store()
    store.begin_run(run)
    for s in run.steps:
        store.add_step(s)
    store.end_run(run)


def test_show_and_diagnoser_view_use_instance_labels(monkeypatch):
    from crashlab.cli import app
    from crashlab.diagnose.compact import compact

    _store(_parallel())
    monkeypatch.setenv("COLUMNS", "160")
    out = CliRunner().invoke(app, ["show", "last"]).output
    assert "@researcher#2" in out and "@researcher#1" in out
    assert "manager → researcher#1 → researcher#2" in out
    out = CliRunner().invoke(app, ["show", "last", "--agent", "researcher#2"]).output
    assert "showing 3 of 8 steps: agent researcher#2" in out
    view = compact(crashlab.load("last"))
    assert "[agent: researcher#2]" in view


def test_json_round_trip_keeps_instances():
    run = crashlab.Run.model_validate(json.loads(_parallel().model_dump_json()))
    assert run.agent_instances()[6] == "researcher#2"
