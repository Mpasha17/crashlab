"""Handoffs between agents are shown as events, worked out from the recording.

Live findings (nov13): the OpenAI Agents SDK (``transfer_to_refunds``) and LlamaIndex
AgentWorkflow (``handoff``) hand the conversation to another agent; the next agent's first
request answers the handoff tool call of the previous one. ``crashlab show`` only showed
the agent label changing. Agents called as tools (``as_tool``, smolagents managed agents,
Haystack sub-agents) start a fresh conversation and are not handoffs."""

from __future__ import annotations

import json
from typing import Any

from typer.testing import CliRunner

import crashlab
from crashlab.store import Store

USER = {"role": "user", "parts": [{"type": "text", "content": "Refund A100"}]}


def _call(cid: str, name: str) -> dict[str, Any]:
    return {"role": "assistant", "parts": [{"type": "tool_call", "id": cid, "name": name,
                                             "arguments": {}}]}  # fmt: skip


def _answer(cid: str, out: str) -> dict[str, Any]:
    return {"role": "tool", "parts": [{"type": "tool_call_response", "id": cid, "response": out}]}


def _chat(idx: int, agent: str | None, messages: list[dict[str, Any]], *,
          call: tuple[str, str] | None = None, text: str = "") -> crashlab.Step:  # fmt: skip
    resp: dict[str, Any] = {"messages": [], "tool_calls": []}
    if call:
        resp["tool_calls"] = [{"id": call[0], "type": "function", "name": call[1], "arguments": {}}]
        resp["messages"] = [_call(*call)]
    if text:
        resp["messages"] = [{"role": "assistant", "parts": [{"type": "text", "content": text}]}]
    return crashlab.Step(
        run_id="r", idx=idx, span_id=f"s{idx}", operation="chat", name="m", request_hash="h",
        request={"system": f"You are {agent}.", "messages": messages}, response=resp,
        attributes={"gen_ai.agent.name": agent} if agent else {},
    )  # fmt: skip


def _tool(idx: int, agent: str, cid: str, name: str) -> crashlab.Step:
    return crashlab.Step(
        run_id="r", idx=idx, span_id=f"t{idx}", operation="execute_tool", name=name,
        request_hash="h", request={"args": {}}, response={"result": "ok"},
        attributes={"gen_ai.tool.call.id": cid, "gen_ai.agent.name": agent},
    )  # fmt: skip


def _run(steps: list[crashlab.Step]) -> crashlab.Run:
    return crashlab.Run(run_id="r", name="t", entrypoint="x:y", steps=steps)


def _oa_handoff(agents: tuple[str | None, str | None] = ("triage", "refunds")) -> crashlab.Run:
    """Recorded shape of OpenAI Agents SDK handoff (run 01M78HVKP3)."""
    a, b = agents
    h = [USER, _call("c1", "transfer_to_refunds"), _answer("c1", '{"assistant": "refunds"}')]
    return _run([
        _chat(0, a, [USER], call=("c1", "transfer_to_refunds")),
        _chat(1, b, h, call=("c2", "lookup_order")),
        _tool(2, "refunds", "c2", "lookup_order"),
        _chat(3, b, h + [_call("c2", "lookup_order"), _answer("c2", "40")], text="Refunded 40."),
    ])  # fmt: skip


def test_oa_sdk_handoff_is_one_event() -> None:
    hs = _oa_handoff().handoffs()
    assert [(h.from_agent, h.to_agent, h.step, h.via_step, h.tool) for h in hs] == [
        ("triage", "refunds", 1, 0, "transfer_to_refunds")
    ]
    assert hs[0].call_id == "c1"


def test_llamaindex_handoff_uses_the_latest_call() -> None:
    """LlamaIndex AgentWorkflow (run 01M78NDAQY): the write agent's request carries the
    research agent's lookup and its handoff call; the handoff is the latest one."""
    hist = [USER, _call("l1", "lookup_release"), _answer("l1", "3.14.0"),
            _call("h1", "handoff"), _answer("h1", "Handed off to WriteAgent")]  # fmt: skip
    run = _run([
        _chat(0, "research", [USER], call=("l1", "lookup_release")),
        _tool(1, "research", "l1", "lookup_release"),
        _chat(2, "research", hist[:3], call=("h1", "handoff")),
        _chat(3, "write", hist, text="Python 3.14.0 was released."),
    ])  # fmt: skip
    assert [(h.from_agent, h.to_agent, h.step, h.via_step, h.tool) for h in run.handoffs()] == [
        ("research", "write", 3, 2, "handoff")
    ]


def test_agent_called_as_tool_is_not_a_handoff() -> None:
    """as_tool / smolagents managed agent: the sub-agent starts a fresh conversation and
    the manager answers its own call."""
    run = _run([
        _chat(0, "manager", [USER], call=("m1", "researcher")),
        _chat(1, "researcher", [{"role": "user", "parts": [{"type": "text", "content": "py?"}]}],
              text="3.14"),
        _chat(2, "manager", [USER, _call("m1", "researcher"), _answer("m1", "3.14")], text="ok"),
    ])  # fmt: skip
    assert run.handoffs() == []


def test_hand_back_and_unlabelled() -> None:
    h = [USER, _call("c1", "transfer_to_refunds"), _answer("c1", "{}")]
    back = h + [_call("c2", "transfer_to_triage"), _answer("c2", "{}")]
    run = _run([
        _chat(0, "triage", [USER], call=("c1", "transfer_to_refunds")),
        _chat(1, "refunds", h, call=("c2", "transfer_to_triage")),
        _chat(2, "triage", back, text="done"),
    ])  # fmt: skip
    assert [(h.from_agent, h.to_agent, h.step) for h in run.handoffs()] == [
        ("triage", "refunds", 1), ("refunds", "triage", 2)
    ]  # fmt: skip
    assert _oa_handoff((None, None)).handoffs() == []
    assert _oa_handoff(("triage", None)).handoffs() == []


def test_chat_completions_shape_tool_call_id() -> None:
    """Requests stored as plain chat-completions messages (tool_call_id) work too."""
    msgs = [{"role": "user", "content": "x"},
            {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "go"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "{}"}]  # fmt: skip
    run = _run([_chat(0, "a", msgs[:1], call=("c1", "go")), _chat(1, "b", msgs, text="hi")])
    assert [(h.from_agent, h.to_agent, h.tool) for h in run.handoffs()] == [("a", "b", "go")]


def test_show_prints_handoffs(monkeypatch) -> None:
    from crashlab.cli import app

    run = _oa_handoff()
    store = Store()
    store.begin_run(run)
    for s in run.steps:
        store.add_step(s)
    store.end_run(run)
    monkeypatch.setenv("COLUMNS", "200")
    out = CliRunner().invoke(app, ["show", "last"]).output
    assert "triage → refunds at step 1 (transfer_to_refunds, step 0)" in out
    assert "⇢ from triage" in out
    data = json.loads(CliRunner().invoke(app, ["show", "last", "--json"]).output)
    want = {"from_agent": "triage", "to_agent": "refunds", "step": 1, "via_step": 0,
            "tool": "transfer_to_refunds", "call_id": "c1"}  # fmt: skip
    assert data["handoffs"] == [want]
    plain = _run([s for s in run.steps if s.idx != 0])
    plain.run_id = "r2"
    store.begin_run(plain)
    store.end_run(plain)
    assert "handoffs" not in json.loads(CliRunner().invoke(app, ["show", "r2", "--json"]).output)


def test_diagnoser_view_marks_handoff() -> None:
    from crashlab.diagnose.compact import compact

    text = compact(_oa_handoff())
    assert "handoff from triage (transfer_to_refunds at step 0)" in text
