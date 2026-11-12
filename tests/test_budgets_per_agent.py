"""within_budget(..., agent=) / summarize_usage(run, agent=): budgets for one agent of a
multi-agent run (a researcher that may spend 2 LLM calls while the writer gets 1)."""

from __future__ import annotations

from typing import Any

import pytest

import crashlab
from crashlab.budgets import summarize_usage, within_budget


def _chat(idx: int, agent: str | None, inp: int, out: int, calls: list[str] = ()) -> Any:
    attrs: dict[str, Any] = {"gen_ai.usage.input_tokens": inp, "gen_ai.usage.output_tokens": out,
                             "crashlab.usage.cost_usd": (inp + out) / 1e6}  # fmt: skip
    if agent:
        attrs["gen_ai.agent.name"] = agent
    resp = {"tool_calls": [{"id": c, "name": "lookup"} for c in calls]}
    return crashlab.Step(run_id="r", idx=idx, span_id=f"s{idx}", operation="chat", name="m",
                         request_hash="h", request={"messages": [{"role": "user",
                         "content": f"{agent}-{idx}"}]}, response=resp, attributes=attrs,
                         duration_ms=100)  # fmt: skip


def _tool(idx: int, agent: str | None, call: str) -> Any:
    attrs: dict[str, Any] = {"gen_ai.tool.call.id": call}
    if agent:
        attrs["gen_ai.agent.name"] = agent
    return crashlab.Step(run_id="r", idx=idx, span_id=f"t{idx}", operation="execute_tool",
                         name="lookup", request_hash="h", request={"args": {}},
                         response={"result": 1}, attributes=attrs, duration_ms=50)  # fmt: skip


def _run() -> crashlab.Run:
    return crashlab.Run(run_id="r", name="t", steps=[
        _chat(0, "researcher", 100, 10, ["c1"]), _tool(1, "researcher", "c1"),
        _chat(2, "researcher", 300, 20), _chat(3, "writer", 1000, 200),
    ])  # fmt: skip


def test_usage_of_one_agent():
    u = summarize_usage(_run(), agent="researcher")
    assert (u.steps, u.llm_calls, u.tool_calls) == (3, 2, 1)
    assert (u.input_tokens, u.output_tokens, u.total_tokens) == (400, 30, 430)
    assert u.cost_usd == pytest.approx(430 / 1e6)
    assert summarize_usage(_run()).total_tokens == 1630  # whole run unchanged


def test_within_budget_per_agent():
    run = _run()
    assert within_budget(run, agent="researcher", max_tokens=500, max_llm_calls=2) is None
    msg = within_budget(run, agent="writer", max_tokens=500)
    assert msg == "budget (agent writer): 1,200 tokens > max_tokens 500"
    assert within_budget(run, max_tokens=500).startswith("budget: 1,630 tokens")


def test_unknown_agent_fails_and_names_the_agents():
    msg = within_budget(_run(), agent="critic", max_tokens=10)
    assert msg == "budget (agent critic): no steps of agent critic (agents: researcher, writer)"


def test_run_wall_time_is_not_an_agent_budget():
    with pytest.raises(ValueError, match="max_seconds"):
        within_budget(_run(), agent="writer", max_seconds=10)
    # the slowest single call of that agent is fine
    assert within_budget(_run(), agent="writer", max_call_seconds=1) is None


def test_agent_instance_labels_work():
    def m(idx: int, msgs: list[dict[str, Any]], calls=(), toks=100) -> Any:
        s = _chat(idx, "researcher", toks, 0, list(calls))
        s.request = {"messages": msgs}
        return s

    a0, b0 = [{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]
    tail = lambda c: [{"role": "assistant", "tool_calls": [{"id": c}]},  # noqa: E731
                      {"role": "tool", "tool_call_id": c, "content": "1"}]  # fmt: skip
    run = crashlab.Run(run_id="r", name="t", steps=[
        m(0, a0, ["ca"]), m(1, b0, ["cb"], toks=50), _tool(2, "researcher", "ca"),
        m(3, a0 + tail("ca")), _tool(4, "researcher", "cb"), m(5, b0 + tail("cb"), toks=50),
    ])  # fmt: skip
    assert summarize_usage(run, agent="researcher#2").input_tokens == 100
    assert summarize_usage(run, agent="researcher#1").input_tokens == 200
