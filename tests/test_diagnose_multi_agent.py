"""Heuristic diagnose on multi-agent runs: blame the reply a check quotes, and name the
field a quote really came from.

Live findings (nov13):
- Haystack orchestrator -> research sub-agent: the research tool returned "7 October
  2025"; the orchestrator's final answer said "July 7, 2025". The check quoted the final
  answer; the heuristic blamed the lookup tool step (its result shares the numbers).
- AutoGen writer + critic: the critic's system prompt says "reply APPROVE" and its reply
  is "APPROVE"; the heuristic quote "APPROVE" was labelled request.system."""

from __future__ import annotations

from typing import Any

import crashlab
from crashlab.diagnose.evidence import validate
from crashlab.diagnose.heuristics import heuristic_raw
from crashlab.models import CheckResult


def _chat(idx: int, text: str | None = None, *, system: str = "", agent: str | None = None,
          calls: list[dict[str, Any]] | None = None) -> crashlab.Step:  # fmt: skip
    resp: dict[str, Any] = {"messages": [], "finish_reason": "stop"}
    if text is not None:
        resp["messages"] = [{"role": "assistant", "parts": [{"type": "text", "content": text}]}]
    if calls:
        resp["tool_calls"] = calls
        resp["finish_reason"] = "tool_calls"
    attrs = {"gen_ai.agent.name": agent} if agent else {}
    return crashlab.Step(
        run_id="r", idx=idx, span_id=f"s{idx}", operation="chat", name="m", request_hash="h",
        request={"system": system, "messages": [{"role": "user", "content": "go"}]},
        response=resp, attributes=attrs,
    )  # fmt: skip


def _tool(idx: int, name: str, args: dict[str, Any], result: Any, agent: str | None = None,
          call_id: str | None = None) -> crashlab.Step:  # fmt: skip
    attrs: dict[str, Any] = {"gen_ai.agent.name": agent} if agent else {}
    if call_id:
        attrs["gen_ai.tool.call.id"] = call_id
    return crashlab.Step(
        run_id="r", idx=idx, span_id=f"t{idx}", operation="execute_tool", name=name,
        request_hash="h", request={"args": args}, response={"result": result}, attributes=attrs,
    )  # fmt: skip


WRONG = "The latest Python release is 3.14.0, released on July 7, 2025."
FACT = "Python 3.14.0 was released on 7 October 2025."


def _haystack_run(*, research_reply: str = FACT, final: str = WRONG) -> crashlab.Run:
    steps = [
        _chat(
            0,
            agent="orchestrator",
            calls=[{"id": "c1", "name": "research", "arguments": '{"messages": "latest python"}'}],
        ),  # fmt: skip
        _tool(
            1,
            "research",
            {"messages": "latest python"},
            {"last_message": research_reply},
            agent="orchestrator",
            call_id="c1",
        ),
        _chat(
            2,
            agent="research",
            calls=[{"id": "c2", "name": "lookup_release", "arguments": '{"project": "python"}'}],
        ),  # fmt: skip
        _tool(3, "lookup_release", {"project": "python"}, FACT, agent="research", call_id="c2"),
        _chat(4, research_reply, agent="research"),
        _chat(5, final, agent="orchestrator"),
    ]
    return crashlab.Run(
        run_id="r", name="hs", steps=steps, output=final, outcome="fail",
        checks=[CheckResult(name="answer_has_date", status="fail", message=final)],
    )  # fmt: skip


def test_check_quoting_the_final_answer_blames_the_reply_not_the_lookup():
    raw = heuristic_raw(_haystack_run())
    assert raw["failing_step"] == 5, raw
    assert "orchestrator" in raw["root_cause"]


def test_wrong_text_first_written_by_the_sub_agent_blames_the_sub_agent():
    wrong_fact = "Python 3.14.0 was released on July 7, 2025."
    run = _haystack_run(research_reply=wrong_fact, final=wrong_fact)
    raw = heuristic_raw(run)
    assert raw["failing_step"] == 4, raw
    assert "research" in raw["root_cause"]


def test_unlabelled_run_gets_the_same_step():
    run = _haystack_run()
    for s in run.steps:
        s.attributes.pop("gen_ai.agent.name", None)
    assert heuristic_raw(run)["failing_step"] == 5


def test_short_or_unquoted_check_messages_keep_the_old_rule():
    run = _haystack_run()
    run.checks = [CheckResult(name="answer_has_date", status="fail", message="date 7 lost")]
    assert heuristic_raw(run)["failing_step"] == 3  # number-matching tool rule, unchanged


def _autogen_run() -> crashlab.Run:
    sys_critic = "You are a critic. Reply APPROVE if the haiku is good, else give feedback."
    steps = [
        _chat(
            0,
            agent="writer",
            calls=[{"id": "c1", "name": "word_count", "arguments": '{"text": "Screen glows"}'}],
        ),  # fmt: skip
        _tool(1, "word_count", {"text": "Screen glows"}, 14, agent="writer", call_id="c1"),
        _chat(2, "APPROVE", system=sys_critic, agent="critic"),
    ]
    return crashlab.Run(
        run_id="r", name="ag", steps=steps, outcome="fail",
        checks=[CheckResult(name="critic_reviewed_a_haiku", status="fail",
                            message="critic at step 2 judged '14', not a haiku")],
    )  # fmt: skip


def test_heuristic_quote_is_attributed_to_the_reply_not_the_system_prompt():
    run = _autogen_run()
    d = crashlab.diagnose(run, heuristic=True, save=False, cache=False)
    assert d.failing_step == 2
    assert [(e.step_idx, e.field, e.quote) for e in d.evidence] == [(2, "response.text", "APPROVE")]


def test_validate_honours_a_claimed_field_only_when_the_quote_is_there():
    run = _autogen_run()
    raw = {"label": "TASK_SPEC_VIOLATION", "failing_step": 2, "confidence": 0.9,
           "root_cause": "x", "evidence": [
               {"step": 2, "quote": "APPROVE", "field": "response.text"},
               {"step": 2, "quote": "APPROVE", "field": "request.messages[0]"},
               {"step": 2, "quote": "APPROVE"}]}  # fmt: skip
    d = validate(raw, run)
    assert [e.field for e in d.evidence] == ["response.text", "request.system", "request.system"]


def test_empty_reply_outranks_a_vague_check_on_another_step():
    # LlamaIndex AgentWorkflow (gpt-oss): the research agent's last turn was reasoning only,
    # no handoff; the checks said "date lost: ''" / "agents: ['research']" and the
    # heuristic blamed step 0
    from crashlab.checks import ChecksConfig, evaluate

    reasoning = {"type": "reasoning", "content": "So we should hand off to WriteAgent."}
    last = _chat(2, agent="research")
    last.response = {"messages": [{"role": "assistant", "parts": [reasoning]}],
                     "finish_reason": "stop"}  # fmt: skip
    last.request = {"messages": [{"role": "user", "content": "go"},
                                 {"role": "tool", "content": FACT}]}  # fmt: skip
    steps = [
        _chat(
            0,
            agent="research",
            calls=[{"id": "c1", "name": "lookup_release", "arguments": '{"project": "Python"}'}],
        ),  # fmt: skip
        _tool(1, "lookup_release", {"project": "Python"}, FACT, agent="research", call_id="c1"),
        last,
    ]
    run = crashlab.Run(
        run_id="r", name="li", steps=steps, output="", outcome="fail",
        checks=[CheckResult(name="report_keeps_date", status="fail", message="date lost: ''"),
                CheckResult(name="research_then_write", status="fail",
                            message="agents: ['research']")],
    )  # fmt: skip
    run.findings = evaluate(run, ChecksConfig())
    raw = heuristic_raw(run)
    assert raw["failing_step"] == 2 and "empty_reply" in raw["root_cause"], raw
    assert raw["label"] == "TASK_SPEC_VIOLATION"
