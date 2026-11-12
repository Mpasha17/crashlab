"""A @crashlab.check that crashes on a programming error (found on a live OpenAI Agents
SDK run: the oracle used ``step.arguments`` instead of ``step.args``) is a bug in the
check, not evidence against the agent. The heuristic diagnoser blamed the agent's chat
step with TASK_SPEC_VIOLATION; it must say the check is broken instead."""

from __future__ import annotations

import pytest

from crashlab.diagnose.compact import compact
from crashlab.diagnose.heuristics import heuristic_raw, is_broken_check
from crashlab.models import CheckResult, Run, Step


def _run(checks: list[CheckResult]) -> Run:
    steps = [
        Step(run_id="r", idx=0, span_id="a", operation="chat", name="m", request_hash="h",
             response={"tool_calls": [{"id": "c1", "name": "issue_refund",
                                       "arguments": {"order_id": "A2", "amount": 15.0}}]}),
        Step(run_id="r", idx=1, span_id="b", operation="execute_tool", name="issue_refund",
             request_hash="h", request={"args": {"order_id": "A2", "amount": 15.0}},
             response={"result": {"refunded": 15.0}}, attributes={"gen_ai.tool.call.id": "c1"}),
        Step(run_id="r", idx=2, span_id="c", operation="chat", name="m", request_hash="h",
             response={"messages": [{"role": "assistant", "content": "Refunded 15."}]}),
    ]  # fmt: skip
    return Run(run_id="r", name="x", status="ok", outcome="fail", checks=checks, steps=steps)


BROKEN = [
    "AttributeError: 'Step' object has no attribute 'arguments'",
    "AttributeError: 'Run' object has no attribute 'tool_call'",
    "NameError: name 'ORDERS' is not defined",
    "ModuleNotFoundError: No module named 'myapp'",
    "ImportError: cannot import name 'x' from 'y'",
    "UnboundLocalError: cannot access local variable 'o' where it is not associated with a value",
]


@pytest.mark.parametrize("msg", BROKEN)
def test_broken_check_is_not_blamed_on_the_agent(msg):
    out = heuristic_raw(_run([CheckResult(name="refund_ok", status="error", message=msg)]))
    assert out["label"] == "UNKNOWN"
    assert out["confidence"] == 0.0
    assert "refund_ok" in out["root_cause"] and "check itself" in out["root_cause"]


@pytest.mark.parametrize(
    "msg",
    [
        "KeyError: 'total'",  # the agent's output lacks a field: may be the agent's fault
        "JSONDecodeError: Expecting value: line 1 column 1 (char 0)",
        "AttributeError: 'NoneType' object has no attribute 'get'",
    ],
)
def test_data_errors_in_a_check_still_count_against_the_run(msg):
    assert not is_broken_check(CheckResult(name="c", status="error", message=msg))


def test_a_failing_check_beside_a_broken_one_still_wins():
    out = heuristic_raw(_run([
        CheckResult(name="refund_ok", status="error", message=BROKEN[0]),
        CheckResult(name="limit", status="fail", message="refunded 15.0 above the limit (step 1)"),
    ]))  # fmt: skip
    assert out["label"] == "TASK_SPEC_VIOLATION"
    assert out["failing_step"] == 0


def test_diagnoser_view_marks_the_broken_check():
    view = compact(_run([CheckResult(name="refund_ok", status="error", message=BROKEN[0])]))
    assert "ERROR refund_ok" in view
    assert "the check itself crashed" in view
