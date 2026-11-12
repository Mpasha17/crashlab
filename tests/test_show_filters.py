"""`crashlab show` step filters (--errors / --tools / --llm / --failing), the
first-failure line and --step detail."""

from __future__ import annotations

import json
import re

from typer.testing import CliRunner

from crashlab.cli import app
from crashlab.models import CheckResult, ErrorInfo
from crashlab.store import Store
from tests.debug_fixtures import chat, make_run, run_id, tool

runner = CliRunner()
ENV = {"COLUMNS": "200", "TERM": "xterm-256color", "NO_COLOR": "1"}
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def invoke(args, env=None):
    res = runner.invoke(app, args, env=env or ENV)
    return res.exit_code, _ANSI.sub("", res.output)


def _failing(s):
    tb = 'Traceback (most recent call last):\n  File "agent.py", line 9\nTimeoutError: slow'
    steps = [
        chat("x", 0, "Refund ORD-1", tool_call=("get_order", {"order_id": "ORD-1"})),
        tool("x", 1, "get_order", {"order_id": "ORD-1"}, {"total": 49}),
        chat("x", 2, None, tool_call=("issue_refund", {"amount": 999})),
        tool("x", 3, "issue_refund", {"amount": 999},
             error=ErrorInfo(type="TimeoutError", message="gateway slow", traceback=tb)),
        chat("x", 4, None, "Your refund is done."),
        tool("x", 5, "send_email", {"to": "c@x"}, {"ok": True}),
        chat("x", 6, None, "Bye."),
        chat("x", 7, None, "Bye again."),
    ]  # fmt: skip
    return make_run(
        s, 1, outcome="fail", steps=steps,
        checks=[CheckResult(name="refund_within_policy", status="fail", message="999 > 49")],
        diagnosis_label="BAD_TOOL_ARGUMENTS", diagnosis_step=2,
    )  # fmt: skip


def _idx_column(out):
    return [int(m.group(1)) for m in re.finditer(r"^│\s+(\d+) │", out, re.M)]


def test_filters_select_steps(crashlab_dir):
    _failing(Store())
    rid = run_id(1)
    assert _idx_column(invoke(["show", rid])[1]) == list(range(8))
    code, out = invoke(["show", rid, "--errors"])
    assert code == 0 and _idx_column(out) == [3]
    assert "showing 1 of 8 steps: steps with an error" in out
    assert _idx_column(invoke(["show", rid, "--tools"])[1]) == [1, 3, 5]
    assert _idx_column(invoke(["show", rid, "--llm"])[1]) == [0, 2, 4, 6, 7]
    assert _idx_column(invoke(["show", rid, "--tools", "--llm"])[1]) == list(range(8))
    code, out = invoke(["show", rid, "--failing"])
    assert _idx_column(out) == [1, 2, 3, 4, 5] and "around step 3" in out
    assert _idx_column(invoke(["show", rid, "--failing", "--tools"])[1]) == [1, 3, 5]


def test_failure_line_jumps_to_step_and_names_checks(crashlab_dir):
    _failing(Store())
    code, out = invoke(["show", "last"])
    assert code == 0
    line = next(ln for ln in out.splitlines() if ln.startswith("→ "))
    assert "first error: step 3 (tool issue_refund, called by chat step 2)" in line
    assert "checks failed: refund_within_policy" in line
    assert "diagnosis: step 2 BAD_TOOL_ARGUMENTS" in line
    assert f"open it: crashlab show {run_id(1)} --step 3" in out


def test_passing_run_has_no_failure_line_and_output_is_unchanged(crashlab_dir):
    make_run(Store(), 1)
    code, out = invoke(["show", "last"])
    assert code == 0 and "→ first" not in out and "showing" not in out
    assert "open it:" not in out


def test_filters_with_no_match_and_no_failing_step(crashlab_dir):
    make_run(Store(), 1)
    code, out = invoke(["show", "last", "--errors"])
    assert code == 0 and "No steps match the filter." in out
    code, out = invoke(["show", "last", "--failing"])
    assert "no failing step recorded" in out and _idx_column(out) == [0, 1, 2]


def test_failing_falls_back_to_the_diagnosis_step(crashlab_dir):
    make_run(Store(), 1, outcome="fail", diagnosis_label="TASK_SPEC_VIOLATION",
             diagnosis_step=2)  # fmt: skip
    code, out = invoke(["show", "last", "--failing"])
    assert "around step 2" in out and _idx_column(out) == [0, 1, 2]


def test_failing_uses_the_step_a_failed_check_cites(crashlab_dir):
    # new-user run: a check-only failure ("step 1: search_docs returned ...") showed every
    # step with "no failing step recorded" although the check names the step
    make_run(Store(), 1, outcome="fail", checks=[
        CheckResult(name="ok_check", status="pass", message="step 6: fine"),
        CheckResult(name="refund_within_policy", status="fail",
                    message="step 2: refunded C03, delivered 45 days ago"),
    ], steps=[chat(rid := run_id(1), i, "hi", "ok") for i in range(8)])  # fmt: skip
    code, out = invoke(["show", rid, "--failing"])
    assert "around step 2 (cited by check refund_within_policy)" in out, out
    assert _idx_column(out) == [0, 1, 2, 3, 4]


def test_failure_line_names_a_finding_and_the_exception(crashlab_dir):
    from crashlab.models import Finding

    make_run(Store(), 1, status="error", outcome="fail",
             error=ErrorInfo(type="crashlab.checks.engine.CircuitBreakerError", message="stop"),
             findings=[Finding(kind="budget", severity="critical", step=2,
                               message="over")])  # fmt: skip
    code, out = invoke(["show", "last"])
    line = next(ln for ln in out.splitlines() if ln.startswith("→ "))
    assert "first budget finding: step 2 (chat gpt-4o-mini)" in line
    assert "raised CircuitBreakerError" in line
    assert "open it: crashlab show" in out and "--step 2" in out


def test_json_respects_filters(crashlab_dir):
    _failing(Store())
    code, out = invoke(["show", "last", "--json", "--tools"])
    data = json.loads(out)
    assert [s["idx"] for s in data["steps"]] == [1, 3, 5]
    data = json.loads(invoke(["show", "last", "--json"])[1])
    assert len(data["steps"]) == 8


def test_step_detail(crashlab_dir):
    _failing(Store())
    code, out = invoke(["show", "last", "--step", "3"])
    assert code == 0, out
    assert "step 3 (tool issue_refund, called by chat step 2)" in out
    assert "TimeoutError: gateway slow" in out and "traceback" in out
    assert 'File "agent.py", line 9' in out and '"amount": 999' in out
    code, out = invoke(["show", "last", "--step", "0"])
    assert "gpt-4o-mini" in out and "100 in · 10 out" in out and "Refund ORD-1" in out
    assert "est. cost" in out and "You are a refund assistant" in out
    data = json.loads(invoke(["show", "last", "--step", "3", "--json"])[1])
    assert data["idx"] == 3 and data["error"]["type"] == "TimeoutError"
    code, out = invoke(["show", "last", "--step", "42"])
    assert code == 1 and "has no step 42 (steps: 0–7)" in out


def test_narrow_terminal_still_fits(crashlab_dir):
    _failing(Store())
    for args in (["show", "last"], ["show", "last", "--failing"], ["show", "last", "--step", "3"]):
        code, out = invoke(args, env={**ENV, "COLUMNS": "80"})
        assert code == 0 and max(len(ln) for ln in out.splitlines()) <= 80, args
