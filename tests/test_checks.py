"""@crashlab.check outcome oracles, refund_bot's oracles and its fault injections."""

from __future__ import annotations

import json

import pytest
from examples.refund_bot import agent as refund
from typer.testing import CliRunner

import crashlab
from crashlab.cli import app
from crashlab.verify import oracles
from tests.fakes import FakeOpenAI, refund_script, text_response, tool_call_response

runner = CliRunner()
ENV = {"COLUMNS": "220", "TERM": "xterm-256color", "NO_COLOR": "1"}


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    oracles.unregister("probe")


def _refund_run(script, msg="Refund my order A17", faults=()):
    client = crashlab.wrap(FakeOpenAI(script))
    refund.run_agent(msg, client=client, model="fake-model", faults=list(faults))
    return crashlab.load("last")


def _status(run):
    return {c.name: c.status for c in run.checks}


# --- refund_bot oracles --------------------------------------------------------------


def test_clean_refund_passes_every_check():
    run = _refund_run(refund_script())
    assert run.outcome == "pass" and run.outcome_reason == "4 check(s) passed"
    assert _status(run) == {
        "refund_amount_equals_order_total": "pass",
        "refund_within_policy": "pass",
        "no_unbacked_refund_claim": "pass",
        "within_step_budget": "pass",
    }


def test_reply_claiming_a_refund_that_never_happened_fails():
    script = [
        tool_call_response(1, "get_order", {"order_id": "A17"}),
        text_response(2, "I have issued a refund for your order A17. Allow 3-5 days."),
    ]
    run = _refund_run(script)
    assert _status(run)["no_unbacked_refund_claim"] == "fail"
    assert run.outcome == "fail"
    assert "step 2: the reply says 'I have issued a refund for your order A17.'" in (
        run.outcome_reason
    )


@pytest.mark.parametrize(
    "reply",
    [
        "I can process a refund for you once I have the order details.",
        "We can issue a refund if you confirm the order id.",
        "I'm sorry, I couldn't retrieve your order, so no refund has been issued yet.",
        "I can't process the refund amount as it's not available.",
    ],
)
def test_offers_and_negations_are_not_refund_claims(reply):
    run = _refund_run([tool_call_response(1, "get_order", {"order_id": "A17"}),
                       text_response(2, reply)])  # fmt: skip
    assert _status(run)["no_unbacked_refund_claim"] == "pass"


def test_wrong_amount_fails():
    run = _refund_run(refund_script(amount=30.0))
    assert run.outcome == "fail"
    assert _status(run)["refund_amount_equals_order_total"] == "fail"
    assert run.outcome_reason == (
        "check refund_amount_equals_order_total failed: step 3: refunded 30 for A17, but the "
        "order total is 49"
    )


def test_out_of_window_refund_violates_policy():
    script = [
        tool_call_response(1, "get_order", {"order_id": "C03"}),
        tool_call_response(2, "issue_refund", {"order_id": "C03", "amount": 120.0}),
        text_response(3, "Refunded."),
    ]
    run = _refund_run(script, "Refund order C03", faults=["policy_drop"])
    assert _status(run)["refund_within_policy"] == "fail"
    assert "delivered 45 days ago (window: 30 days)" in run.outcome_reason
    assert _status(run)["refund_amount_equals_order_total"] == "pass"


def test_refund_without_lookup_violates_policy():
    script = [tool_call_response(1, "issue_refund", {"order_id": "A17", "amount": 49.0}),
              text_response(2, "Refunded.")]  # fmt: skip
    run = _refund_run(script)
    assert "without looking it up first" in run.outcome_reason


def test_lookup_loop_blows_the_step_budget():
    script = [tool_call_response(i, "get_order", {"order_id": "A17"}) for i in range(1, 5)]
    run = _refund_run([*script, text_response(9, "Sorry, the system is down.")],
                      faults=["html_502"])  # fmt: skip
    assert run.usage.steps == 9
    assert _status(run) == {
        "refund_amount_equals_order_total": "pass",
        "refund_within_policy": "pass",
        "no_unbacked_refund_claim": "pass",
        "within_step_budget": "fail",
    }
    assert run.outcome_reason == "check within_step_budget failed: 9 steps, budget is 8"


def test_rejected_refund_attempt_is_not_a_refund():
    # issue_refund raises for amounts above the total; the attempt is an error, not a refund
    script = [tool_call_response(1, "get_order", {"order_id": "A17"}),
              tool_call_response(2, "issue_refund", {"order_id": "A17", "amount": 60.0}),
              text_response(3, "I could not refund that amount.")]  # fmt: skip
    run = _refund_run(script)
    assert run.tool_calls("issue_refund")[0].error is not None
    assert run.outcome == "pass"


# --- faults --------------------------------------------------------------------------


def test_schema_drift_renames_total():
    run = _refund_run(refund_script(), faults=["schema_drift"])
    order = run.tool_results("get_order")[0]
    assert "total" not in order and order["amount_total"] == 49.0
    assert list(order) == [("amount_total" if k == "total" else k) for k in refund.ORDERS["A17"]]
    assert run.input["faults"] == ["schema_drift"]


def test_policy_drop_removes_the_refund_window_line():
    prompt = refund.system_prompt(frozenset({"policy_drop"}))
    assert "30 days" not in prompt and "delivered" not in prompt
    assert "1. Always look up" in prompt and "3. Issue at most one refund" in prompt
    assert "4." not in prompt
    assert "30 days" in refund.SYSTEM_PROMPT


def test_faults_from_env_and_validation(monkeypatch):
    monkeypatch.setenv("FAULT_SCHEMA_DRIFT", "1")
    monkeypatch.setenv("FAULT_POLICY_DROP", "true")
    assert refund.faults_from_env() == ["schema_drift", "policy_drop"]
    with pytest.raises(ValueError, match="unknown faults"):
        refund.run_agent("x", client=crashlab.wrap(FakeOpenAI([])), faults=["nope"])


def test_main_records_model_provider_and_faults(monkeypatch, capsys):
    monkeypatch.setenv("FAULT_HTML_502", "1")
    fake = crashlab.wrap(FakeOpenAI(refund_script()))
    monkeypatch.setattr(refund, "make_client", lambda provider=None: fake)
    refund.main(["Refund A17", "--fault-schema-drift", "--model", "gpt-x"])
    run = crashlab.load("last")
    assert run.input["model"] == "gpt-x" and run.input["provider"] == "openai"
    assert run.input["faults"] == ["html_502", "schema_drift"]
    out = capsys.readouterr().out
    assert "[outcome: pass" in out and "crashlab replay last" in out


# --- check semantics -----------------------------------------------------------------


def _probe_run(**out):
    with crashlab.record("probe") as run:
        run.output = out
    return crashlab.load("last")


def test_check_return_values_and_errors():
    @crashlab.check("probe")
    def returns_none(run):
        pass

    @crashlab.check("probe")
    def returns_false(run):
        """Output must be polite."""
        return False

    @crashlab.check("probe", name="custom-name")
    def returns_message(run):
        return "too rude"

    @crashlab.check("probe")
    def raises(run):
        raise RuntimeError("oracle bug")

    run = _probe_run(text="hi")
    got = {c.name: (c.status, c.message) for c in run.checks}
    assert got == {
        "returns_none": ("pass", None),
        "returns_false": ("fail", "Output must be polite."),
        "custom-name": ("fail", "too rude"),
        "raises": ("error", "RuntimeError: oracle bug"),
    }
    assert run.outcome == "fail"
    assert run.outcome_reason == "check returns_false failed: Output must be polite."


def test_global_checks_apply_to_every_run_and_user_outcome_wins():
    @crashlab.check
    def everything_passes(run):
        return True

    try:
        with crashlab.record("probe") as run:
            run.outcome = "fail"
            run.outcome_reason = "marked by hand"
        stored = crashlab.load("last")
        assert stored.outcome == "fail" and stored.outcome_reason == "marked by hand"
        assert [c.name for c in stored.checks] == ["everything_passes"]
    finally:
        oracles.unregister(None, "everything_passes")


def test_no_checks_keeps_unknown():
    run = _probe_run()
    assert run.outcome == "unknown" and run.checks == []


def test_run_checks_on_a_loaded_run():
    run = _refund_run(refund_script(amount=30.0))
    results = crashlab.run_checks(run)
    assert [r.status for r in results] == ["fail", "pass", "pass", "pass"]


# --- CLI -----------------------------------------------------------------------------


def test_cli_check_updates_outcome():
    run = _refund_run(refund_script())
    # simulate a run recorded before the oracles existed
    run.outcome, run.outcome_reason, run.checks = "unknown", None, []
    crashlab.Store().update_run(run)
    res = runner.invoke(app, ["check", run.run_id], env=ENV)
    assert res.exit_code == 0, res.output
    assert "refund_within_policy" in res.output and "outcome: pass" in res.output
    assert crashlab.load(run.run_id).outcome == "pass"

    bad = _refund_run(refund_script(amount=30.0))
    res = runner.invoke(app, ["check", bad.run_id, "--no-save"], env=ENV)
    assert res.exit_code == 1 and "refunded 30 for A17" in res.output


def test_show_lists_checks():
    _refund_run(refund_script(amount=30.0))
    res = runner.invoke(app, ["show", "last"], env=ENV)
    assert res.exit_code == 0
    assert "checks" in res.output and "fail" in res.output
    assert "refund_amount_equals_order_total" in res.output
    data = json.loads(runner.invoke(app, ["show", "last", "--json"], env=ENV).output)
    assert data["checks"][0]["status"] == "fail"
