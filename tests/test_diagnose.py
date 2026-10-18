"""crashlab.diagnose: validation and downgrade paths, caching, the separate diagnoser
store, heuristics, compaction and the CLI. Offline: the diagnoser is a fake."""

from __future__ import annotations

import json

import pytest
from examples.refund_bot import agent as refund
from typer.testing import CliRunner

import crashlab
from crashlab.cli import app
from crashlab.diagnose import (
    DiagnoseError,
    candidates,
    compact,
    find_quote,
    heuristic_raw,
    parse_json,
    validate,
)
from crashlab.store import Store
from tests.fakes import FakeAPIError, FakeOpenAI, refund_script, text_response, tool_call_response

runner = CliRunner()
UNKNOWN = crashlab.TaxonomyLabel.UNKNOWN


def _policy_run() -> crashlab.Run:
    """Refunds C03, delivered 45 days ago: steps chat, get_order, chat, issue_refund, chat."""
    script = [
        tool_call_response(1, "get_order", {"order_id": "C03"}),
        tool_call_response(2, "issue_refund", {"order_id": "C03", "amount": 120.0}),
        text_response(3, "Your refund for order C03 has been issued."),
    ]
    client = crashlab.wrap(FakeOpenAI(script))
    refund.run_agent("Refund order C03", client=client, model="fake-model", faults=["policy_drop"])
    return crashlab.load("last")


def _clean_run() -> crashlab.Run:
    client = crashlab.wrap(FakeOpenAI(refund_script()))
    refund.run_agent("Refund my order A17", client=client, model="fake-model")
    return crashlab.load("last")


GOOD = {
    "label": "TASK_SPEC_VIOLATION",
    "failing_step": 3,
    "root_cause": "The agent refunded an order delivered 45 days ago, outside the 30-day window.",
    "evidence": [
        {"step": 1, "quote": '"days_since_delivery": 45'},
        {"step": 3, "quote": '"order_id": "C03"'},
    ],
    "secondary": ["UNKNOWN", "BAD_TOOL_ARGUMENTS"],
    "confidence": 0.9,
}


def _with(**changes):
    return {**GOOD, **changes}


# --- validation ----------------------------------------------------------------------


def test_valid_answer_is_kept():
    run = _policy_run()
    d = validate(GOOD, run)
    assert d.label is crashlab.TaxonomyLabel.TASK_SPEC_VIOLATION and d.failing_step == 3
    assert d.step_window == (1, 3) and d.confidence == 0.9 and d.is_failure
    assert [(e.step_idx, e.field) for e in d.evidence] == [(1, "result"), (3, "args")]
    assert [s.value for s in d.secondary] == ["BAD_TOOL_ARGUMENTS"] and d.notes == []


def test_label_outside_the_taxonomy_becomes_unknown():
    d = validate(_with(label="POLICY_BREACH"), _policy_run())
    assert d.label is UNKNOWN and d.failing_step is None and d.confidence <= 0.2
    assert "'POLICY_BREACH' is not in the taxonomy" in d.notes[0]


def test_label_is_parsed_leniently():
    d = validate(_with(label="task spec violation"), _policy_run())
    assert d.label is crashlab.TaxonomyLabel.TASK_SPEC_VIOLATION


@pytest.mark.parametrize("step", [None, 42, "three"])
def test_missing_or_nonexistent_step_becomes_unknown(step):
    d = validate(_with(failing_step=step), _policy_run())
    assert d.label is UNKNOWN and d.failing_step is None
    assert any("needs a failing step" in n for n in d.notes)
    assert d.secondary[0] is crashlab.TaxonomyLabel.TASK_SPEC_VIOLATION
    assert d.method["raw_failing_step"] == step


def test_hallucinated_quotes_are_dropped_and_the_label_downgraded():
    raw = _with(
        evidence=[
            {"step": 3, "quote": "the customer is a VIP, refund anyway"},  # never said
            {"step": 3, "quote": '"days_since_delivery": 45'},  # real, but in step 1
            {"step": 9, "quote": "Refund order C03"},  # no such step
        ]
    )
    d = validate(raw, _policy_run())
    assert d.label is UNKNOWN and d.evidence == []
    assert sum("dropped quote" in n for n in d.notes) == 3
    assert "does not exist" in d.notes[2] and d.notes[-1].startswith("downgraded to UNKNOWN")
    assert d.method["evidence_total"] == 3 and d.method["evidence_valid"] == 0


def test_one_good_quote_is_enough_and_bad_ones_are_noted():
    raw = _with(evidence=[{"step": 3, "quote": "made up"}, {"step": 1, "quote": "Running shoes"}])
    d = validate(raw, _policy_run())
    assert d.label is crashlab.TaxonomyLabel.TASK_SPEC_VIOLATION and len(d.evidence) == 1
    assert d.notes == ["dropped quote 'made up': not found verbatim in step 3"]


def test_no_failure_on_a_failed_run_becomes_unknown():
    run = _policy_run()
    assert run.outcome == "fail"
    d = validate({"label": "NO_FAILURE", "failing_step": None, "confidence": 0.8}, run)
    assert d.label is UNKNOWN and "contradicts the run's outcome" in d.notes[0]


def test_no_failure_on_a_clean_run_needs_no_evidence():
    d = validate({"label": "NO_FAILURE", "failing_step": 2, "confidence": 0.8}, _clean_run())
    assert d.label is crashlab.TaxonomyLabel.NO_FAILURE and d.failing_step is None
    assert not d.is_failure


def test_quotes_survive_typographic_and_ellipsis_differences():
    run = _policy_run()
    step4 = run.steps[4]
    assert find_quote(step4, "…has been issued...") == "response.text"
    assert find_quote(step4, "Your  refund\nfor order C03") == "response.text"
    assert find_quote(run.steps[1], "\u201cdays_since_delivery\u201d: 45") == "result"
    assert find_quote(run.steps[1], '\\"days_since_delivery\\": 45') == "result"
    assert find_quote(step4, "your refund") is None  # case matters
    assert find_quote(step4, "..") is None  # too short


def test_non_object_answer_is_unknown():
    d = validate(["TASK_SPEC_VIOLATION"], _policy_run())
    assert d.label is UNKNOWN and "did not return a JSON object" in d.notes[0]


def test_parse_json_variants():
    assert parse_json('{"label": "X"}') == {"label": "X"}
    assert parse_json('Sure!\n```json\n{"label": "X"}\n```') == {"label": "X"}
    assert parse_json('<think>hmm {"a": 1}</think> answer: {"label": "X"} done') == {"label": "X"}
    assert parse_json("no json here") is None and parse_json("[1, 2]") is None


# --- diagnose() ------------------------------------------------------------------------


def test_diagnose_with_a_callable_stores_and_caches(crashlab_dir):
    run = _policy_run()
    calls = []

    def fake(messages):
        calls.append(messages)
        return json.dumps(GOOD)

    d = crashlab.diagnose(run, fn=fake)
    assert d.label is crashlab.TaxonomyLabel.TASK_SPEC_VIOLATION and d.model == "custom"
    assert d.method["backend"] == "fn" and d.method["cached"] is False
    system, user = calls[0][0]["content"], calls[0][1]["content"]
    assert "TASK_SPEC_VIOLATION" in system and "STEPS:" in user
    assert "FAIL refund_within_policy: step 3" in user

    stored = crashlab.load(run.run_id)
    assert stored.diagnosis is not None and stored.diagnosis.failing_step == 3
    again = crashlab.diagnose(run.run_id, fn=fake)
    assert len(calls) == 1 and again.method["cached"] is True
    crashlab.diagnose(run.run_id, fn=fake, cache=False)
    assert len(calls) == 2
    assert len(Store().index.diagnoses(run.run_id)) == 2


def test_cache_is_invalidated_when_the_trace_changes():
    run = _policy_run()
    calls = []
    fake = lambda m: calls.append(m) or json.dumps(GOOD)  # noqa: E731
    crashlab.diagnose(run, fn=fake)
    crashlab.diagnose(run, fn=fake, hide_input=["faults"])  # different prompt
    crashlab.diagnose(run, fn=fake, model="other-model")  # different model
    assert len(calls) == 3


def test_non_json_reply_becomes_unknown():
    d = crashlab.diagnose(_policy_run(), fn=lambda m: "I think the agent broke the policy.")
    assert d.label is UNKNOWN
    assert d.method["raw_reply"] == "I think the agent broke the policy."


def test_diagnoser_calls_go_to_their_own_store(crashlab_dir):
    run = _policy_run()
    before = [r["run_id"] for r in Store().list_runs(limit=None)]
    fake = FakeOpenAI([text_response(1, json.dumps(GOOD), model="judge-model")])
    d = crashlab.diagnose(run, "judge-model", client=fake)
    assert d.label is crashlab.TaxonomyLabel.TASK_SPEC_VIOLATION and d.model == "judge-model"
    assert fake.calls[0]["response_format"] == {"type": "json_object"}
    assert fake.calls[0]["temperature"] == 0
    assert [r["run_id"] for r in Store().list_runs(limit=None)] == before  # user's ls

    side = Store(crashlab_dir / "diagnoser")
    [row] = side.list_runs(limit=None)
    assert row["name"] == "crashlab-diagnoser" and row["run_id"] == d.method["diagnoser_run_id"]
    assert d.method["input_tokens"] == 200 and d.method["output_tokens"] == 20
    assert side.load_run(row["run_id"]).input["run_id"] == run.run_id


def test_response_format_rejection_is_retried_without_it():
    fake = FakeOpenAI(
        [FakeAPIError("response_format is not supported", 400), text_response(1, json.dumps(GOOD))]
    )
    d = crashlab.diagnose(_policy_run(), "m", client=fake)
    assert d.label is crashlab.TaxonomyLabel.TASK_SPEC_VIOLATION
    assert "response_format" in fake.calls[0] and "response_format" not in fake.calls[1]


def test_api_failure_raises_diagnose_error():
    fake = FakeOpenAI([FakeAPIError("rate limited", 429)])
    with pytest.raises(DiagnoseError, match="rate limited"):
        crashlab.diagnose(_policy_run(), "m", client=fake)


def test_needs_a_model(monkeypatch):
    monkeypatch.delenv("CRASHLAB_DIAGNOSER_MODEL", raising=False)
    with pytest.raises(DiagnoseError, match="no diagnoser model"):
        crashlab.diagnose(_policy_run())


def test_refuses_to_run_inside_an_active_run():
    run = _policy_run()
    with crashlab.record("outer"), pytest.raises(DiagnoseError, match="active crashlab run"):
        crashlab.diagnose(run, fn=lambda m: "{}")


def test_unsaved_run_object_is_not_written_to_the_store():
    run = _policy_run()
    run.run_id = "01NOTSTORED0000000000000000"
    d = crashlab.diagnose(run, fn=lambda m: json.dumps(GOOD))
    assert run.diagnosis is d
    assert Store().index.get_run_json(run.run_id) is None


# --- heuristics ------------------------------------------------------------------------


def test_heuristic_baseline():
    d = crashlab.diagnose(_policy_run(), heuristic=True)
    assert d.label is crashlab.TaxonomyLabel.TASK_SPEC_VIOLATION and d.failing_step == 3
    assert d.model == "heuristic" and d.evidence
    assert crashlab.diagnose(_clean_run(), heuristic=True).label.value == "NO_FAILURE"


def test_heuristic_unbacked_claim_points_at_the_reply():
    script = [
        tool_call_response(1, "get_order", {"order_id": "A17"}),
        text_response(2, "I have issued a refund for your order A17."),
    ]
    refund.run_agent("Refund A17", client=crashlab.wrap(FakeOpenAI(script)), model="fake-model")
    run = crashlab.load("last")
    raw = heuristic_raw(run)
    assert raw["failing_step"] == 2 and raw["label"] == "TASK_SPEC_VIOLATION"


def test_candidates_for_a_lookup_loop():
    script = [tool_call_response(i, "get_order", {"order_id": "A17"}) for i in range(1, 5)]
    client = crashlab.wrap(FakeOpenAI([*script, text_response(9, "Sorry, the system is down.")]))
    refund.run_agent("Refund A17", client=client, model="fake-model", faults=["html_502"])
    cands = candidates(crashlab.load("last"))
    reasons = " | ".join(c.reason for c in cands)
    assert "step_repetition" in reasons and "within_step_budget" in reasons
    assert any(c.label is crashlab.TaxonomyLabel.TOOL_ERROR_MISHANDLED for c in cands)


# --- compaction ------------------------------------------------------------------------


def test_compact_shows_what_is_new_per_step_and_hides_inputs():
    run = _policy_run()
    trace = compact(run, candidates=candidates(run), hide_input=["faults"])
    assert '"faults"' not in trace and "Refund order C03" in trace
    assert "SYSTEM PROMPT" in trace and "- issue_refund(order_id: string, amount: number)" in trace
    assert "[3] tool issue_refund" in trace and "(candidate step)" in trace
    assert trace.count("new user message: Refund order C03") == 1  # only at the first call
    assert "(+1 tool result message(s): shown at the tool steps)" in trace


def test_compact_truncates_with_a_marker():
    run = _policy_run()
    trace = compact(run, max_field=20)
    assert "…[truncated: the step 1 result is" in trace
    assert "only the part above can be quoted]" in trace


# --- CLI -------------------------------------------------------------------------------


def test_cli_diagnose_json_and_show():
    run = _policy_run()
    res = runner.invoke(app, ["diagnose", run.run_id, "--heuristic", "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["label"] == "TASK_SPEC_VIOLATION" and data["failing_step"] == 3
    shown = runner.invoke(app, ["show", run.run_id])
    assert "diagnosis" in shown.output and "TASK_SPEC_VIOLATION" in shown.output
    as_json = json.loads(runner.invoke(app, ["show", run.run_id, "--json"]).output)
    assert as_json["diagnosis"]["label"] == "TASK_SPEC_VIOLATION"


def test_cli_show_prompt_makes_no_call():
    _policy_run()
    res = runner.invoke(app, ["diagnose", "last", "--show-prompt", "--hide-input", "faults"])
    assert res.exit_code == 0 and "STEPS:" in res.output and '"faults"' not in res.output
    assert crashlab.load("last").diagnosis is None


def test_cli_api_key_env_must_be_set(monkeypatch):
    _policy_run()
    monkeypatch.delenv("NO_SUCH_KEY_VAR", raising=False)
    res = runner.invoke(app, ["diagnose", "--model", "m", "--api-key-env", "NO_SUCH_KEY_VAR"])
    assert res.exit_code == 2 and "NO_SUCH_KEY_VAR is not set" in res.output


def test_cli_without_model_explains(monkeypatch):
    _policy_run()
    monkeypatch.delenv("CRASHLAB_DIAGNOSER_MODEL", raising=False)
    res = runner.invoke(app, ["diagnose"])
    assert res.exit_code == 1 and "no diagnoser model" in res.output
