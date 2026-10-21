"""crashlab.propose_fixes / try_fix: parsing, validation, the leakage guard, storage and
the one-shot fork. Offline: the fix model is a fake."""

from __future__ import annotations

import json

import pytest

import crashlab
from crashlab.diagnose import DiagnoseError, instance_literals, validate, validate_fix
from tests.fakes import FakeOpenAI, text_response, tool_call_response
from tests.test_diagnose import GOOD, _clean_run, _policy_run

RULE = {
    "kind": "system_prompt",
    "op": "insert_line",
    "after": "3. Issue at most one refund per order.",
    "text": "4. Only refund delivered orders, within 30 days of delivery.",
}


def _diagnosed():
    run = _policy_run()
    crashlab.diagnose(run, fn=lambda m: json.dumps(GOOD))
    return crashlab.load(run.run_id)


def _reply(*patches, **extra):
    fixes = [
        {"patch": p, "rationale": "the 30-day rule is missing", "expected_effect": "refuses"}
        for p in patches
    ]
    for f in fixes:
        f.update(extra)
    return lambda m: json.dumps({"fixes": fixes})


def test_good_fix_is_kept_saved_and_usable_by_fork(crashlab_dir):
    run = _diagnosed()
    calls = []
    fixes = crashlab.propose_fixes(run, fn=lambda m: calls.append(m) or _reply(RULE)(m))
    assert [f.kind for f in fixes] == ["system_prompt"] and fixes[0].target == "system prompt"
    assert fixes[0].rank == 1 and fixes[0].expected_effect == "refuses"
    assert "TASK_SPEC_VIOLATION" in calls[0][1]["content"] and "STEPS:" in calls[0][1]["content"]
    stored = crashlab.load(run.run_id).diagnosis
    assert stored.fixes[0].as_patch.text.startswith("4. Only refund")
    assert stored.fix_method["kept"] == 1 and stored.label.value == "TASK_SPEC_VIOLATION"
    patchset = crashlab.patch.load(stored.fixes[0].file)
    assert patchset.patches[0] == stored.fixes[0].as_patch
    again = crashlab.propose_fixes(crashlab.load(run.run_id), fn=lambda m: calls.append(m))
    assert len(calls) == 1 and again[0].as_patch == fixes[0].as_patch  # cached


def test_instance_literals_are_the_task_data_not_the_policy():
    lits = instance_literals(_policy_run())
    assert {"C03", "120", "45"} <= lits
    assert "30" not in lits  # not in this run's data (the window is policy)


@pytest.mark.parametrize(
    ("patch", "reason"),
    [
        ({"kind": "rewrite_agent"}, "is not one of"),
        ({"kind": "system_prompt", "op": "insert_line", "text": "x"}, "does not parse"),
        ({**RULE, "text": "4. Never refund order C03."}, "leaks this run's specifics ['C03']"),
        ({**RULE, "text": "4. Refund at most 120 dollars."}, "leaks this run's specifics ['120']"),
        ({**RULE, "after": "no such line"}, "does not apply to the recorded run"),
        ({"kind": "system_prompt", "op": "replace", "text": "Be good."}, "not minimal"),
        ({**RULE, "text": "Be careful. " * 80}, "not minimal"),
        ({**RULE, "text": "a\nb\nc\nd\ne"}, "more than 4 new lines"),
        ({"kind": "tool_description", "tool": "cancel_order", "description": "x"}, "no tool"),
        ({"kind": "message", "step": 1, "index": 0, "content": "x"}, "not an LLM call"),
        ({"kind": "tool_result", "step": 42, "result": {}}, "no step 42"),
    ],
)
def test_bad_proposals_are_rejected(patch, reason):
    fix, why = validate_fix({"patch": patch}, _policy_run(), 1)
    assert fix is None and reason in why


def test_a_justified_literal_is_kept_with_a_note():
    item = {
        "patch": {**RULE, "text": "4. Orders delivered 45 or more days ago are final."},
        "literal_justification": "45 days is the shop's published cut-off",
    }
    fix, why = validate_fix(item, _policy_run(), 1)
    assert why is None and "justified" in fix.notes[0]


def test_tool_result_counterfactual_is_test_only():
    patch = {"kind": "tool_result", "step": 1, "result": {"status": "delivered"}}
    fix, _ = validate_fix({"patch": patch}, _policy_run(), 1)
    assert fix.test_only and fix.target == "step 1" and "not a fix" in fix.notes[0]


def test_tool_description_fix_is_kept():
    patch = {
        "kind": "tool_description",
        "tool": "issue_refund",
        "op": "append",
        "description": "Only for delivered orders inside the refund window.",
    }
    fix, why = validate_fix({"patch": patch}, _policy_run(), 2)
    assert why is None and fix.target == "issue_refund" and fix.fix_id == "fix-2"


def test_mixed_reply_keeps_valid_ranks_and_explains_rejections():
    leaky = {**RULE, "text": "4. Never refund order C03."}
    other = {**RULE, "text": "4. Refuse refunds for orders delivered over a month ago."}
    run = _diagnosed()
    fixes = crashlab.propose_fixes(run, fn=_reply(leaky, RULE, RULE, other, {"kind": "x"}))
    assert [f.fix_id for f in fixes] == ["fix-1", "fix-2"]
    notes = run.diagnosis.fix_notes
    assert any("leaks" in n for n in notes) and any("duplicate" in n for n in notes)
    assert any("'x' is not one of" in n for n in notes)


def test_at_most_three_fixes():
    texts = [f"4. Rule {w}: only refund recent orders." for w in "abcde"]
    run = _diagnosed()
    fixes = crashlab.propose_fixes(run, fn=_reply(*[{**RULE, "text": t} for t in texts]))
    assert len(fixes) == 3 and "kept the first 3" in run.diagnosis.fix_notes[-1]


def test_non_json_reply_gives_no_fixes():
    run = _diagnosed()
    assert crashlab.propose_fixes(run, fn=lambda m: "add a rule about 30 days") == []
    assert "did not return" in run.diagnosis.fix_notes[0]
    assert run.diagnosis.fix_method["raw_reply"].startswith("add a rule")


def test_no_fix_for_a_clean_run_and_no_call():
    run = _clean_run()
    crashlab.diagnose(run, heuristic=True)
    assert crashlab.propose_fixes(run, fn=lambda m: pytest.fail("called")) == []
    assert "NO_FAILURE" in run.diagnosis.fix_notes[0]


def test_needs_a_diagnosis_first():
    with pytest.raises(DiagnoseError, match="no diagnosis"):
        crashlab.propose_fixes(_policy_run(), fn=lambda m: "{}")


def test_accepts_a_diagnosis_object():
    run = _policy_run()
    d = validate(GOOD, run)
    fixes = crashlab.propose_fixes(d, run=run, fn=_reply(RULE), save=False)
    assert fixes and d.fixes == fixes


def test_try_fix_forks_once_and_reports_the_checks():
    run = _diagnosed()
    [fix] = crashlab.propose_fixes(run, fn=_reply(RULE))
    client = crashlab.wrap(
        FakeOpenAI(
            [
                tool_call_response(1, "get_order", {"order_id": "C03"}),
                text_response(2, "Sorry, C03 was delivered too long ago to refund."),
            ]
        )
    )
    tried = crashlab.try_fix(run, fix, client=client)
    assert tried["fork_step"] == 0 and tried["outcome"] == "pass" and tried["failed_checks"] == []
    branch = crashlab.load(tried["run_id"])
    assert (
        branch.forked_from == run.run_id and "4. Only refund" in branch.steps[0].request["system"]
    )
