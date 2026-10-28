"""Restore proposals (prompt/tool text from past passing runs), the ask-a-person filter
and placement ranking of fix proposals. Offline: the fix model is a fake."""

from __future__ import annotations

import json

import crashlab
from crashlab.diagnose import validate_fix
from crashlab.diagnose.fixes import rank_fixes
from crashlab.diagnose.restore import _prompt_hunks, good_runs, restore_proposals
from crashlab.store import Store
from tests.test_diagnose import GOOD, _clean_run, _policy_run
from tests.test_verify import policy_client  # noqa: F401 - a fixture

APPEND = {"kind": "system_prompt", "op": "append", "text": "Never refund after 30 days."}


def _diagnosed_with_good(n_good: int = 2):
    good = [_clean_run() for _ in range(n_good)]
    run = _policy_run()
    crashlab.diagnose(run, fn=lambda m: json.dumps(GOOD))
    return crashlab.load(run.run_id), good


def _reply(*patches):
    items = [{"patch": p, "rationale": "r", "expected_effect": "e"} for p in patches]
    return lambda m: json.dumps({"fixes": items})


def test_missing_rule_is_restored_at_its_original_position(crashlab_dir):
    run, good = _diagnosed_with_good()
    assert all(g.outcome == "pass" for g in good)
    fixes = crashlab.propose_fixes(run, fn=_reply(APPEND))
    first = fixes[0]
    # the policy list is renumbered, so the block is substituted back as it was
    assert first.fix_id == "fix-1" and first.as_patch.op == "substitute"
    assert "30 days" in first.as_patch.text
    sys_of = lambda r: r.steps[0].request["system"]  # noqa: E731
    assert first.as_patch.edit(sys_of(run)) == sys_of(good[0])
    assert first.notes[0].startswith("restored from past passing runs")
    assert "2 of 2 past passing" in first.rationale
    assert all(g.run_id in first.rationale for g in good)
    assert fixes[1].as_patch.op == "append"  # the model's proposal ranks after
    d = crashlab.load(run.run_id).diagnosis
    assert d.fix_method["restored"] == 1 and d.fix_method["ranker"]


def test_no_good_runs_no_restore(crashlab_dir):
    run = _policy_run()
    crashlab.diagnose(run, fn=lambda m: json.dumps(GOOD))
    fixes = crashlab.propose_fixes(crashlab.load(run.run_id), fn=_reply(APPEND))
    assert [f.as_patch.op for f in fixes] == ["append"]


def test_good_runs_skip_forks_and_failures(crashlab_dir):
    run, good = _diagnosed_with_good(1)
    store = Store()
    assert [g.run_id for g in good_runs(store, run)] == [good[0].run_id]


def test_only_text_most_good_runs_agree_on_is_restored():
    class R:  # a stand-in for a Run: steps with one chat request
        def __init__(self, rid, system):
            self.run_id, self.steps = rid, [type("S", (), {"operation": "chat",
                                             "request": {"system": system}})()]  # fmt: skip

    fail = R("f", "You help.\nBe brief.")
    good = [R(f"g{i}", f"You help.\nCustomer: user{i}\nBe brief.") for i in range(3)]
    assert restore_proposals(fail, good) == []  # a per-run line, not a rule
    same = [R(f"g{i}", "You help.\nNever share keys.\nBe brief.") for i in range(3)]
    out = restore_proposals(fail, same[:2] + good[:1])
    assert len(out) == 1 and out[0]["patch"]["text"] == "Never share keys."
    assert out[0]["patch"]["after"] == "You help."


def test_prompt_hunks_cover_insert_replace_delete_and_first_line():
    hunks = _prompt_hunks(
        "A rule.\nOld wording.\nInjected.\nEnd.", "Top.\nA rule.\nNew wording.\nEnd."
    )
    ops = [(p["op"], p.get("after"), p.get("before"), p.get("match")) for _, p, _ in hunks]
    assert ("insert_line", None, "A rule.", None) in ops
    assert any(o[0] in ("replace_line", "substitute") for o in ops)


def test_changed_tool_description_is_restored():
    def run(rid, desc):
        req = {"system": "s", "tools": [{"type": "function", "function": {
            "name": "issue_refund", "description": desc}}]}  # fmt: skip
        step = type("S", (), {"operation": "chat", "request": req})()
        return type("Run", (), {"run_id": rid, "steps": [step]})()

    out = restore_proposals(
        run("f", "Refund anything."), [run("g", "Refund delivered orders only.")]
    )
    assert out[0]["patch"] == {
        "kind": "tool_description", "tool": "issue_refund",
        "description": "Refund delivered orders only.",
    }  # fmt: skip


def test_ask_a_person_patches_are_rejected(crashlab_dir):
    run = _policy_run()
    for text in (
        "If the order is older than 30 days, let me know before refunding.",
        "Ask the customer to confirm before issuing a refund.",
        "Would you like me to refund it anyway?",
        "Wait for the user's approval before refunding.",
    ):
        fix, why = validate_fix({"patch": {**APPEND, "text": text}}, run, 1)
        assert fix is None and "asks a person" in why, text
    ok, _ = validate_fix(
        {"patch": {**APPEND, "text": "Ask the customer for the order id."}}, run, 1
    )
    assert ok is not None  # asking for missing input is fine


def test_ranking_prefers_in_place_and_imperative(crashlab_dir):
    run = _policy_run()
    line = "1. Look up the order before acting."
    first = run.steps[0].request["system"].splitlines()[0][:30]
    items = [
        {**APPEND},
        {
            "kind": "system_prompt",
            "op": "insert_line",
            "after": first,
            "text": "Try to only refund recent orders.",
        },  # fmt: skip
        {"kind": "system_prompt", "op": "insert_line", "after": first, "text": line},
    ]
    fixes = [validate_fix({"patch": p}, run, i + 1)[0] for i, p in enumerate(items)]
    ranked = rank_fixes(fixes)
    assert [f.as_patch.text for f in ranked] == [line, items[1]["text"], APPEND["text"]]
    assert [f.fix_id for f in ranked] == ["fix-1", "fix-2", "fix-3"]


def test_fix_verify_verifies_the_restore_before_a_weak_model_fix(policy_client, monkeypatch):  # noqa: F811
    # live: the same rule appended was 0/3 on LangChain, restored in place 3/3
    from typer.testing import CliRunner

    from crashlab.cli import app
    from crashlab.diagnose import fixes as fixes_mod

    weak = {**APPEND, "text": "Be polite."}
    monkeypatch.setattr(
        fixes_mod, "_ask", lambda run, msgs, model, **kw: (_reply(weak)(msgs), {"backend": "f"})
    )
    run, _ = _diagnosed_with_good(1)
    res = CliRunner().invoke(app, ["fix", run.run_id, "-m", "fake", "--verify", "-k", "2"])
    assert res.exit_code == 0, res.output
    attempts = crashlab.load(run.run_id).fix_attempts
    assert len(attempts) == 1 and attempts[0]["fix_id"] == "fix-1"
    assert attempts[0]["verdict"] in ("VERIFIED", "LIKELY")
    assert "substitute" in attempts[0]["patch"] or "replace" in attempts[0]["patch"]


def test_restore_proposals_without_a_fix_model_or_when_it_fails(crashlab_dir, monkeypatch):
    from crashlab.diagnose import DiagnoseError
    from crashlab.diagnose import fixes as fixes_mod

    monkeypatch.delenv("CRASHLAB_DIAGNOSER_MODEL", raising=False)
    run, _ = _diagnosed_with_good(1)
    fixes = crashlab.propose_fixes(run)
    assert len(fixes) == 1 and fixes[0].notes[0].startswith("restored from")
    d = crashlab.load(run.run_id).diagnosis
    assert any("no fix model" in n for n in d.fix_notes)

    def boom(*a, **k):
        raise DiagnoseError("429 quota exceeded")

    monkeypatch.setattr(fixes_mod, "_ask", boom)
    fixes = crashlab.propose_fixes(crashlab.load(run.run_id), "some-model")
    d = crashlab.load(run.run_id).diagnosis
    assert len(fixes) == 1 and any("the fix model failed (429" in n for n in d.fix_notes)
    assert d.fix_method["cache_key"] is None


def test_no_model_and_nothing_to_restore_still_errors(crashlab_dir, monkeypatch):
    import pytest

    from crashlab.diagnose import DiagnoseError

    monkeypatch.delenv("CRASHLAB_DIAGNOSER_MODEL", raising=False)
    run = _policy_run()
    crashlab.diagnose(run, fn=lambda m: json.dumps(GOOD))
    with pytest.raises(DiagnoseError, match="no fix model"):
        crashlab.propose_fixes(crashlab.load(run.run_id))


def test_fix_prompt_lists_prompt_lines_and_the_restore_already_proposed(crashlab_dir):
    run, _ = _diagnosed_with_good(1)
    seen = []
    crashlab.propose_fixes(run, fn=lambda m: seen.append(m[1]["content"]) or _reply(APPEND)(m))
    user = seen[0]
    assert "SYSTEM PROMPT LINES" in user and "\n1: You are the refund assistant" in user
    assert "ALREADY PROPOSED" in user and "30 days" in user.split("ALREADY PROPOSED")[1]
