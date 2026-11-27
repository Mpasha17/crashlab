"""crashlab.repair: diagnose -> propose -> verify -> iterate, offline. The agent's model
is the rule-following fake from test_verify; the fix model is a canned fn."""

from __future__ import annotations

import json
import re

import pytest
from typer.testing import CliRunner

import crashlab
from crashlab.cli import app
from crashlab.repair import RepairError, repairs
from tests.test_diagnose import GOOD, _clean_run, _policy_run
from tests.test_fixes import RULE
from tests.test_verify import APPEND, PolicyModel

runner = CliRunner()
ANSI = re.compile(r"\x1b\[[0-9;]*m")
WEAK = {"kind": "system_prompt", "op": "append", "text": "4. Only refund orders with a receipt."}


@pytest.fixture(autouse=True)
def _model_fixes_only(monkeypatch):
    from crashlab.diagnose import fixes as fixes_mod

    monkeypatch.setattr(fixes_mod, "good_runs", lambda *a, **k: [])


def _diagnoser(msgs):
    return json.dumps(GOOD)


def _fixer(*patches):
    def fn(msgs):
        return json.dumps(
            {
                "fixes": [
                    {"patch": p, "rationale": "missing rule", "expected_effect": "refuses"}
                    for p in patches
                ]
            }
        )

    return fn


def _learner(calls):
    """Proposes the weak rule first; once a REJECTED attempt is fed back, the real one."""

    def fn(msgs):
        text = msgs[-1]["content"]
        calls.append(text)
        return _fixer(APPEND if "PREVIOUS FIXES" in text else WEAK)(msgs)

    return fn


@pytest.fixture
def agent_client(monkeypatch):
    from crashlab.verify import runner as r

    model = crashlab.wrap(PolicyModel())
    real = r._trial
    monkeypatch.setattr(r, "_trial", lambda *a: real(*a[:4], model, *a[5:]))
    return model


def test_known_patch_fixes_the_run(agent_client):
    run = _policy_run()
    res = crashlab.repair(run, diagnoser=_diagnoser, propose_fn=_fixer(APPEND), trials=3)
    assert res.status == "fixed" and res.fixed, res.evidence
    assert res.verified_pass_rate == 1.0 and res.iterations == 1
    assert res.patch is not None and res.patch.as_patch.text == APPEND["text"]
    [a] = res.attempts
    assert (a.verdict, a.passes, a.trials, a.control, a.accepted) == ("VERIFIED", 3, 3, "0/3", True)
    assert res.diagnosis["label"] == "TASK_SPEC_VIOLATION"
    # audit trail: the verification and the repair record are in the store
    stored = crashlab.load(run.run_id)
    assert stored.fix_attempts[-1]["verdict"] == "VERIFIED"
    [rec] = repairs(run.run_id)
    assert rec["status"] == "fixed" and rec["attempts"][0]["verification_id"] == a.verification_id


def test_not_fixable_returns_not_fixed(agent_client):
    run = _policy_run()
    res = crashlab.repair(
        run, diagnoser=_diagnoser, propose_fn=_fixer(WEAK), trials=2, max_iterations=2
    )
    assert res.status == "not_fixed" and res.patch is None
    assert [a.verdict for a in res.attempts] == ["REJECTED"]  # round 2 had nothing new
    assert any("no new candidate" in e for e in res.evidence)


def test_iteration_feeds_failures_back(agent_client):
    run = _policy_run()
    calls: list[str] = []
    res = crashlab.repair(run, diagnoser=_diagnoser, propose_fn=_learner(calls), trials=3)
    assert res.status == "fixed" and res.iterations == 2
    assert [(a.iteration, a.verdict) for a in res.attempts] == [(1, "REJECTED"), (2, "VERIFIED")]
    assert "PREVIOUS FIXES" not in calls[0]
    assert "REJECTED" in calls[1] and WEAK["text"] in calls[1]


def test_judge_and_oracle_gate_passing_trials(agent_client):
    run = _policy_run()
    res = crashlab.repair(
        run,
        diagnoser=_diagnoser,
        propose_fn=_fixer(APPEND),
        trials=2,
        max_iterations=1,
        oracle=lambda child: False,
    )
    [a] = res.attempts
    assert a.verdict in ("VERIFIED", "LIKELY") and not a.accepted and a.pass_rate == 0.0
    assert a.judge_failures and res.status == "not_fixed"


def test_judge_gates_with_a_fake_judge_model(agent_client):
    run = _policy_run()
    low = lambda msgs: json.dumps({"score": 0.1, "reasoning": "unsafe", "evidence": []})  # noqa: E731
    judge = crashlab.judges.custom("The reply is polite.", name="polite")
    res = crashlab.repair(
        run,
        diagnoser=_diagnoser,
        propose_fn=_fixer(APPEND),
        trials=3,
        max_iterations=1,
        judges=[judge],
        judge_fn=low,
    )
    [a] = res.attempts
    assert a.verdict == "VERIFIED" and a.pass_rate == 0.0 and not a.accepted
    assert any("judge polite" in n for n in a.judge_failures)
    high = lambda msgs: json.dumps({"score": 0.9, "reasoning": "ok", "evidence": []})  # noqa: E731
    run2 = _policy_run()
    res2 = crashlab.repair(
        run2, diagnoser=_diagnoser, propose_fn=_fixer(APPEND), trials=3,
        judges=[judge], judge_fn=high,
    )  # fmt: skip
    assert res2.status == "fixed" and res2.verified_pass_rate == 1.0


def test_dry_run_verifies_nothing(agent_client):
    run = _policy_run()
    res = crashlab.repair(run, diagnoser=_diagnoser, propose_fn=_fixer(RULE, APPEND), dry_run=True)
    assert res.dry_run and res.status == "not_fixed"
    assert [a.verdict for a in res.attempts] == ["PROPOSED", "PROPOSED"]
    assert not crashlab.load(run.run_id).fix_attempts


def test_passing_run_and_bad_options_are_usage_errors():
    clean = _clean_run()
    with pytest.raises(RepairError, match="passed"):
        crashlab.repair(clean)
    with pytest.raises(RepairError, match="at least 1"):
        crashlab.repair(clean, trials=0)


# --- CLI -----------------------------------------------------------------------------


def _fake_models(monkeypatch, fix_fn):
    from crashlab.diagnose import diagnoser as diag_mod
    from crashlab.diagnose import fixes as fixes_mod

    monkeypatch.setattr(
        diag_mod, "_ask", lambda run, msgs, model, **kw: (_diagnoser(msgs), {"backend": "f"})
    )
    monkeypatch.setattr(
        fixes_mod, "_ask", lambda run, msgs, model, **kw: (fix_fn(msgs), {"backend": "f"})
    )


def test_cli_repair_writes_patch_and_test(agent_client, monkeypatch, tmp_path):
    _fake_models(monkeypatch, _fixer(APPEND))
    run = _policy_run()
    patch_path, test_dir = tmp_path / "fix.json", tmp_path / "regress"
    res = runner.invoke(
        app,
        [
            "repair", run.run_id, "-m", "fake", "--trials", "3",
            "--write-patch", str(patch_path), "--export-test", str(test_dir),
        ],
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    out = ANSI.sub("", res.output)
    assert "FIXED" in out and "VERIFIED" in out and "3/3" in out
    ps = crashlab.patch.load(patch_path)
    assert ps.patches[0].text == APPEND["text"]
    tests = list(test_dir.glob("test_*.py"))
    assert len(tests) == 1 and "fixed" in tests[0].read_text()
    assert list((test_dir / "cassettes").glob("*_fixed.jsonl"))


def test_cli_repair_json_not_fixed_exit_1(agent_client, monkeypatch):
    _fake_models(monkeypatch, _fixer(WEAK))
    run = _policy_run()
    res = runner.invoke(app, ["repair", run.run_id, "-m", "fake", "-k", "1", "--json"])
    assert res.exit_code == 1, res.output
    data = json.loads(res.stdout)
    assert data["ok"] and data["status"] == "not_fixed" and data["attempts"]


def test_cli_repair_dry_run_and_usage_errors(agent_client, monkeypatch):
    _fake_models(monkeypatch, _fixer(APPEND))
    run = _policy_run()
    res = runner.invoke(app, ["repair", run.run_id, "-m", "fake", "--dry-run"])
    assert res.exit_code == 0 and "proposed" in ANSI.sub("", res.output)
    res = runner.invoke(app, ["repair", "nope-no-such-run"])
    assert res.exit_code == 2
    clean = _clean_run()
    res = runner.invoke(app, ["repair", clean.run_id, "--json"])
    assert res.exit_code == 2 and "passed" in json.loads(res.stdout)["error"]


def test_cli_repair_help():
    res = runner.invoke(app, ["repair", "--help"])
    out = ANSI.sub("", res.output)
    assert res.exit_code == 0
    for flag in ("--max-iterations", "--candidates", "--trials", "--judge", "--write-patch",
                 "--export-test", "--json", "--dry-run"):  # fmt: skip
        assert flag in out
    top = ANSI.sub("", runner.invoke(app, ["--help"]).output)
    assert "repair" in top


def test_offline_heuristic_repair_restores_past_passing_prompt(agent_client, monkeypatch):
    """No diagnoser and no fix model: heuristic diagnosis + restore proposals only."""
    from crashlab.diagnose import fixes as fixes_mod
    from crashlab.diagnose import restore

    monkeypatch.setattr(fixes_mod, "good_runs", restore.good_runs)
    monkeypatch.delenv("CRASHLAB_DIAGNOSER_MODEL", raising=False)
    _clean_run()
    run = _policy_run()
    res = crashlab.repair(run, trials=3, regress="none")
    assert res.attempts and res.attempts[0].source == "restore", res.evidence
    assert res.status == "fixed", [a.to_dict() for a in res.attempts]
