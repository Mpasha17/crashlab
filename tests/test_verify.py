"""crashlab.verify: stats, verdicts, errored trials, regressions, the budget cap and the
CLI. Offline: the agent's model is a rule-following fake."""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from typer.testing import CliRunner

import crashlab
from crashlab.cli import app
from crashlab.models import ArmStats, RegressionResult
from crashlab.verify import is_infra_error, pass_hat_k, verdict, verifications, wilson
from tests.fakes import FakeAPIError, FakeOpenAI, text_response, tool_call_response
from tests.test_diagnose import _clean_run
from tests.test_fixes import RULE, _diagnosed, _reply

runner = CliRunner()
APPEND = {"kind": "system_prompt", "op": "append", "text": RULE["text"]}


class _RuleFollower:
    """Answers like a model that obeys the system prompt: refuses C03 when the 30-day
    rule is present, refunds otherwise; ``BREAK`` in the prompt makes it refund 1.00;
    ``fail`` lists call numbers that raise a provider error instead."""

    def __init__(self, owner: PolicyModel) -> None:
        self.owner = owner

    def create(self, **kw: Any) -> Any:
        o = self.owner
        o.n += 1
        if o.n in o.fail:
            raise FakeAPIError("503 model overloaded, high demand", 503)
        msgs = kw["messages"]
        system = " ".join(m["content"] for m in msgs if m["role"] == "system")
        user = next(m["content"] for m in msgs if m["role"] == "user")
        oid = re.search(r"\b([A-Z]\d{2})\b", user).group(1)
        tools = [m for m in msgs if m["role"] == "tool"]
        if not tools:
            return tool_call_response(o.n, "get_order", {"order_id": oid})
        if len(tools) == 1:
            if oid == "C03" and ("30 days" in system and "Only refund" in system):
                return text_response(o.n, f"Sorry, {oid} is outside the refund window.")
            total = json.loads(tools[0]["content"]).get("total", 0)
            amount = 1.0 if "BREAK" in system else total
            return tool_call_response(o.n, "issue_refund", {"order_id": oid, "amount": amount})
        return text_response(o.n, f"Your refund for order {oid} has been issued.")


class PolicyModel(FakeOpenAI):
    def __init__(self, fail: set[int] = frozenset()) -> None:
        super().__init__([])
        self.n = 0
        self.fail = set(fail)
        self.chat.completions = _RuleFollower(self)


def _setup(patch=APPEND):
    clean = _clean_run()
    run = _diagnosed()
    crashlab.propose_fixes(run, fn=_reply(patch))
    return crashlab.load(run.run_id), clean


# --- stats ---------------------------------------------------------------------------


def test_wilson_matches_known_values():
    lo, hi = wilson(5, 5)
    assert round(lo, 3) == 0.566 and hi == 1.0
    lo, hi = wilson(0, 5)
    assert lo == 0.0 and round(hi, 3) == 0.434
    lo, hi = wilson(3, 10)
    assert (round(lo, 3), round(hi, 3)) == (0.108, 0.603)
    assert wilson(0, 0) == (0.0, 1.0)


def test_pass_hat_k():
    assert pass_hat_k(5, 5, 5) == 1.0 and pass_hat_k(4, 5, 5) == 0.0
    assert pass_hat_k(8, 10, 5) == pytest.approx(56 / 252)
    assert pass_hat_k(0, 0, 5) is None


def _arm(p, f, e=0):
    n = p + f
    return ArmStats(
        passes=p, fails=f, errored=e, rate=p / n if n else None, ci95=wilson(p, n) if n else None
    )


@pytest.mark.parametrize(
    ("patched", "control", "reg", "want"),
    [
        (_arm(5, 0), _arm(0, 5), [], "VERIFIED"),
        (_arm(4, 1), _arm(0, 5), [], "LIKELY"),  # 4/5 low 0.38 < 0.43: overlap
        (_arm(5, 0), _arm(0, 5), [RegressionResult(run_id="X", status="fail")], "REJECTED"),
        (_arm(2, 3), _arm(2, 3), [], "REJECTED"),  # no improvement
        (_arm(0, 0, 5), _arm(0, 5), [], "INCONCLUSIVE"),  # every patched trial errored
        (_arm(3, 0, 2), _arm(0, 3, 2), [], "LIKELY"),  # sweep, but trials missing
        (_arm(5, 0), None, [], "LIKELY"),  # no control
        (_arm(0, 5), None, [], "REJECTED"),
        (_arm(5, 0), _arm(0, 5), [RegressionResult(run_id="X", status="errored")], "LIKELY"),
    ],
)
def test_verdicts(patched, control, reg, want):
    got, reasons = verdict(patched, control, reg, k=5)
    assert got == want and reasons


def test_infra_errors_are_recognised():
    assert is_infra_error(FakeAPIError("x", 429))
    assert is_infra_error(crashlab.models.ErrorInfo(type="InternalServerError", message="503"))
    assert not is_infra_error(crashlab.models.ErrorInfo(type="KeyError", message="'total'"))
    assert not is_infra_error(None)


# --- runner --------------------------------------------------------------------------


def test_verify_good_fix_is_verified_with_controls_and_regression():
    run, clean = _setup()
    v = crashlab.verify(run, "fix-1", k=3, client=crashlab.wrap(PolicyModel()))
    assert (v.patched.passes, v.patched.n, v.control.passes, v.control.n) == (3, 3, 0, 3)
    assert v.verdict == "VERIFIED", v.reasons
    assert [(r.run_id, r.status) for r in v.regression] == [(clean.run_id, "pass")]
    assert len(v.trials) == 7 and all(t.run_id for t in v.trials)
    child = crashlab.load(v.trials[0].run_id)
    assert f"verify:{v.verification_id}" in child.tags and child.forked_from == run.run_id
    stored = crashlab.load(run.run_id)
    assert stored.fix_attempts[-1]["verdict"] == "VERIFIED"
    assert stored.fix_attempts[-1]["patched"] == "3/3"
    assert verifications(crashlab.store.Store(), run.run_id)[0].verification_id == v.verification_id


def test_regression_run_the_patch_cannot_apply_to_is_skipped_not_passed():
    run, clean = _setup(RULE)  # anchors on line 3 of the faulty prompt only
    v = crashlab.verify(run, 1, k=3, client=crashlab.wrap(PolicyModel()))
    [r] = v.regression
    assert r.status == "skipped" and "does not apply" in r.note
    assert v.verdict == "LIKELY"


def test_patch_that_breaks_a_good_run_is_rejected():
    bad = {**APPEND, "text": "4. Only refund within 30 days of delivery. BREAK"}
    run, clean = _setup(bad)
    v = crashlab.verify(run, 1, k=2, client=crashlab.wrap(PolicyModel()))
    assert v.patched.passes == 2 and v.control.passes == 0
    assert v.verdict == "REJECTED" and v.regressions[0].run_id == clean.run_id
    assert "regression" in v.reasons[0]


def test_errored_trials_are_not_fails():
    run, _ = _setup()
    # calls 1 and 3 fail: the first patched trial and the first control (2 calls each
    # when it errors at its first live call)
    v = crashlab.verify(run, 1, k=3, regress="none", client=crashlab.wrap(PolicyModel({1})))
    assert v.patched.errored == 1 and v.patched.fails == 0 and v.patched.passes == 2
    errored = [t for t in v.trials if t.status == "errored"]
    assert errored and "503" in errored[0].error
    assert v.verdict == "LIKELY" and any("errored" in r for r in v.reasons)


def test_no_control_and_explicit_regression_list():
    run, clean = _setup()
    v = crashlab.verify(
        run, 1, k=2, control=False, regress=[clean.run_id], client=crashlab.wrap(PolicyModel())
    )
    assert v.control is None and len(v.trials) == 3 and v.verdict == "LIKELY"


def test_budget_cap_stops_new_trials(monkeypatch):
    from crashlab.verify import runner as r

    monkeypatch.setattr(r, "_live_usage", lambda run: (0.01, 100))
    run, _ = _setup()
    v = crashlab.verify(run, 1, k=5, max_cost_usd=0.025, client=crashlab.wrap(PolicyModel()))
    assert len(v.trials) == 3 and v.budget_exhausted and v.cost_usd == pytest.approx(0.03)
    assert v.tokens == 300 and all(r.status == "skipped" for r in v.regression)
    assert any("budget" in x for x in v.reasons)


def test_fix_from_a_patch_file_and_bad_refs(tmp_path):
    run, _ = _setup()
    f = run.diagnosis.fixes[0]
    v = crashlab.verify(run, f.file, k=1, regress="none", client=crashlab.wrap(PolicyModel()))
    assert v.patched.passes == 1 and v.fix_file == f.file
    with pytest.raises(crashlab.VerifyError, match="no fix 'fix-9'"):
        crashlab.verify(run, "fix-9", k=1)


# --- CLI -----------------------------------------------------------------------------


@pytest.fixture
def policy_client(monkeypatch):
    from crashlab.verify import runner as r

    model = crashlab.wrap(PolicyModel())
    real = r._trial
    monkeypatch.setattr(r, "_trial", lambda *a: real(*a[:4], model, *a[5:]))
    return model


def test_cli_verify_and_report(policy_client):
    run, _ = _setup()
    res = runner.invoke(app, ["verify", run.run_id, "--fix", "fix-1", "-k", "2"])
    assert res.exit_code == 0, res.output
    assert "VERIFIED" not in res.output or "patched" in res.output
    assert "2/2" in res.output and "0/2" in res.output and "regression" in res.output.lower()
    res = runner.invoke(app, ["report", run.run_id])
    assert res.exit_code == 0 and "Fix attempts" in res.output and "2/2" in res.output


def test_cli_verify_json(policy_client):
    run, _ = _setup()
    res = runner.invoke(app, ["verify", run.run_id, "-k", "1", "--no-control", "--json"])
    data = json.loads(res.stdout)
    assert data["patched"]["passes"] == 1 and data["control"] is None


def test_cli_fix_verify_one_command(policy_client, monkeypatch):
    from crashlab.diagnose import fixes as fixes_mod

    monkeypatch.setattr(
        fixes_mod, "_ask", lambda run, msgs, model, **kw: (_reply(APPEND)(msgs), {"backend": "f"})
    )
    _clean_run()
    run = _diagnosed()
    res = runner.invoke(app, ["fix", run.run_id, "-m", "fake", "--verify", "-k", "2"])
    assert res.exit_code == 0, res.output
    assert "fix-1" in res.output and "2/2" in res.output
    assert crashlab.load(run.run_id).fix_attempts


def test_pace_sleeps_between_trials(monkeypatch):
    from crashlab.verify import runner as r

    naps = []
    monkeypatch.setattr(r.time, "sleep", naps.append)
    run, _ = _setup()
    crashlab.verify(run, 1, k=2, regress="none", pace_s=4, client=crashlab.wrap(PolicyModel()))
    assert naps == [4, 4, 4]


def test_a_patched_fork_replays_offline_with_its_patch():
    pytest.importorskip("openai")  # the agent builds its own client on replay
    run, _ = _setup()
    v = crashlab.verify(run, 1, k=1, regress="none", client=crashlab.wrap(PolicyModel()))
    patched = next(t for t in v.trials if t.role == "patched")
    rep = crashlab.replay(patched.run_id)
    assert rep.outcome == "pass" and rep.replay_of == patched.run_id
    assert "4. Only refund" in rep.steps[0].request["system"]


def test_attempt_times_are_local():
    from crashlab.cli import _local_minute

    assert _local_minute("2026-10-22T03:32:26+00:00") == (
        __import__("datetime")
        .datetime(2026, 10, 22, 3, 32, 26, tzinfo=__import__("datetime").timezone.utc)
        .astimezone()
        .strftime("%Y-%m-%d %H:%M")
    )


def test_one_usable_trial_per_arm_is_inconclusive_not_likely():
    """Seen live: a provider's daily quota errored 4 of 5 trials in each arm; 1/1 vs 0/1
    must not read as LIKELY."""
    from crashlab.models import ArmStats, RegressionResult
    from crashlab.verify.stats import verdict

    p = ArmStats(passes=1, fails=0, errored=4, rate=1.0, ci95=(0.21, 1.0))
    c = ArmStats(passes=0, fails=1, errored=4, rate=0.0, ci95=(0.0, 0.79))
    reg = [RegressionResult(run_id="r1", status="errored")]
    v, why = verdict(p, c, reg, k=5)
    assert v == "INCONCLUSIVE" and "too few usable trials" in why[0]
    full = ArmStats(passes=3, fails=0, rate=1.0, ci95=(0.44, 1.0))
    zero = ArmStats(passes=0, fails=3, rate=0.0, ci95=(0.0, 0.56))
    assert verdict(full, zero, [], k=3)[0] == "VERIFIED"


class _CapsAtTen(PolicyModel):
    """With the fix in the prompt, refunds a *different* amount than the failing run did
    (a side-effect call the recording never made), so forks hit the side-effect guard."""

    def __init__(self) -> None:
        super().__init__()
        inner = self.chat.completions

        class _C:
            def create(_self, **kw: Any) -> Any:
                system = " ".join(m["content"] for m in kw["messages"] if m["role"] == "system")
                tools = [m for m in kw["messages"] if m["role"] == "tool"]
                if "CAP" in system and len(tools) == 1:
                    self.n += 1
                    return tool_call_response(
                        self.n, "issue_refund", {"order_id": "C03", "amount": 10.0}
                    )
                return inner.create(**kw)

        self.chat.completions = _C()


def test_blocked_side_effect_trials_are_not_counted_and_can_run_live():
    cap = {**APPEND, "text": "4. CAP refunds outside the window at 10.00."}
    run, _ = _setup(cap)
    v = crashlab.verify(run, 1, k=2, regress="none", client=crashlab.wrap(_CapsAtTen()))
    patched = [t for t in v.trials if t.role == "patched"]
    assert all(t.status == "errored" and "SideEffectBlockedError" in t.error for t in patched)
    assert v.verdict == "INCONCLUSIVE"
    assert any("--allow-live-side-effects" in r for r in v.reasons)
    # in a sandbox the side-effect tool runs live and the trials are graded
    v2 = crashlab.verify(
        run,
        1,
        k=2,
        regress="none",
        client=crashlab.wrap(_CapsAtTen()),
        allow_live_side_effects=True,
    )
    assert all(t.status in ("pass", "fail") for t in v2.trials if t.role == "patched")
    assert not any("--allow-live-side-effects" in r for r in v2.reasons)


def test_cli_verify_accepts_allow_live_side_effects(policy_client):
    run, _ = _setup()
    res = runner.invoke(
        app, ["verify", run.run_id, "-k", "1", "--regress", "none", "--allow-live-side-effects"]
    )
    assert res.exit_code == 0, res.output
