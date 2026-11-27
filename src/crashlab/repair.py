"""``crashlab.repair``: diagnose a failed run, propose fixes, verify them, iterate.

The closed loop over the pieces crashlab already has:

1. **Diagnose** the run (:func:`crashlab.diagnose`; heuristic when no model is set).
2. **Propose** candidate patches (:func:`crashlab.propose_fixes`): restore proposals
   from past passing runs always, plus LLM-proposed patches (system prompt edits, tool
   description / schema edits, model params ...) when a fix model or ``propose_fn`` is
   given.
3. **Verify** each candidate with :func:`crashlab.verify`: ``trials`` patched forks vs
   ``trials`` unpatched controls, flake-aware (Wilson intervals, errored trials not
   counted), plus optional judges / an ``oracle`` on every passing patched fork.
4. **Pick** the best passing patch, or feed the failed attempts back (they are in
   ``run.fix_attempts``, which the fix prompt shows the model) and go again, up to
   ``max_iterations``.

Every attempt is a stored verification (``crashlab report`` lists them) and the whole
repair is saved under ``<store>/runs/<id>/repairs/<repair_id>.json`` for audit. With a
fake ``propose_fn`` and a fake agent client the loop is deterministic and offline.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from ._ids import new_ulid
from .models import Fix, Run, Verification

Status = Literal["fixed", "partial", "not_fixed"]
PASSING = ("VERIFIED", "LIKELY")
DEFAULT_TRIALS = 3
DEFAULT_MIN_PASS_RATE = 0.8


class RepairError(ValueError):
    """The run cannot be repaired as asked (not failed, no diagnosis, bad options)."""


@dataclass
class Attempt:
    """One candidate patch and how it fared."""

    iteration: int
    fix_id: str
    kind: str
    patch: str  # one-line description
    source: str = "model"  # "restore" (past passing runs) or "model"
    verdict: str = "PROPOSED"  # VERIFIED / LIKELY / INCONCLUSIVE / REJECTED / PROPOSED
    passes: int = 0
    trials: int = 0
    control: str | None = None  # "0/3"
    pass_rate: float | None = None  # after judges / oracle
    judge_failures: list[str] = field(default_factory=list)
    accepted: bool = False
    verification_id: str | None = None
    cost_usd: float = 0.0
    reasons: list[str] = field(default_factory=list)
    file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RepairResult:
    """What :func:`repair` found: ``status``, the chosen ``patch`` (a
    :class:`~crashlab.models.Fix`, ``None`` when nothing passed) and every attempt."""

    run_id: str
    status: Status
    patch: Fix | None = None
    attempts: list[Attempt] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    verified_pass_rate: float | None = None
    cost: float = 0.0
    iterations: int = 0
    diagnosis: dict[str, Any] | None = None
    repair_id: str = ""
    file: str | None = None
    dry_run: bool = False

    @property
    def fixed(self) -> bool:
        return self.status == "fixed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "repair_id": self.repair_id,
            "run_id": self.run_id,
            "status": self.status,
            "patch": self.patch.model_dump(mode="json", exclude_none=True) if self.patch else None,
            "verified_pass_rate": self.verified_pass_rate,
            "cost_usd": round(self.cost, 6),
            "iterations": self.iterations,
            "dry_run": self.dry_run,
            "diagnosis": self.diagnosis,
            "evidence": list(self.evidence),
            "attempts": [a.to_dict() for a in self.attempts],
            "file": self.file,
        }

    def write_patch(self, path: str | Path) -> Path:
        """Write the chosen patch in crashlab's patch format (``crashlab fork --patch``);
        YAML for ``.yaml`` / ``.yml`` (needs PyYAML), JSON otherwise."""
        if self.patch is None or self.patch.as_patch is None:
            raise RepairError(f"run {self.run_id}: no passing patch to write")
        return write_patch(self.patch, path)


def write_patch(fix: Fix, path: str | Path) -> Path:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {"patches": [fix.as_patch.model_dump(mode="json", exclude_none=True)]}  # type: ignore[union-attr]
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RepairError("YAML patch files need PyYAML: use a .json path") from exc
        head = f"# crashlab repair: {fix.after}\n"
        p.write_text(head + yaml.safe_dump(data, sort_keys=False), "utf-8")
    else:
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", "utf-8")
    return p


def _digest(fix: Fix) -> str:
    from .verify import patch_digest

    return patch_digest(fix.as_patch)


def _source(fix: Fix) -> str:
    return "restore" if any(n.startswith("restored from") for n in fix.notes) else "model"


def _diagnose(run: Run, store: Any, diagnoser: Any, diagnose_kw: dict[str, Any]) -> Any:
    from .diagnose import diagnose

    if run.diagnosis is not None and run.diagnosis.is_failure and diagnoser is None:
        return run.diagnosis
    if callable(diagnoser):
        return diagnose(run, fn=diagnoser, store=store, **diagnose_kw)
    if isinstance(diagnoser, str) and diagnoser:
        return diagnose(run, diagnoser, store=store, **diagnose_kw)
    import os

    from .diagnose import ENV_MODEL

    if os.environ.get(ENV_MODEL):
        return diagnose(run, store=store, **diagnose_kw)
    return diagnose(run, heuristic=True, store=store, **diagnose_kw)


def _check_trial_runs(
    store: Any,
    v: Verification,
    judges: list[Any],
    oracle: Callable[[Run], Any] | None,
    judge_kw: dict[str, Any],
) -> tuple[int, list[str]]:
    """Passing patched trials that also pass every judge and the oracle; failure notes."""
    from .judges import get_judge

    chosen = [get_judge(j, judge_kw.get("threshold")) for j in judges]
    kw = {k: val for k, val in judge_kw.items() if k != "threshold"}
    passes, notes = 0, []
    for t in v.trials:
        if t.role != "patched" or t.status != "pass" or not t.run_id:
            continue
        child = store.load_run(t.run_id)
        ok = True
        for j in chosen:
            r = j.evaluate(child, store=store, **kw)
            if not r.skipped and not r.passed:
                ok = False
                notes.append(f"judge {r.name} {r.score:.2f} < {r.threshold:.2f} on {t.run_id}")
        if oracle is not None:
            try:
                good = oracle(child)
            except Exception as e:  # noqa: BLE001 - a crashing oracle fails the trial
                good, notes = False, [*notes, f"oracle raised {type(e).__name__}: {e}"[:200]]
            if not good:
                ok = False
                notes.append(f"oracle rejected {t.run_id}")
        passes += ok
    return passes, notes


def _score(a: Attempt) -> tuple[int, float, int]:
    order = {"VERIFIED": 3, "LIKELY": 2, "INCONCLUSIVE": 1}
    return (int(a.accepted), a.pass_rate or 0.0, order.get(a.verdict, 0))


def repair(
    run: Run | str = "last",
    *,
    max_iterations: int = 3,
    candidates: int = 3,
    trials: int = DEFAULT_TRIALS,
    verify_runs: int | None = None,
    oracle: Callable[[Run], Any] | None = None,
    judges: Iterable[Any] | None = None,
    judge_model: str | None = None,
    judge_fn: Callable[..., Any] | None = None,
    judge_threshold: float | None = None,
    diagnoser: str | Callable[..., Any] | None = None,
    model: str | None = None,
    propose_fn: Callable[..., Any] | None = None,
    client: Any = None,
    control: bool = True,
    regress: Any = None,
    min_pass_rate: float = DEFAULT_MIN_PASS_RATE,
    max_cost_usd: float | None = None,
    allow_live_side_effects: bool = False,
    dry_run: bool = False,
    store_dir: Any = None,
    save: bool = True,
    on_attempt: Callable[[Attempt], None] | None = None,
) -> RepairResult:
    """Repair a failed run: diagnose -> propose -> verify -> iterate. Returns a
    :class:`RepairResult` (``fixed`` / ``partial`` / ``not_fixed``).

    - ``diagnoser``: a diagnoser model name or ``fn(messages) -> reply`` (heuristic when
      neither it nor ``$CRASHLAB_DIAGNOSER_MODEL`` is set and the run has no diagnosis).
    - ``model`` / ``propose_fn``: the fix model or ``fn(messages) -> reply`` (defaults to
      the diagnoser). Without either only restore proposals from past passing runs are
      tried.
    - ``trials`` (alias ``verify_runs``): forks per arm in each verification.
    - ``judges`` / ``oracle``: every passing patched fork must also pass these
      (``judge_model`` / ``judge_fn`` / ``judge_threshold`` go to the judges).
    - ``min_pass_rate``: a VERIFIED / LIKELY patch whose judged pass rate reaches it is
      accepted; ``fixed`` needs an accepted VERIFIED patch, ``partial`` an accepted
      LIKELY one or any patch that beat the control.
    - ``client``, ``regress``, ``control``, ``allow_live_side_effects``: as in
      :func:`crashlab.verify`. ``max_cost_usd`` caps the total spend of the repair.
    - ``dry_run``: diagnose and propose only, verify nothing.
    """
    from .diagnose import DiagnoseError, propose_fixes
    from .store import Store
    from .verify import VerifyError, verify

    if verify_runs is not None:
        trials = verify_runs
    if max_iterations < 1 or candidates < 1 or trials < 1:
        raise RepairError("max_iterations, candidates and trials must be at least 1")
    store = Store(store_dir)
    run = store.load_run(run) if isinstance(run, str) else store.load_run(run.run_id)
    if run.forked_from or run.replay_of:
        raise RepairError(f"{run.run_id} is a fork/replay; repair the original run")
    if run.outcome == "pass" and run.status != "error":
        raise RepairError(f"run {run.run_id} passed its checks; nothing to repair")

    res = RepairResult(run_id=run.run_id, status="not_fixed", dry_run=dry_run)
    res.repair_id = new_ulid()
    judge_list = list(judges or [])
    judge_kw: dict[str, Any] = {"threshold": judge_threshold}
    if judge_model:
        judge_kw["model"] = judge_model
    if judge_fn is not None:
        judge_kw["judge_fn"] = judge_fn
    fix_fn = propose_fn if propose_fn is not None else (diagnoser if callable(diagnoser) else None)
    fix_model = model or (diagnoser if isinstance(diagnoser, str) else None)

    try:
        d = _diagnose(run, store, diagnoser, {})
    except DiagnoseError as e:
        res.evidence.append(f"diagnosis failed: {e}"[:300])
        return _finish(res, store, run, save)
    run = store.load_run(run.run_id)
    run.diagnosis = d
    res.diagnosis = {
        "label": d.label.value,
        "failing_step": d.failing_step,
        "root_cause": d.root_cause,
        "confidence": d.confidence,
        "model": d.model,
    }
    if d.root_cause:
        res.evidence.append(f"diagnosis: {d.label.value} at step {d.failing_step}: {d.root_cause}")
    for e in d.evidence[:3]:
        res.evidence.append(f'step {e.step_idx} {e.field}: "{e.quote}"')
    if not d.is_failure:
        res.evidence.append(f"no fix proposed: the diagnosis is {d.label.value}")
        return _finish(res, store, run, save)

    seen: set[str] = set()
    for it in range(1, max_iterations + 1):
        res.iterations = it
        try:
            fixes = propose_fixes(
                run,
                fix_model,
                fn=fix_fn,
                store=store,
                max_fixes=max(candidates, 1) + 2,  # room for already-tried ones we skip
            )
        except DiagnoseError as e:
            res.evidence.append(f"iteration {it}: no candidates ({e})"[:300])
            break
        notes = run.diagnosis.fix_notes if run.diagnosis else []
        res.evidence += [f"iteration {it}: {n}" for n in notes if n.startswith("rejected")][:3]
        fresh = [f for f in fixes if f.as_patch is not None and _digest(f) not in seen]
        test_only = [f for f in fresh if f.test_only]
        for f in test_only:
            res.evidence.append(f"{f.fix_id} is a test-only counterfactual, not a fix: {f.after}")
        fresh = [f for f in fresh if not f.test_only][:candidates]
        if not fresh:
            res.evidence.append(f"iteration {it}: no new candidate patch")
            break
        for f in fresh:
            seen.add(_digest(f))
            a = Attempt(
                iteration=it,
                fix_id=f.fix_id,
                kind=f.kind,
                patch=f.after,
                source=_source(f),
                file=f.file,
            )
            if dry_run:
                res.attempts.append(a)
                if on_attempt:
                    on_attempt(a)
                continue
            budget = None if max_cost_usd is None else max(max_cost_usd - res.cost, 0.0)
            if budget is not None and budget <= 0:
                a.verdict, a.reasons = "SKIPPED", ["repair budget reached"]
                res.attempts.append(a)
                if on_attempt:
                    on_attempt(a)
                continue
            try:
                v = verify(
                    run,
                    f,
                    k=trials,
                    control=control,
                    regress=regress,
                    client=client,
                    store_dir=store.root,
                    max_cost_usd=budget,
                    allow_live_side_effects=allow_live_side_effects,
                    save=save,
                )
            except (VerifyError, ValueError) as e:
                a.verdict, a.reasons = "REJECTED", [f"verify failed: {e}"[:300]]
                res.attempts.append(a)
                if on_attempt:
                    on_attempt(a)
                continue
            res.cost += v.cost_usd
            a.verdict, a.verification_id, a.cost_usd = v.verdict, v.verification_id, v.cost_usd
            a.passes, a.trials = v.patched.passes, v.patched.n
            a.control = f"{v.control.passes}/{v.control.n}" if v.control else None
            a.reasons = list(v.reasons[:4])
            judged = a.passes
            if judge_list or oracle is not None:
                judged, a.judge_failures = _check_trial_runs(store, v, judge_list, oracle, judge_kw)
            a.pass_rate = (judged / a.trials) if a.trials else None
            ctrl_rate = (v.control.rate or 0.0) if v.control else 0.0
            a.accepted = (
                v.verdict in PASSING
                and a.pass_rate is not None
                and a.pass_rate >= min_pass_rate
                and a.pass_rate > ctrl_rate
            )
            res.attempts.append(a)
            if on_attempt:
                on_attempt(a)
            if a.accepted and v.verdict == "VERIFIED":
                break
        run = store.load_run(run.run_id)  # fix_attempts now carry this round's verdicts
        if dry_run or any(a.accepted and a.verdict == "VERIFIED" for a in res.attempts):
            break
        if max_cost_usd is not None and res.cost >= max_cost_usd:
            res.evidence.append("repair budget reached")
            break

    _pick(res, run)
    return _finish(res, store, run, save)


def _pick(res: RepairResult, run: Run) -> None:
    tried = [a for a in res.attempts if a.verdict not in ("PROPOSED", "SKIPPED")]
    if not tried:
        return
    best = max(tried, key=_score)
    if best.accepted and best.verdict == "VERIFIED":
        res.status = "fixed"
    elif best.accepted or (best.pass_rate or 0) > 0 and best.verdict in PASSING:
        res.status = "partial"
    else:
        return
    res.verified_pass_rate = best.pass_rate
    res.patch = _fix_for(run, best)
    res.evidence.insert(
        0,
        f"{best.fix_id} {best.verdict}: patched {best.passes}/{best.trials}"
        + (f" vs control {best.control}" if best.control else "")
        + (f", judged pass rate {best.pass_rate:.2f}" if best.pass_rate is not None else ""),
    )


def _fix_for(run: Run, a: Attempt) -> Fix | None:
    from .verify import resolve_fix

    if a.file and Path(a.file).is_file():
        f = resolve_fix(run, a.file)
        f.fix_id, f.after = a.fix_id, a.patch
        return f
    for f in run.diagnosis.fixes if run.diagnosis else []:
        if f.fix_id == a.fix_id:
            return f
    return None


def _finish(res: RepairResult, store: Any, run: Run, save: bool) -> RepairResult:
    if not save or store.index.get_run_json(run.run_id) is None:
        return res
    path = store.run_dir(run.run_id) / "repairs" / f"{res.repair_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    res.file = str(path)
    path.write_text(json.dumps(res.to_dict(), indent=2, ensure_ascii=False), "utf-8")
    return res


def repairs(run_id: str, *, store_dir: Any = None) -> list[dict[str, Any]]:
    """Stored repair records of ``run_id``, oldest first."""
    from .store import Store

    store = Store(store_dir)
    d = store.run_dir(store.resolve_run_id(run_id)) / "repairs"
    files = sorted(d.glob("*.json")) if d.is_dir() else []
    return [json.loads(p.read_text("utf-8")) for p in files]


__all__ = ["Attempt", "RepairError", "RepairResult", "repair", "repairs", "write_patch"]
