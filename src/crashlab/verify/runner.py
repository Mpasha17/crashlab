"""``crashlab.verify``: does a fix really fix the failing run, and break nothing else?

- **Patched arm:** fork the failing run ``k`` times with the fix, live from the step it
  applies to.
- **Control arm:** fork it ``k`` times from the same step with *no* patch: the noise
  floor (how often the agent passes anyway).
- **Regression:** rerun past good runs from step 0 with the patch; any pass -> fail
  rejects the fix.

Provider failures (429, 5xx, connection errors) are ``errored`` and never count as a
fail. ``max_cost_usd`` stops new trials once the spend reaches it.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .._ids import new_ulid
from ..models import Fix, RegressionResult, Run, Trial, Verification
from .stats import arm, verdict

INFRA_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}
_INFRA_RE = re.compile(
    r"rate.?limit|too many requests|overloaded|high demand|unavailable|timed? ?out|"
    r"connection|internalservererror|apiconnection|apitimeout|\b(429|500|502|503|504|529)\b",
    re.I,
)
DEFAULT_K = 5
DEFAULT_REGRESSION_LIMIT = 10


class VerifyError(ValueError):
    pass


def is_infra_error(error: Any) -> bool:
    """True for a provider failure (rate limit, overload, 5xx, network), as opposed to
    the agent misbehaving."""
    if error is None:
        return False
    code = getattr(error, "status_code", None)
    if code in INFRA_STATUS:
        return True
    text = f"{getattr(error, 'type', type(error).__name__)} {getattr(error, 'message', error)}"
    return bool(_INFRA_RE.search(text))


def _live_usage(run: Run) -> tuple[float, int]:
    cost, tokens = 0.0, 0
    for s in run.steps:
        if s.operation != "chat" or s.source not in ("live", "patched"):
            continue
        cost += float(s.attributes.get("crashlab.usage.cost_usd") or 0)
        tokens += int(s.attributes.get("crashlab.usage.total_tokens") or 0)
    return cost, tokens


def resolve_fix(run: Run, fix: Any) -> Fix:
    """A :class:`Fix`, its id/rank on the run's diagnosis (``"fix-2"``, ``2``) or a patch
    file (``fix-1.yaml``)."""
    from ..patch import load as load_patch

    if isinstance(fix, Fix):
        return fix
    ref = str(fix)
    fixes = run.diagnosis.fixes if run.diagnosis else []
    for f in fixes:
        if ref in (f.fix_id, str(f.rank)) or (f.file and Path(f.file).name == ref):
            return f
    path = Path(ref).expanduser()
    if path.is_file():
        ps = load_patch(str(path))
        if len(ps.patches) != 1:
            raise VerifyError(f"{path} holds {len(ps.patches)} patches; verify takes one")
        p = ps.patches[0]
        return Fix(fix_id=path.stem, kind=p.kind, after=p.describe(), as_patch=p, file=str(path))
    have = ", ".join(f.fix_id for f in fixes) or "none (run `crashlab fix` first)"
    raise VerifyError(f"no fix {ref!r} on run {run.run_id} (fixes: {have}) and no such file")


def _default_regression(store: Any, run: Run, limit: int) -> list[str]:
    out: list[str] = []
    for row in store.list_runs(limit=None, name=run.name):
        rid = row["run_id"]
        if rid == run.run_id or row.get("outcome") != "pass":
            continue
        try:
            past = store.load_run(rid)
        except Exception:  # noqa: BLE001 - a broken old run is just not a candidate
            continue
        if past.forked_from or past.replay_of or any(t.startswith("verify") for t in past.tags):
            continue
        out.append(rid)
        if len(out) >= limit:
            break
    return out


def resolve_regression(store: Any, run: Run, regress: Any, limit: int) -> list[str]:
    """``None``: past passing runs of the same agent (newest ``limit``); ``"none"`` / ``[]``:
    no regression; ``"tag:X"``: passing runs tagged X; else run ids (list or comma text)."""
    if regress is None:
        return _default_regression(store, run, limit)
    if isinstance(regress, str):
        if regress.strip().lower() in ("", "none", "no"):
            return []
        if regress.startswith("tag:"):
            tag = regress[4:]
            rows = store.list_runs(limit=None)
            ids = [r["run_id"] for r in rows if r["run_id"] != run.run_id]
            return [i for i in ids if tag in store.load_run(i).tags][:limit]
        regress = [x for x in regress.split(",") if x.strip()]
    return [store.resolve_run_id(str(x).strip()) for x in regress]


BLOCKED = "SideEffectBlockedError"


def _failed_note(root: Any, t: Trial) -> str | None:
    """``"failed: check (message); ..."`` for a regression run that failed its checks."""
    if t.status != "fail" or not t.run_id:
        return None
    try:
        from ..store import Store

        checks = Store(root).load_run(t.run_id).checks
    except Exception:  # noqa: BLE001 - a note is best effort
        checks = []
    bad = [f"{c.name} ({c.message})" if c.message else c.name
           for c in checks if c.status != "pass"]  # fmt: skip
    if not bad and t.failed_checks:
        bad = list(t.failed_checks)
    return f"failed: {'; '.join(bad)}"[:300] if bad else None


def _blocked_steps(run: Run) -> list[str]:
    return [f"{s.name} (step {s.idx})" for s in run.steps if s.source == "blocked"]


def _trial(
    role: str,
    source: Run,
    at: int,
    patch: Any,
    client: Any,
    root: Any,
    tag: str,
    allow_live_side_effects: bool = False,
) -> Trial:
    from ..replay import fork
    from ..store import Store

    try:
        branch = fork(
            source,
            at_step=at,
            patch=patch,
            client=client,
            store_dir=root,
            allow_live_side_effects=allow_live_side_effects,
        )
    except Exception as e:  # noqa: BLE001 - the harness failing says nothing about the fix
        msg = f"{type(e).__name__}: {' '.join(str(e).split())[:200]}"
        return Trial(role=role, source_run=source.run_id, status="errored", error=msg)  # type: ignore[arg-type]
    cost, tokens = _live_usage(branch)
    branch.tags.append(tag)
    branch.tags.append(f"verify-{role}")
    Store(root).update_run(branch)
    failed = [c.name for c in branch.checks if c.status != "pass"]
    err = f"{branch.error.type}: {branch.error.message[:200]}" if branch.error else None
    if branch.status == "error" and is_infra_error(branch.error):
        status = "errored"
    elif blocked := _blocked_steps(branch):
        # the agent made a side-effect tool call the source run never made (often just a
        # different argument, e.g. "widgets" for "widget"); crashlab refused to run it live
        # (even if the agent caught the error and carried on), so the trial cannot say
        # whether the run passes. Not counted, in any arm: --allow-live-side-effects
        # against a sandbox grades it.
        status = "errored"
        err = f"{BLOCKED}: {', '.join(blocked)} blocked (side_effects=True, not in the recording)"
    else:
        status = "pass" if branch.outcome == "pass" else "fail"
    return Trial(
        role=role,  # type: ignore[arg-type]
        source_run=source.run_id,
        run_id=branch.run_id,
        status=status,  # type: ignore[arg-type]
        outcome=branch.outcome,
        failed_checks=failed,
        error=err,
        cost_usd=cost,
        tokens=tokens,
    )


def verify(
    run: Run | str = "last",
    fix: Any = 1,
    *,
    k: int = DEFAULT_K,
    control: bool = True,
    regress: Any = None,
    regression_limit: int = DEFAULT_REGRESSION_LIMIT,
    client: Any = None,
    store_dir: Any = None,
    max_cost_usd: float | None = None,
    on_trial: Callable[[Trial], None] | None = None,
    save: bool = True,
    pace_s: float = 0.0,
    allow_live_side_effects: bool = False,
) -> Verification:
    """Verify ``fix`` on the failing ``run``; returns (and stores) a :class:`Verification`.

    See the module docstring for the arms. ``client`` is passed to every fork (the agent's
    own client otherwise). ``regress``: see :func:`resolve_regression`. ``pace_s``: sleep
    between trials (free-tier rate limits). ``allow_live_side_effects``: run
    ``side_effects=True`` tools live in the forks (sandboxes only); otherwise a trial that
    needs a side-effect call the recording never made is blocked and not counted."""
    from ..diagnose.fixes import _fork_step
    from ..patch import as_patchset
    from ..store import Store

    if k < 1:
        raise VerifyError("k must be at least 1")
    store = Store(store_dir)
    if isinstance(run, str):
        run = store.load_run(run)
    f = resolve_fix(run, fix)
    if f.as_patch is None:
        raise VerifyError(f"{f.fix_id} has no patch")
    patch = f.as_patch
    at = _fork_step(patch, run)
    vid = new_ulid()
    tag = f"verify:{vid}"
    v = Verification(
        verification_id=vid,
        run_id=run.run_id,
        fix_id=f.fix_id,
        patch_description=patch.describe(),
        fix_file=f.file,
        k=k,
        fork_step=at,
        max_cost_usd=max_cost_usd,
    )
    if f.test_only:
        v.reasons.append("test-only counterfactual (tool_result): proves a cause, not a fix")

    def spent() -> bool:
        return max_cost_usd is not None and v.cost_usd >= max_cost_usd

    def record(t: Trial) -> None:
        if pace_s and v.trials:
            time.sleep(pace_s)
        v.trials.append(t)
        v.cost_usd += t.cost_usd
        v.tokens += t.tokens
        if on_trial:
            on_trial(t)

    plan: list[tuple[str, Any]] = []
    for _ in range(k):
        plan.append(("patched", patch))
        if control:
            plan.append(("control", None))
    for role, p in plan:
        if spent():
            v.budget_exhausted = True
            break
        record(_trial(role, run, at, p, client, store.root, tag, allow_live_side_effects))

    reg_ids = resolve_regression(store, run, regress, regression_limit)
    global_patch = patch.scope != "step"
    for rid in reg_ids:
        if spent():
            v.budget_exhausted = True
            v.regression.append(RegressionResult(run_id=rid, status="skipped", note="budget"))
            continue
        if not global_patch:
            note = "step-bound patch only applies to the failing run"
            v.regression.append(RegressionResult(run_id=rid, status="skipped", note=note))
            continue
        past = store.load_run(rid)
        try:
            from ..patch import preview_patches

            preview_patches(as_patchset(patch), past, fork_step=0)
        except Exception as e:  # noqa: BLE001 - e.g. an insert anchor that run lacks
            note = f"patch does not apply: {' '.join(str(e).split())[:150]}"
            v.regression.append(RegressionResult(run_id=rid, status="skipped", note=note))
            continue
        t = _trial(
            "regression",
            past,
            _fork_step(patch, past),
            patch,
            client,
            store.root,
            tag,
            allow_live_side_effects,
        )
        record(t)
        v.regression.append(
            RegressionResult(
                run_id=rid,
                status=t.status,
                trial_run=t.run_id,
                note=t.error or _failed_note(store.root, t),
            )
        )

    v.patched = arm([t for t in v.trials if t.role == "patched"], k)
    v.control = arm([t for t in v.trials if t.role == "control"], k) if control else None
    v.verdict, reasons = verdict(  # type: ignore[assignment]
        v.patched,
        v.control,
        v.regression,
        k=k,
        original_failed=run.outcome != "pass",
        budget_exhausted=v.budget_exhausted,
    )
    blocked = [t for t in v.trials if t.status == "errored" and BLOCKED in (t.error or "")]
    if blocked:
        reasons.append(
            f"{len(blocked)} trial(s) stopped at a side_effects=True tool call the recording "
            "never made (crashlab never runs those live in a fork), not counted; register a "
            "fake (crashlab.fake_tool) or rerun with --allow-live-side-effects against a "
            "sandbox to count them"
        )
    v.reasons = reasons + v.reasons
    if save:
        save_verification(store, run, v)
    return v


def save_verification(store: Any, run: Run, v: Verification) -> Path:
    path = store.run_dir(run.run_id) / "verifications" / f"{v.verification_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(v.model_dump(mode="json"), indent=2), "utf-8")
    fresh = store.load_run(run.run_id)
    fresh.fix_attempts.append(
        {
            "verification_id": v.verification_id,
            "fix_id": v.fix_id,
            "patch": v.patch_description,
            "verdict": v.verdict,
            "patched": f"{v.patched.passes}/{v.patched.n}",
            "control": f"{v.control.passes}/{v.control.n}" if v.control else None,
            "regressions": len(v.regressions),
            "cost_usd": round(v.cost_usd, 6),
            "created_at": v.created_at.isoformat(),
            "file": str(path),
        }
    )
    store.update_run(fresh)
    run.fix_attempts = fresh.fix_attempts
    return path


def verifications(store: Any, run_id: str) -> list[Verification]:
    """Stored verifications of ``run_id``, oldest first."""
    d = store.run_dir(run_id) / "verifications"
    files: Iterable[Path] = sorted(d.glob("*.json")) if d.is_dir() else []
    return [Verification.model_validate_json(p.read_text("utf-8")) for p in files]
