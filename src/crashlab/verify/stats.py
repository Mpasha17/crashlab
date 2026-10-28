"""Statistics for ``crashlab verify``: Wilson intervals, pass^k and the verdict."""

from __future__ import annotations

from math import comb, sqrt

from ..models import ArmStats, RegressionResult, Trial

Z95 = 1.959963984540054


def wilson(passes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for ``passes`` out of ``n`` (``(0, 1)`` when ``n`` is 0)."""
    if n <= 0:
        return (0.0, 1.0)
    p = passes / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def pass_hat_k(passes: int, n: int, k: int) -> float | None:
    """Unbiased estimate of P(all k trials pass) from ``passes`` of ``n`` (tau-bench's
    pass^k); with n == k it is 1.0 only if every trial passed."""
    if n <= 0 or k <= 0 or k > n:
        return None
    return comb(passes, k) / comb(n, k)


def arm(trials: list[Trial], k: int) -> ArmStats:
    passes = sum(t.status == "pass" for t in trials)
    fails = sum(t.status == "fail" for t in trials)
    errored = sum(t.status == "errored" for t in trials)
    n = passes + fails
    return ArmStats(
        passes=passes,
        fails=fails,
        errored=errored,
        rate=passes / n if n else None,
        ci95=wilson(passes, n) if n else None,
        pass_hat_k=pass_hat_k(passes, n, min(k, n)) if n else None,
    )


def verdict(
    patched: ArmStats,
    control: ArmStats | None,
    regression: list[RegressionResult],
    *,
    k: int,
    original_failed: bool = True,
    budget_exhausted: bool = False,
) -> tuple[str, list[str]]:
    """``(verdict, reasons)``.

    - REJECTED: a past good run fails with the patch, or the patch does no better than
      the unpatched control.
    - VERIFIED: no regression, and the patched Wilson lower bound clears the control's
      upper bound (or control failed every trial while the patched arm passed every one,
      over at least 3 trials each).
    - LIKELY: better than control, no regression, not significant (or no control).
    - INCONCLUSIVE: too few non-errored trials to say anything (fewer than min(2, k) in
      the patched arm, or in a control arm that ran).
    """
    reasons: list[str] = []
    fails = [r for r in regression if r.status == "fail"]
    if fails:
        ids = ", ".join(r.run_id for r in fails)
        return "REJECTED", [
            f"regression: {len(fails)} past good run(s) fail with the patch ({ids})"
        ]
    if patched.n == 0:
        why = f"no usable patched trial ({patched.errored} errored)"
        return "INCONCLUSIVE", [why + ("; budget exhausted" if budget_exhausted else "")]
    need = min(2, k)
    thin = [
        f"{name} {arm_.n}/{need}"
        for name, arm_ in (("patched", patched), ("control", control))
        if arm_ is not None and (arm_.n or arm_.errored) and arm_.n < need
    ]
    if thin:
        return "INCONCLUSIVE", [
            f"too few usable trials ({', '.join(thin)} needed; "
            f"{patched.errored + (control.errored if control else 0)} errored)"
            + ("; budget exhausted" if budget_exhausted else "")
        ]
    if patched.errored:
        reasons.append(f"{patched.errored} patched trial(s) errored, not counted")
    if control is not None and control.errored:
        reasons.append(f"{control.errored} control trial(s) errored, not counted")
    if budget_exhausted:
        reasons.append("budget cap reached before every trial ran")
    reg_err = [r for r in regression if r.status in ("errored", "skipped")]
    if reg_err:
        reasons.append(f"{len(reg_err)} regression run(s) errored or skipped, not checked")
    p_lo, _ = patched.ci95 or (0.0, 1.0)
    if control is None or control.n == 0:
        base = 0.0 if original_failed else None
        reasons.append("no usable control trials: compared with the original run only")
        if base is not None and patched.passes > 0:
            return "LIKELY", reasons
        return ("REJECTED" if patched.passes == 0 else "INCONCLUSIVE"), reasons
    if (patched.rate or 0) <= (control.rate or 0):
        reasons.insert(
            0,
            f"no improvement: patched {patched.passes}/{patched.n} vs control "
            f"{control.passes}/{control.n}",
        )
        return "REJECTED", reasons
    _, c_hi = control.ci95 or (0.0, 1.0)
    clean_sweep = control.passes == 0 and patched.fails == 0 and min(patched.n, control.n) >= 3
    if p_lo > c_hi or clean_sweep:
        if patched.n < k or control.n < k or reg_err:
            reasons.insert(0, "significant, but some trials or regression runs are missing")
            return "LIKELY", reasons
        reasons.insert(
            0,
            f"patched {patched.passes}/{patched.n} (CI low {p_lo:.2f}) vs control "
            f"{control.passes}/{control.n} (CI high {c_hi:.2f})",
        )
        return "VERIFIED", reasons
    reasons.insert(
        0,
        f"better than control ({patched.passes}/{patched.n} vs {control.passes}/{control.n}) "
        "but the intervals overlap; raise -k",
    )
    return "LIKELY", reasons
