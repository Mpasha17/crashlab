"""Verification: outcome oracles (``@crashlab.check``) and k-rerun fix verification."""

from __future__ import annotations

from .oracles import CheckSpec, apply_outcome, check, checks_for, run_checks, unregister
from .runner import VerifyError, is_infra_error, resolve_fix, verifications, verify
from .stats import pass_hat_k, verdict, wilson

__all__ = [
    "CheckSpec",
    "VerifyError",
    "apply_outcome",
    "check",
    "checks_for",
    "is_infra_error",
    "pass_hat_k",
    "resolve_fix",
    "run_checks",
    "unregister",
    "verdict",
    "verifications",
    "verify",
    "wilson",
]
