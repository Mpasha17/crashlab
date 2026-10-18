"""Verification: outcome oracles (``@crashlab.check``); k-rerun verification comes later."""

from __future__ import annotations

from .oracles import CheckSpec, apply_outcome, check, checks_for, run_checks, unregister

__all__ = ["CheckSpec", "apply_outcome", "check", "checks_for", "run_checks", "unregister"]
