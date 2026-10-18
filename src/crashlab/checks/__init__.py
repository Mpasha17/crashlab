"""Runtime checks: cheap deterministic detectors that run while an agent is recorded
(and on stored runs with ``crashlab check``) and report :class:`~crashlab.models.Finding`s.

Not to be confused with ``@crashlab.check`` outcome oracles, which decide whether a run
passed. Findings describe *how* it behaved: loops, tool error streaks, bad tool
arguments, suspicious tool results, blown budgets, length cut-offs.
"""

from .config import (
    BadToolArgsConfig,
    BudgetConfig,
    ChecksConfig,
    LengthCutoffConfig,
    StepRepetitionConfig,
    SuspiciousToolResultConfig,
    ToolErrorStreakConfig,
    pyproject_section,
    resolve_config,
)
from .detectors import KINDS, classify_result
from .engine import CheckEngine, CircuitBreakerError, evaluate, max_severity, replace_findings

__all__ = [
    "KINDS",
    "BadToolArgsConfig",
    "BudgetConfig",
    "CheckEngine",
    "ChecksConfig",
    "CircuitBreakerError",
    "LengthCutoffConfig",
    "StepRepetitionConfig",
    "SuspiciousToolResultConfig",
    "ToolErrorStreakConfig",
    "classify_result",
    "evaluate",
    "max_severity",
    "pyproject_section",
    "replace_findings",
    "resolve_config",
]
