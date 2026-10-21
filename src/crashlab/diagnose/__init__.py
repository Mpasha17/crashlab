"""Diagnosis: which step of a recorded run went wrong, why, and the verbatim evidence."""

from .compact import compact
from .diagnoser import (
    DIAGNOSER_RUN_NAME,
    ENV_MODEL,
    DiagnoseError,
    cache_key,
    diagnose,
    parse_json,
)
from .evidence import validate
from .fields import find_quote, step_fields
from .fixes import FIX_PROMPT_VERSION, instance_literals, propose_fixes, try_fix, validate_fix
from .heuristics import Candidate, candidates, heuristic_raw
from .prompt import PROMPT_VERSION
from .taxonomy import DEFINITIONS, FAILURE_LABELS, parse_label

__all__ = [
    "DEFINITIONS",
    "DIAGNOSER_RUN_NAME",
    "FIX_PROMPT_VERSION",
    "ENV_MODEL",
    "FAILURE_LABELS",
    "PROMPT_VERSION",
    "Candidate",
    "DiagnoseError",
    "cache_key",
    "candidates",
    "compact",
    "diagnose",
    "find_quote",
    "heuristic_raw",
    "instance_literals",
    "parse_json",
    "parse_label",
    "propose_fixes",
    "step_fields",
    "try_fix",
    "validate",
    "validate_fix",
]
