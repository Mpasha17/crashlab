"""Runtime-check configuration: defaults < ``[tool.crashlab.checks]`` in pyproject.toml <
``crashlab.record(..., checks=...)``.

::

    [tool.crashlab.checks]
    circuit_breaker = true          # stop the agent on a critical finding (opt-in)

    [tool.crashlab.checks.budget]
    max_steps = 30
    max_cost_usd = 0.05

    [tool.crashlab.checks.step_repetition]
    ignore_tools = ["poll_job"]
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True

    @model_validator(mode="before")
    @classmethod
    def _bool_shorthand(cls, data: Any) -> Any:
        if isinstance(data, bool):  # step_repetition = false
            return {"enabled": data}
        return data


class StepRepetitionConfig(_Cfg):
    """Same tool + same arguments repeated, identical assistant output, or the same
    (near-identical) LLM request sent again."""

    threshold: int = 3  # identical tool calls within `window` steps
    window: int = 10
    critical_at: int = 5  # severity critical (stops the run with the circuit breaker)
    llm_output_threshold: int = 2  # identical assistant text replies in a row
    request_threshold: int = 2  # near-identical LLM requests (not retries after an error)
    ignore_tools: list[str] = Field(default_factory=list)  # e.g. polling tools


class ToolErrorStreakConfig(_Cfg):
    """Consecutive failures of the same tool."""

    threshold: int = 3
    critical_at: int = 5
    count_error_results: bool = True  # an HTML 502 page / {"error": ...} result counts


class BadToolArgsConfig(_Cfg):
    """Tool-call arguments from the model that are not valid JSON, name an unknown tool,
    or do not match the tool's declared JSON schema."""


class SuspiciousToolResultConfig(_Cfg):
    """HTML / error pages, error bodies, empty or truncated results, and schema drift
    against earlier results of the same tool."""

    html: bool = True
    error_bodies: bool = True
    empty: bool = True
    truncated: bool = True
    drift: bool = True
    baseline_runs: int = 5  # also compare against this many earlier runs of the agent


class BudgetConfig(_Cfg):
    """Limits on steps / tokens / cost / wall time. Exceeding one is critical."""

    max_steps: int | None = None
    max_llm_calls: int | None = None
    max_tool_calls: int | None = None
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_seconds: float | None = None
    warn_at: float | None = 0.8  # warn when a limit is this far used; None: no warning


class LengthCutoffConfig(_Cfg):
    """The model stopped because it hit max_tokens / the context limit."""


class ChecksConfig(BaseModel):
    """Which built-in runtime checks run, with what thresholds."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    circuit_breaker: bool = False  # raise CircuitBreakerError on a critical finding
    step_repetition: StepRepetitionConfig = Field(default_factory=StepRepetitionConfig)
    tool_error_streak: ToolErrorStreakConfig = Field(default_factory=ToolErrorStreakConfig)
    bad_tool_args: BadToolArgsConfig = Field(default_factory=BadToolArgsConfig)
    suspicious_tool_result: SuspiciousToolResultConfig = Field(
        default_factory=SuspiciousToolResultConfig
    )
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    length_cutoff: LengthCutoffConfig = Field(default_factory=LengthCutoffConfig)

    @model_validator(mode="before")
    @classmethod
    def _aliases(cls, data: Any) -> Any:
        return _apply_aliases(data) if isinstance(data, dict) else data

    def merged(self, overrides: dict[str, Any] | None) -> ChecksConfig:
        """A copy with ``overrides`` (same shape as the config, partial) deep-merged in."""
        if not overrides:
            return self
        return ChecksConfig.model_validate(
            _deep_merge(self.model_dump(), _apply_aliases(dict(overrides)))
        )


_SECTIONS = (
    "step_repetition",
    "tool_error_streak",
    "bad_tool_args",
    "suspicious_tool_result",
    "budget",
    "length_cutoff",
)


def _apply_aliases(data: dict[str, Any]) -> dict[str, Any]:
    """``loop`` = ``step_repetition``; budget limits may sit at the top level
    (``checks={"max_steps": 20}``); ``{"budget": False}`` disables a detector."""
    data = dict(data)
    if "loop" in data:
        data["step_repetition"] = data.pop("loop")
    for key in [k for k in data if k in BudgetConfig.model_fields and k != "enabled"]:
        budget = data.get("budget")
        budget = dict(budget) if isinstance(budget, dict) else {}
        budget[key] = data.pop(key)
        data["budget"] = budget
    for key in _SECTIONS:
        if isinstance(data.get(key), bool):
            data[key] = {"enabled": data[key]}
    return data


def _deep_merge(base: dict[str, Any], upd: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in upd.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


# --- pyproject ----------------------------------------------------------------------

_pp_lock = threading.Lock()
_pp_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def _toml_load(path: Path) -> dict[str, Any]:
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover - 3.10
        try:
            import tomli as tomllib
        except ImportError:
            return {}
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except (OSError, ValueError):
        return {}


def find_pyproject(start: str | os.PathLike[str] | None = None) -> Path | None:
    here = Path(start or os.getcwd()).resolve()
    for d in (here, *here.parents):
        candidate = d / "pyproject.toml"
        if candidate.is_file():
            return candidate
    return None


def pyproject_section(name: str, start: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """``[tool.crashlab.<name>]`` from the nearest pyproject.toml (cached by mtime)."""
    if os.environ.get("CRASHLAB_NO_PYPROJECT"):
        return {}
    path = find_pyproject(start)
    if path is None:
        return {}
    key = str(path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    with _pp_lock:
        hit = _pp_cache.get(key)
        if hit is None or hit[0] != mtime:
            hit = (mtime, (_toml_load(path).get("tool") or {}).get("crashlab") or {})
            _pp_cache[key] = hit
    section = hit[1].get(name)
    return dict(section) if isinstance(section, dict) else {}


def resolve_config(value: Any = None, *, base: dict[str, Any] | None = None) -> ChecksConfig:
    """Turn ``record(checks=...)`` into a config.

    ``None`` / ``True``: defaults + pyproject. ``False``: everything off. A dict is merged
    over defaults + pyproject; a :class:`ChecksConfig` is used as given. ``base`` (a stored
    run's ``check_config``) replaces defaults + pyproject, e.g. when re-checking a run.
    """
    if isinstance(value, ChecksConfig):
        return value
    if value is False:
        return ChecksConfig(enabled=False)
    if base is not None:
        cfg = ChecksConfig.model_validate(base)
    else:
        cfg = ChecksConfig().merged(pyproject_section("checks"))
    if isinstance(value, dict):
        cfg = cfg.merged(value)
    elif value not in (None, True):
        raise TypeError(f"checks= must be a ChecksConfig, dict, bool or None, not {value!r}")
    return cfg
