"""Outcome oracles: ``@crashlab.check`` functions that turn a run into pass / fail.

A check receives the finished :class:`~crashlab.models.Run` (with its steps) and passes
when it returns ``True`` or ``None``. It fails when it returns ``False`` (message: the
docstring's first line) or a non-empty string (the message), or raises ``AssertionError``
(message: the assertion text). Any other exception is recorded as ``error`` and also makes
the run fail. Checks run automatically when a run ends; ``crashlab check <run>`` re-runs
them on a stored run.
"""

from __future__ import annotations

import inspect
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, overload

from ..models import CheckResult, Run

CheckFn = Callable[[Run], Any]


@dataclass(frozen=True)
class CheckSpec:
    name: str
    agent: str | None  # None: applies to every run
    fn: CheckFn


_lock = threading.Lock()
_registry: dict[tuple[str | None, str], CheckSpec] = {}


@overload
def check(fn: CheckFn, /) -> CheckFn: ...
@overload
def check(
    agent: str | None = None, /, *, name: str | None = None
) -> Callable[[CheckFn], CheckFn]: ...


def check(agent: Any = None, /, *, name: str | None = None) -> Any:
    """Register an outcome oracle for runs of ``agent`` (all runs if omitted).

    ::

        @crashlab.check("refund-bot")
        def refund_not_above_total(run):
            for step in run.tool_calls("issue_refund", ok_only=True):
                assert step.args["amount"] <= 49.0, f"refunded {step.args['amount']}"
    """

    def decorate(fn: CheckFn, agent_name: str | None) -> CheckFn:
        spec = CheckSpec(name=name or fn.__name__, agent=agent_name, fn=fn)
        with _lock:  # re-registering (module reload) replaces the old version
            _registry[(agent_name, spec.name)] = spec
        fn.__crashlab_check__ = spec  # type: ignore[attr-defined]
        return fn

    if callable(agent):
        return decorate(agent, None)
    return lambda fn: decorate(fn, agent)


def checks_for(run_name: str) -> list[CheckSpec]:
    with _lock:
        return [s for (a, _), s in _registry.items() if a is None or a == run_name]


def unregister(agent: str | None = None, name: str | None = None) -> None:
    """Remove registered checks (all, one agent's, or one by name). Mainly for tests."""
    with _lock:
        for key in list(_registry):
            if (agent is None or key[0] == agent) and (name is None or key[1] == name):
                del _registry[key]


def _doc_line(fn: CheckFn) -> str | None:
    doc = inspect.getdoc(fn)
    return doc.strip().splitlines()[0] if doc else None


def run_check(spec: CheckSpec, run: Run) -> CheckResult:
    try:
        value = spec.fn(run)
    except AssertionError as exc:
        return CheckResult(name=spec.name, status="fail", message=str(exc) or _doc_line(spec.fn))
    except Exception as exc:
        return CheckResult(name=spec.name, status="error",
                           message=f"{type(exc).__name__}: {exc}")  # fmt: skip
    if value is None or value is True:
        return CheckResult(name=spec.name, status="pass")
    if value is False:
        return CheckResult(name=spec.name, status="fail", message=_doc_line(spec.fn))
    if isinstance(value, str):
        return CheckResult(
            name=spec.name, status="fail" if value else "pass", message=value or None
        )
    return CheckResult(name=spec.name, status="pass" if value else "fail")


def run_checks(run: Run, specs: list[CheckSpec] | None = None) -> list[CheckResult]:
    """Evaluate the registered checks for ``run`` (does not modify or store the run)."""
    return [run_check(spec, run) for spec in (specs if specs is not None else checks_for(run.name))]


def apply_outcome(run: Run, results: list[CheckResult]) -> None:
    """Set ``run.checks`` and derive ``outcome`` (an exception already made it ``fail``)."""
    run.checks = results
    if not results:
        return
    failed = [r for r in results if r.status != "pass"]
    if run.status == "error" and run.outcome == "fail":
        return
    if failed:
        run.outcome = "fail"
        first = failed[0]
        run.outcome_reason = f"check {first.name} {first.status}ed" + (
            f": {first.message}" if first.message else ""
        )
    else:
        run.outcome = "pass"
        run.outcome_reason = f"{len(results)} check(s) passed"
