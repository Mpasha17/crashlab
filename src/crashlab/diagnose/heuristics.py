"""Candidate culprit steps from cheap signals, and the heuristic-only diagnosis (a
rule table over those signals; the baseline the LLM diagnoser is measured against)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..models import Run, Step, TaxonomyLabel
from .fields import step_fields

_STEP_REF = re.compile(r"\bsteps? (\d+)\b")
_NOT_CALLED = re.compile(r"\b(?:never|not) (?:been )?called\b|\bwithout calling\b", re.I)
_API_ERRORS = (
    "RateLimit",
    "APIConnection",
    "APITimeout",
    "InternalServer",
    "ServiceUnavailable",
    "APIStatus",
    "Timeout",
    "BadGateway",
    "Overloaded",
)
# finding kind -> (label the rule table gives it, rank: lower is a stronger signal)
_FINDING_RULES: dict[str, tuple[TaxonomyLabel | None, int]] = {
    "bad_tool_args": (TaxonomyLabel.BAD_TOOL_ARGUMENTS, 1),
    "step_repetition": (TaxonomyLabel.STEP_REPETITION, 3),
    "tool_error_streak": (TaxonomyLabel.TOOL_ERROR_MISHANDLED, 4),
    "suspicious_tool_result": (TaxonomyLabel.TOOL_ERROR_MISHANDLED, 5),
    "length_cutoff": (None, 6),
    "budget": (None, 7),
    "no_llm_calls": (None, 8),
    "failed_side_effect_claim": (TaxonomyLabel.TOOL_ERROR_MISHANDLED, 4),
    # the agent followed instructions found in data instead of the user's / system prompt
    "tool_result_instructions": (TaxonomyLabel.TASK_SPEC_VIOLATION, 2),
}


@dataclass
class Candidate:
    step: int
    reason: str
    label: TaxonomyLabel | None = None  # what the rule table would say
    quotes: list[str] = field(default_factory=list)  # verbatim from the step
    rank: int = 0  # lower = stronger signal

    def as_dict(self) -> dict[str, Any]:
        label = self.label.value if self.label else None
        return {"step": self.step, "reason": self.reason, "label": label}


def _quote(step: Step) -> str:
    """A short verbatim excerpt that identifies ``step``."""
    fields = step_fields(step)
    for name in ("error", "response.tool_calls[0]", "args", "result", "response.text"):
        if fields.get(name):
            return fields[name][:160]
    return next(iter(fields.values()), "")[:160]


def _finding_step(f: Any) -> int | None:
    steps = list(f.steps or ([f.step] if f.step is not None else []))
    if not steps:
        return None
    if f.kind == "step_repetition" and len(steps) > 1:
        return steps[1]  # the first repeat, not the original call
    return steps[0]


def candidates(run: Run) -> list[Candidate]:
    """Candidate culprit steps with the reason each was picked, strongest first."""
    steps = {s.idx: s for s in run.steps}
    out: list[Candidate] = []

    for s in run.steps:  # an LLM API call that failed: usually external
        if (
            s.error is not None
            and s.operation == "chat"
            and any(k in s.error.type for k in _API_ERRORS)
        ):
            reason = f"the LLM call failed: {s.error.type}"
            out.append(Candidate(s.idx, reason, TaxonomyLabel.EXTERNAL_FAILURE, [_quote(s)], 0))
            break

    for f in run.findings:
        if f.kind == "fork":
            continue
        idx = _finding_step(f)
        if idx is None or idx not in steps:
            continue
        label, rank = _FINDING_RULES.get(f.kind, (None, 8))
        if f.kind == "suspicious_tool_result":  # an HTML / error page, drift: upstream
            label, rank = TaxonomyLabel.EXTERNAL_FAILURE, 6 if "changed shape" in f.message else 5
        quotes = [e.quote for e in f.evidence if e.step_idx == idx][:2] or [_quote(steps[idx])]
        reason = f"runtime check {f.kind} ({f.severity}): {f.message}"
        out.append(Candidate(idx, reason, label, quotes, rank))

    for c in run.checks:
        if c.status not in ("fail", "error"):
            continue
        refs = [int(m) for m in _STEP_REF.findall(c.message or "") if int(m) in steps]
        idx = refs[0] if refs else (run.steps[-1].idx if run.steps else None)
        if idx is None:
            continue
        reason = f"check {c.name} failed: {c.message or ''}".strip()
        label = TaxonomyLabel.TASK_SPEC_VIOLATION
        if _NOT_CALLED.search(c.message or ""):  # "... but issue_refund was never called"
            label = TaxonomyLabel.WRONG_TOOL_SELECTION
        # a check that names no step (a budget, the final answer) is weaker evidence of
        # *where* it went wrong than a detector that does
        out.append(Candidate(idx, reason, label, [_quote(steps[idx])], 2 if refs else 7))

    for s in run.steps:
        if s.source == "blocked":
            out.append(Candidate(s.idx, "a side-effect call was blocked", None, [_quote(s)], 8))
    first_error = next((s for s in run.steps if s.error is not None), None)
    if first_error is not None:
        reason = f"step raised {first_error.error.type}"  # type: ignore[union-attr]
        out.append(Candidate(first_error.idx, reason, None, [_quote(first_error)], 8))

    if run.outcome == "fail" and run.steps and not out:
        last = run.steps[-1]
        out.append(Candidate(last.idx, "last step of a failed run", None, [_quote(last)], 9))

    best: dict[int, Candidate] = {}
    for c in sorted(out, key=lambda c: (c.rank, c.step)):
        if c.step not in best:
            best[c.step] = c
        else:
            best[c.step].reason += f"; {c.reason}"
    return sorted(best.values(), key=lambda c: (c.rank, c.step))


def heuristic_raw(run: Run, cands: list[Candidate] | None = None) -> dict[str, Any]:
    """The rule-table diagnosis as diagnoser-style JSON (validated like an LLM's answer):
    a passing run with no error-level finding, bad tool arguments or error / HTML tool
    result is NO_FAILURE; otherwise the strongest
    labelled candidate wins; otherwise UNKNOWN."""
    cands = candidates(run) if cands is None else cands
    serious = [
        f
        for f in run.findings
        if f.kind != "fork"
        and (
            f.rank >= 2
            or f.kind == "bad_tool_args"
            or (f.kind == "suspicious_tool_result" and "changed shape" not in f.message)
        )
    ]
    if run.outcome != "fail" and run.status != "error" and not serious:
        return {
            "label": "NO_FAILURE",
            "failing_step": None,
            "confidence": 0.5,
            "root_cause": "No failed check and no error-level runtime finding.",
            "evidence": [],
        }
    for c in cands:
        if c.label is not None:
            return {
                "label": c.label.value,
                "failing_step": c.step,
                "confidence": 0.5,
                "root_cause": f"Heuristic: {c.reason}",
                "evidence": [{"step": c.step, "quote": q} for q in c.quotes if q],
            }
    return {
        "label": "UNKNOWN",
        "failing_step": cands[0].step if cands else None,
        "confidence": 0.0,
        "root_cause": "No rule matched the run's signals.",
        "evidence": [],
    }
