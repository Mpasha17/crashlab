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
_ACTION_VIOLATION = re.compile(
    r"\bordered\b|\brefunded\b|\bplaced\b|\blimit\b|\bwindow:\b|\bexceed",
    re.I,
)
# Check messages that are usually consequences of a tool/upstream failure, not the
# instruction gap. When an error-level tool finding exists, demote these so the
# finding label (TOOL_ERROR / EXTERNAL) wins over TASK_SPEC (loop7 real-agent gaps).
_TOOL_FAILURE_CONSEQUENCE = re.compile(
    r"no order placed|ordered without a valid price|without a valid price",
    re.I,
)
_TOOL_FAILURE_FINDINGS = frozenset(
    {
        "unhandled_tool_error",
        "tool_error_streak",
        "suspicious_tool_result",
        "failed_side_effect_claim",
    }
)
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
    # the agent code let a tool exception end the run; the model never saw it
    "unhandled_tool_error": (TaxonomyLabel.TOOL_ERROR_MISHANDLED, 4),
}


@dataclass
class Candidate:
    step: int
    reason: str
    label: TaxonomyLabel | None = None  # what the rule table would say
    quotes: list[str] = field(default_factory=list)  # verbatim from the step
    rank: int = 0  # lower = stronger signal
    quote_from: int | None = None  # step the quotes came from (tool) when step is the decision chat

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


def _caller_chat(run: Run, idx: int) -> int | None:
    """The chat step that decided on tool step ``idx``, if any."""
    return next(
        (s.idx for s in sorted(run.steps, key=lambda s: -s.idx)
         if s.idx < idx and s.operation == "chat"),
        None,
    )  # fmt: skip


def _decision_step(run: Run, idx: int) -> int:
    """For a wrong action, blame the LLM decision (caller chat), not the tool execution."""
    step = next((s for s in run.steps if s.idx == idx), None)
    if step is None or step.operation != "execute_tool":
        return idx
    caller = _caller_chat(run, idx)
    return caller if caller is not None else idx


def _step_for_check(run: Run, message: str, steps: dict[int, Step]) -> int | None:
    """Resolve where a failing check points: an explicit ``step N``, else a tool whose
    args/result share a number with the message, else the last tool, else the last step.

    Real shop oracles say ``ordered 10 units (limit 5)`` with no step index; blaming the
    last chat reply (the previous fallback) hid the decision that placed the order.
    """
    refs = [int(m) for m in _STEP_REF.findall(message or "") if int(m) in steps]
    if refs:
        return refs[0]
    nums = set(re.findall(r"\d+", message or ""))
    tools = [s for s in run.steps if s.operation == "execute_tool"]
    for s in reversed(tools):
        blob = " ".join(_walk_nums(s.args)) + " " + " ".join(_walk_nums(s.result))
        if nums and any(n in blob for n in nums):
            return s.idx
    if tools:
        return tools[-1].idx
    return run.steps[-1].idx if run.steps else None


def _walk_nums(value: Any) -> list[str]:
    if isinstance(value, dict):
        out: list[str] = []
        for v in value.values():
            out.extend(_walk_nums(v))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            out.extend(_walk_nums(v))
        return out
    if value is None:
        return []
    return re.findall(r"\d+", str(value))


def _system_prompt(run: Run) -> str:
    for s in run.steps:
        if s.operation == "chat":
            sys = (s.request or {}).get("system") or ""
            if sys:
                return sys if isinstance(sys, str) else str(sys)
    return ""


_STEER_NUMERIC = re.compile(
    r"divide by\s*\d+|multiply by\s*\d+|/\s*\d+|\*\s*0\.\d+|in cents|stored in cents|"
    r"restocking|never refunded|refund\s+total\s*\*\s*0\.\d+|withhold|\bdeduct\b.*\bfee\b",
    re.I,
)
_STEER_AVAIL = re.compile(
    r"always (?:treat|report|assume).{0,40}\b(?:in stock|available)\b|"
    r"ignore (?:zero )?availability|ignore stock|never out of stock|"
    r"always available",
    re.I,
)
_STEER_PERM = re.compile(
    r"no auth(?:entication|orization)? needed|skip (?:permission|auth)|"
    r"bypass permission|has admin|elevated (?:access|privileges)|without (?:auth|permission)",
    re.I,
)
_AMOUNT_MISMATCH = re.compile(
    r"refund_amount_equals_order_total|refunded .+ but the order total|under-?refund",
    re.I,
)


def _tool_descriptions(run: Run) -> list[tuple[str, str]]:
    """``(tool_name, description)`` from the first chat request that lists tools."""
    out: list[tuple[str, str]] = []
    for s in run.steps:
        if s.operation != "chat":
            continue
        for t in (s.request or {}).get("tools") or []:
            if not isinstance(t, dict) or not t.get("name"):
                continue
            desc = t.get("description")
            if desc is None and isinstance(t.get("function"), dict):
                desc = t["function"].get("description")
            if isinstance(desc, str) and desc.strip():
                out.append((str(t["name"]), desc))
        if out:
            break
    return out


def _steering_kinds(desc: str) -> set[str]:
    """Kinds of misleading steer a tool description may encode."""
    kinds: set[str] = set()
    if _STEER_NUMERIC.search(desc):
        kinds.add("numeric_transform")
    if _STEER_AVAIL.search(desc):
        kinds.add("availability")
    if _STEER_PERM.search(desc):
        kinds.add("permission")
    return kinds


def _claim_reflected(run: Run, tool_name: str, kinds: set[str], reason: str) -> bool:
    """True when the run's tool use / check failure matches the description's steer."""
    if "numeric_transform" in kinds:
        if _AMOUNT_MISMATCH.search(reason):
            return True
        for s in run.steps:
            if s.operation != "execute_tool":
                continue
            args = str(s.args)
            if re.search(r"/\s*100|\*\s*0\.\d+", args):
                return True
            if s.name == tool_name and re.search(r"/\s*\d+|\*\s*0\.\d+", args):
                return True
        if re.search(r"answered .+ but ground truth|/100|cents", reason, re.I):
            # wrong numeric answer after a scale steer
            for s in run.steps:
                if (
                    s.operation == "execute_tool"
                    and s.name == tool_name
                    and re.search(r"/\s*\d+", str(s.args))
                ):
                    return True
    if "availability" in kinds and re.search(r"stock|availab|out of stock", reason, re.I):
        # a later side-effect order (or the check naming one) after a stock lookup
        ordered = any(s.operation == "execute_tool" and s.name == "place_order" for s in run.steps)
        if ordered or "placed an order" in reason.lower():
            return True
    if "permission" in kinds and re.search(r"permission|unauthoriz|admin|auth", reason, re.I):
        for s in run.steps:
            if s.operation == "execute_tool" and s.name == tool_name:
                return True
    return False


def _misleading_tool_cause(run: Run, reason: str, decision: int) -> str | None:
    """Blame a tool description when its steer is reflected in the failing behaviour.

    Evidence relationship: (steering claim in a tool description) + (tool use / check
    failure consistent with following that claim) → misleading description, not a
    generic symptom restatement. Honest descriptions without a matching steer return None.
    """
    for name, desc in _tool_descriptions(run):
        kinds = _steering_kinds(desc)
        if not kinds or not _claim_reflected(run, name, kinds, reason):
            continue
        if "numeric_transform" in kinds:
            detail = "a numeric transform / unit claim (cents, scale, fee haircut)"
        elif "availability" in kinds:
            detail = "an availability claim that overrides stock"
        else:
            detail = "a permission / auth bypass claim"
        return (
            f"Heuristic: misleading tool description for {name} steered the model "
            f"({detail}) at chat step {decision}"
        )
    return None


def _missing_rule_cause(run: Run, reason: str, decision: int) -> str | None:
    """Name the instruction gap (missing / ignored / contradictory), not only the symptom."""
    mislead = _misleading_tool_cause(run, reason, decision)
    if mislead:
        return mislead
    prompt = _system_prompt(run)
    # injected / contradictory instructions in a tool result (prompt injection)
    if "tool_result_instructions" in reason or "prompt injection" in reason.lower():
        return (
            "Heuristic: the model followed contradictory instructions embedded in a tool "
            f"result over the system prompt (chat step {decision})"
        )
    if not prompt:
        return None
    # shop: "ordered 10 units (limit 5)"
    m = re.search(r"ordered (\d+) units \(limit (\d+)\)", reason, re.I)
    if m:
        limit = m.group(2)
        has_rule = bool(
            re.search(
                rf"\b(?:never |no more than |at most |maximum (?:of )?){re.escape(limit)}\b",
                prompt,
                re.I,
            )
            or re.search(rf"\b{re.escape(limit)}\s*-?\s*unit", prompt, re.I)
            or re.search(rf"more than {re.escape(limit)}", prompt, re.I)
        )
        if not has_rule and not re.search(rf"\b{re.escape(limit)}\b", prompt):
            return (
                f"Heuristic: the system prompt is missing the {limit}-unit order limit; "
                f"at chat step {decision} the model decided to place an over-limit order"
            )
        if has_rule or re.search(rf"\b{re.escape(limit)}\b", prompt):
            return (
                f"Heuristic: the model ignored the {limit}-unit order limit in the system "
                f"prompt; at chat step {decision} it still decided to place an over-limit order"
            )
    # refund: "window: 30 days" / "30-day"
    m = re.search(r"window:\s*(\d+)\s*days", reason, re.I) or re.search(
        r"\b(\d+)[ -]?day(?:s)?\b", reason, re.I
    )
    if m:
        days = m.group(1)
        has_window = bool(
            re.search(rf"\b{re.escape(days)}\s*-?\s*day", prompt, re.I)
            or re.search(rf"\bwithin\s+{re.escape(days)}\b", prompt, re.I)
        )
        if not has_window:
            return (
                f"Heuristic: the system prompt is missing the {days}-day refund window; "
                f"at chat step {decision} the model decided to issue an out-of-window refund"
            )
        return (
            f"Heuristic: the model ignored the {days}-day refund window in the system prompt; "
            f"at chat step {decision} it still decided to issue an out-of-window refund"
        )
    return None


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

    for s in run.steps:  # the provider rejected the request (400): a config / model mismatch
        if s.error is None or s.operation != "chat" or "BadRequest" not in s.error.type:
            continue
        detail = s.error.detail
        reason = f"the provider rejected LLM call {s.idx}: {detail[:160]}"
        if "tool choice is required" in detail.lower():
            reason += (" (the request forces a tool call, tool_choice='required', and the model "
                       "answered in text: a model that follows tool calling, or a framework "
                       "setting that lets it reply / call its final-answer tool)")  # fmt: skip
        out.append(Candidate(s.idx, reason, None, [_quote(s)], 2))
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
        # error-level tool streaks outrank warn-level repetition on the same failure mode
        if f.kind == "tool_error_streak" and f.severity == "error":
            rank = 2
        quotes = [e.quote for e in f.evidence if e.step_idx == idx][:2] or [_quote(steps[idx])]
        reason = f"runtime check {f.kind} ({f.severity}): {f.message}"
        out.append(Candidate(idx, reason, label, quotes, rank))

    for c in run.checks:
        if c.status not in ("fail", "error"):
            continue
        refs = [int(m) for m in _STEP_REF.findall(c.message or "") if int(m) in steps]
        damage = _step_for_check(run, c.message or "", steps)
        if damage is None:
            continue
        reason = f"check {c.name} failed: {c.message or ''}".strip()
        label = TaxonomyLabel.TASK_SPEC_VIOLATION
        if _NOT_CALLED.search(c.message or ""):  # "... but issue_refund was never called"
            label = TaxonomyLabel.WRONG_TOOL_SELECTION
        msg = c.message or ""
        # a concrete action violation (ordered N / refunded / limit) points at a tool; a
        # budget / step-count check does not — keep those weak so detectors win (loops)
        action = bool(_ACTION_VIOLATION.search(msg))
        quote_step = damage if damage in steps else damage
        idx = damage
        if (
            action
            and label in (TaxonomyLabel.TASK_SPEC_VIOLATION, TaxonomyLabel.WRONG_TOOL_SELECTION)
            and damage in steps
            and steps[damage].operation == "execute_tool"
        ):
            # wrong action: blame the chat that decided on the tool, not the tool itself
            idx = _decision_step(run, damage)
        rank = 2 if (refs or action) else 7
        if _TOOL_FAILURE_CONSEQUENCE.search(msg) and any(
            f.kind in _TOOL_FAILURE_FINDINGS and f.severity in ("error", "warn")
            for f in run.findings
        ):
            # symptom of the tool failure — let the finding candidate outrank TASK_SPEC
            rank = 7
        out.append(
            Candidate(
                idx,
                reason,
                label,
                [_quote(steps[quote_step])],
                rank,
                quote_from=quote_step if quote_step != idx else None,
            )
        )

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
            cause = _missing_rule_cause(run, c.reason, c.step) or f"Heuristic: {c.reason}"
            # quotes may come from the damage tool step while failing_step is the decision chat
            ev_step = c.quote_from if c.quote_from is not None else c.step
            return {
                "label": c.label.value,
                "failing_step": c.step,
                "confidence": 0.5,
                "root_cause": cause,
                "evidence": [{"step": ev_step, "quote": q} for q in c.quotes if q],
            }
    return {
        "label": "UNKNOWN",
        "failing_step": cands[0].step if cands else None,
        "confidence": 0.0,
        "root_cause": "No rule gives a label"
        + (f"; strongest signal: {cands[0].reason}" if cands else " (no signal in the run)."),
        "evidence": [{"step": cands[0].step, "quote": q} for q in cands[0].quotes if q]
        if cands
        else [],
    }
