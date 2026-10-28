"""Turn the diagnoser's raw JSON into a validated :class:`Diagnosis`.

Rules (the anti-hallucination guarantee): the label must be in the taxonomy; a failure
label needs an existing failing step and at least one quote found verbatim in the step it
cites; quotes that are not found are dropped. Anything else becomes ``UNKNOWN`` with
confidence capped, and ``notes`` says why. A run whose outcome is ``fail`` cannot be
``NO_FAILURE``."""

from __future__ import annotations

from typing import Any

from ..models import Diagnosis, Evidence, Run, TaxonomyLabel
from .fields import find_quote, quote_in_run_error
from .taxonomy import parse_label

UNKNOWN_CONFIDENCE_CAP = 0.2
MAX_EVIDENCE = 6


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def _confidence(value: Any) -> float:
    try:
        c = float(value)
    except (TypeError, ValueError):
        return 0.0
    if c > 1.0 and c <= 100.0:  # "85" meaning 85%
        c /= 100.0
    return min(max(c, 0.0), 1.0)


_NOT_QUOTE = ("step", "step_idx", "field", "why", "reason", "explanation", "note")


def validate(raw: Any, run: Run, **extra: Any) -> Diagnosis:
    """Build the :class:`Diagnosis` for ``run`` from the diagnoser's JSON object ``raw``."""
    notes: list[str] = []
    if not isinstance(raw, dict):
        return Diagnosis(
            run_id=run.run_id, notes=["the diagnoser did not return a JSON object"], **extra
        )
    steps = {s.idx: s for s in run.steps}
    raw_label = raw.get("label")
    label = parse_label(raw_label)
    if label is None:
        notes.append(f"label {raw_label!r} is not in the taxonomy")
        label = TaxonomyLabel.UNKNOWN
    step = _int(raw.get("failing_step", raw.get("step")))
    if raw.get("failing_step") is not None and step is None:
        notes.append(f"failing_step {raw.get('failing_step')!r} is not a step index")
    if step is not None and step not in steps:
        notes.append(f"failing_step {step} does not exist (the run has steps 0-{len(steps) - 1})")
        step = None
    confidence = _confidence(raw.get("confidence"))
    root_cause = str(raw.get("root_cause") or raw.get("explanation") or "").strip()

    evidence: list[Evidence] = []
    items = raw.get("evidence") or []
    if not isinstance(items, list):
        items = [items]
    total = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        total += 1
        idx = _int(item.get("step", item.get("step_idx")))
        quote = item.get("quote")
        if not isinstance(quote, str) or not quote.strip():
            # {"step": 3, "result": "<quote>"}: the field name used as the key (Gemini
            # flash-lite). The quote is still checked verbatim against the step below.
            others = [
                v
                for k, v in item.items()
                if k not in _NOT_QUOTE and isinstance(v, str) and v.strip()
            ]
            quote = others[0] if len(others) == 1 else None
        if not isinstance(quote, str) or not quote.strip():
            notes.append("dropped an evidence item without a quote")
            continue
        short = quote if len(quote) <= 60 else quote[:57] + "..."
        if idx is None or idx not in steps:
            notes.append(f"dropped quote {short!r}: step {item.get('step')!r} does not exist")
            continue
        fld = find_quote(steps[idx], quote)
        if fld is None and quote_in_run_error(run, quote):
            fld = "run.error"  # the run's exception, which surfaced after this step
        if fld is None:
            notes.append(f"dropped quote {short!r}: not found verbatim in step {idx}")
            continue
        if len(evidence) < MAX_EVIDENCE:
            evidence.append(Evidence(step_idx=idx, field=fld, quote=quote.strip()))

    secondary = []
    for s in raw.get("secondary") or []:
        lbl = parse_label(s)
        if lbl is not None and lbl not in (label, TaxonomyLabel.UNKNOWN, TaxonomyLabel.NO_FAILURE):
            secondary.append(lbl)

    downgrade = None
    if label not in (TaxonomyLabel.UNKNOWN, TaxonomyLabel.NO_FAILURE):
        if step is None:
            downgrade = f"{label.value} needs a failing step that exists in the run"
        elif not evidence:
            downgrade = f"{label.value} has no evidence quote found verbatim in the cited steps"
    elif label is TaxonomyLabel.NO_FAILURE and run.outcome == "fail":
        downgrade = (
            f"NO_FAILURE contradicts the run's outcome (fail: {run.outcome_reason or 'see checks'})"
        )
    if downgrade:
        notes.append(f"downgraded to UNKNOWN: {downgrade}")
        if label is not TaxonomyLabel.NO_FAILURE:
            secondary = [label, *[s for s in secondary if s is not label]]
        label = TaxonomyLabel.UNKNOWN
    if label is TaxonomyLabel.UNKNOWN:
        confidence = min(confidence, UNKNOWN_CONFIDENCE_CAP)
    if label is TaxonomyLabel.NO_FAILURE:
        step = None
    window = None
    if step is not None and label is not TaxonomyLabel.UNKNOWN:
        cited = [e.step_idx for e in evidence] + [step]
        window = (min(cited), max(cited))
    method = dict(extra.pop("method", {}) or {})
    method.update(
        {
            "evidence_total": total,
            "evidence_valid": len(evidence),
            "raw_label": raw_label if isinstance(raw_label, str) else None,
            "raw_failing_step": raw.get("failing_step"),
        }
    )
    return Diagnosis(
        run_id=run.run_id,
        failing_step=step if label is not TaxonomyLabel.UNKNOWN else None,
        step_window=window,
        label=label,
        secondary=secondary[:3],
        confidence=round(confidence, 3),
        root_cause=root_cause,
        evidence=evidence,
        explanation=str(raw.get("explanation") or "").strip() if raw.get("root_cause") else "",
        notes=notes,
        method=method,
        **extra,
    )
