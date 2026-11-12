"""Group failed runs by failure signature: ``crashlab cluster`` and :func:`cluster_runs`.

A signature is built only from recorded facts, never from an LLM:

- ``name``: the agent name (runs of different agents never share a cluster)
- ``error``: the run's exception type and normalized message (ids, long numbers,
  timestamps, hex / UUIDs, e-mails and long quoted values replaced by placeholders;
  small integers such as HTTP status codes are kept)
- ``step``: the first step that recorded an error (operation, name, exception type and
  normalized message), or else the first error/critical runtime finding (step operation,
  name and finding kind). The step index is not part of it: the same bug often fails at
  step 3 in one run and step 5 in another
- ``checks``: the names of ``@crashlab.check`` oracles that did not pass, sorted
- ``label``: the stored diagnosis's primary taxonomy label, when there is one
- ``findings``: kinds of error/critical runtime findings, sorted

Two runs share a cluster only when every selected component is equal, so the default
(all components) is conservative: it splits rather than merges. Pass fewer components
(``by=("name", "error")``) to group more loosely.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from ._runscan import ScanStats, iter_runs
from .models import SEVERITY_RANK, ErrorInfo, Run, Step
from .store import Store

__all__ = [
    "COMPONENTS",
    "Cluster",
    "ClusterResult",
    "FailurePoint",
    "cluster_runs",
    "failure_point",
    "failure_signature",
    "is_failed",
    "normalize_message",
]

COMPONENTS = ("name", "error", "step", "checks", "label", "findings")
_ERROR_RANK = SEVERITY_RANK["error"]
MESSAGE_WIDTH = 160

_SUBS: list[tuple[re.Pattern[str], str]] = [
    # ISO timestamps / dates before plain numbers
    (re.compile(r"\b\d{4}-\d{2}-\d{2}"
                r"(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?\b"), "<ts>"),
    (re.compile(r"\b[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\b"), "<uuid>"),
    (re.compile(r"\b[0-9A-HJKMNP-TV-Z]{26}\b"), "<id>"),  # ULID (crashlab run ids)
    (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "<email>"),
    (re.compile(r"\b(?:0x)?(?=[0-9a-fA-F]*\d)(?=[0-9a-fA-F]*[a-fA-F])[0-9a-fA-F]{8,}\b"), "<hex>"),
    # ids such as call_abc123 / req-8f2k1: a word with letters and 4+ digits
    (re.compile(r"\b(?=\w*[A-Za-z])(?=\w*\d{4,})[\w-]{6,}\b"), "<id>"),
    (re.compile(r"\b\d+\.\d+|\b\d{4,}(?!\d)"), "<n>"),  # 1.25s, 12345ms
    (re.compile(r"'[^'\n]{41,}'|\"[^\"\n]{41,}\""), "<str>"),
    (re.compile(r"\s+"), " "),
]  # fmt: skip


def normalize_message(text: str | None, width: int = MESSAGE_WIDTH) -> str:
    """Strip the parts of an error message that differ between occurrences of the same
    failure. Conservative: words, small integers (status codes, counts) and short quoted
    values (``KeyError('sku')``) are kept."""
    if not text:
        return ""
    out = text
    for pat, repl in _SUBS:
        out = pat.sub(repl, out)
    out = out.strip()
    return out if len(out) <= width else out[: width - 1] + "…"


def _error_text(err: ErrorInfo) -> str:
    code = f" [{err.status_code}]" if err.status_code else ""
    return f"{err.type}{code}: {normalize_message(err.detail)}"


def _failing_checks(run: Run) -> list[str]:
    return sorted({c.name for c in run.checks if c.status != "pass"})


def _serious_findings(run: Run) -> list[Any]:
    return [f for f in run.findings if f.rank >= _ERROR_RANK]


def is_failed(run: Run, *, flagged: bool = False) -> bool:
    """Status ``error``, outcome ``fail``, a check that did not pass, or (with
    ``flagged``) an error / critical runtime finding."""
    if run.status == "error" or run.outcome == "fail" or run.error is not None:
        return True
    if _failing_checks(run):
        return True
    return flagged and bool(_serious_findings(run))


@dataclass
class FailurePoint:
    """Where a run first went wrong, from recorded facts (see :func:`failure_point`)."""

    step: int | None  # first step with an error, else first serious finding's step
    step_kind: str | None  # "error" or the finding kind
    step_description: str | None
    failing_checks: list[str] = field(default_factory=list)
    diagnosis_step: int | None = None
    diagnosis_label: str | None = None
    run_error: str | None = None


def failure_point(run: Run) -> FailurePoint:
    """The first step that recorded an error (or, if none did, the first step an
    error / critical runtime finding fired on), the checks that did not pass, the
    stored diagnosis's step and label, and the run's exception."""
    step: Step | None = next((s for s in run.steps if s.error is not None), None)
    idx, kind = (step.idx, "error") if step is not None else (None, None)
    if step is None:
        serious = sorted(
            (f for f in _serious_findings(run) if f.step is not None), key=lambda f: f.step
        )
        if serious:
            idx, kind = serious[0].step, serious[0].kind
    d = run.diagnosis
    return FailurePoint(
        step=idx,
        step_kind=kind,
        step_description=run.describe_step(idx) if idx is not None else None,
        failing_checks=_failing_checks(run),
        diagnosis_step=d.failing_step if d is not None else None,
        diagnosis_label=d.label.value if d is not None else None,
        run_error=f"{run.error.type}: {run.error.detail}" if run.error else None,
    )


def failure_signature(run: Run, by: tuple[str, ...] | list[str] = COMPONENTS) -> dict[str, Any]:
    """The run's signature restricted to the ``by`` components (see module doc)."""
    bad = [c for c in by if c not in COMPONENTS]
    if bad:
        raise ValueError(
            f"unknown signature component(s) {bad}; choose from {', '.join(COMPONENTS)}"
        )
    sig: dict[str, Any] = {}
    if "name" in by:
        sig["name"] = run.name
    if "error" in by:
        sig["error"] = _error_text(run.error) if run.error is not None else None
    if "step" in by:
        first = next((s for s in run.steps if s.error is not None), None)
        if first is not None and first.error is not None:
            sig["step"] = f"{first.operation} {first.name} → {_error_text(first.error)}"
        else:
            steps = {s.idx: s for s in run.steps}
            serious = sorted(
                (f for f in _serious_findings(run) if f.step is not None), key=lambda f: f.step
            )
            if serious and serious[0].step in steps:
                s = steps[serious[0].step]
                sig["step"] = f"{s.operation} {s.name} ⚑ {serious[0].kind}"
            else:
                sig["step"] = None
    if "checks" in by:
        sig["checks"] = _failing_checks(run)
    if "label" in by:
        sig["label"] = run.diagnosis.label.value if run.diagnosis is not None else None
    if "findings" in by:
        sig["findings"] = sorted({f.kind for f in _serious_findings(run)})
    return sig


def _sig_id(sig: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(sig, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[
        :10
    ]


def _summary(sig: dict[str, Any]) -> str:
    bits = []
    if sig.get("error"):
        bits.append(sig["error"])
    if sig.get("step") and sig.get("step") != sig.get("error"):
        bits.append(f"first failure: {sig['step']}")
    if sig.get("checks"):
        bits.append("checks failed: " + ", ".join(sig["checks"]))
    if sig.get("label"):
        bits.append(f"diagnosis: {sig['label']}")
    if sig.get("findings"):
        bits.append("findings: " + ", ".join(sig["findings"]))
    if not bits:
        rest = [k for k in sig if k != "name"]
        bits.append(f"failed; nothing recorded for {', '.join(rest)}" if rest else "failed")
    return " · ".join(bits)


@dataclass
class Cluster:
    cluster_id: str
    size: int
    signature: dict[str, Any]
    summary: str
    representative: str  # newest run in the cluster
    run_ids: list[str]  # newest first
    first_seen: str
    last_seen: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClusterResult:
    clusters: list[Cluster]
    by: list[str]
    scanned: int  # runs read
    failed: int  # runs that counted as failed and were clustered
    unreadable: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "by": self.by,
            "scanned": self.scanned,
            "failed": self.failed,
            "unreadable": self.unreadable,
            "clusters": [c.to_dict() for c in self.clusters],
        }


def cluster_runs(
    *,
    store: Store | None = None,
    store_dir: str | None = None,
    by: tuple[str, ...] | list[str] = COMPONENTS,
    name: str | None = None,
    since: datetime | None = None,
    run_prefix: str | None = None,
    flagged: bool = False,
    min_size: int = 1,
    limit: int | None = None,
) -> ClusterResult:
    """Cluster the failed runs in the store (see :func:`is_failed`) by
    :func:`failure_signature`. Largest clusters first; ties by most recent run.
    ``limit`` caps how many runs (newest first) are read."""
    by = tuple(by)
    failure_signature(Run(run_id="x", name="x"), by)  # validate ``by`` up front
    store = store or Store(store_dir)
    stats = ScanStats()
    groups: dict[str, tuple[dict[str, Any], list[Run]]] = {}
    failed = 0
    for run in iter_runs(store, since=since, run_prefix=run_prefix, name=name, stats=stats):
        if limit is not None and stats.scanned > limit:
            stats.scanned -= 1
            break
        if not is_failed(run, flagged=flagged):
            continue
        failed += 1
        sig = failure_signature(run, by)
        key = _sig_id(sig)
        groups.setdefault(key, (sig, []))[1].append(run)
    clusters = []
    for key, (sig, runs) in groups.items():
        if len(runs) < min_size:
            continue
        runs.sort(key=lambda r: r.run_id, reverse=True)
        clusters.append(
            Cluster(
                cluster_id=key,
                size=len(runs),
                signature=sig,
                summary=_summary(sig),
                representative=runs[0].run_id,
                run_ids=[r.run_id for r in runs],
                first_seen=min(r.started_at for r in runs).isoformat(),
                last_seen=max(r.started_at for r in runs).isoformat(),
            )
        )
    clusters.sort(key=lambda c: (c.size, c.representative), reverse=True)
    return ClusterResult(
        clusters=clusters,
        by=list(by),
        scanned=stats.scanned,
        failed=failed,
        unreadable=stats.unreadable,
    )
