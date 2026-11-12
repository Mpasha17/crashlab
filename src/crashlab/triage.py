"""Triage a store full of failures: ``crashlab triage`` and :func:`triage_runs`.

Failed runs are grouped with :func:`crashlab.cluster.cluster_runs`; each cluster's
representative (its newest run) gets a diagnosis: the one already stored with it, or a
new offline **heuristic** diagnosis (rules over checks, findings and errors; no LLM, no
keys), saved with the run. Only representatives are diagnosed. The result is ranked
like the clusters (largest first, then most recent) and says for each cluster which
diagnoser produced its label, so a heuristic guess is never shown as an LLM's answer.

The default signature leaves out ``label`` (the stored diagnosis), so diagnosing a
representative does not split its cluster on the next triage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from .cluster import COMPONENTS, cluster_runs
from .models import Run
from .store import Store

__all__ = ["DEFAULT_BY", "TriageItem", "TriageResult", "triage_runs"]

DEFAULT_BY = tuple(c for c in COMPONENTS if c != "label")
_QUOTE = 100


@dataclass
class TriageItem:
    rank: int
    cluster_id: str
    size: int
    signature: dict[str, Any]
    summary: str
    representative: str
    run_ids: list[str]
    first_seen: str
    last_seen: str
    label: str | None = None
    confidence: float | None = None
    failing_step: int | None = None
    step_description: str | None = None
    agent: str | None = None
    root_cause: str = ""
    evidence: str | None = None  # one line: step N field: quote
    diagnoser: str | None = None  # "heuristic" or the diagnoser model
    diagnosis_source: str | None = None  # "new" (diagnosed now) / "stored" (reused)
    diagnosis_error: str | None = None
    commands: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TriageResult:
    items: list[TriageItem]
    by: list[str]
    scanned: int
    failed: int
    clusters_total: int
    diagnosed: int  # representatives diagnosed now
    reused: int  # representatives whose stored diagnosis was used
    unreadable: list[str]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["items"] = [i.to_dict() for i in self.items]
        return d


def _one_line(text: str, width: int = _QUOTE) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _fill(item: TriageItem, run: Run, source: str) -> None:
    d = run.diagnosis
    if d is None:
        return
    item.diagnosis_source = source
    item.diagnoser = d.model or "heuristic"
    item.label = d.label.value
    item.confidence = d.confidence
    item.failing_step = d.failing_step
    item.root_cause = d.root_cause
    if d.failing_step is not None:
        item.step_description = run.describe_step(d.failing_step)
        labels = run.agent_instances() if run.agents else {}
        step = next((s for s in run.steps if s.idx == d.failing_step), None)
        item.agent = labels.get(d.failing_step) or (step.agent if step else None)
    if d.evidence:
        e = d.evidence[0]
        item.evidence = f"step {e.step_idx} {e.field}: {_one_line(e.quote)}"


def triage_runs(
    *,
    store: Store | None = None,
    store_dir: str | None = None,
    by: tuple[str, ...] | list[str] = DEFAULT_BY,
    name: str | None = None,
    since: datetime | None = None,
    flagged: bool = False,
    limit: int | None = 10,
    rediagnose: bool = False,
    save: bool = True,
) -> TriageResult:
    """Cluster the failed runs, diagnose each of the top ``limit`` clusters'
    representatives offline (heuristic) unless a diagnosis is already stored with it
    (``rediagnose=True`` replaces it with a new heuristic one), and return them ranked.
    ``save=False`` diagnoses in memory only."""
    from .diagnose import diagnose

    store = store or Store(store_dir)
    res = cluster_runs(store=store, by=by, name=name, since=since, flagged=flagged)
    items: list[TriageItem] = []
    diagnosed = reused = 0
    for rank, c in enumerate(res.clusters[: limit or None], 1):
        rep = c.representative
        item = TriageItem(
            rank=rank, cluster_id=c.cluster_id, size=c.size, signature=c.signature,
            summary=c.summary, representative=rep, run_ids=c.run_ids,
            first_seen=c.first_seen, last_seen=c.last_seen,
            commands=[f"crashlab show {rep} --failing", f"crashlab fix {rep}",
                      f"crashlab export-test {rep}"],
        )  # fmt: skip
        try:
            run = store.load_run(rep)
            if run.diagnosis is not None and not rediagnose:
                reused += 1
                _fill(item, run, "stored")
            else:
                diagnose(run, heuristic=True, store=store, save=save, cache=False)
                diagnosed += 1
                _fill(item, run, "new")
        except Exception as exc:  # one bad run must not stop the triage
            item.diagnosis_error = f"{type(exc).__name__}: {exc}"
        items.append(item)
    return TriageResult(
        items=items, by=list(res.by), scanned=res.scanned, failed=res.failed,
        clusters_total=len(res.clusters), diagnosed=diagnosed, reused=reused,
        unreadable=res.unreadable,
    )  # fmt: skip
