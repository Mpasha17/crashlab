"""Flake detection: the same agent, given the same input, sometimes fails.

``find_flakes`` groups stored runs that are *the same input* and reports every outcome
signal that differs between them: the run raising (by exception type), the run's outcome,
each ``@crashlab.check``, and each runtime-check finding kind.

**Same input** means all of:

- the same agent ``name``;
- the same recorded ``run.input`` (the agent's bound arguments as stored, after
  redaction), compared as canonical JSON (key order does not matter, values do);
- the same ``code_version`` (git commit; ``-dirty`` kept), unless
  ``across_versions=True``. Different commits are different code, not a flake.

Strict replays and forks are left out (a replay serves the recording, a fork changes
it), as are runs still recording. Model, prompt files, tools' outside state and the
environment are *not* part of the key: when the recorded request models differ inside a
group the report says so, since that, not chance, may explain the difference.

Numbers are counts. The 95% interval reported for a failure rate is a Wilson score
interval and assumes the runs are independent trials of the same thing; with a handful
of runs it is wide, and it is a description of these runs, not a guarantee.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .models import SEVERITY_RANK, Run
from .verify.stats import wilson

__all__ = ["FlakeGroup", "Signal", "find_flakes", "input_key", "render_text"]


def input_key(run: Run) -> str:
    """Digest of the run's recorded input as canonical JSON."""
    text = json.dumps(run.input, sort_keys=True, separators=(",", ":"), default=str,
                      ensure_ascii=False)  # fmt: skip
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class Signal:
    """One outcome that differs between runs of the same input.

    ``bad`` of ``n`` runs showed it (``n`` counts only runs where it could be observed:
    e.g. runs that evaluated that check)."""

    name: str  # "raised", "outcome", "check:<name>", "finding:<kind>"
    bad: int
    n: int
    ci95: tuple[float, float]
    bad_runs: list[str]
    good_runs: list[str]
    detail: str = ""  # e.g. exception types seen

    @property
    def rate(self) -> float:
        return self.bad / self.n if self.n else 0.0


@dataclass
class FlakeGroup:
    name: str
    input_key: str
    input: dict[str, Any]
    code_version: str | None
    runs: list[str]  # oldest first
    signals: list[Signal] = field(default_factory=list)  # only the ones that differ
    models: list[str] = field(default_factory=list)  # request models seen in the group
    distinct_outputs: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def flaky(self) -> bool:
        return bool(self.signals)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["flaky"] = self.flaky
        for s, sd in zip(self.signals, d["signals"], strict=True):
            sd["rate"] = round(s.rate, 4)
            sd["ci95"] = [round(x, 4) for x in s.ci95]
        return d


def _disabled(section: Any) -> bool:
    return section is False or (isinstance(section, dict) and section.get("enabled") is False)


def _observations(run: Run, min_rank: int) -> dict[str, tuple[bool, str]]:
    """signal name -> (bad?, detail) for the signals observable on this run."""
    obs: dict[str, tuple[bool, str]] = {}
    err = run.error.type if run.status == "error" and run.error is not None else ""
    obs["raised"] = (run.status == "error", err)
    if run.outcome in ("pass", "fail"):
        obs["outcome"] = (run.outcome == "fail", "")
    for c in run.checks:
        obs[f"check:{c.name}"] = (c.status != "pass", c.status)
    cfg = run.check_config
    if isinstance(cfg, dict) and cfg.get("enabled", True):
        from .checks import KINDS

        enabled = [k for k in KINDS if not _disabled(cfg.get(k))]
        found = {f.kind for f in run.findings if SEVERITY_RANK.get(f.severity, 0) >= min_rank}
        for kind in enabled:
            obs[f"finding:{kind}"] = (kind in found, "")
    return obs


def _models(run: Run) -> set[str]:
    return {str(s.attributes.get("gen_ai.request.model") or s.name)
            for s in run.steps if s.operation == "chat"}  # fmt: skip


def _output_key(run: Run) -> str:
    return json.dumps(run.output, sort_keys=True, default=str)


def find_flakes(
    store: Any = None,
    *,
    name: str | None = None,
    limit: int | None = 500,
    min_runs: int = 2,
    min_severity: str = "warn",
    across_versions: bool = False,
    include_stable: bool = False,
    runs: list[Run] | None = None,
) -> list[FlakeGroup]:
    """Groups of same-input runs (at least ``min_runs``), flaky ones first.

    ``store``: a :class:`~crashlab.Store` (default: the current one); ``runs``: use these
    runs instead (steps loaded). ``limit``: newest runs scanned. ``min_severity``: the
    lowest finding severity counted as a bad signal. ``include_stable``: also return
    groups where every signal agreed."""
    if min_runs < 2:
        raise ValueError("min_runs must be at least 2")
    if min_severity not in SEVERITY_RANK:
        raise ValueError(f"min_severity must be one of {', '.join(SEVERITY_RANK)}")
    min_rank = SEVERITY_RANK[min_severity]
    loader = None
    if runs is None:
        from .store import Store

        store = store if store is not None else Store()
        loader = store.load_run
        runs = []
        for row in store.list_runs(limit=limit, name=name):
            if row.get("status") == "running" or row.get("replay_of"):
                continue
            try:  # the index row has everything but the steps: load those per group
                runs.append(Run.model_validate(json.loads(row["run_json"])))
            except Exception:  # an unreadable run is not evidence either way
                continue
    elif name is not None:
        runs = [r for r in runs if r.name == name]
    groups: dict[tuple[str, str, str | None], list[Run]] = {}
    for r in runs:
        if r.status == "running" or r.replay_of is not None or r.forked_from is not None:
            continue
        version = None if across_versions else r.code_version
        groups.setdefault((r.name, input_key(r), version), []).append(r)
    if loader is not None:
        for members in groups.values():
            if len(members) >= min_runs:
                for i, r in enumerate(members):
                    # unreadable cassette: keep the index copy (no steps, no model list)
                    with contextlib.suppress(Exception):
                        members[i] = loader(r.run_id)
    out: list[FlakeGroup] = []
    for (agent, key, version), members in groups.items():
        if len(members) < min_runs:
            continue
        members.sort(key=lambda r: (r.started_at, r.run_id))
        g = FlakeGroup(name=agent, input_key=key, input=members[0].input,
                       code_version=version, runs=[r.run_id for r in members])  # fmt: skip
        per_run = [(r, _observations(r, min_rank)) for r in members]
        names = sorted({s for _, o in per_run for s in o},
                       key=lambda s: (s != "raised", s != "outcome", s))  # fmt: skip
        for sig in names:
            seen = [(r, o[sig]) for r, o in per_run if sig in o]
            bad = [r.run_id for r, (b, _) in seen if b]
            good = [r.run_id for r, (b, _) in seen if not b]
            if len(seen) < min_runs or not bad or not good:
                continue
            details = sorted({d for _, (b, d) in seen if b and d})
            g.signals.append(Signal(name=sig, bad=len(bad), n=len(seen),
                                    ci95=wilson(len(bad), len(seen)), bad_runs=bad,
                                    good_runs=good, detail=", ".join(details)))  # fmt: skip
        models: set[str] = set()
        for r in members:
            models |= _models(r)
        g.models = sorted(models)
        g.distinct_outputs = len({_output_key(r) for r in members if r.status == "ok"})
        if len({frozenset(_models(r)) for r in members if _models(r)}) > 1:
            g.notes.append("runs called different models: the model, not chance, may "
                           "explain the difference")  # fmt: skip
        if across_versions and len({r.code_version for r in members}) > 1:
            g.notes.append("runs span several code versions (--across-versions)")
        elif version is not None and version.endswith("-dirty"):
            g.notes.append("uncommitted changes (-dirty): the code may differ between runs")
        if g.flaky or include_stable:
            out.append(g)
    out.sort(key=lambda g: (not g.flaky, -max((s.bad for s in g.signals), default=0),
                            -len(g.runs), g.name))  # fmt: skip
    return out


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _preview(value: Any, width: int = 70) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return text if len(text) <= width else text[: width - 1] + "\u2026"


def render_text(groups: list[FlakeGroup], *, show_stable: bool = False) -> list[str]:
    """Plain-text report lines (what ``crashlab flakes`` prints). ``groups``: from
    ``find_flakes(..., include_stable=True)`` so the summary can count stable ones."""
    flaky = [g for g in groups if g.flaky]
    lines: list[str] = []
    for g in groups if show_stable else flaky:
        ver = f"  code {g.code_version[:12]}" if g.code_version else ""
        state = "FLAKY" if g.flaky else "stable"
        lines.append(f"{state}  {g.name}  input {_preview(g.input)}{ver}  "
                     f"({len(g.runs)} runs, {g.distinct_outputs} distinct outputs)")  # fmt: skip
        last_pair: tuple[str, str] | None = None
        for s in g.signals:
            lo, hi = s.ci95
            what = f"  [{s.detail}]" if s.detail else ""
            lines.append(f"  {s.name}: {s.bad}/{s.n} runs ({_pct(s.rate)}, 95% CI "
                         f"{_pct(lo)}-{_pct(hi)}){what}")  # fmt: skip
            pair = (s.good_runs[0], s.bad_runs[0])
            if pair != last_pair:  # good run first, bad run second
                lines.append(f"    compare: crashlab diff {pair[0]} {pair[1]}")
            last_pair = pair
        if g.models:
            lines.append(f"  models: {', '.join(g.models)}")
        for n in g.notes:
            lines.append(f"  note: {n}")
    if not groups:
        lines.append("No agent input was run at least twice (same agent, input and code "
                     "version), so there is nothing to compare.")  # fmt: skip
    else:
        runs = sum(len(g.runs) for g in groups)
        lines.append(f"{len(flaky)} flaky of {len(groups)} repeated input(s) ({runs} runs). "
                     "Counts are of these runs; the CI is a Wilson 95% interval that "
                     "assumes independent runs.")  # fmt: skip
    return lines
