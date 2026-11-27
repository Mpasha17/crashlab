"""Batch evaluation suites and candidate-patch sweeps for recorded agent runs.

``eval_runs`` grades many recordings (from the store or exported ``.jsonl``
cassettes) with ``@crashlab.check`` oracles, built-in runtime checks and optional
budget gates, and can emit JUnit XML / Markdown for CI.

``sweep_runs`` forks the same set of runs with one candidate patch and classifies
each as fixed, regressed, still passing, still failing, or errored.
"""

from __future__ import annotations

import glob
import importlib
import importlib.util
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from xml.etree.ElementTree import Element, SubElement, tostring

from .budgets import summarize_usage, within_budget
from .checks import evaluate, resolve_config
from .models import SEVERITY_RANK, Finding, Run
from .patch import ModelParamPatch, PatchSet, SystemPromptPatch, as_patchset
from .patch import load as load_patch
from .replay import DivergenceError, ForkError, ReplayError, fork
from .replay import replay as replay_run
from .store import Store, import_run
from .verify.oracles import apply_outcome, run_checks

__all__ = [
    "EvalReport",
    "EvalResult",
    "SweepReport",
    "SweepResult",
    "collect_runs",
    "eval_runs",
    "expand_cassette_paths",
    "load_check_modules",
    "load_export",
    "sweep_runs",
]

EvalStatus = Literal["pass", "fail", "error"]
SweepVerdict = Literal["fixed", "regressed", "still_passing", "still_failing", "errored"]
_SEVERITIES = tuple(SEVERITY_RANK)
_OUTCOMES = ("pass", "fail", "unknown")


# --- loading -------------------------------------------------------------------------


def load_export(path: str | Path) -> Run:
    """Load one exported cassette (``crashlab export`` JSONL) into a :class:`Run`."""
    from .store import EXPORT_HEADER, EXPORT_VERSION, compact
    from .store.cassette import line_to_step, read_lines

    src = Path(path)
    lines = read_lines(src)
    if not lines or lines[0].get("type") != EXPORT_HEADER:
        raise ValueError(f"{src}: not a crashlab export (first line must be the run header)")
    if int(lines[0].get("version", 0)) > EXPORT_VERSION:
        raise ValueError(f"{src}: export version {lines[0].get('version')} is newer than supported")
    run = Run.model_validate(lines[0]["run"])
    body, _problems = compact.expand(lines[1:])
    unknown: list[str] = []
    steps = [line_to_step(line, unknown) for line in body]
    run.steps = [
        s if s.run_id == run.run_id else s.model_copy(update={"run_id": run.run_id}) for s in steps
    ]
    return run


def expand_cassette_paths(files: Iterable[str | Path]) -> list[Path]:
    """Resolve file paths, directories and globs to exported ``.jsonl`` cassettes.

    A directory yields every ``.jsonl`` under it (recursive). A glob uses
    :func:`glob.glob` with ``recursive=True``. Paths that are not ``.jsonl`` files
    are skipped; a pattern or path that matches nothing raises ``FileNotFoundError``.
    """
    out: list[Path] = []
    seen: set[Path] = set()
    for item in files:
        raw = str(item)
        p = Path(raw).expanduser()
        matches: list[Path]
        if any(ch in raw for ch in "*?["):
            matches = [Path(m) for m in sorted(glob.glob(raw, recursive=True))]
        elif p.is_dir():
            matches = sorted(p.rglob("*.jsonl"))
        elif p.is_file():
            matches = [p]
        else:
            matches = [Path(m) for m in sorted(glob.glob(raw, recursive=True))]
        jsonl = [m for m in matches if m.is_file() and m.suffix.lower() == ".jsonl"]
        if not jsonl:
            raise FileNotFoundError(f"no exported .jsonl cassette at {raw}")
        for m in jsonl:
            key = m.resolve()
            if key not in seen:
                seen.add(key)
                out.append(m)
    return out


def load_check_modules(modules: Iterable[str | Path] | None) -> list[str]:
    """Import Python files or modules so ``@crashlab.check`` oracles register."""
    loaded: list[str] = []
    for item in modules or []:
        path = Path(str(item)).expanduser()
        if path.suffix == ".py" and path.is_file():
            name = "crashlab_eval_" + "_".join(path.resolve().parts[-3:]).replace(".", "_").replace(
                "-", "_"
            )
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot import check module {path}")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            loaded.append(name)
        else:
            importlib.import_module(str(item))
            loaded.append(str(item))
    return loaded


def _parse_since(value: datetime | str | None) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    from ._runscan import parse_since

    return parse_since(value)


def _wanted(run: Run, *, name: str | None, tag: str | None, outcome: str | None,
            since: datetime | None) -> bool:  # fmt: skip
    if name is not None and run.name != name:
        return False
    if tag is not None and tag not in (run.tags or []):
        return False
    if outcome is not None and run.outcome != outcome:
        return False
    if since is not None:
        started = run.started_at
        if started.tzinfo is None:
            from datetime import timezone

            started = started.replace(tzinfo=timezone.utc)
        if started < since:
            return False
    return True


def _collect_runs(
    store: Store | None,
    files: Iterable[str | Path] | str | Path | None,
    *,
    name: str | None,
    tag: str | None,
    outcome: str | None,
    since: datetime | str | None,
    limit: int | None,
) -> list[Run]:
    after = _parse_since(since)
    collected: list[Run] = []
    if files is not None:
        paths = [files] if isinstance(files, (str, Path)) else list(files)
        store_for_import = store
        for path in expand_cassette_paths(paths):
            if store_for_import is not None:
                run_id, _added = import_run(store_for_import, path, replace=False)
                run = store_for_import.load_run(run_id)
            else:
                run = load_export(path)
            if _wanted(run, name=name, tag=tag, outcome=outcome, since=after):
                collected.append(run)
    else:
        if store is None:
            store = Store()
        from ._runscan import ScanStats, iter_runs

        stats = ScanStats()
        for run in iter_runs(store, since=after, name=name, stats=stats):
            if _wanted(run, name=None, tag=tag, outcome=outcome, since=None):
                collected.append(run)
    if limit is not None and limit >= 0:
        collected = collected[:limit]
    return collected


collect_runs = _collect_runs


# --- judging -------------------------------------------------------------------------


def _import_entrypoint(run: Run) -> None:
    """Best-effort import of the recorded entry point so its oracles register."""
    if not run.entrypoint:
        return
    from ._import_guard import process_wide
    from .replay.divergence import ReplayError
    from .replay.engine import placeholder_credentials, resolve_entrypoint

    try:
        with process_wide(run.entrypoint), placeholder_credentials():
            resolve_entrypoint(run.entrypoint)
    except (ReplayError, ImportError, OSError, SyntaxError):
        return


def _duration_seconds(run: Run) -> float | None:
    if run.usage.wall_time_ms is not None:
        return run.usage.wall_time_ms / 1000.0
    if run.ended_at is not None:
        try:
            return (run.ended_at - run.started_at).total_seconds()
        except TypeError:
            return None
    return None


def _usage_numbers(run: Run) -> tuple[float | None, int, float | None]:
    summary = summarize_usage(run)
    tokens = summary.total_tokens if summary.total_tokens else (run.usage.total_tokens or 0)
    cost = summary.cost_usd if summary.cost_usd is not None else run.usage.cost_usd
    return cost, tokens, _duration_seconds(run)


def _finding_dicts(findings: Sequence[Finding]) -> list[dict[str, Any]]:
    return [
        {"kind": f.kind, "severity": f.severity, "message": f.message, "step": f.step}
        for f in findings
    ]


def _judge(
    run: Run,
    *,
    store: Store | None,
    max_cost_usd: float | None,
    max_tokens: int | None,
    max_seconds: float | None,
    max_steps: int | None,
    min_finding_severity: str,
    rule_specs: list[Any] | None = None,
    rubric: Any | None = None,
    judges: list[Any] | None = None,
    judge_options: dict[str, Any] | None = None,
) -> EvalResult:
    working = run.model_copy()
    _import_entrypoint(working)
    results = run_checks(working)
    cfg = resolve_config(None, base=working.check_config)
    findings = evaluate(working, cfg, store=store)
    if rule_specs:
        from .verify.oracles import run_checks as _run_given

        results = list(results) + _run_given(working, rule_specs)
    if results:
        apply_outcome(working, results)
    failing = [r for r in results if r.status != "pass"]
    floor = SEVERITY_RANK.get(min_finding_severity, SEVERITY_RANK["error"])
    serious = [f for f in findings if f.rank >= floor]
    budget = None
    limits = {
        "max_cost_usd": max_cost_usd,
        "max_tokens": max_tokens,
        "max_seconds": max_seconds,
        "max_steps": max_steps,
    }
    if any(v is not None for v in limits.values()):
        budget = within_budget(working, **{k: v for k, v in limits.items() if v is not None})
    reasons: list[str] = []
    if failing:
        reasons.append(
            ", ".join(
                f"check {r.name} {r.status}ed" + (f": {r.message}" if r.message else "")
                for r in failing
            )
        )
    if serious:
        reasons.append(", ".join(f"[{f.severity}] {f.kind}: {f.message}" for f in serious))
    if budget:
        reasons.append(budget)
    if not results and working.outcome == "fail":
        reasons.append(working.outcome_reason or "stored outcome is fail")
    if working.status == "error":
        err = working.error
        reasons.append(
            f"run status error: {err.type}: {err.message}" if err else "run status error"
        )

    if any(r.status == "error" for r in failing):
        status: EvalStatus = "error"
    elif reasons:
        status = "fail"
    else:
        status = "pass"

    cost, tokens, duration = _usage_numbers(working)
    score = grade = rubric_passed = None
    if rubric is not None:
        from .rubrics import score_run

        scored = score_run(working, rubric)
        score, grade, rubric_passed = scored.score, scored.grade, scored.passed
        if not scored.passed:
            reasons.append(
                f"rubric score {scored.score:.1f} ({scored.grade}) "
                f"below min_score {rubric.min_score:g}"
            )
            if status == "pass":
                status = "fail"
    judge_dicts: list[dict[str, Any]] = []
    for judge in judges or []:
        verdict = judge.evaluate(working, **(judge_options or {}))
        judge_dicts.append(verdict.to_dict())
        if not verdict.passed:
            reasons.append(
                f"judge {verdict.name} {verdict.score:.2f} < {verdict.threshold:g}: "
                f"{verdict.reasoning}"
            )
            if status == "pass":
                status = "fail"
    return EvalResult(
        run_id=working.run_id,
        name=working.name,
        status=status,
        reason="; ".join(reasons) if reasons else None,
        failing_checks=[r.name for r in failing],
        findings=_finding_dicts(serious),
        cost_usd=cost,
        tokens=tokens,
        duration_seconds=duration,
        outcome=working.outcome,
        score=score,
        grade=grade,
        rubric_passed=rubric_passed,
        judges=judge_dicts,
    )


# --- reports -------------------------------------------------------------------------


@dataclass
class EvalResult:
    """One run's grade in an :class:`EvalReport`."""

    run_id: str
    name: str
    status: EvalStatus
    reason: str | None = None
    failing_checks: list[str] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    cost_usd: float | None = None
    tokens: int = 0
    duration_seconds: float | None = None
    outcome: str = "unknown"
    path: str | None = None
    score: float | None = None
    grade: str | None = None
    rubric_passed: bool | None = None
    judges: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalReport:
    """Batch evaluation of recorded (or replayed) runs."""

    results: list[EvalResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == "pass")

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == "fail")

    @property
    def errored(self) -> int:
        return sum(1 for r in self.results if r.status == "error")

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd or 0.0 for r in self.results)

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens for r in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "errored": self.errored,
            "total": self.total,
            "pass_rate": self.pass_rate,
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "results": [r.to_dict() for r in self.results],
        }

    def to_junit_xml(self) -> str:
        """JUnit XML: ``<testsuites><testsuite><testcase>…</testcase></testsuite></testsuites>``."""
        suites = Element(
            "testsuites",
            {
                "name": "crashlab eval",
                "tests": str(self.total),
                "failures": str(self.failed),
                "errors": str(self.errored),
            },
        )
        suite = SubElement(
            suites,
            "testsuite",
            {
                "name": "crashlab.eval",
                "tests": str(self.total),
                "failures": str(self.failed),
                "errors": str(self.errored),
                "skipped": "0",
            },
        )
        for result in self.results:
            time = f"{result.duration_seconds:.3f}" if result.duration_seconds is not None else "0"
            case = SubElement(
                suite,
                "testcase",
                {"classname": result.name, "name": result.run_id, "time": time},
            )
            if result.status == "fail":
                node = SubElement(
                    case,
                    "failure",
                    {"message": result.reason or "failed", "type": "fail"},
                )
                node.text = _case_body(result)
            elif result.status == "error":
                node = SubElement(
                    case,
                    "error",
                    {"message": result.reason or "error", "type": "error"},
                )
                node.text = _case_body(result)
        return '<?xml version="1.0" encoding="UTF-8"?>\n' + tostring(suites, encoding="unicode")

    def to_markdown(self) -> str:
        """GitHub-flavored Markdown summary table of the evaluation."""
        rate = f"{self.pass_rate:.0%}" if self.total else "n/a"
        head = (
            f"# crashlab eval\n\n"
            f"**{self.passed} passed**, {self.failed} failed, {self.errored} errored "
            f"of {self.total} runs ({rate} pass). "
            f"Cost ${_fmt_cost(self.total_cost_usd)}, {self.total_tokens:,} tokens.\n"
        )
        has_scores = any(r.score is not None for r in self.results)
        has_judges = any(r.judges for r in self.results)
        headers = ["run", "agent", "result", "checks", "findings", "cost", "tokens", "duration"]
        if has_scores:
            headers[3:3] = ["score", "grade"]
        if has_judges:
            headers.insert(len(headers) - 3, "judges")
        rows = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for r in self.results:
            checks = ", ".join(r.failing_checks) or "—"
            findings = ", ".join(f"{f['kind']}:{f['severity']}" for f in r.findings) or "—"
            dur = f"{r.duration_seconds:.2f}s" if r.duration_seconds is not None else "—"
            cells = [
                _md(r.run_id),
                _md(r.name),
                r.status,
                _md(checks),
                _md(findings),
                f"${_fmt_cost(r.cost_usd)}" if r.cost_usd is not None else "—",
                f"{r.tokens:,}",
                dur,
            ]
            if has_scores:
                score = f"{r.score:.1f}" if r.score is not None else "—"
                cells[3:3] = [score, r.grade or "—"]
            if has_judges:
                verdicts = ", ".join(
                    f"{j['name']} {j['score']:.2f}{'' if j['passed'] else ' ✗'}" for j in r.judges
                )
                cells.insert(len(cells) - 3, _md(verdicts or "—"))
            rows.append("| " + " | ".join(cells) + " |")
        body = "\n".join(rows)
        fails = [r for r in self.results if r.status != "pass"]
        if fails:
            detail = "\n\n## Failures\n\n" + "\n".join(
                f"- `{r.run_id}` ({r.name}): {_md(r.reason or r.status)}" for r in fails
            )
        else:
            detail = ""
        return head + "\n" + body + detail + "\n"


def _case_body(result: EvalResult) -> str:
    parts = [result.reason or result.status]
    if result.failing_checks:
        parts.append("checks: " + ", ".join(result.failing_checks))
    if result.findings:
        parts.append(
            "findings: " + ", ".join(f"{f['kind']} ({f['severity']})" for f in result.findings)
        )
    return "\n".join(parts)


def _fmt_cost(value: float | None) -> str:
    if value is None:
        return "?"
    if value == 0:
        return "0"
    if abs(value) >= 0.0001:
        return f"{value:.4f}"
    return f"{value:.6f}"


def _md(value: Any) -> str:
    return " ".join(str(value).replace("|", "\\|").split())


def _load_eval_rubric(rubric: str | Path | Any | None, *, min_score: float | None) -> Any | None:
    """Return a Rubric when ``rubric`` or ``min_score`` is set, else None."""
    if rubric is None and min_score is None:
        return None
    from .rubrics import DEFAULT_RUBRIC, Rubric, RubricError, load_rubric

    if rubric is None:
        loaded = DEFAULT_RUBRIC
    elif isinstance(rubric, Rubric):
        loaded = rubric
    else:
        try:
            loaded = load_rubric(rubric)
        except RubricError as exc:
            raise ValueError(str(exc)) from exc
    return loaded.with_min_score(min_score)


# --- eval_runs -----------------------------------------------------------------------


def eval_runs(
    store: Store | None = None,
    *,
    files: Iterable[str | Path] | str | Path | None = None,
    name: str | None = None,
    tag: str | None = None,
    outcome: str | None = None,
    since: datetime | str | None = None,
    limit: int | None = None,
    check_modules: Iterable[str | Path] | None = None,
    replay: bool = False,
    max_cost_usd: float | None = None,
    max_tokens: int | None = None,
    max_seconds: float | None = None,
    max_steps: int | None = None,
    min_finding_severity: str = "error",
    rules: str | Path | Iterable[Any] | bool | None = None,
    rubric: str | Path | Any | None = None,
    min_score: float | None = None,
    judges: Iterable[str | Any] | None = None,
    judge_threshold: float | None = None,
    judge_fn: Any = None,
    judge_model: str | None = None,
    judge_cache: bool | None = None,
) -> EvalReport:
    """Grade many recorded runs with oracles, runtime checks and optional budgets.

    Runs come from ``store`` (filtered by ``name`` / ``tag`` / ``outcome`` / ``since``
    / ``limit``) or from exported ``.jsonl`` cassette ``files`` (paths, directories
    or globs). ``check_modules`` are imported first so ``@crashlab.check`` oracles in
    those files are registered. ``rules`` is a rule file path, ``"auto"`` / ``None``
    (discover ``.crashlab/rules.*`` or ``[tool.crashlab.rules]``), ``"off"`` / ``False``
    to skip, or an in-memory list of rules. ``rubric`` is a rubric file path or a
    :class:`~crashlab.rubrics.Rubric`; ``min_score`` overrides the rubric's passing
    score. When either is set, each run is also graded 0–100 and fails the suite if
    it scores below the threshold. ``judges`` (names of :mod:`crashlab.judges` built-ins
    or :class:`~crashlab.judges.Judge` objects) add LLM-as-judge verdicts: a run fails if
    any judge scores below its threshold (``judge_threshold`` overrides every judge's);
    ``judge_fn`` / ``judge_model`` / ``judge_cache`` configure how they run. With
    ``replay=True`` each run is replayed offline first;
    :class:`~crashlab.replay.DivergenceError` and
    :class:`~crashlab.replay.ReplayError` are recorded as failures.
    """
    if min_finding_severity not in SEVERITY_RANK:
        raise ValueError(
            f"min_finding_severity must be one of {', '.join(_SEVERITIES)}, "
            f"not {min_finding_severity!r}"
        )
    if outcome is not None and outcome not in _OUTCOMES:
        raise ValueError(f"outcome must be one of {', '.join(_OUTCOMES)}, not {outcome!r}")
    if replay and store is None:
        store = Store()
    load_check_modules(check_modules)
    from .assertions import RuleError, load_rules, rules_to_checks

    try:
        rule_specs = rules_to_checks(load_rules(rules)) if rules is not False else []
    except RuleError as exc:
        raise ValueError(str(exc)) from exc
    rubric_obj = _load_eval_rubric(rubric, min_score=min_score)
    judge_list: list[Any] = []
    judge_opts: dict[str, Any] = {"judge_fn": judge_fn, "model": judge_model, "store": store}
    if judge_cache is not None:
        judge_opts["cache"] = judge_cache
    if judges:
        from .judges import get_judge, resolve_judge_config

        judge_list = [get_judge(j, judge_threshold) for j in judges]
        if judge_fn is None and not resolve_judge_config(judge_model)[0]:
            from .judges import _no_model_error

            raise ValueError(str(_no_model_error()))
    runs = _collect_runs(
        store, files, name=name, tag=tag, outcome=outcome, since=since, limit=limit
    )
    report = EvalReport()
    for run in runs:
        try:
            judged = run
            if replay:
                judged = replay_run(run, store_dir=store.root if store is not None else None)
            result = _judge(
                judged,
                store=store,
                max_cost_usd=max_cost_usd,
                max_tokens=max_tokens,
                max_seconds=max_seconds,
                max_steps=max_steps,
                min_finding_severity=min_finding_severity,
                rule_specs=rule_specs,
                rubric=rubric_obj,
                judges=judge_list or None,
                judge_options=judge_opts,
            )
        except (DivergenceError, ReplayError) as exc:
            cost, tokens, duration = _usage_numbers(run)
            result = EvalResult(
                run_id=run.run_id,
                name=run.name,
                status="fail",
                reason=f"{type(exc).__name__}: {exc}",
                cost_usd=cost,
                tokens=tokens,
                duration_seconds=duration,
                outcome=run.outcome,
            )
        except Exception as exc:  # a broken check / store must not abort the suite
            result = EvalResult(
                run_id=run.run_id,
                name=run.name,
                status="error",
                reason=f"{type(exc).__name__}: {exc}",
                outcome=run.outcome,
            )
        report.results.append(result)
    return report


# --- sweep ---------------------------------------------------------------------------


@dataclass
class SweepResult:
    """One run's before/after comparison in a :class:`SweepReport`."""

    run_id: str
    name: str
    verdict: SweepVerdict
    baseline_status: str
    patched_status: str | None = None
    patched_run_id: str | None = None
    reason: str | None = None
    cost_delta_usd: float | None = None
    token_delta: int = 0
    duration_delta_seconds: float | None = None
    baseline: dict[str, Any] | None = None
    patched: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SweepReport:
    """Candidate-patch sweep across many recorded runs."""

    results: list[SweepResult] = field(default_factory=list)
    patch_description: str = ""

    @property
    def fixes(self) -> int:
        return sum(1 for r in self.results if r.verdict == "fixed")

    @property
    def regressions(self) -> int:
        return sum(1 for r in self.results if r.verdict == "regressed")

    @property
    def unchanged(self) -> int:
        return sum(1 for r in self.results if r.verdict in ("still_passing", "still_failing"))

    @property
    def errored(self) -> int:
        return sum(1 for r in self.results if r.verdict == "errored")

    @property
    def still_passing(self) -> int:
        return sum(1 for r in self.results if r.verdict == "still_passing")

    @property
    def still_failing(self) -> int:
        return sum(1 for r in self.results if r.verdict == "still_failing")

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def cost_delta_usd(self) -> float:
        return sum(r.cost_delta_usd or 0.0 for r in self.results)

    @property
    def token_delta(self) -> int:
        return sum(r.token_delta for r in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch": self.patch_description,
            "fixes": self.fixes,
            "regressions": self.regressions,
            "unchanged": self.unchanged,
            "still_passing": self.still_passing,
            "still_failing": self.still_failing,
            "errored": self.errored,
            "total": self.total,
            "cost_delta_usd": self.cost_delta_usd,
            "token_delta": self.token_delta,
            "results": [r.to_dict() for r in self.results],
        }

    def to_markdown(self) -> str:
        """GitHub-flavored Markdown summary of the sweep."""
        patch = f" Patch: {self.patch_description}." if self.patch_description else ""
        head = (
            f"# crashlab sweep\n\n"
            f"**{self.fixes} fixed**, {self.regressions} regressed, "
            f"{self.unchanged} unchanged, {self.errored} errored "
            f"of {self.total} runs.{patch} "
            f"Cost Δ ${_fmt_cost(self.cost_delta_usd)}, tokens Δ {self.token_delta:+,}.\n"
        )
        rows = [
            "| run | agent | verdict | baseline | patched | cost Δ | tokens Δ |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in self.results:
            cost = f"${_fmt_cost(r.cost_delta_usd)}" if r.cost_delta_usd is not None else "—"
            rows.append(
                "| "
                + " | ".join(
                    [
                        _md(r.run_id),
                        _md(r.name),
                        r.verdict.replace("_", " "),
                        r.baseline_status,
                        r.patched_status or "—",
                        cost,
                        f"{r.token_delta:+,}",
                    ]
                )
                + " |"
            )
        body = "\n".join(rows)
        notable = [r for r in self.results if r.verdict in ("fixed", "regressed", "errored")]
        if notable:
            detail = "\n\n## Notable runs\n\n" + "\n".join(
                f"- `{r.run_id}` ({r.name}): **{r.verdict.replace('_', ' ')}**"
                + (f" — {_md(r.reason)}" if r.reason else "")
                for r in notable
            )
        else:
            detail = ""
        return head + "\n" + body + detail + "\n"


def _resolve_patch(patch: Any) -> PatchSet:
    if isinstance(patch, (str, Path)):
        path = Path(patch).expanduser()
        if path.is_file() or path.suffix.lower() in {".json", ".yaml", ".yml"}:
            return load_patch(path)
        return as_patchset(patch)
    return as_patchset(patch)


def _apply_scope(patches: PatchSet, *, model: str | None, agent: str | None) -> PatchSet:
    items = list(patches.patches)
    if agent:
        scoped = []
        for p in items:
            if getattr(p, "scope", "request") == "step":
                scoped.append(p)
            else:
                scoped.append(p.model_copy(update={"agent": agent}))
        items = scoped
    out = PatchSet(patches=items)
    if model:
        out = out + ModelParamPatch(model=model, agent=agent)
    return out


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None and b is None:
        return None
    return (b or 0.0) - (a or 0.0)


def _verdict(baseline: EvalStatus, patched: EvalStatus | None, *, errored: bool) -> SweepVerdict:
    if errored or patched is None:
        return "errored"
    if patched == "error" and baseline != "error":
        return "errored" if baseline == "fail" else "regressed"
    good_b, good_p = baseline == "pass", patched == "pass"
    if not good_b and good_p:
        return "fixed"
    if good_b and not good_p:
        return "regressed"
    if good_b and good_p:
        return "still_passing"
    return "still_failing"


def sweep_runs(
    runs: Sequence[Run] | Iterable[Run],
    patch: Any,
    *,
    store: Store | None = None,
    model: str | None = None,
    at_step: int = 0,
    agent: str | None = None,
    max_cost_usd: float | None = None,
    max_tokens: int | None = None,
    min_finding_severity: str = "error",
) -> SweepReport:
    """Apply ``patch`` to each run with :func:`crashlab.fork` and compare outcomes.

    ``patch`` is a :class:`~crashlab.patch.PatchSet`, a patch, or a path to a
    JSON / YAML patch file. ``model`` and ``agent`` are applied on top (a model
    change, and scoping request patches to one agent). Each run is classified as
    ``fixed`` (bad → good), ``regressed`` (good → bad), ``still_passing``,
    ``still_failing`` or ``errored``.
    """
    patches = _apply_scope(_resolve_patch(patch), model=model, agent=agent)
    report = SweepReport(patch_description=patches.describe())
    judge_kw: dict[str, Any] = {
        "store": store,
        "max_cost_usd": max_cost_usd,
        "max_tokens": max_tokens,
        "max_seconds": None,
        "max_steps": None,
        "min_finding_severity": min_finding_severity,
    }
    for run in runs:
        baseline = _judge(run, **judge_kw)
        patched_eval: EvalResult | None = None
        err: str | None = None
        try:
            branch = fork(
                run,
                at_step=at_step,
                patch=patches,
                store_dir=store.root if store is not None else None,
            )
            patched_eval = _judge(branch, **judge_kw)
        except (DivergenceError, ReplayError, ForkError) as exc:
            err = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # a failed fork must not abort the rest of the sweep
            err = f"{type(exc).__name__}: {exc}"
        verdict = _verdict(
            baseline.status,
            patched_eval.status if patched_eval is not None else None,
            errored=err is not None,
        )
        cost_delta = _delta(baseline.cost_usd, patched_eval.cost_usd if patched_eval else None)
        token_delta = (patched_eval.tokens if patched_eval else 0) - baseline.tokens
        dur_delta = _delta(
            baseline.duration_seconds,
            patched_eval.duration_seconds if patched_eval else None,
        )
        report.results.append(
            SweepResult(
                run_id=run.run_id,
                name=run.name,
                verdict=verdict,
                baseline_status=baseline.status,
                patched_status=patched_eval.status if patched_eval else None,
                patched_run_id=patched_eval.run_id if patched_eval else None,
                reason=err or (patched_eval.reason if patched_eval else baseline.reason),
                cost_delta_usd=cost_delta,
                token_delta=token_delta,
                duration_delta_seconds=dur_delta,
                baseline=baseline.to_dict(),
                patched=patched_eval.to_dict() if patched_eval else None,
            )
        )
    return report


# used by the CLI to build a patch from flags
def patch_from_flags(
    *,
    patch_file: str | Path | None = None,
    system_prompt: str | None = None,
    model: str | None = None,
    agent: str | None = None,
) -> PatchSet:
    """Combine a patch file with an inline system-prompt append and / or model change."""
    patches = PatchSet()
    if patch_file is not None:
        patches = patches + _resolve_patch(patch_file)
    if system_prompt is not None:
        patches = patches + SystemPromptPatch(op="append", text=system_prompt)
    return _apply_scope(patches, model=model, agent=agent)
