"""``crashlab score``: grade recorded runs with a weighted multi-criteria rubric."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.text import Text

from .cli import Table, _console, _e, app
from .store import RunNotFoundError, Store

_GRADE_STYLE = {"A": "green", "B": "green", "C": "yellow", "D": "yellow", "F": "red"}


@app.command("score")
def score_cmd(
    run_refs: list[str] | None = typer.Argument(  # noqa: B008
        None, help="Run ids / prefixes / 'last'. Default: last."
    ),
    all_runs: bool = typer.Option(False, "--all", help="Score every stored run (newest first)."),
    name: str | None = typer.Option(None, "--name", help="With --all: only this agent."),
    rubric_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--rubric",
        help="Rubric file (JSON, TOML or YAML). Default: built-in rubric.",
        metavar="PATH",
    ),
    min_score: float | None = typer.Option(
        None, "--min-score", help="Passing score (0–100). Default: the rubric's min_score (70)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
    fail_under: float | None = typer.Option(
        None,
        "--fail-under",
        help="Exit 1 if any run's score is below this value (0–100).",
    ),
) -> None:
    """Grade recorded runs 0–100 with a weighted rubric (outcome, safety, tools, cost, latency).

    Loads a JSON / TOML / YAML rubric with --rubric, or uses the built-in default
    (weights: outcome 0.35, safety 0.25, tool efficiency 0.15, cost 0.15, latency 0.10).
    Exit 0 if every run meets --min-score, 1 if any fall short (or under --fail-under),
    2 if there are no runs or the rubric is invalid.
    """
    from .rubrics import DEFAULT_RUBRIC as default_rubric
    from .rubrics import RubricError, load_rubric, score_runs

    console = _console(stderr=as_json)
    try:
        rubric = load_rubric(rubric_path) if rubric_path is not None else default_rubric
        rubric = rubric.with_min_score(min_score)
    except RubricError as exc:
        if as_json:
            typer.echo(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        else:
            console.print(f"[red]{_e(exc)}[/]")
        raise typer.Exit(code=2) from exc

    store = Store()
    runs = []
    if all_runs:
        refs = [r["run_id"] for r in store.list_runs(limit=None, name=name)]
        if not refs:
            msg = f"no runs to score in {store.root}"
            if as_json:
                typer.echo(json.dumps({"ok": False, "error": msg, "results": []}, indent=2))
            else:
                console.print(f"[red]{_e(msg)}[/]")
            raise typer.Exit(code=2)
        try:
            runs = [store.load_run(ref) for ref in refs]
        except RunNotFoundError as exc:
            console.print(f"[red]{_e(exc)}[/]")
            raise typer.Exit(code=2) from exc
    else:
        refs = list(run_refs or ["last"])
        for ref in refs:
            try:
                runs.append(store.load_run(ref))
            except RunNotFoundError as exc:
                if as_json:
                    typer.echo(json.dumps({"ok": False, "error": str(exc)}, indent=2))
                else:
                    console.print(f"[red]{_e(exc)}[/]")
                raise typer.Exit(code=2) from exc

    report = score_runs(runs, rubric)
    if as_json:
        payload: dict[str, Any] = report.to_dict()
        payload["ok"] = report.failed == 0
        typer.echo(json.dumps(payload, indent=2, default=str))
    else:
        _print_score_report(report)

    under = fail_under
    if under is not None and any(r.score < under for r in report.results):
        raise typer.Exit(code=1)
    if report.failed:
        raise typer.Exit(code=1)


def _print_score_report(report: Any) -> None:
    console = _console()
    table = Table(title="crashlab score", title_justify="left")
    table.add_column("run", no_wrap=True, style="cyan")
    table.add_column("name", overflow="fold")
    table.add_column("score", justify="right", no_wrap=True)
    table.add_column("grade", no_wrap=True)
    table.add_column("result", no_wrap=True)
    table.add_column("weakest", overflow="fold")
    for result in report.results:
        weakest = None
        if result.breakdown:
            weakest = min(result.breakdown, key=lambda c: (c.score, c.name))
        table.add_row(
            result.run_id or "-",
            result.name or "-",
            f"{result.score:.1f}",
            Text(result.grade, style=_GRADE_STYLE.get(result.grade, "")),
            Text("pass" if result.passed else "fail", style="green" if result.passed else "red"),
            f"{weakest.name} ({weakest.score:.2f})" if weakest is not None else "-",
        )
    console.print(table)
    if len(report.results) == 1:
        result = report.results[0]
        breakdown = Table(title="breakdown", title_justify="left")
        breakdown.add_column("criterion", overflow="fold")
        breakdown.add_column("kind", no_wrap=True)
        breakdown.add_column("score", justify="right", no_wrap=True)
        breakdown.add_column("points", justify="right", no_wrap=True)
        breakdown.add_column("detail", overflow="fold")
        for item in result.breakdown:
            breakdown.add_row(
                item.name,
                item.kind,
                f"{item.score:.2f}",
                f"{item.contribution:.1f}",
                item.message or "-",
            )
        console.print(breakdown)
    summary = (
        f"{report.passed} passed, {report.failed} failed of {report.total} "
        f"(mean {report.mean:.1f}, median {report.median:.1f}, "
        f"min {report.min:.1f}, max {report.max:.1f}"
    )
    if report.weakest:
        summary += f", weakest {report.weakest}"
    summary += f", min_score {report.min_score:g})"
    console.print(summary)
