"""Operational commands registered on the main ``crashlab`` app: ``doctor``, ``init``
and ``profile``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.text import Text

from .cli import Table, _console, _cost_text, _e, app
from .store import RunNotFoundError, Store

_STATUS_STYLE = {"ok": "green", "warn": "yellow", "error": "red"}


@app.command("doctor")
def doctor_cmd(
    fix: bool = typer.Option(
        False,
        "--fix",
        help="Apply safe repairs: initialize a missing/empty store, VACUUM/REINDEX the index.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
) -> None:
    """Diagnose the local crashlab store, optional packages, redaction and pricing.

    Checks the store directory and SQLite index (integrity, schema, size, run count),
    orphaned child runs, a dangling last run, installed extras, redaction / env vars,
    and whether observed models have a list price. --fix never deletes recorded runs.
    """
    from .doctor import run_doctor

    report = run_doctor(fix=fix)
    if as_json:
        typer.echo(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        _print_doctor(report)
    if report.errors:
        raise typer.Exit(code=1)


@app.command("init")
def init_cmd(
    target_dir: Path = typer.Argument(  # noqa: B008
        Path("."),
        help="Project directory to scaffold (default: current directory).",
    ),
    ci: str = typer.Option("github", "--ci", help="CI provider: github or none."),
    checks: bool = typer.Option(
        True, "--checks/--no-checks", help="Write a starter crashlab_checks.py."
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite existing scaffold files."),
) -> None:
    """Scaffold a crashlab eval CI workflow and starter @crashlab.check oracles.

    Writes .github/workflows/crashlab-eval.yml (when --ci github) and crashlab_checks.py,
    and adds .crashlab/ to .gitignore. Existing files are left alone unless --force.
    """
    from .scaffold import init_project

    console = _console()
    try:
        result = init_project(target_dir, ci=ci, checks=checks, force=force)
    except ValueError as exc:
        console.print(f"[red]{_e(exc)}[/]")
        raise typer.Exit(code=2) from exc
    except OSError as exc:
        console.print(f"[red]could not write scaffold:[/] {_e(exc)}")
        raise typer.Exit(code=2) from exc
    if result.created:
        console.print("created: " + ", ".join(_e(p) for p in result.created))
    if result.updated:
        console.print("updated: " + ", ".join(_e(p) for p in result.updated))
    if result.skipped:
        console.print("skipped: " + ", ".join(_e(p) for p in result.skipped))
    for note in result.notes:
        console.print(f"[dim]{_e(note)}[/]")
    if not (result.created or result.updated or result.skipped or result.notes):
        console.print(f"nothing to do in {_e(result.target_dir)}")


@app.command("profile")
def profile_cmd(
    run_ref: str = typer.Argument("last", help="Run id, unique prefix, or 'last'."),
    compare_models: str | None = typer.Option(
        None,
        "--compare-models",
        help="Comma-separated model names for what-if cost estimates.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the profile as JSON."),
) -> None:
    """Break down a run's time, tokens, cost, cache opportunities and model swaps.

    Reports LLM vs tool vs other overhead, the slowest and costliest steps, a tool
    leaderboard, repeated prompt-prefix tokens, and list-price what-if swaps.
    """
    from .profiler import profile_run

    console = _console(stderr=as_json)
    store = Store()
    try:
        run = store.load_run(run_ref)
    except RunNotFoundError as exc:
        console.print(f"[red]{_e(exc)}[/]")
        raise typer.Exit(code=2) from exc
    models = None
    if compare_models:
        models = [m.strip() for m in compare_models.split(",") if m.strip()]
        if not models:
            console.print("[red]--compare-models is empty[/]")
            raise typer.Exit(code=2)
    profile = profile_run(run, compare_models=models)
    if as_json:
        typer.echo(json.dumps(profile.to_dict(), indent=2, default=str))
        return
    _print_profile(profile)


def _print_doctor(report: Any) -> None:
    console = _console()
    table = Table(title=report.summary, title_justify="left", box=None)
    table.add_column("check", style="bold", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("message", overflow="fold")
    for check in report.checks:
        label = check.status + (" (fixed)" if check.fixed else "")
        table.add_row(
            check.name, Text(label, style=_STATUS_STYLE.get(check.status, "")), check.message
        )
    console.print(table)
    if report.fixes_applied:
        console.print("fixes: " + "; ".join(_e(f) for f in report.fixes_applied))


def _print_profile(profile: Any) -> None:
    console = _console()
    console.print(
        f"profile {_e(profile.run_id)}  ({_e(profile.name)})  "
        f"duration {_fmt_ms(profile.total_duration_ms)}  "
        f"tokens {profile.input_tokens:,} in / {profile.output_tokens:,} out  "
        f"cost {_cost_text(profile.cost_usd)}"
    )
    console.print(
        f"LLM {_fmt_ms(profile.llm_duration_ms)} ({profile.llm_pct:.1f}%)  "
        f"tools {_fmt_ms(profile.tool_duration_ms)} ({profile.tool_pct:.1f}%)  "
        f"other {_fmt_ms(profile.other_duration_ms)} ({profile.other_pct:.1f}%)"
    )
    if profile.steps:
        table = Table(title="steps", title_justify="left", box=None)
        table.add_column("#", justify="right", no_wrap=True)
        table.add_column("op", no_wrap=True)
        table.add_column("name")
        table.add_column("time", justify="right", no_wrap=True)
        table.add_column("%", justify="right", no_wrap=True)
        table.add_column("in", justify="right", no_wrap=True)
        table.add_column("out", justify="right", no_wrap=True)
        table.add_column("cost", justify="right", no_wrap=True)
        for s in profile.steps:
            table.add_row(
                str(s.idx),
                s.category,
                s.name,
                _fmt_ms(s.duration_ms),
                f"{s.duration_pct:.1f}",
                str(s.input_tokens) if s.input_tokens else "",
                str(s.output_tokens) if s.output_tokens else "",
                _cost_text(s.cost_usd) if s.cost_usd is not None else "",
            )
        console.print(table)
    if profile.slowest_steps:
        bits = [f"{s.idx} {s.name} {_fmt_ms(s.duration_ms)}" for s in profile.slowest_steps[:3]]
        console.print("slowest: " + ", ".join(bits))
    if profile.costliest_steps:
        bits = [f"{s.idx} {s.name} {_cost_text(s.cost_usd)}" for s in profile.costliest_steps[:3]]
        console.print("costliest: " + ", ".join(bits))
    if profile.tools:
        table = Table(title="tools", title_justify="left", box=None)
        table.add_column("tool", style="bold")
        table.add_column("calls", justify="right")
        table.add_column("total", justify="right")
        table.add_column("mean", justify="right")
        table.add_column("max", justify="right")
        for t in profile.tools:
            table.add_row(
                t.name,
                str(t.calls),
                _fmt_ms(t.total_duration_ms),
                _fmt_ms(t.mean_duration_ms),
                _fmt_ms(t.max_duration_ms),
            )
        console.print(table)
    cache = profile.cache
    if cache.consecutive_pairs:
        console.print(
            f"cache opportunity: ~{cache.estimated_redundant_tokens:,} redundant token(s) "
            f"across {cache.consecutive_pairs} consecutive LLM pair(s) "
            f"(of {cache.total_input_tokens:,} input tokens)"
        )
    elif profile.steps:
        console.print("cache opportunity: no consecutive LLM steps to compare")
    if profile.what_if:
        table = Table(title="what-if cost", title_justify="left", box=None)
        table.add_column("model", style="bold")
        table.add_column("est. cost", justify="right")
        table.add_column("note")
        for swap in profile.what_if:
            cost = _cost_text(swap.estimated_cost_usd) if swap.priced else "?"
            table.add_row(swap.model, cost, swap.note or "")
        console.print(table)


def _fmt_ms(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:.2f}s"
    return f"{value:.0f}ms"
