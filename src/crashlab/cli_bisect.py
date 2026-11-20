"""``crashlab bisect GOOD BAD``: compare a passing run with a failing run."""

from __future__ import annotations

import json
from typing import Any

import typer
from rich.text import Text

from .bisect import bisect_runs
from .cli import Table, _console, _e, app
from .store import RunNotFoundError, Store


@app.command("bisect")
def bisect_cmd(
    good: str = typer.Argument(..., help="Passing run (id, unique prefix, or 'last')."),
    bad: str = typer.Argument(..., help="Failing run (id, unique prefix, or 'last')."),
    isolate: bool = typer.Option(
        False,
        "--isolate",
        help="Fork the failing run with each single revert to find the minimal causal change.",
    ),
    agent: str | None = typer.Option(
        None, "--agent", help="Only this agent's steps and isolation patches."
    ),
    model: str | None = typer.Option(
        None, "--model", help="Only chat steps whose model name contains this."
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable report."),
) -> None:
    """Compare a passing run and a failing run; optionally isolate the minimal causal change."""
    console = _console()
    store = Store()
    try:
        report = bisect_runs(good, bad, store=store, isolate=isolate, agent=agent, model=model)
    except RunNotFoundError as exc:
        console.print(f"[red]{_e(exc)}[/]")
        raise typer.Exit(code=1) from exc

    if as_json:
        typer.echo(json.dumps(report.to_dict(), indent=2, default=str))
        return

    console.print(Text(report.summary()))
    head = Table(title=f"A {report.good.run_id} (good)  vs  B {report.bad.run_id} (bad)",
                 title_justify="left")  # fmt: skip
    head.add_column("", style="bold", no_wrap=True)
    head.add_column("A (good)", overflow="fold")
    head.add_column("B (bad)", overflow="fold")
    head.add_row("name", report.good.name, report.bad.name)
    head.add_row("outcome", report.good.outcome, report.bad.outcome)
    head.add_row("code", report.good.code_version or "-", report.bad.code_version or "-")
    if report.first_divergence is not None:
        head.add_row("first divergence", f"step {report.first_divergence}", "")
    console.print(head)

    if report.factors:
        ft = Table(title="suspected factors (closest first)", title_justify="left")
        ft.add_column("kind", style="bold", no_wrap=True)
        ft.add_column("at", justify="right", no_wrap=True)
        ft.add_column("summary", overflow="fold")
        for factor in report.factors:
            at = "-" if factor.first_step is None else str(factor.first_step)
            ft.add_row(factor.kind, at, factor.summary)
        console.print(ft)

    if isolate:
        if report.causal is not None:
            console.print(f"[green]minimal causal change:[/] {report.causal.patch_description}")
        elif report.trials:
            console.print("[yellow]no single revert restored outcome=pass[/]")
        else:
            console.print("[dim]no reversible candidates to isolate[/]")
        if report.trials:
            tt = Table(title="isolation trials", title_justify="left")
            tt.add_column("factor", style="bold", no_wrap=True)
            tt.add_column("outcome", no_wrap=True)
            tt.add_column("patch", overflow="fold")
            for trial in report.trials:
                mark = "pass" if trial.restored else (trial.outcome or "error")
                detail: Any = trial.patch_description
                if trial.error:
                    detail = f"{detail} ({trial.error})"
                tt.add_row(trial.factor_kind, mark, str(detail))
            console.print(tt)
