"""``crashlab stress``: re-run a recorded agent under built-in chaos scenarios."""

from __future__ import annotations

import json
from typing import Any

import typer
from rich.table import Table

from .cli import _console, _e, app
from .store import Store
from .stress import SCENARIOS, StressReport, stress_run

_SCENARIO_HELP = ", ".join(SCENARIOS)


@app.command("stress")
def stress_cmd(
    run_ref: str = typer.Argument("last", help="Run id, unique id prefix, or 'last'."),
    scenario: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--scenario",
        help=f"Only these scenarios (repeatable). One of: {_SCENARIO_HELP}.",
    ),
    at_step: int = typer.Option(0, "--at-step", help="Fork at this step (default 0)."),
    agent: str | None = typer.Option(
        None, "--agent", help="Only inject into this agent's steps (multi-agent runs)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
    fail_on_crash: bool = typer.Option(
        False, "--fail-on-crash", help="Exit 1 if a scenario crashed or failed its checks."
    ),
) -> None:
    """Run chaos scenarios and report whether the agent recovered.

    Re-runs the recorded entry point under built-in faults that target the tools
    and models the run actually used (a 502 from a tool, a tool timeout, an
    empty or drifted tool result, an LLM rate limit, an empty reply, malformed
    tool-call arguments). Same requirements as `crashlab fork`: the entry point
    must be importable and a live client available. Scenarios that cannot run
    are reported as skipped or needs_live_agent, not as crashes.
    """
    from .store import RunNotFoundError

    console = _console()
    store = Store()
    try:
        store.load_run(run_ref)
    except RunNotFoundError as exc:
        console.print(f"[red]{_e(exc)}[/]")
        raise typer.Exit(code=1) from exc
    try:
        report = stress_run(
            run_ref, scenarios=scenario or None, at_step=at_step, agent=agent, store=store
        )
    except ValueError as exc:
        console.print(f"[red]{_e(exc)}[/]")
        raise typer.Exit(code=2) from exc
    if as_json:
        typer.echo(json.dumps(report.as_dict(), indent=2, default=str))
    else:
        _print_report(console, report)
    if fail_on_crash and any(r.outcome in ("crashed", "failed_checks") for r in report.results):
        raise typer.Exit(code=1)


def _print_report(console: Any, report: StressReport) -> None:
    table = Table(title=f"crashlab stress  source {report.source_run_id}", title_justify="left")
    table.add_column("scenario", no_wrap=True, style="cyan")
    table.add_column("target", overflow="fold")
    table.add_column("outcome", no_wrap=True)
    table.add_column("survived", no_wrap=True)
    table.add_column("checks", no_wrap=True)
    table.add_column("detail", overflow="fold")
    style = {
        "survived": "green",
        "crashed": "red",
        "failed_checks": "red",
        "skipped": "yellow",
        "needs_live_agent": "yellow",
    }
    for r in report.results:
        checks = {True: "pass", False: "fail", None: "—"}.get(r.checks_passed, "—")
        table.add_row(
            r.scenario,
            r.target or "—",
            f"[{style.get(r.outcome, 'white')}]{r.outcome}[/]",
            "yes" if r.survived else "no",
            checks,
            r.detail or ("· ".join(r.findings[:2]) if r.findings else ""),
        )
    console.print(table)
    ran = [r for r in report.results if r.outcome not in ("skipped", "needs_live_agent")]
    if not ran:
        console.print("No scenario could be executed (needs a live agent / client).")
    elif report.survived:
        console.print(f"[green]survived {sum(r.survived for r in ran)}/{len(ran)} executed[/]")
    else:
        bad = sum(1 for r in ran if not r.survived)
        console.print(f"[red]{bad}/{len(ran)} executed scenario(s) did not survive[/]")


__all__ = ["stress_cmd"]
