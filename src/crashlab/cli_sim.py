"""``crashlab simulate``: run a multi-turn scenario against a live agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.text import Text

from .cli import _CHECK_STYLE, Table, _console, _e, app
from .store import Store

_TURN_STYLE = {"pass": "green", "fail": "red", "skipped": "yellow"}


@app.command("simulate")
def simulate_cmd(
    scenario_path: Path = typer.Argument(  # noqa: B008
        ..., help="Scenario file (JSON, TOML or YAML).", metavar="SCENARIO_PATH"
    ),
    agent: str = typer.Option(..., "--agent", help="Live entry point as module:fn."),
    max_turns: int | None = typer.Option(
        None, "--max-turns", help="Stop after this many executed turns."
    ),
    rules: Path | None = typer.Option(  # noqa: B008
        None, "--rules", help="Declarative rule file to grade the finished run.", metavar="PATH"
    ),
    tag: list[str] | None = typer.Option(  # noqa: B008
        None, "--tag", help="Tag the recorded run. Repeatable."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the result as JSON."),
    fail_on_fail: bool = typer.Option(
        False, "--fail-on-fail", help="Exit 1 when the simulation does not pass."
    ),
) -> None:
    """Run a multi-turn scenario against a live agent.

    Catches context rot, constraint forgetting and pushback sycophancy.
    """
    from .replay.divergence import ReplayError
    from .replay.engine import resolve_entrypoint
    from .simulate import SimulateError, simulate

    console = _console(stderr=as_json)
    if max_turns is not None and max_turns < 0:
        console.print("[red]--max-turns must be >= 0[/]")
        raise typer.Exit(code=2)
    store = Store()
    try:
        fn = resolve_entrypoint(agent)
        result = simulate(
            fn,
            scenario_path,
            store=store,
            rules=str(rules) if rules is not None else None,
            max_turns=max_turns,
            tags=tag,
        )
    except (SimulateError, ReplayError, ValueError, FileNotFoundError, ImportError, OSError) as exc:
        if as_json:
            typer.echo(json.dumps({"ok": False, "error": str(exc)}, indent=2, default=str))
        else:
            console.print(f"[red]{_e(exc)}[/]")
        raise typer.Exit(code=2) from exc
    if as_json:
        payload = result.to_dict()
        payload["ok"] = result.passed
        typer.echo(json.dumps(payload, indent=2, default=str))
    else:
        _print_result(console, result)
    if fail_on_fail and not result.passed:
        raise typer.Exit(code=1)


def _print_result(console: Any, result: Any) -> None:
    status = "pass" if result.passed else "fail"
    color = "green" if result.passed else "red"
    run = f"  run {result.run_id}" if result.run_id else ""
    console.print(f"[{color}]{status}[/] {result.turns_executed} turn(s) executed{run}")
    table = Table(title="crashlab simulate", title_justify="left")
    table.add_column("turn", no_wrap=True, style="cyan")
    table.add_column("status", no_wrap=True)
    table.add_column("user", overflow="fold")
    table.add_column("reply", overflow="fold")
    table.add_column("detail", overflow="fold")
    for turn in result.turn_results:
        mark = "skipped" if turn.skipped else "pass" if turn.passed else "fail"
        detail = turn.skip_reason or (turn.errors[0] if turn.errors else "")
        if turn.tools and not detail:
            detail = "tools: " + ", ".join(turn.tools)
        table.add_row(
            str(turn.turn),
            Text(mark, style=_TURN_STYLE.get(mark, "")),
            (turn.user or "")[:80],
            (turn.reply or "")[:80],
            detail,
        )
    console.print(table)
    for check in result.check_results:
        style = _CHECK_STYLE.get(check.status, "")
        msg = f"  {check.message}" if check.message else ""
        console.print(Text(f"check {check.name}: {check.status}{msg}", style=style))
    for err in result.errors:
        console.print(f"[red]error: {_e(err)}[/]")


__all__ = ["simulate_cmd"]
