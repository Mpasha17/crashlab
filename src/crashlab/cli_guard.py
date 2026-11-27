"""``crashlab guard-report``: what ``crashlab.guard`` did during a recorded run."""

from __future__ import annotations

import json
from typing import Any

import typer

from .cli import Table, _console, _e, app
from .store import RunNotFoundError, Store


def _detail(entry: dict[str, Any]) -> str:
    detail = entry.get("detail") or {}
    if not isinstance(detail, dict):
        return str(detail)
    parts = []
    for key, value in detail.items():
        if key in ("errors", "problems") and isinstance(value, list):
            value = len(value)
        parts.append(f"{key}={value}")
    return ", ".join(parts)[:160]


@app.command("guard-report")
def guard_report_cmd(
    run_ref: str = typer.Argument("last", help="Run id, unique prefix, or 'last'."),
    as_json: bool = typer.Option(False, "--json", help="Print the interventions as JSON."),
) -> None:
    """List the interventions crashlab.guard made in a run (blocks, warnings, retries,
    fallbacks, argument repairs, redactions).

    Reads the ``crashlab.guard`` metadata on each step and the run's ``guard.*``
    findings. Replays of a guarded run show the recorded interventions too. Exit 0
    (also when there are none), 2 if the run does not exist.
    """
    from .guard import interventions

    console = _console(stderr=as_json)
    try:
        run = Store().load_run(run_ref)
    except RunNotFoundError as exc:
        if as_json:
            typer.echo(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        else:
            console.print(f"[red]{_e(str(exc))}[/]")
        raise typer.Exit(code=2) from exc
    rows = interventions(run)
    findings = [f for f in run.findings if f.kind.startswith("guard.")]
    if as_json:
        typer.echo(
            json.dumps(
                {
                    "ok": True,
                    "run_id": run.run_id,
                    "name": run.name,
                    "status": run.status,
                    "interventions": rows,
                    "findings": [f.model_dump(mode="json") for f in findings],
                },
                indent=2,
                default=str,
            )
        )
        return
    if not rows and not findings:
        console.print(f"no guard interventions in run {_e(run.run_id)} ({_e(run.name)})")
        return
    table = Table(title=f"guard interventions: {run.run_id} ({run.name})")
    for col in ("step", "call", "policy", "action", "detail"):
        table.add_column(col)
    for r in rows:
        table.add_row(
            str(r["step"]),
            _e(f"{r['operation']} {r['call']}"),
            _e(str(r.get("policy"))),
            _e(str(r.get("action"))),
            _e(_detail(r)),
        )
    console.print(table)
    stopped = [f for f in findings if f.stopped_run]
    blocked = sum(1 for r in rows if r.get("action") in ("blocked", "stopped"))
    console.print(
        f"{len(rows)} intervention(s), {blocked} blocked call(s)"
        + (f"; run stopped by {_e(stopped[0].kind)}" if stopped else "")
    )
