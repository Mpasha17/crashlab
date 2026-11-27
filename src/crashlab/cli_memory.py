"""``crashlab memory`` and ``crashlab state``: memory / state debugging for multi-turn runs."""

from __future__ import annotations

import json
from typing import Any

import typer

from .cli import Table, _console, _e, app
from .store import RunNotFoundError, Store


def _load(run_ref: str, as_json: bool, console: Any) -> Any:
    try:
        return Store().load_run(run_ref)
    except RunNotFoundError as exc:
        if as_json:
            typer.echo(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        else:
            console.print(f"[red]{_e(str(exc))}[/]")
        raise typer.Exit(code=2) from exc


@app.command("memory")
def memory_cmd(
    run_ref: str = typer.Argument("last", help="Run id, unique prefix, or 'last'."),
    as_json: bool = typer.Option(False, "--json", help="Print the findings as JSON."),
    judge: bool = typer.Option(
        False, "--judge", help="Ask the judge model to confirm each finding (sends run data)."
    ),
    model: str | None = typer.Option(None, "--model", help="Judge model (with --judge)."),
) -> None:
    """Find where a multi-turn agent forgot, contradicted or lost context.

    Runs the memory.* detectors on a stored run: context truncation (messages silently
    dropped from a later request), forgotten facts (asked again or contradicted),
    self-contradiction, stale tool results and unbounded prompt growth. Each finding
    carries turn-level evidence. Settings come from the run's check config
    (the memory section of the checks config). With --judge the model set by $CRASHLAB_JUDGE_MODEL
    confirms or dismisses each finding. Exit 0 if clean (or all dismissed by the judge),
    1 if findings, 2 if the run does not exist or the judge cannot run.
    """
    from .judges import JudgeError
    from .memory import confirm_with_judge, memory_findings

    console = _console(stderr=as_json)
    run = _load(run_ref, as_json, console)
    findings = memory_findings(run)
    verdicts: list[dict[str, Any]] = []
    if judge and findings:
        try:
            verdicts = confirm_with_judge(run, findings, model=model, store=Store())
        except JudgeError as exc:
            if as_json:
                typer.echo(json.dumps({"ok": False, "error": str(exc)}, indent=2))
            else:
                console.print(f"[red]{_e(str(exc))}[/]")
            raise typer.Exit(code=2) from exc
    dismissed = {(v["kind"], v["key"]) for v in verdicts if not v["confirmed"]}
    real = [f for f in findings if (f.kind, f.key) not in dismissed]
    if as_json:
        typer.echo(
            json.dumps(
                {
                    "ok": not real,
                    "run_id": run.run_id,
                    "name": run.name,
                    "findings": [f.model_dump(mode="json") for f in findings],
                    "judge": verdicts,
                },
                indent=2,
                default=str,
            )
        )
        raise typer.Exit(code=0 if not real else 1)
    if not findings:
        console.print(f"no memory problems in run {_e(run.run_id)} ({_e(run.name)})")
        return
    by_key = {(v["kind"], v["key"]): v for v in verdicts}
    table = Table(title=f"memory findings: {run.run_id} ({run.name})")
    for col in ("kind", "severity", "steps", "what"):
        table.add_column(col)
    if verdicts:
        table.add_column("judge")
    for f in findings:
        row = [_e(f.kind), f.severity, ", ".join(str(i) for i in f.steps), _e(f.message)]
        if verdicts:
            v = by_key.get((f.kind, f.key))
            row.append("-" if v is None else ("confirmed" if v["confirmed"] else "dismissed"))
        table.add_row(*row)
    console.print(table)
    for f in findings:
        for ev in f.evidence:
            console.print(f"  [dim]step {ev.step_idx} {_e(ev.field)}:[/] {_e(ev.quote[:160])}")
    console.print(
        f"{len(real)} memory problem(s)"
        + (f" ({len(findings) - len(real)} dismissed by the judge)" if dismissed else "")
    )
    raise typer.Exit(code=0 if not real else 1)


def _cut(value: Any, width: int = 60) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= width else text[: width - 1] + "…"


@app.command("state")
def state_cmd(
    run_ref: str = typer.Argument("last", help="Run id, unique prefix, or 'last'."),
    diff: bool = typer.Option(False, "--diff", help="Show key-level changes between snapshots."),
    name: str | None = typer.Option(None, "--name", "-n", help="Only snapshots with this name."),
    as_json: bool = typer.Option(False, "--json", help="Print the snapshots (and diffs) as JSON."),
) -> None:
    """Show the state snapshots a run recorded with crashlab.snapshot(name, state).

    Lists each snapshot in order (step, turn, size); with --diff also the key-level
    changes (added, removed, changed) from the previous snapshot of the same name.
    Exit 0 (also when there are none), 2 if the run does not exist.
    """
    from .memory import snapshots, state_diff

    console = _console(stderr=as_json)
    run = _load(run_ref, as_json, console)
    snaps = snapshots(run, name)
    rows: list[dict[str, Any]] = []
    last: dict[str, Any] = {}
    for sn in snaps:
        row = sn.to_dict()
        if diff:
            prev = last.get(sn.name)
            row["changes"] = (
                [] if prev is None else [c.to_dict() for c in state_diff(prev, sn.state)]
            )
            row["first"] = prev is None
        last[sn.name] = sn.state
        rows.append(row)
    if as_json:
        typer.echo(
            json.dumps({"ok": True, "run_id": run.run_id, "snapshots": rows}, indent=2, default=str)
        )
        return
    if not rows:
        console.print(f"no state snapshots in run {_e(run.run_id)} ({_e(run.name)}); "
                      "record them with crashlab.snapshot(name, state)")  # fmt: skip
        return
    table = Table(title=f"state snapshots: {run.run_id} ({run.name})")
    for col in ("step", "turn", "name", "chars", "state" if not diff else "changes"):
        table.add_column(col)
    for r in rows:
        if diff:
            if r["first"]:
                shown = "(first snapshot)"
            elif not r["changes"]:
                shown = "(no change)"
            else:
                shown = "\n".join(
                    f"{c['op']} {_e(c['path'])}"
                    + (
                        f": {_e(_cut(c['old']))} -> {_e(_cut(c['new']))}"
                        if c["op"] == "changed"
                        else f": {_e(_cut(c['new']))}"
                        if c["op"] == "added"
                        else f": {_e(_cut(c['old']))}"
                    )
                    for c in r["changes"][:12]
                )
        else:
            shown = _e(_cut(r["state"], 90))
        table.add_row(str(r["step"]), "-" if r["turn"] is None else str(r["turn"]),
                      _e(r["name"]), str(r["chars"]), shown)  # fmt: skip
    console.print(table)
