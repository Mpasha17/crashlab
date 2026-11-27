"""``crashlab repair``: diagnose -> propose -> verify -> iterate, in one command."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .cli import Table, _console, _e, app
from .store import RunNotFoundError


@app.command("repair")
def repair_cmd(
    run_ref: str = typer.Argument("last", help="Run id, unique id prefix, or 'last'."),
    max_iterations: int = typer.Option(
        3, "--max-iterations", min=1, help="Propose/verify rounds before giving up."
    ),
    candidates: int = typer.Option(
        3, "--candidates", min=1, help="Candidate patches verified per round."
    ),
    trials: int = typer.Option(
        3, "--trials", "-k", min=1, help="Patched forks (and unpatched controls) per candidate."
    ),
    model: str | None = typer.Option(
        None, "--model", "-m", help="Diagnoser / fix model (default $CRASHLAB_DIAGNOSER_MODEL)."
    ),
    judge: list[str] = typer.Option(  # noqa: B008
        [], "--judge", help="Judge every passing patched fork too (repeatable), e.g. safety."
    ),
    judge_model: str | None = typer.Option(None, "--judge-model", help="Model for --judge."),
    min_pass_rate: float = typer.Option(
        0.8, "--min-pass-rate", min=0.0, max=1.0, help="Judged pass rate a patch must reach."
    ),
    regress: str | None = typer.Option(
        None, "--regress", help="Regression runs: ids, tag:X or 'none' (default: past passes)."
    ),
    max_cost: float | None = typer.Option(
        None, "--max-cost", help="Stop verifying once the repair has spent this much USD."
    ),
    write_patch: Path | None = typer.Option(  # noqa: B008
        None, "--write-patch", help="Write the chosen patch here (.json, or .yaml)."
    ),
    export_test: Path | None = typer.Option(  # noqa: B008
        None, "--export-test", help="Write an offline pytest regression test into this folder."
    ),
    allow_live_side_effects: bool = typer.Option(
        False, "--allow-live-side-effects", help="Run side-effect tools live in forks (sandbox)."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Diagnose and propose candidates only; verify nothing."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the repair result as JSON."),
) -> None:
    """Repair a failed run: diagnose it, propose candidate patches, verify each by
    forking the run (patched vs control, flake-aware), feed failures back and iterate,
    then write the verified patch and a regression test.

    Exit 0 when a patch is VERIFIED (or with --dry-run), 1 when not fixed or only
    partially fixed, 2 on a usage error (unknown run, a run that passed).
    """
    from .diagnose import DiagnoseError
    from .export import ExportError
    from .export import export_test as _export_test
    from .repair import Attempt, RepairError, repair

    console = _console(stderr=as_json)

    def fail(msg: str) -> None:
        if as_json:
            typer.echo(json.dumps({"ok": False, "error": msg}, indent=2))
        else:
            console.print(f"[red]{_e(msg)}[/]")
        raise typer.Exit(code=2)

    def show(a: Attempt) -> None:
        if as_json:
            return
        mark = "[green]✓[/]" if a.accepted else ("[dim]·[/]" if dry_run else "[red]✗[/]")
        rate = f" rate {a.pass_rate:.2f}" if a.pass_rate is not None else ""
        ctrl = f" vs control {a.control}" if a.control else ""
        stats = f"{a.passes}/{a.trials}{ctrl}{rate}" if not dry_run else "proposed"
        console.print(
            f"{mark} round {a.iteration} {_e(a.fix_id)} [{_e(a.verdict)}] {stats}: "
            f"{_e(a.patch[:90])}"
        )

    try:
        res = repair(
            run_ref,
            max_iterations=max_iterations,
            candidates=candidates,
            trials=trials,
            diagnoser=model,
            judges=judge or None,
            judge_model=judge_model,
            min_pass_rate=min_pass_rate,
            regress=regress,
            max_cost_usd=max_cost,
            allow_live_side_effects=allow_live_side_effects,
            dry_run=dry_run,
            on_attempt=show,
        )
    except (RepairError, RunNotFoundError, DiagnoseError, ValueError) as exc:
        fail(str(exc))
        return
    written: dict[str, list[str]] = {}
    if res.patch is not None and write_patch is not None:
        written["patch"] = [str(res.write_patch(write_patch))]
    if res.patch is not None and export_test is not None:
        try:
            paths = _export_test(res.run_id, fix=res.patch.fix_id, out_dir=export_test)
            written["test"] = [str(p) for p in paths]
        except ExportError as exc:
            written["test_error"] = [str(exc)]
    if as_json:
        typer.echo(json.dumps({"ok": True, **res.to_dict(), "written": written}, indent=2))
    else:
        if res.attempts:
            t = Table(title=f"repair {res.run_id[:10]}", show_lines=False)
            for col in ("round", "fix", "kind", "verdict", "patched", "control", "rate", "patch"):
                t.add_column(col)
            for a in res.attempts:
                t.add_row(
                    str(a.iteration),
                    a.fix_id,
                    a.kind,
                    a.verdict + (" ✓" if a.accepted else ""),
                    f"{a.passes}/{a.trials}" if a.trials else "-",
                    a.control or "-",
                    f"{a.pass_rate:.2f}" if a.pass_rate is not None else "-",
                    _e(a.patch[:60]),
                )
            console.print(t)
        color = {"fixed": "green", "partial": "yellow"}.get(res.status, "red")
        console.print(f"[bold {color}]{res.status.upper()}[/] after {res.iterations} round(s)")
        for line in res.evidence[:6]:
            console.print(f"  [dim]{_e(line[:200])}[/]")
        if res.patch is not None:
            console.print(f"patch: {_e(res.patch.after)}")
        for kind, paths in written.items():
            for p in paths:
                style = "red" if kind == "test_error" else "dim"
                console.print(f"[{style}]{kind}: {_e(p)}[/]")
        if res.cost:
            console.print(f"[dim]cost ${res.cost:.4f}[/]")
    if dry_run or res.status == "fixed":
        raise typer.Exit(code=0)
    raise typer.Exit(code=1)
