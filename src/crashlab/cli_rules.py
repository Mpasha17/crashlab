"""``crashlab rules``: validate and test declarative policy rule files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.text import Text

from .cli import _CHECK_STYLE, Table, _console, _e, _resolve, app
from .store import Store

rules_app = typer.Typer(
    name="rules",
    help="Validate and test declarative policy rule files.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)

app.add_typer(
    rules_app,
    name="rules",
    help="Validate and test declarative policy rule files.",
    rich_help_panel="Evaluate",
    no_args_is_help=True,
)


def _rules_from_arg(path: str | None) -> tuple[str, list[Any]]:
    from .assertions import RuleError, discover_rules, load_rules

    if path is None or str(path).strip().lower() in {"", "auto"}:
        found, rules = discover_rules()
        if found is None:
            raise RuleError(
                "no rules file found (looked for .crashlab/rules.json, "
                ".crashlab/rules.toml, .crashlab/rules.yaml, .crashlab/rules.yml, "
                "and [tool.crashlab.rules] in pyproject.toml)"
            )
        return str(found), rules
    source = str(Path(path).expanduser())
    return source, load_rules(path)


@rules_app.command("validate")
def rules_validate(
    path: str | None = typer.Argument(
        None,
        help="Rule file (JSON, TOML or YAML). Default: auto-discover.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print a JSON report."),
) -> None:
    """Validate the syntax and schema of a declarative rule file."""
    from .assertions import RuleError

    console = _console()
    try:
        source, rules = _rules_from_arg(path)
        if not rules:
            raise RuleError(f"no rules in {source}")
    except RuleError as exc:
        if as_json:
            typer.echo(json.dumps({"ok": False, "path": path, "error": str(exc)}, indent=2))
        else:
            console.print(f"[red]{_e(exc)}[/]")
        raise typer.Exit(code=1) from exc
    payload = {
        "ok": True,
        "path": source,
        "count": len(rules),
        "rules": [r.to_dict() for r in rules],
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=2, default=str))
        return
    console.print(f"[green]ok[/] {_e(source)}: {len(rules)} rule(s)")
    for rule in rules:
        extra = []
        if rule.agent:
            extra.append(f"agent={rule.agent}")
        if rule.name:
            extra.append(f"name={rule.name}")
        suffix = f" ({', '.join(extra)})" if extra else ""
        console.print(f"  {rule.id}  {rule.type}{suffix}")


@rules_app.command("test")
def rules_test(
    path: str = typer.Argument(..., help="Rule file (JSON, TOML or YAML), or 'auto'."),
    run_refs: list[str] | None = typer.Argument(  # noqa: B008
        None, help="Run ids / prefixes / 'last'. Default: last."
    ),
    all_runs: bool = typer.Option(False, "--all", help="Test every stored run (newest first)."),
    name: str | None = typer.Option(None, "--name", help="With --all: only this agent."),
    as_json: bool = typer.Option(False, "--json", help="Print a JSON report."),
) -> None:
    """Evaluate a rule file against one or more stored runs. Exit 1 on a failure."""
    from .assertions import RuleError, evaluate_rules

    console = _console()
    try:
        source, rules = _rules_from_arg(path)
        if not rules:
            raise RuleError(f"no rules in {source}")
    except RuleError as exc:
        if as_json:
            typer.echo(json.dumps({"ok": False, "path": path, "error": str(exc)}, indent=2))
        else:
            console.print(f"[red]{_e(exc)}[/]")
        raise typer.Exit(code=2) from exc

    store = Store()
    if all_runs:
        refs = [r["run_id"] for r in store.list_runs(limit=None, name=name)]
        if not refs:
            msg = f"no runs to test in {store.root}"
            if as_json:
                typer.echo(json.dumps({"ok": False, "path": source, "error": msg, "results": []}))
            else:
                console.print(f"[red]{_e(msg)}[/]")
            raise typer.Exit(code=2)
    else:
        refs = list(run_refs or ["last"])

    report: list[dict[str, Any]] = []
    failed = False
    table = Table(title="crashlab rules test", title_justify="left")
    table.add_column("run", no_wrap=True, style="cyan")
    table.add_column("name", overflow="fold")
    table.add_column("rule", overflow="fold")
    table.add_column("status", no_wrap=True)
    table.add_column("message", overflow="fold")
    for ref in refs:
        run = _resolve(store, ref)
        results = evaluate_rules(run, rules)
        if any(r.status != "pass" for r in results):
            failed = True
        report.append(
            {
                "run_id": run.run_id,
                "name": run.name,
                "checks": [c.model_dump() for c in results],
                "ok": all(c.status == "pass" for c in results),
            }
        )
        for result in results:
            table.add_row(
                run.run_id,
                run.name,
                result.name,
                Text(result.status, style=_CHECK_STYLE.get(result.status, "")),
                result.message or "-",
            )
    if as_json:
        typer.echo(
            json.dumps({"ok": not failed, "path": source, "results": report}, indent=2, default=str)
        )
    else:
        console.print(table)
        n_fail = sum(1 for row in report if not row["ok"])
        console.print(f"{len(report) - n_fail} run(s) passed, {n_fail} failed")
    if failed:
        raise typer.Exit(code=1)
