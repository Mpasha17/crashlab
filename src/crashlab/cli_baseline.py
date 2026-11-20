"""``crashlab baseline save`` / ``crashlab baseline check`` registered on the main app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.text import Text

from .cli import Table, _console, _e, app
from .store import Store

baseline_app = typer.Typer(
    name="baseline",
    help="Snapshot fleet metrics and fail CI on regressions.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)

app.add_typer(
    baseline_app,
    name="baseline",
    help="Snapshot fleet metrics and fail CI on regressions.",
    rich_help_panel="Evaluate",
    no_args_is_help=True,
)


def _since(value: str | None) -> str | None:
    if not value:
        return None
    from ._runscan import parse_since

    try:
        parse_since(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--since") from None
    return value


def _pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def _num(value: float | None, *, kind: str) -> str:
    if value is None:
        return "-"
    if kind == "cost":
        return "$0" if value == 0 else f"${value:.4f}" if value >= 0.0001 else f"${value:.6f}"
    if kind == "dur":
        if value < 1:
            return f"{value * 1000:.0f}ms"
        return f"{value:.2f}s"
    if kind == "pct":
        return _pct(value)
    if kind == "ratio":
        return f"{value:.2f}×"
    return f"{value:g}"


def _kind_for(metric: str) -> str:
    if "cost" in metric:
        return "cost"
    if "duration" in metric or "latency" in metric:
        return "dur"
    if "pass_rate" in metric or "error_rate" in metric or metric.endswith("_rate"):
        return "pct"
    return "num"


@baseline_app.command("save")
def baseline_save(
    out: Path | None = typer.Option(  # noqa: B008
        None, "--out", "-o", help="JSON path (default: <store>/baseline.json)."
    ),
    since: str | None = typer.Option(
        None, "--since", help="Started within 30m / 2h / 7d / 1w, or after an ISO date."
    ),
    name: str | None = typer.Option(None, "--name", help="Only runs of this agent name."),
    limit: int = typer.Option(0, "--limit", "-n", help="Max runs to scan, newest first (0 = all)."),
) -> None:
    """Write a fleet-metrics snapshot (pass rate, p95 cost and latency, tools) to JSON."""
    from .baseline import save_baseline

    store = Store()
    path = out if out is not None else store.root / "baseline.json"
    try:
        payload = save_baseline(store.root, path, since=_since(since), name=name, limit=limit)
    except ValueError as exc:
        _console(stderr=True).print(f"[red]{_e(exc)}[/]")
        raise typer.Exit(code=2) from exc
    except OSError as exc:
        _console(stderr=True).print(f"[red]cannot write {_e(path)}: {_e(exc)}[/]")
        raise typer.Exit(code=2) from exc
    console = _console()
    console.print(
        f"wrote {_e(path)}  {payload['n']} run(s)  pass {_pct(payload['pass_rate'])}  "
        f"p95 cost {_num(payload.get('p95_cost_usd'), kind='cost')}  "
        f"p95 {_num(payload.get('p95_duration_s'), kind='dur')}"
    )


@baseline_app.command("check")
def baseline_check(
    baseline: Path | None = typer.Argument(  # noqa: B008
        None, help="Baseline JSON (default: <store>/baseline.json)."
    ),
    since: str | None = typer.Option(
        None, "--since", help="Started within 30m / 2h / 7d / 1w, or after an ISO date."
    ),
    name: str | None = typer.Option(None, "--name", help="Only runs of this agent name."),
    limit: int = typer.Option(0, "--limit", "-n", help="Max runs to scan, newest first (0 = all)."),
    min_pass_rate_delta: float = typer.Option(
        -0.05,
        "--min-pass-rate-delta",
        help="Fail when pass rate drops by more than this (default: -0.05).",
    ),
    max_cost_ratio: float = typer.Option(
        1.25, "--max-cost-ratio", help="Fail when p95 cost exceeds this multiple of baseline."
    ),
    max_latency_ratio: float = typer.Option(
        1.5, "--max-latency-ratio", help="Fail when p95 latency exceeds this multiple of baseline."
    ),
    fail_on_regression: bool = typer.Option(
        False, "--fail-on-regression", help="Exit 1 when a regression is found (CI)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the comparison as JSON."),
) -> None:
    """Compare the current fleet against a saved baseline and report regressions."""
    from .baseline import BaselineError, compare_baseline

    store = Store()
    path = baseline if baseline is not None else store.root / "baseline.json"
    try:
        result = compare_baseline(
            path,
            store.root,
            min_pass_rate_delta=min_pass_rate_delta,
            max_cost_ratio=max_cost_ratio,
            max_latency_ratio=max_latency_ratio,
            since=_since(since),
            name=name,
            limit=limit,
        )
    except BaselineError as exc:
        _console(stderr=True).print(f"[red]{_e(exc)}[/]")
        raise typer.Exit(code=2) from exc
    except ValueError as exc:
        _console(stderr=True).print(f"[red]{_e(exc)}[/]")
        raise typer.Exit(code=2) from exc
    if as_json:
        typer.echo(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        _print_result(result)
    if fail_on_regression and not result.ok:
        raise typer.Exit(code=1)


def _print_result(result: Any) -> None:
    console = _console()
    style = "green" if result.ok else "red"
    console.print(Text(result.summary(), style=style))
    table = Table(title="vs baseline", title_justify="left", box=None)
    table.add_column("metric", style="bold")
    table.add_column("baseline", justify="right")
    table.add_column("current", justify="right")
    table.add_column("Δ", justify="right")
    table.add_column("", no_wrap=True)
    for row in result.comparisons:
        kind = _kind_for(str(row["metric"]))
        delta = row.get("delta")
        ratio = row.get("ratio")
        if row.get("skipped"):
            change = "skipped"
        elif ratio is not None:
            change = _num(ratio, kind="ratio")
        else:
            change = _num(delta, kind=kind) if delta is not None else "-"
        flag = "" if row.get("skipped") else ("ok" if row.get("ok") else "REGRESSED")
        table.add_row(
            str(row["metric"]),
            _num(row.get("baseline"), kind=kind),
            _num(row.get("current"), kind=kind),
            change,
            flag,
        )
    console.print(table)
    for reg in result.regressions:
        console.print(f"[red]{_e(reg.message)}[/]")
