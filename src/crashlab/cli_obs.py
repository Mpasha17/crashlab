"""Observability commands registered on the main ``crashlab`` app: ``stats``,
``timeline`` and ``ui``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from .cli import Table, _console, _cost_text, _e, app
from .store import RunNotFoundError, Store

# --- crashlab stats -----------------------------------------------------------------


def _since(value: str | None) -> Any:
    if not value:
        return None
    from ._runscan import parse_since

    try:
        return parse_since(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--since") from None


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _dur(value: float | None) -> str:
    if value is None:
        return "-"
    if value < 1:
        return f"{value * 1000:.0f}ms"
    return f"{value:.2f}s"


@app.command("stats")
def stats_cmd(
    since: str | None = typer.Option(
        None, "--since", help="Started within 30m / 2h / 7d / 1w, or after an ISO date."
    ),
    name: str | None = typer.Option(None, "--name", help="Only runs of this agent name."),
    limit: int = typer.Option(0, "--limit", "-n", help="Max runs to scan, newest first (0 = all)."),
    as_json: bool = typer.Option(False, "--json", help="Print JSON."),
    html: Path | None = typer.Option(  # noqa: B008
        None, "--html", help="Write a self-contained HTML fleet report to this file."
    ),
) -> None:
    """Summarise recorded runs: pass / fail rates, cost, tokens, latency, tools and findings.

    Scans the store (newest first). Cost figures are list-price estimates, the same
    numbers as ``crashlab show``. --html writes a shareable single file (no external assets).
    """
    from .stats import compute_stats, render_fleet_html

    store = Store()
    try:
        stats = compute_stats(store, since=_since(since), name=name, limit=limit or None)
    except ValueError as exc:
        _console(stderr=True).print(f"[red]{_e(exc)}[/]")
        raise typer.Exit(code=2) from exc
    html_path = None
    if html is not None:
        try:
            out = Path(html)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(render_fleet_html(stats), "utf-8")
            html_path = out
        except OSError as exc:
            _console(stderr=True).print(f"[red]cannot write {_e(html)}: {_e(exc)}[/]")
            raise typer.Exit(code=2) from exc
    if as_json:
        typer.echo(json.dumps(stats.to_dict(), indent=2, default=str))
        return
    console = _console()
    if html_path is not None:
        console.print(f"wrote {_e(html_path)}")
    if stats.n == 0:
        extra = f" ({len(stats.unreadable)} unreadable)" if stats.unreadable else ""
        console.print(f"no runs in {_e(store.root)}{extra}")
        return
    console.print(
        f"{stats.n} runs  pass {_pct(stats.pass_rate)}  fail {_pct(stats.fail_rate)}  "
        f"error {_pct(stats.error_rate)}  unknown {_pct(stats.unknown_rate)}"
    )
    cost = _cost_text(stats.total_cost_usd)
    if stats.cost_unknown:
        cost += f"  ({stats.cost_unknown} run(s) with unknown cost)"
    console.print(
        f"cost {cost}  mean {_cost_text(stats.mean_cost_usd)}  "
        f"median {_cost_text(stats.median_cost_usd)}  p95 {_cost_text(stats.p95_cost_usd)}"
    )
    console.print(
        f"tokens {stats.total_tokens:,}  duration mean {_dur(stats.mean_duration_s)}  "
        f"p95 {_dur(stats.p95_duration_s)}"
    )
    if stats.by_agent:
        table = Table(title="by agent", title_justify="left", box=None)
        table.add_column("agent", style="bold")
        table.add_column("runs", justify="right")
        table.add_column("pass", justify="right")
        table.add_column("fail", justify="right")
        table.add_column("tokens", justify="right")
        table.add_column("cost", justify="right")
        table.add_column("mean dur", justify="right")
        for a in stats.by_agent:
            table.add_row(
                a.name, str(a.runs), str(a.pass_n), str(a.fail_n), f"{a.total_tokens:,}",
                _cost_text(a.total_cost_usd), _dur(a.mean_duration_s),
            )  # fmt: skip
        console.print(table)
    if stats.by_model:
        table = Table(title="by model", title_justify="left", box=None)
        table.add_column("model", style="bold")
        table.add_column("calls", justify="right")
        table.add_column("in tok", justify="right")
        table.add_column("out tok", justify="right")
        table.add_column("cost", justify="right")
        table.add_column("errors", justify="right")
        for m in stats.by_model:
            table.add_row(
                m.model, str(m.calls), f"{m.input_tokens:,}", f"{m.output_tokens:,}",
                _cost_text(m.total_cost_usd), str(m.error_n),
            )  # fmt: skip
        console.print(table)
    if stats.tools:
        table = Table(title="tools", title_justify="left", box=None)
        table.add_column("tool", style="bold")
        table.add_column("calls", justify="right")
        table.add_column("errors", justify="right")
        table.add_column("err rate", justify="right")
        table.add_column("p95", justify="right")
        for t in stats.tools:
            table.add_row(
                t.name, str(t.calls), str(t.error_n), _pct(t.error_rate), _dur(t.p95_duration_s),
            )  # fmt: skip
        console.print(table)
    if stats.findings:
        table = Table(title="runtime findings", title_justify="left", box=None)
        table.add_column("kind", style="bold")
        table.add_column("severity")
        table.add_column("n", justify="right")
        for f in stats.findings:
            table.add_row(f.kind, f.severity, str(f.count))
        console.print(table)
    if stats.failing_checks:
        table = Table(title="failing checks", title_justify="left", box=None)
        table.add_column("check", style="bold")
        table.add_column("fail", justify="right")
        table.add_column("error", justify="right")
        for c in stats.failing_checks:
            table.add_row(c.name, str(c.fail_n), str(c.error_n))
        console.print(table)
    if stats.slowest:
        table = Table(title="slowest", title_justify="left", box=None)
        table.add_column("run")
        table.add_column("name")
        table.add_column("duration", justify="right")
        for r in stats.slowest:
            table.add_row(r.run_id, r.name, _dur(r.duration_s))
        console.print(table)
    if stats.most_expensive:
        table = Table(title="most expensive", title_justify="left", box=None)
        table.add_column("run")
        table.add_column("name")
        table.add_column("cost", justify="right")
        for r in stats.most_expensive:
            table.add_row(r.run_id, r.name, _cost_text(r.cost_usd))
        console.print(table)
    if stats.unreadable:
        console.print(f"[yellow]unreadable: {', '.join(_e(x) for x in stats.unreadable)}[/]")


# --- crashlab timeline --------------------------------------------------------------


@app.command("timeline")
def timeline_cmd(
    run_ref: str = typer.Argument("last", help="Run id, unique id prefix, or 'last'."),
    no_children: bool = typer.Option(
        False, "--no-children", help="Do not include linked child runs."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the waterfall as JSON events."),
) -> None:
    """Print an ASCII waterfall of a run's steps: timing bars, tokens, cost and errors.

    Linked child runs are indented under the parent unless ``--no-children``.
    """
    from .timeline import render_timeline, timeline_events

    store = Store()
    try:
        run = store.load_run(run_ref)
    except RunNotFoundError as exc:
        _console().print(f"[red]{_e(exc)}[/]")
        raise typer.Exit(code=1) from exc
    if as_json:
        events = timeline_events(run, store=store, include_children=not no_children)
        typer.echo(json.dumps([e.to_dict() for e in events], indent=2, default=str))
        return
    typ = _console()
    typ.print(render_timeline(run, store=store, include_children=not no_children, width=typ.width))


# --- crashlab ui --------------------------------------------------------------------


@app.command("ui")
def ui_cmd(
    port: int = typer.Option(8765, "--port", help="Local port (0 picks an ephemeral port)."),
    no_open: bool = typer.Option(False, "--no-open", help="Do not open a browser."),
) -> None:
    """Open a local web trace explorer for recorded runs (localhost only).

    Serves a self-contained viewer and a JSON API on 127.0.0.1. Stop with Ctrl-C.
    """
    from .ui import create_server

    if port < 0:
        _console(stderr=True).print("[red]--port must be 0 or a positive port[/]")
        raise typer.Exit(code=2)
    store = Store()
    try:
        server = create_server(store, host="127.0.0.1", port=port)
    except OSError as exc:
        _console(stderr=True).print(f"[red]cannot bind 127.0.0.1:{port}: {_e(exc)}[/]")
        raise typer.Exit(code=2) from exc
    host, bound = server.server_address[:2]
    url = f"http://127.0.0.1:{bound}/"
    _console().print(f"crashlab ui  {url}  (store {_e(store.root)}; Ctrl-C to stop)")
    if not no_open:
        import webbrowser

        try:
            webbrowser.open(url)
        except Exception as exc:  # opening a browser is a convenience, not a requirement
            _console(stderr=True).print(f"[yellow]could not open a browser: {_e(exc)}[/]")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        raise typer.Exit(code=0) from None
    finally:
        server.shutdown()
        server.server_close()
