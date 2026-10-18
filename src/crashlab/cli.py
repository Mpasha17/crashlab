"""``crashlab`` command line: ``ls`` and ``show`` (more commands land with replay/diagnose)."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ._version import __version__
from .models import Run, Step
from .store import ENV_DIR, RunNotFoundError, Store

app = typer.Typer(
    name="crashlab",
    help="Crash-test your AI agents. Inspect locally recorded agent runs.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)

_STATUS_STYLE = {"ok": "green", "error": "red", "running": "yellow"}
_OUTCOME_STYLE = {"pass": "green", "fail": "red", "unknown": "dim"}


def _console() -> Console:
    return Console()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"crashlab {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True, help="Show version."
    ),
    store_dir: str | None = typer.Option(
        None, "--dir", help=f"Store directory (default: ${ENV_DIR} or ./.crashlab)."
    ),
) -> None:
    """Crash-test your AI agents."""
    if store_dir:
        os.environ[ENV_DIR] = store_dir


def _local_time(value: str | datetime | None) -> str:
    if not value:
        return "-"
    if isinstance(value, datetime):
        dt = value
    else:  # Python 3.10's fromisoformat() does not accept a trailing "Z"
        text = str(value)
        dt = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _short(value: Any, width: int = 60) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


@app.command("ls")
def ls(
    limit: int = typer.Option(20, "--limit", "-n", help="Max runs to show (0 = all)."),
    name: str | None = typer.Option(None, "--name", help="Only runs of this agent name."),
) -> None:
    """List recorded runs, newest first."""
    console = _console()
    store = Store()
    rows = store.list_runs(limit=limit or None, name=name)
    if not rows:
        console.print(f"No runs recorded yet in [bold]{store.root}[/].")
        return
    table = Table(title=f"crashlab runs ({store.root})", title_justify="left")
    table.add_column("id", no_wrap=True, style="cyan")
    table.add_column("name")
    table.add_column("status")
    table.add_column("outcome")
    table.add_column("steps", justify="right")
    table.add_column("tokens", justify="right")
    table.add_column("started", no_wrap=True)
    for r in rows:
        table.add_row(
            r["run_id"],
            r["name"],
            Text(r["status"], style=_STATUS_STYLE.get(r["status"], "")),
            Text(r["outcome"], style=_OUTCOME_STYLE.get(r["outcome"], "")),
            str(r["steps"]),
            str(r["total_tokens"]),
            _local_time(r["started_at"]),
        )
    console.print(table)


def _part_text(part: dict[str, Any], width: int) -> str:
    t = part.get("type")
    if t == "text":
        return _short(part.get("content"), width)
    if t == "tool_call":
        return f"→ {part.get('name')}({_short(part.get('arguments'), 40)})"
    if t == "tool_call_response":
        return _short(part.get("response"), width)
    if t in ("image", "audio", "file"):
        return f"[{t}]"
    return f"[{t}]"


def _message_text(msg: Any, width: int) -> str:
    if not isinstance(msg, dict):
        return _short(msg, width)
    if "parts" not in msg:  # day-1 cassette shape
        return f"{msg.get('role', '?')}: {_short(msg.get('content'), width)}"
    body = " ".join(_part_text(p, width) for p in msg.get("parts") or [])
    return f"{msg.get('role', '?')}: {_short(body, width)}"


def _step_io(step: Step) -> tuple[str, str]:
    if step.operation == "chat":
        msgs = step.request.get("messages") or []
        inp = _message_text(msgs[-1], 50) if msgs else ""
        resp = step.response or {}
        calls = resp.get("tool_calls") or []
        if calls:
            out = "; ".join(f"→ {c.get('name')}({_short(c.get('arguments'), 40)})" for c in calls)
        else:
            out = ""
            for m in resp.get("messages") or []:
                if "parts" in m:
                    out = _short(
                        "".join(p.get("content") or "" for p in m["parts"] if p["type"] == "text"),
                        60,
                    )
                else:
                    out = _short(m.get("content"), 60)
                if out:
                    break
            reason = resp.get("finish_reason")
            if reason not in (None, "stop"):
                out += f" [{reason}]"
        if step.attributes.get("crashlab.stream"):
            out += " (stream" + (
                ")" if step.attributes.get("crashlab.stream.complete", True) else ", incomplete)"
            )
        return inp, out
    if step.operation in ("execute_tool", "invoke_agent"):
        return _short(step.args, 50), _short(step.result, 60) if step.response else ""
    return _short(step.request, 50), _short(step.response, 60)


def _resolve(store: Store, ref: str) -> Run:
    try:
        return store.load_run(ref)
    except RunNotFoundError as exc:
        _console().print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc


@app.command("show")
def show(
    run_ref: str = typer.Argument("last", help="Run id, unique id prefix, or 'last'."),
    as_json: bool = typer.Option(False, "--json", help="Print the run and steps as JSON."),
) -> None:
    """Show a run's metadata and step table."""
    store = Store()
    run = _resolve(store, run_ref)
    if as_json:
        typer.echo(run.model_dump_json(indent=2))
        return
    console = _console()
    u = run.usage
    meta = Table.grid(padding=(0, 2))
    meta.add_column(style="bold")
    meta.add_column()
    meta.add_row("run", run.run_id)
    meta.add_row("name", run.name)
    meta.add_row(
        "status",
        Text(run.status, style=_STATUS_STYLE.get(run.status, ""))
        + Text("  outcome: ")
        + Text(run.outcome, style=_OUTCOME_STYLE.get(run.outcome, "")),
    )
    if run.outcome_reason:
        meta.add_row("reason", run.outcome_reason)
    meta.add_row("entrypoint", run.entrypoint or "-")
    meta.add_row("input", _short(run.input, 100))
    meta.add_row("output", _short(run.output, 100))
    if run.error:
        meta.add_row("error", Text(f"{run.error.type}: {_short(run.error.message, 90)}", "red"))
    meta.add_row("started", f"{_local_time(run.started_at)} (local)")
    meta.add_row(
        "usage",
        f"{u.steps} steps · {u.llm_calls} llm · {u.tool_calls} tool · {u.errors} errors · "
        f"{u.input_tokens}+{u.output_tokens}={u.total_tokens} tokens"
        + (f" · {u.wall_time_ms / 1000:.2f}s" if u.wall_time_ms is not None else ""),
    )
    meta.add_row("code", run.code_version or "-")
    console.print(Panel(meta, title="crashlab run", title_align="left"))

    table = Table(show_lines=False)
    table.add_column("idx", justify="right", style="cyan")
    table.add_column("op")
    table.add_column("name")
    table.add_column("args / input", overflow="fold")
    table.add_column("result / output", overflow="fold")
    table.add_column("error", style="red", overflow="fold")
    for step in run.steps:
        inp, out = _step_io(step)
        err = f"{step.error.type}: {_short(step.error.message, 50)}" if step.error else ""
        table.add_row(str(step.idx), step.operation, step.name, inp, out, err)
    if run.steps:
        console.print(table)
    else:
        console.print("[dim]No steps recorded.[/]")


if __name__ == "__main__":  # pragma: no cover
    app()
