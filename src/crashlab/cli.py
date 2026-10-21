"""``crashlab`` command line: ``ls``, ``show``, ``replay``, ``check``, ``diagnose``,
``fix``, ``report``, ``fork``, ``diff``, ``export``, ``import``."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table as _RichTable
from rich.text import Text

from ._version import __version__
from .models import SEVERITY_RANK, CheckResult, Diagnosis, Finding, Run, Step
from .store import ENV_DIR, RunNotFoundError, Store


class Table(_RichTable):
    """Plain strings in cells are recorded data (tool output, LLM text, reasons), never
    markup: Rich would eat ``[budget]`` or ``[/]`` in them."""

    def add_row(self, *renderables, **kwargs):  # type: ignore[no-untyped-def,override]
        cells = [Text(r) if isinstance(r, str) else r for r in renderables]
        return super().add_row(*cells, **kwargs)


def _e(value: object) -> str:
    """Escape recorded data interpolated into a markup string."""
    return escape(str(value))


app = typer.Typer(
    name="crashlab",
    help="Crash-test your AI agents. Inspect, replay, check, fork and diff recorded agent runs.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)

_STATUS_STYLE = {"ok": "green", "error": "red", "running": "yellow"}
_OUTCOME_STYLE = {"pass": "green", "fail": "red", "unknown": "dim"}
_CHECK_STYLE = {"pass": "green", "fail": "red", "error": "red"}
_SEV_STYLE = {"info": "dim", "warn": "yellow", "error": "red", "critical": "bold red"}
_SEVERITIES = ("info", "warn", "error", "critical")


def _cost_text(value: float | None) -> str:
    if value is None:
        return "?"
    return "$0" if value == 0 else f"${value:.4f}" if value >= 0.0001 else f"${value:.6f}"


def _findings_table(findings: list[Finding], title: str = "runtime findings") -> Table:
    table = Table(title=title, title_justify="left", show_header=False, box=None)
    table.add_column("severity", no_wrap=True)
    table.add_column("kind", style="bold", no_wrap=True)
    table.add_column("steps", no_wrap=True, style="cyan")
    table.add_column("message", overflow="fold")
    for f in sorted(findings, key=lambda f: (-f.rank, f.step if f.step is not None else -1)):
        steps = (
            ",".join(str(i) for i in f.steps)
            if f.steps
            else ("-" if f.step is None else str(f.step))
        )
        msg = Text(f.message)
        if f.stopped_run:
            msg = Text("STOPPED THE RUN: ", style="bold red") + msg
        if f.evidence:
            q = f.evidence[0]
            quote = " ".join(q.quote.split())[:90]
            msg += Text(f"\n  step {q.step_idx} {q.field}: ", style="dim") + Text(
                repr(quote), style="dim"
            )
        table.add_row(Text(f.severity, style=_SEV_STYLE.get(f.severity, "")), f.kind, steps, msg)
    return table


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
    flagged: bool = typer.Option(
        False, "--flagged", "-f", help="Only runs with runtime-check findings."
    ),
    min_severity: str = typer.Option(
        "warn", "--min-severity", help="With --flagged: info, warn, error or critical."
    ),
) -> None:
    """List recorded runs, newest first."""
    console = _console()
    store = Store()
    if min_severity not in _SEVERITIES:
        console.print(f"[red]--min-severity must be one of {', '.join(_SEVERITIES)}[/]")
        raise typer.Exit(code=2)
    rows = store.list_runs(
        limit=limit or None, name=name, flagged=min_severity if flagged else None
    )
    if not rows:
        what = "flagged runs" if flagged else "runs recorded yet"
        console.print(f"No {what} in [bold]{_e(store.root)}[/].")
        return
    narrow = console.width < 130
    id_len = _unique_prefix_len(r["run_id"] for r in store.list_runs()) if narrow else 0
    table = Table(
        title=f"crashlab runs ({store.root})",
        title_justify="left",
        box=None if narrow else box.HEAVY_HEAD,
        pad_edge=not narrow,
        header_style="bold underline" if narrow else "table.header",
    )
    # id and the small columns never wrap; name wraps. On a narrow terminal the id is
    # shortened to the shortest prefix unique in the store (any command accepts it) and
    # the tokens column is dropped, so 80 columns stay readable.
    table.add_column("id", no_wrap=True, style="cyan")
    table.add_column("name", min_width=8)
    table.add_column("status", no_wrap=True)
    table.add_column("outcome", no_wrap=True)
    table.add_column("steps", justify="right", no_wrap=True)
    if not narrow:
        table.add_column("tokens", justify="right", no_wrap=True)
    table.add_column("cost", justify="right", no_wrap=True)
    table.add_column("findings", no_wrap=True)
    table.add_column("started", no_wrap=True)
    for r in rows:
        sev = r.get("max_severity")
        found = (
            Text(f"{r['flag_count']} {sev}", style=_SEV_STYLE.get(sev or "", ""))
            if (r.get("flag_count"))
            else Text("")
        )
        cells: list[Any] = [
            r["run_id"][:id_len] if narrow else r["run_id"],
            r["name"] + _relation_suffix(r),
            Text(r["status"], style=_STATUS_STYLE.get(r["status"], "")),
            Text(r["outcome"], style=_OUTCOME_STYLE.get(r["outcome"], "")),
            str(r["steps"]),
        ]
        if not narrow:
            cells.append(str(r["total_tokens"]))
        cells += [
            _cost_text(r.get("cost_usd")) if r.get("cost_usd") is not None else "",
            found,
            # MM-DD HH:MM[:SS], local time
            _local_time(r["started_at"])[5:16] if narrow else _local_time(r["started_at"])[5:],
        ]
        table.add_row(*cells)
    console.print(table)


def _finding_diff_json(fd: Any) -> dict[str, Any]:
    def side(fs: list[Finding]) -> list[dict[str, Any]]:
        return [{"severity": f.severity, "steps": f.steps, "message": f.message} for f in fs]

    return {
        "kind": fd.kind,
        "worst_a": fd.worst_a,
        "worst_b": fd.worst_b,
        "changed": fd.changed,
        "a": side(fd.a),
        "b": side(fd.b),
    }


def _unique_prefix_len(run_ids: Any, minimum: int = 10) -> int:
    """Shortest prefix length (at least ``minimum``) that tells all ``run_ids`` apart."""
    ids = sorted(set(run_ids))
    need = minimum
    for a, b in zip(ids, ids[1:], strict=False):
        common = 0
        while common < min(len(a), len(b)) and a[common] == b[common]:
            common += 1
        need = max(need, common + 1)
    return need


def _relation_suffix(row: dict[str, Any]) -> str:
    if row.get("replay_of"):
        return " (replay)"
    if row.get("parent_run_id"):  # index column for Run.forked_from
        return f" (fork@{row.get('fork_at')})"
    return ""


_SOURCE_MARK = {"cassette": " ↺", "patched": " ✎", "blocked": " ⊘"}


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
        _console().print(f"[red]{_e(exc)}[/]")
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
    if run.replay_of:
        meta.add_row("replay of", run.replay_of)
    if run.forked_from:
        meta.add_row("forked from", f"{run.forked_from} at step {run.fork_step}")
        for i, line in enumerate(run.patch.describe_lines() if run.patch else [], 1):
            meta.add_row("patch" if i == 1 else "", f"{i}. {line}")
    meta.add_row("entrypoint", run.entrypoint or "-")
    meta.add_row("input", _short(run.input, 100))
    meta.add_row("output", _short(run.output, 100))
    if run.error:
        meta.add_row("error", Text(f"{run.error.type}: {_short(run.error.message, 90)}", "red"))
    meta.add_row("started", f"{_local_time(run.started_at)} (local)")
    from .pricing import run_cost, step_cost

    cost = run_cost(run)
    meta.add_row(
        "usage",
        f"{u.steps} steps · {u.llm_calls} llm · {u.tool_calls} tool · {u.errors} errors · "
        f"{u.input_tokens}+{u.output_tokens}={u.total_tokens} tokens"
        + (f" · {u.wall_time_ms / 1000:.2f}s" if u.wall_time_ms is not None else "")
        + (f" · est. {_cost_text(cost)}" if u.llm_calls else ""),
    )
    meta.add_row("code", run.code_version or "-")
    console.print(Panel(meta, title="crashlab run", title_align="left"))
    if run.checks:
        console.print(_checks_table(run.checks))
    if run.findings:
        console.print(_findings_table(run.findings))
    if run.diagnosis is not None:
        console.print(_diagnosis_panel(run.diagnosis))
    marks: dict[int, Finding] = {}
    for f in run.findings:
        for i in f.steps or ([f.step] if f.step is not None else []):
            if i not in marks or f.rank > marks[i].rank:
                marks[i] = f

    table = Table(show_lines=False)
    table.add_column("idx", justify="right", style="cyan")
    table.add_column("op")
    table.add_column("name")
    table.add_column("args / input", overflow="fold")
    table.add_column("result / output", overflow="fold")
    table.add_column("error", style="red", overflow="fold")
    table.add_column("$", justify="right", no_wrap=True, style="dim")
    table.add_column("", no_wrap=True)
    for step in run.steps:
        inp, out = _step_io(step)
        err = f"{step.error.type}: {_short(step.error.message, 50)}" if step.error else ""
        op = step.operation + _SOURCE_MARK.get(step.source, "")
        c = step.attributes.get("crashlab.usage.cost_usd") if step.operation == "chat" else None
        c = step_cost(step) if c is None and step.operation == "chat" else c
        usd = _cost_text(c) if step.operation == "chat" else ""
        mark = marks.get(step.idx)
        flag = Text("⚑ " + mark.kind, style=_SEV_STYLE.get(mark.severity, "")) if mark else Text("")
        table.add_row(str(step.idx), op, step.name, inp, out, err, usd, flag)
    if run.steps:
        console.print(table)
    else:
        console.print("[dim]No steps recorded.[/]")
    if any(s.source != "live" for s in run.steps):
        console.print("[dim]↺ served from the cassette  ✎ patched by the fork  ⊘ blocked[/]")


def _checks_table(results: list[CheckResult]) -> Table:
    table = Table(title="checks", title_justify="left", show_header=False, box=None)
    table.add_column("status", no_wrap=True)
    table.add_column("check", style="bold", no_wrap=True)
    table.add_column("message", overflow="fold")
    for r in results:
        table.add_row(Text(r.status, style=_CHECK_STYLE.get(r.status, "")), r.name, r.message or "")
    return table


_LABEL_STYLE = {"NO_FAILURE": "green", "UNKNOWN": "yellow"}


def _diagnosis_panel(d: Diagnosis) -> Panel:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column(overflow="fold")
    label = Text(d.label.value, style=_LABEL_STYLE.get(d.label.value, "bold red"))
    head = (Text(f"step {d.failing_step} · ") if d.failing_step is not None else Text()) + label
    if d.step_window and d.step_window[0] != d.step_window[1]:
        head += Text(f"  (steps {d.step_window[0]}-{d.step_window[1]})", style="dim")
    grid.add_row("verdict", head)
    grid.add_row("confidence", f"{d.confidence:.2f}")
    if d.root_cause:
        grid.add_row("root cause", d.root_cause)
    for i, ev in enumerate(d.evidence):
        quote = " ".join(ev.quote.split())
        quote = quote if len(quote) <= 160 else quote[:159] + "…"
        grid.add_row(
            "evidence" if i == 0 else "",
            Text(f"step {ev.step_idx} {ev.field}: ", style="dim") + Text(repr(quote)),
        )
    if d.secondary:
        grid.add_row("also", ", ".join(s.value for s in d.secondary))
    for i, note in enumerate(d.notes):
        grid.add_row("notes" if i == 0 else "", Text(note, style="yellow"))
    m = d.method or {}
    bits = [d.model or "?"]
    if m.get("input_tokens") is not None:
        bits.append(f"{m.get('input_tokens', 0)}+{m.get('output_tokens', 0)} tokens")
    if m.get("cost_usd") is not None:
        bits.append(_cost_text(m.get("cost_usd")))
    if m.get("seconds") is not None:
        bits.append(f"{m['seconds']:.1f}s")
    if m.get("cached"):
        bits.append("cached")
    grid.add_row("diagnoser", Text(" · ".join(bits), style="dim"))
    return Panel(grid, title="diagnosis", title_align="left")


def _fixes_panel(d: Diagnosis, run: Run) -> Panel:
    from rich.console import Group
    from rich.syntax import Syntax

    from .patch import PatchError, PatchSet

    parts: list[Any] = []
    for f in d.fixes:
        head = Text(f"{f.fix_id}  ", style="bold") + Text(f.kind, style="cyan")
        head += Text(f" → {f.target}")
        if f.test_only:
            head += Text("  (test only: a counterfactual, not a fix)", style="yellow")
        parts.append(head)
        parts.append(Text("  " + f.after))
        if f.rationale:
            parts.append(Text("  why: ", style="dim") + Text(f.rationale))
        if f.expected_effect:
            parts.append(Text("  expect: ", style="dim") + Text(f.expected_effect))
        for n in f.notes:
            parts.append(Text("  note: " + n, style="yellow"))
        if f.as_patch is not None:
            at = f.tried["fork_step"] if f.tried else (f.as_patch.target_step() or 0)
            try:
                diff = PatchSet(patches=[f.as_patch]).render_diff(run, fork_step=at)
                body = "\n".join(diff.splitlines()[1:]).strip()
                if body:
                    parts.append(Syntax(body, "diff", theme="ansi_dark", word_wrap=True))
            except PatchError as exc:
                parts.append(Text(f"  (diff unavailable: {exc})", style="dim"))
        if f.file:
            parts.append(Text("  file: ", style="dim") + Text(f.file))
        if f.tried:
            t = f.tried
            style = _OUTCOME_STYLE.get(t.get("outcome", ""), "")
            line = Text("  tried: fork ") + Text(str(t.get("run_id"))[:10], style="cyan")
            line += Text(f" from step {t.get('fork_step')} → outcome ")
            line += Text(str(t.get("outcome")), style=style)
            if t.get("failed_checks"):
                line += Text(f" (failed: {', '.join(t['failed_checks'])})", style="red")
            if t.get("error"):
                line += Text(f"  error: {t['error']}", style="red")
            parts.append(line)
        parts.append(Text(""))
    for n in d.fix_notes:
        parts.append(Text(n, style="yellow"))
    if not parts:
        parts.append(Text("no fix proposed", style="dim"))
    m = d.fix_method or {}
    if m.get("model"):
        bits = [str(m["model"])]
        if m.get("input_tokens") is not None:
            bits.append(f"{m.get('input_tokens', 0)}+{m.get('output_tokens', 0)} tokens")
        if m.get("cost_usd") is not None:
            bits.append(_cost_text(m["cost_usd"]))
        bits.append(f"{m.get('kept', 0)}/{m.get('proposed', 0)} proposals kept")
        if m.get("cached"):
            bits.append("cached")
        parts.append(Text("fix model: " + " · ".join(bits), style="dim"))
    return Panel(Group(*parts), title="proposed fixes", title_align="left")


def _next_hints(run: Run) -> list[str]:
    rid = run.run_id[:10]
    d = run.diagnosis
    hints: list[str] = []
    if d is None:
        hints.append(f"crashlab diagnose {rid} --model MODEL [--base-url URL --api-key-env VAR]")
        hints.append(f"crashlab diagnose {rid} --heuristic        # no LLM")
    elif d.is_failure and not d.fixes:
        hints.append(f"crashlab fix {rid} --model MODEL           # propose patches")
    for f in (d.fixes if d else [])[:1]:
        if f.file:
            at = f.as_patch.target_step() if f.as_patch is not None else None
            hints.append(f"crashlab fork {rid} --at {at or 0} --patch {f.file}")
        if not f.tried:
            hints.append(f"crashlab fix {rid} --model MODEL --try     # fork once with fix-1")
    if run.entrypoint:
        hints.append(f"crashlab replay {rid}                     # strict offline replay")
    return hints


def _print_report(console: Console, run: Run) -> None:
    head = Table.grid(padding=(0, 2))
    head.add_column(style="bold")
    head.add_column(overflow="fold")
    head.add_row("run", f"{run.run_id}  ({run.name})")
    head.add_row(
        "outcome",
        Text(run.outcome, style=_OUTCOME_STYLE.get(run.outcome, ""))
        + Text(f"  status: {run.status}"),
    )
    if run.outcome_reason:
        head.add_row("reason", run.outcome_reason)
    if run.forked_from:
        head.add_row("forked from", f"{run.forked_from} at step {run.fork_step}")
    head.add_row("output", _short(run.output, 160))
    u = run.usage
    head.add_row(
        "usage",
        f"{u.steps} steps · {u.llm_calls} llm · {u.tool_calls} tool · {u.total_tokens} tokens",
    )
    console.print(Panel(head, title="crashlab report", title_align="left"))
    if run.checks:
        console.print(_checks_table(run.checks))
    if run.findings:
        console.print(_findings_table(run.findings))
    if run.diagnosis is not None:
        console.print(_diagnosis_panel(run.diagnosis))
        if run.diagnosis.is_failure or run.diagnosis.fixes or run.diagnosis.fix_notes:
            console.print(_fixes_panel(run.diagnosis, run))
    hints = _next_hints(run)
    if hints:
        console.print(Text("next:", style="bold"))
        for h in hints:
            console.print(Text("  " + h, style="dim"))


@app.command("report")
def report_cmd(
    run_ref: str = typer.Argument("last", help="Run id, unique id prefix, or 'last'."),
) -> None:
    """One-screen report of a run: outcome, checks and findings, the stored diagnosis
    (label, step, root cause, quotes), proposed fixes with diffs, and what to run next."""
    _print_report(_console(), _resolve(Store(), run_ref))


def _key_from(console: Console, api_key_env: str | None) -> str | None:
    if not api_key_env:
        return None
    key = os.environ.get(api_key_env)
    if not key:
        console.print(f"[red]${_e(api_key_env)} is not set[/]")
        raise typer.Exit(code=2)
    return key


def _run_fixes(
    console,
    store,
    run,
    *,
    model,
    base_url,
    api_key,
    cache,
    try_it,
    max_fixes,
    hide_input,
    max_chars,
):
    from .diagnose import DiagnoseError, propose_fixes, try_fix

    try:
        fixes = propose_fixes(
            run,
            model,
            base_url=base_url,
            api_key=api_key,
            cache=cache,
            store=store,
            max_fixes=max_fixes,
            hide_input=hide_input,
            max_field=max_chars,
        )
    except DiagnoseError as exc:
        console.print(f"[red]{_e(run.run_id)}: {_e(exc)}[/]")
        raise typer.Exit(code=1) from exc
    if try_it and fixes:
        top = next((f for f in fixes if not f.test_only), fixes[0])
        console.print(f"[dim]trying {_e(top.fix_id)}: forking {_e(run.run_id[:10])} live ...[/]")
        try:
            try_fix(run, top, store=store)
        except Exception as exc:  # the experiment failed to start: report, keep the fixes
            top.tried = {
                "run_id": None,
                "fork_step": None,
                "outcome": "error",
                "failed_checks": [],
                "error": f"{type(exc).__name__}: {exc}"[:300],
            }
        store.save_diagnosis(run, run.diagnosis, (run.diagnosis.method or {}).get("cache_key"))
    return fixes


@app.command("diagnose")
def diagnose_cmd(
    run_refs: list[str] = typer.Argument(  # noqa: B008
        None, help="Run ids / prefixes / 'last' (default: last)."
    ),
    model: str | None = typer.Option(
        None, "--model", "-m", help="Diagnoser model (default: $CRASHLAB_DIAGNOSER_MODEL)."
    ),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        help="OpenAI-compatible endpoint for the diagnoser (Groq, Gemini, Ollama, vLLM ...).",
    ),
    api_key_env: str | None = typer.Option(
        None,
        "--api-key-env",
        help="Name of the env var that holds the diagnoser's API key (never the key itself).",
    ),
    heuristic: bool = typer.Option(
        False, "--heuristic", help="No LLM call: rule-based diagnosis from checks and findings."
    ),
    no_cache: bool = typer.Option(False, "--no-cache", help="Ask again even if cached."),
    save: bool = typer.Option(True, "--save/--no-save", help="Store the diagnosis with the run."),
    hide_input: list[str] = typer.Option(  # noqa: B008
        [], "--hide-input", help="Run input key to hide from the diagnoser (repeatable)."
    ),
    max_chars: int = typer.Option(600, "--max-chars", help="Truncate trace fields at N chars."),
    show_prompt: bool = typer.Option(
        False, "--show-prompt", help="Print the compacted trace the diagnoser reads; no call."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the diagnosis as JSON."),
    fix: bool = typer.Option(
        False, "--fix", help="Also propose fixes (a second LLM call) and print the report."
    ),
    try_it: bool = typer.Option(False, "--try", help="With --fix: fork once with fix-1."),
) -> None:
    """Diagnose stored runs: failure label, culprit step, root cause and verbatim evidence
    from one LLM call, validated against the trace (unsupported answers become UNKNOWN)."""
    from .diagnose import DiagnoseError, candidates, compact, diagnose

    console = _console()
    store = Store()
    api_key = None
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            console.print(f"[red]${_e(api_key_env)} is not set[/]")
            raise typer.Exit(code=2)
    out: list[dict[str, Any]] = []
    for ref in run_refs or ["last"]:
        run = _resolve(store, ref)
        if show_prompt:
            trace = compact(
                run, candidates=candidates(run), max_field=max_chars, hide_input=hide_input
            )
            typer.echo(trace)
            continue
        try:
            d = diagnose(
                run,
                model,
                base_url=base_url,
                api_key=api_key,
                heuristic=heuristic,
                cache=not no_cache,
                save=save,
                store=store,
                hide_input=hide_input,
                max_field=max_chars,
            )
        except DiagnoseError as exc:
            console.print(f"[red]{_e(run.run_id)}: {_e(exc)}[/]")
            raise typer.Exit(code=1) from exc
        if fix:
            _run_fixes(
                console,
                store,
                run,
                model=model,
                base_url=base_url,
                api_key=api_key,
                cache=not no_cache,
                try_it=try_it,
                max_fixes=3,
                hide_input=hide_input,
                max_chars=max_chars,
            )
            d = run.diagnosis or d
        if as_json:
            out.append(d.model_dump(mode="json"))
        elif fix:
            _print_report(console, run)
        else:
            console.print(f"[bold]{_e(run.run_id)}[/] {_e(run.name)} · outcome {_e(run.outcome)}")
            console.print(_diagnosis_panel(d))
    if as_json:
        typer.echo(json.dumps(out[0] if len(out) == 1 else out, indent=2, ensure_ascii=False))


@app.command("fix")
def fix_cmd(
    run_ref: str = typer.Argument("last", help="Run id, unique id prefix, or 'last'."),
    model: str | None = typer.Option(
        None, "--model", "-m", help="Fix model (default: $CRASHLAB_DIAGNOSER_MODEL)."
    ),
    base_url: str | None = typer.Option(None, "--base-url", help="OpenAI-compatible endpoint."),
    api_key_env: str | None = typer.Option(
        None, "--api-key-env", help="Name of the env var holding the API key."
    ),
    try_it: bool = typer.Option(
        False, "--try", help="Fork once with the top fix (live, the agent's own client)."
    ),
    max_fixes: int = typer.Option(3, "--max", help="Keep at most N fixes (1-3)."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Ask again even if cached."),
    hide_input: list[str] = typer.Option(  # noqa: B008
        [], "--hide-input", help="Run input key to hide from the models (repeatable)."
    ),
    max_chars: int = typer.Option(600, "--max-chars", help="Truncate trace fields at N chars."),
    as_json: bool = typer.Option(False, "--json", help="Print the fixes as JSON."),
) -> None:
    """Propose 1-3 ranked typed patches for a diagnosed run (diagnosing it first if
    needed), validated against the run and saved as patch files for `crashlab fork --patch`."""
    from .diagnose import DiagnoseError, diagnose

    console = _console()
    store = Store()
    api_key = _key_from(console, api_key_env)
    run = _resolve(store, run_ref)
    if run.diagnosis is None or no_cache:
        try:
            diagnose(
                run,
                model,
                base_url=base_url,
                api_key=api_key,
                cache=not no_cache,
                store=store,
                hide_input=hide_input,
                max_field=max_chars,
            )
        except DiagnoseError as exc:
            console.print(f"[red]{_e(run.run_id)}: {_e(exc)}[/]")
            raise typer.Exit(code=1) from exc
    fixes = _run_fixes(
        console,
        store,
        run,
        model=model,
        base_url=base_url,
        api_key=api_key,
        cache=not no_cache,
        try_it=try_it,
        max_fixes=max(1, min(max_fixes, 3)),
        hide_input=hide_input,
        max_chars=max_chars,
    )
    if as_json:
        typer.echo(
            json.dumps([f.model_dump(mode="json") for f in fixes], indent=2, ensure_ascii=False)
        )
        return
    _print_report(console, run)


@app.command("replay")
def replay_cmd(
    run_ref: str = typer.Argument("last", help="Run id, unique id prefix, or 'last'."),
    allow_network: bool = typer.Option(
        False, "--allow-network", help="Do not block network connections during the replay."
    ),
) -> None:
    """Re-run a recorded run offline: every LLM and tool call is served from its cassette.

    Exits 1 with a diff when the run does not reproduce (DivergenceError).
    """
    from .replay import DivergenceError, ReplayError, replay

    console = _console()
    store = Store()
    source = _resolve(store, run_ref)
    console.print(
        f"Replaying [cyan]{source.run_id}[/] ({_e(source.name)}, {len(source.steps)} steps)"
        + (" offline" if not allow_network else "")
        + " ..."
    )
    try:
        rep = replay(source, offline=not allow_network, store_dir=store.root)
    except DivergenceError as err:
        console.print(Panel(Text(str(err)), title="[red]replay diverged[/]", title_align="left"))
        if err.replay_run_id:
            console.print(
                f"Diverged replay stored as [cyan]{err.replay_run_id}[/] "
                f"(crashlab show {err.replay_run_id})"
            )
        raise typer.Exit(code=1) from err
    except ReplayError as err:
        console.print(f"[red]cannot replay:[/] {_e(err)}")
        raise typer.Exit(code=2) from err
    served = sum(1 for s in rep.steps if s.source == "cassette")
    console.print(
        f"[green]reproduced[/]: {served}/{len(rep.steps)} steps served from the cassette, "
        + (
            f"same error as recorded ({_e(rep.error.type)})."
            if rep.error
            else "identical requests and output."
        )
    )
    console.print(
        f"replay run [cyan]{rep.run_id}[/] (replay of {source.run_id})  status: "
        f"{rep.status}  outcome: " + f"[{_OUTCOME_STYLE.get(rep.outcome, '')}]{rep.outcome}[/]"
    )
    if rep.checks:
        console.print(_checks_table(rep.checks))


@app.command("check")
def check_cmd(
    run_refs: list[str] = typer.Argument(  # noqa: B008
        None, help="Run ids / prefixes / 'last' (default: last)."
    ),
    all_runs: bool = typer.Option(False, "--all", help="Check every stored run (newest first)."),
    name: str | None = typer.Option(None, "--name", help="With --all: only this agent."),
    save: bool = typer.Option(True, "--save/--no-save", help="Store outcome and findings."),
    findings_only: bool = typer.Option(
        False, "--findings-only", help="Only the runtime checks (skip @crashlab.check oracles)."
    ),
    oracles_only: bool = typer.Option(
        False, "--oracles-only", help="Only the @crashlab.check oracles."
    ),
    max_steps: int | None = typer.Option(None, "--max-steps", help="Budget: steps."),
    max_tokens: int | None = typer.Option(None, "--max-tokens", help="Budget: total tokens."),
    max_cost: float | None = typer.Option(None, "--max-cost", help="Budget: USD."),
    max_seconds: float | None = typer.Option(None, "--max-seconds", help="Budget: wall time."),
    baseline: int | None = typer.Option(
        None, "--baseline", help="Compare tool results with this many other runs (drift)."
    ),
    fail_on: str | None = typer.Option(
        None, "--fail-on", help="Exit 1 if a finding is at least: info, warn, error, critical."
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable results."),
) -> None:
    """Check stored runs: the @crashlab.check outcome oracles (pass / fail) and the
    built-in runtime checks (findings). Exit 1 if an oracle fails (or --fail-on hits)."""
    from .checks import evaluate, replace_findings, resolve_config

    console = _console()
    store = Store()
    if fail_on is not None and fail_on not in _SEVERITIES:
        console.print(f"[red]--fail-on must be one of {', '.join(_SEVERITIES)}[/]")
        raise typer.Exit(code=2)
    if all_runs:
        refs = [r["run_id"] for r in store.list_runs(limit=None, name=name)]
    else:
        refs = list(run_refs or ["last"])
    overrides: dict[str, Any] = {}
    budget = {
        "max_steps": max_steps,
        "max_tokens": max_tokens,
        "max_cost_usd": max_cost,
        "max_seconds": max_seconds,
    }
    if any(v is not None for v in budget.values()):
        overrides["budget"] = {k: v for k, v in budget.items() if v is not None}
    if baseline is not None:
        overrides["suspicious_tool_result"] = {"baseline_runs": baseline}
    many = len(refs) > 1
    failed = False
    report = []
    summary = Table(title="crashlab check", title_justify="left") if many else None
    if summary is not None:
        for col in ("run", "name", "outcome", "findings"):
            if col == "run":
                summary.add_column(col, no_wrap=True, style="cyan")
            else:
                summary.add_column(col, overflow="fold")
    for ref in refs:
        run = _resolve(store, ref)
        results = None if findings_only else _run_oracles(console, run, quiet=many or as_json)
        found = None
        if not oracles_only:
            cfg = resolve_config(overrides or None, base=run.check_config)
            found = evaluate(run, cfg, store=store)
            replace_findings(run, found)
            run.check_config = cfg.model_dump(mode="json")
            from .pricing import run_cost

            run.usage.cost_usd = run_cost(run)
        if results is not None and run.outcome == "fail":
            failed = True
        if found and fail_on and any(f.rank >= SEVERITY_RANK[fail_on] for f in found):
            failed = True
        if save:
            store.update_run(run)
        report.append(
            {
                "run_id": run.run_id,
                "name": run.name,
                "outcome": run.outcome,
                "outcome_reason": run.outcome_reason,
                "checks": [c.model_dump() for c in run.checks],
                "findings": [f.model_dump(mode="json") for f in (found or [])],
                "cost_usd": run.usage.cost_usd,
            }
        )
        if as_json:
            continue
        if summary is not None:
            kinds = ", ".join(
                f"{f.kind}:{f.severity}" for f in sorted(found or [], key=lambda f: -f.rank)
            )
            summary.add_row(
                run.run_id,
                run.name,
                Text(run.outcome, style=_OUTCOME_STYLE.get(run.outcome, "")),
                kinds or "-",
            )
            continue
        if results:
            console.print(_checks_table(results))
            console.print(
                f"outcome: [{_OUTCOME_STYLE.get(run.outcome, '')}]{run.outcome}[/]"
                + (f"  ({run.outcome_reason})" if run.outcome_reason else "")
            )
        elif results is not None:
            console.print(
                f"No @crashlab.check oracles registered for runs named {_e(repr(run.name))}."
            )
        if found is not None:
            if found:
                console.print(_findings_table(found))
            else:
                console.print("runtime checks: no findings")
    if as_json:
        typer.echo(json.dumps(report if many else report[0], indent=2, default=str))
    elif summary is not None:
        console.print(summary)
    if failed:
        raise typer.Exit(code=1)


def _run_oracles(console: Console, run: Run, *, quiet: bool) -> list[CheckResult]:
    from .replay.divergence import ReplayError
    from .replay.engine import resolve_entrypoint
    from .verify.oracles import apply_outcome, checks_for, run_checks

    if run.entrypoint:  # importing the agent's module registers its checks
        try:
            resolve_entrypoint(run.entrypoint)
        except ReplayError as err:
            if not quiet:
                console.print(f"[yellow]could not import {_e(run.entrypoint)}:[/] {_e(err)}")
    specs = checks_for(run.name)
    if not specs:
        return []
    results = run_checks(run, specs)
    apply_outcome(run, results)
    return results


# --- fork ------------------------------------------------------------------------------


def _json_or_text(value: str) -> Any:
    try:
        return json.loads(value)
    except ValueError:
        return value


def _flag_patches(
    *,
    system_append: str | None,
    system_prepend: str | None,
    system_replace: str | None,
    system_insert_after: tuple[str, str] | None,
    system_sub: tuple[str, str] | None,
    system_delete_line: str | None,
    model: str | None,
    params: list[str],
    tool_description: tuple[str, str] | None,
    tool_result: tuple[int, str] | None,
    message: tuple[int, int, str] | None,
    inject_user: tuple[int, str] | None,
) -> list[Any]:
    from . import patch as P

    out: list[Any] = []
    if system_replace is not None:
        out.append(P.SystemPromptPatch(op="replace", text=system_replace))
    if system_insert_after and system_insert_after[0] is not None:
        anchor, text = system_insert_after
        out.append(P.SystemPromptPatch(op="insert_line", after=anchor, text=text))
    if system_sub and system_sub[0] is not None:
        out.append(P.SystemPromptPatch(op="substitute", old=system_sub[0], text=system_sub[1]))
    if system_delete_line is not None:
        out.append(P.SystemPromptPatch(op="delete_line", match=system_delete_line))
    if system_prepend is not None:
        out.append(P.SystemPromptPatch(op="prepend", text=system_prepend))
    if system_append is not None:
        out.append(P.SystemPromptPatch(op="append", text=system_append))
    if model is not None or params:
        kv: dict[str, Any] = {}
        for item in params:
            key, sep, value = item.partition("=")
            if not sep or not key.strip():
                raise typer.BadParameter(f"--param expects key=value, not {item!r}")
            kv[key.strip()] = _json_or_text(value)
        out.append(P.ModelParamPatch(model=model, params=kv))
    if tool_description and tool_description[0] is not None:
        out.append(
            P.ToolDescriptionPatch(tool=tool_description[0], description=tool_description[1])
        )
    if tool_result and tool_result[0] is not None:
        out.append(P.ToolResultPatch(step=tool_result[0], result=_json_or_text(tool_result[1])))
    if message and message[0] is not None:
        out.append(P.MessagePatch(step=message[0], index=message[1], content=message[2]))
    if inject_user and inject_user[0] is not None:
        out.append(P.InjectMessagePatch(step=inject_user[0], role="user", content=inject_user[1]))
    return out


@app.command("fork")
def fork_cmd(
    run_ref: str = typer.Argument("last", help="Run id, unique id prefix, or 'last'."),
    at: int = typer.Option(
        ...,
        "--at",
        help="Fork step N: steps < N replay from the cassette, LLM calls from N go "
        "live with the patch applied.",
    ),
    patch_files: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--patch",
        "-p",
        help="Patch file (.json / .yaml): one patch, a list, or {patches: [...]}. Repeatable.",
    ),
    system_append: str | None = typer.Option(
        None, "--system-append", help="Append to the system prompt."
    ),
    system_prepend: str | None = typer.Option(
        None, "--system-prepend", help="Prepend to the system prompt."
    ),
    system_replace: str | None = typer.Option(
        None, "--system-replace", help="Replace the system prompt."
    ),
    system_insert_after: tuple[str, str] = typer.Option(
        (None, None),
        "--system-insert-after",
        metavar="ANCHOR TEXT",
        help="Insert TEXT as a new line after the system-prompt line containing ANCHOR.",
    ),
    system_sub: tuple[str, str] = typer.Option(
        (None, None),
        "--system-sub",
        metavar="OLD NEW",
        help="Replace OLD with NEW in the system prompt.",
    ),
    system_delete_line: str | None = typer.Option(
        None,
        "--system-delete-line",
        metavar="MATCH",
        help="Delete the system-prompt line containing MATCH.",
    ),
    model: str | None = typer.Option(None, "--model", help="Use this model from step N on."),
    params: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--param",
        metavar="KEY=VALUE",
        help="Request parameter (JSON value), e.g. temperature=0.7. Repeatable.",
    ),
    tool_description: tuple[str, str] = typer.Option(
        (None, None),
        "--tool-description",
        metavar="TOOL TEXT",
        help="Replace a tool's description.",
    ),
    tool_result: tuple[int, str] = typer.Option(
        (None, None),
        "--tool-result",
        metavar="STEP JSON",
        help="Serve this result at tool step STEP instead of running the tool.",
    ),
    message: tuple[int, int, str] = typer.Option(
        (None, None, None),
        "--message",
        metavar="STEP INDEX TEXT",
        help="Replace message INDEX of the request at chat step STEP.",
    ),
    inject_user: tuple[int, str] = typer.Option(
        (None, None),
        "--inject-user",
        metavar="STEP TEXT",
        help="Append a user message to the request at chat step STEP.",
    ),
    tools: str = typer.Option(
        "cassette_if_match",
        "--tools",
        help="cassette_if_match (recorded result for identical calls, else live) or live.",
    ),
    allow_live_side_effects: bool = typer.Option(
        False,
        "--allow-live-side-effects",
        help="Run side_effects=True tools live after the fork step (sandboxes only).",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the patch diff, run nothing."),
    save_patch: str | None = typer.Option(
        None, "--save-patch", metavar="FILE", help="Write the combined patch as JSON."
    ),
) -> None:
    """Fork a run at step N with a patch and continue it live ("what if?").

    Steps before N are served from the cassette exactly; from N on the agent's LLM calls
    go live with the patch applied (keys come from the environment as usual). The fork is
    stored as a new run; compare it with [bold]crashlab diff PARENT FORK[/].
    """
    from . import patch as P
    from .replay import DivergenceError, ReplayError, fork

    console = _console()
    store = Store()
    source = _resolve(store, run_ref)
    if tools not in ("cassette_if_match", "live"):
        raise typer.BadParameter("--tools must be cassette_if_match or live")
    try:
        patches = P.PatchSet()
        for path in patch_files or []:
            patches = patches + P.load(path)
        patches = patches + _flag_patches(
            system_append=system_append,
            system_prepend=system_prepend,
            system_replace=system_replace,
            system_insert_after=system_insert_after,
            system_sub=system_sub,
            system_delete_line=system_delete_line,
            model=model,
            params=params or [],
            tool_description=tool_description,
            tool_result=tool_result,
            message=message,
            inject_user=inject_user,
        )
        n = len(source.steps)
        fork_step = at + n if at < 0 else at
        rendered = patches.render_diff(source, fork_step=fork_step)
    except (P.PatchError, ValueError, OSError) as err:
        console.print(f"[red]bad patch:[/] {_e(err)}")
        raise typer.Exit(code=2) from err
    if save_patch:
        Path(save_patch).write_text(patches.to_json(indent=2) + "\n", encoding="utf-8")
        console.print(f"patch written to {_e(save_patch)}")
    title = f"patch for {source.run_id} at step {fork_step}"
    body = rendered if patches else f"no patch: the fork re-runs live from step {fork_step}"
    console.print(Panel(Text(body), title=title, title_align="left"))
    if dry_run:
        return
    prefix = f"steps 0..{fork_step - 1} from the cassette, " if fork_step else ""
    console.print(
        f"Forking [cyan]{source.run_id}[/] ({_e(source.name)}) at step {fork_step}: "
        f"{prefix}live from step {fork_step} ..."
    )
    try:
        branch = fork(
            source,
            at_step=at,
            patch=patches,
            tools=tools,  # type: ignore[arg-type]
            allow_live_side_effects=allow_live_side_effects,
            store_dir=store.root,
        )
    except DivergenceError as err:
        console.print(
            Panel(Text(str(err)), title="[red]fork prefix diverged[/]", title_align="left")
        )
        raise typer.Exit(code=1) from err
    except (ReplayError, P.PatchError) as err:
        console.print(f"[red]cannot fork:[/] {_e(err)}")
        raise typer.Exit(code=2) from err
    if branch.error:
        console.print(
            f"[red]the fork raised[/] {_e(branch.error.type)}: "
            f"{_e(_short(branch.error.message, 160))}"
        )
    notes = [f for f in branch.findings if f.kind == "fork"]
    for note in notes:
        console.print(f"[yellow]note[/] {_e(note.message)}")
    found = [f for f in branch.findings if f.kind != "fork" and f.rank >= SEVERITY_RANK["warn"]]
    if found:
        console.print(_findings_table(found, "runtime findings in the fork"))
    _print_outcomes(console, source, branch)
    console.print(
        f"fork run [cyan]{branch.run_id}[/] (fork of {source.run_id} at step {fork_step})  "
        f"compare: crashlab diff {source.run_id} {branch.run_id}"
    )


def _outcome_text(run: Run) -> Text:
    return Text(run.outcome, style=_OUTCOME_STYLE.get(run.outcome, ""))


def _print_outcomes(console: Console, a: Run, b: Run) -> None:
    from .compare import diff

    console.print(
        Text("outcome: ")
        + _outcome_text(a)
        + Text(" -> ")
        + _outcome_text(b)
        + Text(f"   ({b.outcome_reason})" if b.outcome_reason else "")
    )
    rd = diff(a, b)
    if rd.checks:
        console.print(_checks_diff_table(rd))


# --- diff ------------------------------------------------------------------------------

_STEP_STYLE = {
    "same": "green",
    "response": "yellow",
    "request": "yellow",
    "different": "red",
    "only_a": "magenta",
    "only_b": "magenta",
}
_STEP_LABEL = {
    "same": "same",
    "response": "≠ response",
    "request": "≠ request",
    "different": "≠ call",
    "only_a": "only A",
    "only_b": "only B",
}


def _check_cell(result: CheckResult | None) -> Text:
    if result is None:
        return Text("-", style="dim")
    msg = f" {_short(result.message, 70)}" if result.message and result.status != "pass" else ""
    return Text(result.status, style=_CHECK_STYLE.get(result.status, "")) + Text(msg)


def _checks_diff_table(rd: Any) -> Table:
    table = Table(title="checks", title_justify="left")
    table.add_column("check", style="bold", no_wrap=True)
    table.add_column("A", overflow="fold")
    table.add_column("B", overflow="fold")
    for c in rd.checks:
        table.add_row(c.name + (" *" if c.changed else ""), _check_cell(c.a), _check_cell(c.b))
    return table


def _step_cell(step: Step | None) -> str:
    if step is None:
        return ""
    inp, out = _step_io(step)
    mark = _SOURCE_MARK.get(step.source, "")
    tool_io = f"{inp} → {out}" if out else inp
    body = out if step.operation == "chat" else tool_io
    if step.error:
        body += f" [{step.error.type.rsplit('.', 1)[-1]}]"
    return f"{step.operation}{mark} {step.name}: {_short(body, 70)}"


def _usage_rows(a: Run, b: Run) -> list[tuple[str, str, str, str]]:
    def delta(x: float | None, y: float | None, fmt: str = "{:+g}") -> str:
        if x is None or y is None or x == y:
            return ""
        return fmt.format(y - x)

    ua, ub = a.usage, b.usage
    rows = [
        ("steps", str(ua.steps), str(ub.steps), delta(ua.steps, ub.steps)),
        ("llm calls", str(ua.llm_calls), str(ub.llm_calls), delta(ua.llm_calls, ub.llm_calls)),
        ("tool calls", str(ua.tool_calls), str(ub.tool_calls), delta(ua.tool_calls, ub.tool_calls)),
        ("errors", str(ua.errors), str(ub.errors), delta(ua.errors, ub.errors)),
        (
            "tokens in",
            str(ua.input_tokens),
            str(ub.input_tokens),
            delta(ua.input_tokens, ub.input_tokens),
        ),
        (
            "tokens out",
            str(ua.output_tokens),
            str(ub.output_tokens),
            delta(ua.output_tokens, ub.output_tokens),
        ),
        (
            "tokens total",
            str(ua.total_tokens),
            str(ub.total_tokens),
            delta(ua.total_tokens, ub.total_tokens),
        ),
    ]
    from .pricing import run_cost

    ca, cb = run_cost(a), run_cost(b)
    if ua.llm_calls or ub.llm_calls:
        rows.append(("est. cost", _cost_text(ca), _cost_text(cb), delta(ca, cb, "{:+.6f}")))
    if ua.wall_time_ms is not None or ub.wall_time_ms is not None:
        wa, wb = (ua.wall_time_ms or 0) / 1000, (ub.wall_time_ms or 0) / 1000
        rows.append(("wall time", f"{wa:.2f}s", f"{wb:.2f}s", f"{wb - wa:+.2f}s"))
    return rows


@app.command("diff")
def diff_cmd(
    run_a: str = typer.Argument(..., help="First run (id, prefix or 'last')."),
    run_b: str | None = typer.Argument(
        None, help="Second run. Default: if A is a fork or replay, compare it with its parent."
    ),
    full: bool = typer.Option(False, "--full", help="Show the diff of every differing step."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable summary."),
) -> None:
    """Compare two runs: outcome and checks, usage side by side, step-by-step divergence."""
    from .compare import diff

    console = _console()
    store = Store()
    a = _resolve(store, run_a)
    if run_b is None:
        parent = a.forked_from or a.replay_of
        if parent is None:
            console.print("[red]give two runs[/] (A is not a fork or replay of another run)")
            raise typer.Exit(code=2)
        a, b = _resolve(store, parent), a
    else:
        b = _resolve(store, run_b)
    rd = diff(a, b)
    if as_json:
        typer.echo(
            json.dumps(
                {
                    "a": a.run_id,
                    "b": b.run_id,
                    "relation": rd.relation,
                    "outcome": [a.outcome, b.outcome],
                    "status": [a.status, b.status],
                    "first_divergence": rd.first_divergence,
                    "summary": rd.summary(),
                    "checks": {
                        c.name: [c.a.status if c.a else None, c.b.status if c.b else None]
                        for c in rd.checks
                    },
                    "steps": [
                        {"idx": s.idx, "status": s.status, "fields": s.fields} for s in rd.steps
                    ],
                    "findings": [_finding_diff_json(f) for f in rd.findings],
                    "usage": {"a": a.usage.model_dump(), "b": b.usage.model_dump()},
                },
                indent=2,
            )
        )
        return

    head = Table(title=f"A {a.run_id}  vs  B {b.run_id}", title_justify="left")
    head.add_column("", style="bold", no_wrap=True)
    head.add_column("A", overflow="fold")
    head.add_column("B", overflow="fold")
    head.add_column("Δ", justify="right", no_wrap=True)
    if rd.relation:
        head.add_row("relation", rd.relation, "", "")
    head.add_row("name", a.name, b.name, "")
    head.add_row(
        "status",
        Text(a.status, style=_STATUS_STYLE.get(a.status, "")),
        Text(b.status, style=_STATUS_STYLE.get(b.status, "")),
        "",
    )
    head.add_row(
        "outcome", _outcome_text(a), _outcome_text(b), "changed" if a.outcome != b.outcome else ""
    )
    if a.outcome_reason or b.outcome_reason:
        head.add_row("reason", a.outcome_reason or "", b.outcome_reason or "", "")
    models_a = ", ".join(dict.fromkeys(s.name for s in a.chat_steps())) or "-"
    models_b = ", ".join(dict.fromkeys(s.name for s in b.chat_steps())) or "-"
    head.add_row("model", models_a, models_b, "")
    for label, va, vb, dv in _usage_rows(a, b):
        head.add_row(label, va, vb, dv)
    if a.patch or b.patch:
        head.add_row("patch", a.patch_description or "", b.patch_description or "", "")
    head.add_row("output", _short(a.output, 80), _short(b.output, 80), "")
    console.print(head)
    if rd.checks:
        console.print(_checks_diff_table(rd))
    if rd.findings:
        ft = Table(title="runtime findings", title_justify="left")
        ft.add_column("kind", style="bold", no_wrap=True)
        ft.add_column("A", overflow="fold")
        ft.add_column("B", overflow="fold")
        for fd in rd.findings:
            cells = []
            for fs, worst in ((fd.a, fd.worst_a), (fd.b, fd.worst_b)):
                text = (
                    Text(f"{len(fs)} {worst}", style=_SEV_STYLE.get(worst or "", ""))
                    if fs
                    else Text("-", style="dim")
                )
                if fs:
                    text += Text(f" ({fs[0].message[:70]})", style="dim")
                cells.append(text)
            ft.add_row(fd.kind + (" *" if fd.changed else ""), *cells)
        console.print(ft)

    steps = Table(title="steps", title_justify="left")
    steps.add_column("idx", justify="right", style="cyan")
    steps.add_column("A", overflow="fold")
    steps.add_column("B", overflow="fold")
    steps.add_column("", no_wrap=True)
    for sd in rd.steps:
        label = _STEP_LABEL[sd.status]
        if sd.fields and sd.status in ("request", "response"):
            label += " (" + ", ".join(sd.fields[:3]) + ")"
        steps.add_row(
            str(sd.idx),
            _step_cell(sd.a),
            _step_cell(sd.b),
            Text(label, style=_STEP_STYLE[sd.status]),
        )
    if rd.steps:
        console.print(steps)
    shown = [s for s in rd.steps if not s.same and s.detail()]
    for sd in shown if full else shown[:1]:
        console.print(
            Panel(
                Text(sd.detail()),
                title=f"step {sd.idx}: {_STEP_LABEL[sd.status]}",
                title_align="left",
            )
        )
    console.print(rd.summary())


# --- export / import -------------------------------------------------------------------


@app.command("export")
def export_cmd(
    run_ref: str = typer.Argument("last", help="Run id, unique id prefix, or 'last'."),
    output: Path | None = typer.Option(  # noqa: B008
        None, "--output", "-o", help="File to write (default: <run_id>.jsonl)."
    ),
) -> None:
    """Write one run (metadata + cassette) to a portable JSONL file, e.g. to share a failing
    run or check it into a repo. Redaction runs again on the way out."""
    from .store import export_run

    store = Store()
    run_id = store.resolve_run_id(run_ref)
    path = export_run(store, run_id, output or Path(f"{run_id}.jsonl"))
    _console().print(f"exported {run_id} -> {_e(path)}")


@app.command("import")
def import_cmd(
    paths: list[Path] = typer.Argument(..., help="Exported .jsonl files."),  # noqa: B008
    replace: bool = typer.Option(False, "--replace", help="Overwrite runs that already exist."),
) -> None:
    """Add exported runs to the store (``$CRASHLAB_DIR`` or ./.crashlab)."""
    from .store import import_run

    console = _console()
    store = Store()
    for path in paths:
        try:
            run_id, added = import_run(store, path, replace=replace)
        except (OSError, ValueError) as err:
            console.print(f"[red]{_e(path)}: {_e(err)}[/]")
            raise typer.Exit(code=2) from err
        console.print(f"{'imported' if added else 'already there'}: {run_id}  ({_e(path)})")
    console.print(f"store: {_e(store.root)}")


if __name__ == "__main__":  # pragma: no cover
    app()
