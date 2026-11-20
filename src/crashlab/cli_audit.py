"""Inspect commands for prompt composition and tool-schema audits.

``crashlab context`` and ``crashlab audit-tools`` are registered on the main
``crashlab`` app.
"""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.text import Text

from .cli import Table, _console, _e, app
from .store import RunNotFoundError, Store

_SEV_STYLE = {"info": "dim", "warn": "yellow", "error": "red", "critical": "bold red"}


@app.command("context")
def context_cmd(
    run_ref: str = typer.Argument("last", help="Run id, unique prefix, or 'last'."),
    step: int | None = typer.Option(
        None, "--step", help="Only break down this chat step (0-based index)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
) -> None:
    """Break down a run's prompt by section, tokens and growth.

    For every LLM chat step, reports character counts and estimated token share
    (chars // 4, scaled to recorded input_tokens when available) for the system
    prompt, tool definitions, user / assistant messages and tool results, plus
    a per-tool output size leaderboard, repeated messages, and turn-over-turn
    growth. --step N limits the report to that chat step.
    """
    from .context_audit import analyze_context

    console = _console(stderr=as_json)
    run = _load_run(run_ref, console)
    try:
        report = analyze_context(run, step=step)
    except ValueError as exc:
        console.print(f"[red]{_e(exc)}[/]")
        raise typer.Exit(code=2) from exc
    if as_json:
        typer.echo(json.dumps(report.to_dict(), indent=2, default=str))
        return
    _print_context(report)


@app.command("audit-tools")
def audit_tools_cmd(
    run_ref: str = typer.Argument("last", help="Run id, unique prefix, or 'last'."),
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
    fail_on_error: bool = typer.Option(
        False,
        "--fail-on-error",
        help="Exit 1 when the audit found an error-severity schema issue.",
    ),
) -> None:
    """Audit recorded tool schemas for provider-breaking issues.

    Inspects tool definitions on LLM steps and the tools that were actually
    called. Flags required fields missing from properties (the Groq / Gemini
    rejection), missing or vague descriptions, parameters without a type or
    description, duplicate or near-identical tool names, oversized schemas,
    and tools defined on every step but never called.
    """
    from .context_audit import audit_tool_schemas

    console = _console(stderr=as_json)
    run = _load_run(run_ref, console)
    report = audit_tool_schemas(run)
    if as_json:
        typer.echo(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        _print_audit(report)
    if fail_on_error and not report.ok:
        raise typer.Exit(code=1)


def _load_run(run_ref: str, console: Console):
    store = Store()
    try:
        return store.load_run(run_ref)
    except RunNotFoundError as exc:
        console.print(f"[red]{_e(exc)}[/]")
        raise typer.Exit(code=2) from exc


def _print_context(report: object) -> None:
    from .context_audit import ContextReport

    assert isinstance(report, ContextReport)
    console = _console()
    n = len(report.steps)
    step_label = "chat step" if n == 1 else "chat steps"
    console.print(
        f"context {_e(report.run_id)}  ({_e(report.name)})  {n} {step_label}"
        + (f"  (filtered to step {report.step})" if report.step is not None else "")
    )
    if not report.steps:
        console.print("no chat steps to analyse")
        return
    for ctx in report.steps:
        model = ctx.model or "?"
        recorded = (
            f"{ctx.input_tokens:,} recorded in" if ctx.input_tokens is not None else "no usage"
        )
        delta = _fmt_delta(ctx.delta_chars, "chars")
        console.print(
            f"step {ctx.idx}  {_e(model)}  {ctx.total_chars:,} chars  "
            f"~{ctx.estimated_tokens:,} tok  {recorded}  {delta}"
        )
        table = Table(box=None)
        table.add_column("section", style="bold")
        table.add_column("chars", justify="right", no_wrap=True)
        table.add_column("tokens", justify="right", no_wrap=True)
        table.add_column("share", justify="right", no_wrap=True)
        for section in ctx.sections:
            table.add_row(
                section.name,
                f"{section.chars:,}",
                f"{section.estimated_tokens:,}",
                f"{section.share_pct:.1f}%",
            )
        console.print(table)
        if ctx.tool_outputs:
            bits = [f"{t.name} {t.chars:,} chars ×{t.occurrences}" for t in ctx.tool_outputs[:5]]
            console.print("  tool outputs: " + ", ".join(_e(b) for b in bits))
        if ctx.repeats:
            for rep in ctx.repeats:
                console.print(f"  repeated {rep.role} ×{rep.count}: {_e(rep.preview)}")
    if report.tool_outputs:
        table = Table(title="tool outputs (context footprint)", title_justify="left", box=None)
        table.add_column("tool", style="bold")
        table.add_column("chars", justify="right")
        table.add_column("tokens", justify="right")
        table.add_column("times", justify="right")
        table.add_column("steps")
        for item in report.tool_outputs:
            table.add_row(
                item.name,
                f"{item.chars:,}",
                f"{item.estimated_tokens:,}",
                str(item.occurrences),
                ",".join(str(i) for i in item.steps),
            )
        console.print(table)
    # Cross-turn repeats that are not already printed under a single step.
    extra = [r for r in report.repeats if not r.within_step]
    if extra:
        table = Table(title="repeated messages", title_justify="left", box=None)
        table.add_column("role", no_wrap=True)
        table.add_column("count", justify="right")
        table.add_column("steps")
        table.add_column("preview", overflow="fold")
        for rep in extra:
            table.add_row(
                rep.role,
                str(rep.count),
                ",".join(str(i) for i in rep.steps),
                rep.preview,
            )
        console.print(table)


def _print_audit(report: object) -> None:
    from .context_audit import ToolAuditReport

    assert isinstance(report, ToolAuditReport)
    console = _console()
    n_tools = len(report.tools)
    n_issues = len(report.issues)
    status = "ok" if report.ok else "errors"
    console.print(
        f"audit-tools {_e(report.run_id)}  ({_e(report.name)})  "
        f"{n_tools} tool(s), {n_issues} issue(s)  {status}"
    )
    if report.tools:
        table = Table(title="tools", title_justify="left", box=None)
        table.add_column("tool", style="bold")
        table.add_column("calls", justify="right")
        table.add_column("chars", justify="right")
        table.add_column("status")
        for tool in report.tools:
            if tool.call_count:
                status = f"{tool.call_count} call(s)"
            elif tool.name in report.unused:
                status = "unused"
            elif tool.defined_steps:
                status = "defined, never called"
            else:
                status = "called, not defined"
            table.add_row(tool.name, str(tool.call_count), f"{tool.chars:,}", status)
        console.print(table)
    if report.called:
        console.print("called: " + ", ".join(_e(n) for n in report.called))
    if report.unused:
        console.print("unused: " + ", ".join(_e(n) for n in report.unused))
    if report.issues:
        table = Table(title="issues", title_justify="left", box=None)
        table.add_column("severity", no_wrap=True)
        table.add_column("kind", style="bold", no_wrap=True)
        table.add_column("tool")
        table.add_column("message", overflow="fold")
        for issue in report.issues:
            table.add_row(
                Text(issue.severity, style=_SEV_STYLE.get(issue.severity, "")),
                issue.kind,
                issue.tool or "",
                issue.message,
            )
        console.print(table)
    elif report.tools:
        console.print("no schema issues")


def _fmt_delta(delta: int | None, unit: str) -> str:
    if delta is None:
        return "Δ —"
    sign = "+" if delta > 0 else ""
    return f"Δ {sign}{delta:,} {unit}"
