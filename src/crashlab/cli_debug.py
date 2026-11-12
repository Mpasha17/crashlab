"""Debug-session commands registered on the main ``crashlab`` app: ``search``,
``cluster`` and the step filters / step detail behind ``crashlab show``."""

from __future__ import annotations

import contextlib
import json
from typing import Any

import typer
from rich.text import Text

from .cli import (
    _OUTCOME_STYLE,
    _STATUS_STYLE,
    Table,
    _console,
    _cost_text,
    _e,
    _local_time,
    _unique_prefix_len,
    app,
)
from .models import Run, Step
from .store import Store

_TRIAGE_BY = ("name", "error", "step", "checks", "findings")


def _clip(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


# --- crashlab search ----------------------------------------------------------------


def _since(value: str | None) -> Any:
    if not value:
        return None
    from ._runscan import parse_since

    try:
        return parse_since(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--since") from None


@app.command("search")
def search_cmd(
    query: str | None = typer.Argument(
        None, help="Text to find (case-insensitive substring). Omit to list runs by filters."
    ),
    tool: str | None = typer.Option(None, "--tool", help="Only runs that called this tool."),
    model: str | None = typer.Option(
        None, "--model", help="Only runs whose model name contains this."
    ),
    check: str | None = typer.Option(
        None, "--check", help="Only runs where this @crashlab.check did not pass."
    ),
    label: str | None = typer.Option(
        None, "--label", help="Only runs diagnosed with this label (primary or secondary)."
    ),
    errors: bool = typer.Option(
        False, "--errors", help="Only runs that raised or have a step error."
    ),
    status: str | None = typer.Option(None, "--status", help="ok, error or running."),
    outcome: str | None = typer.Option(None, "--outcome", help="pass, fail or unknown."),
    name: str | None = typer.Option(None, "--name", help="Only runs of this agent name."),
    since: str | None = typer.Option(
        None, "--since", help="Started within 30m / 2h / 7d / 1w, or after an ISO date."
    ),
    run_prefix: str | None = typer.Option(None, "--run", help="Run id prefix."),
    fields: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--in",
        help="Only look in: run, error, message, tool, model, check, finding, diagnosis "
        "(repeatable).",
    ),
    regex: bool = typer.Option(False, "--regex", "-E", help="QUERY is a regular expression."),
    case_sensitive: bool = typer.Option(False, "--case-sensitive", "-s"),
    limit: int = typer.Option(50, "--limit", "-n", help="Max matching runs (0 = all)."),
    as_json: bool = typer.Option(False, "--json", help="Print the matches as JSON."),
) -> None:
    """Search recorded runs for text and filter them.

    Looks in run ids and names, input / output, run and step errors, chat messages,
    tool names / arguments / results, model names, check names / messages, runtime
    findings and diagnosis labels. Newest runs first. Exits 1 when nothing matched,
    like grep.
    """
    from .search import search_runs

    console = _console()
    try:
        result = search_runs(
            query,
            store=Store(),
            tool=tool,
            model=model,
            check=check,
            label=label,
            errors=errors,
            status=status,
            outcome=outcome,
            name=name,
            since=_since(since),
            run_prefix=run_prefix,
            fields=fields or None,
            regex=regex,
            case_sensitive=case_sensitive,
            limit=limit or None,
        )
    except ValueError as exc:
        console.print(f"[red]{_e(exc)}[/]")
        raise typer.Exit(code=2) from None
    if as_json:
        typer.echo(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        raise typer.Exit(code=0 if result.matches else 1)
    if not result.matches:
        what = f"matching {query!r}" if query else "matching the filters"
        console.print(f"No runs {what} ({result.scanned} scanned).")
        _unreadable_note(console, result.unreadable)
        raise typer.Exit(code=1)
    narrow = console.width < 110
    id_len = _unique_prefix_len(Store().index.find_run_ids(""))  # unique store-wide
    table = Table(show_lines=False, pad_edge=not narrow)
    table.add_column("run", style="cyan", no_wrap=True)
    if not narrow:
        table.add_column("name")
    table.add_column("outcome", no_wrap=True)
    table.add_column("step", justify="right", no_wrap=True)
    table.add_column("field", no_wrap=True)
    table.add_column("match", overflow="fold")
    for m in result.matches:
        rows = m.hits or [None]
        for i, hit in enumerate(rows):
            first = i == 0
            cells: list[Any] = [m.run_id[:id_len] if first else ""]
            if not narrow:
                cells.append(m.name if first else "")
            cells.append(
                Text(m.outcome, style=_OUTCOME_STYLE.get(m.outcome, "")) if first else Text("")
            )
            if hit is None:
                cells += ["", "", _local_time(m.started_at)]
            else:
                cells += [
                    "" if hit.step is None else str(hit.step),
                    hit.field,
                    _highlight(hit.snippet, query, regex=regex, case_sensitive=case_sensitive),
                ]
            table.add_row(*cells)
        if m.hits_truncated:
            table.add_row(
                *([""] * (len(table.columns) - 1)),
                Text(f"… {m.hits_truncated} more hit(s) in this run", style="dim"),
            )
    console.print(table)
    hits = sum(len(m.hits) + m.hits_truncated for m in result.matches)
    tail = f", {hits} hit(s)" if query else ""
    console.print(
        f"[dim]{len(result.matches)} run(s){tail} · {result.scanned} scanned"
        + (
            " · stopped at --limit; older runs not scanned (--limit 0 for all)"
            if result.truncated
            else ""
        )
        + "[/]"
    )
    _unreadable_note(console, result.unreadable)


def _unreadable_note(console: Any, unreadable: list[str]) -> None:
    if unreadable:
        shown = ", ".join(unreadable[:3]) + (" …" if len(unreadable) > 3 else "")
        console.print(f"[yellow]skipped {len(unreadable)} unreadable run(s): {_e(shown)}[/]")


def _highlight(snippet: str, query: str | None, *, regex: bool, case_sensitive: bool) -> Text:
    import re

    text = Text(snippet)
    if query:
        flags = 0 if case_sensitive else re.IGNORECASE
        with contextlib.suppress(re.error):
            text.highlight_regex(
                re.compile(query if regex else re.escape(query), flags), style="bold yellow"
            )
    return text


# --- crashlab cluster ---------------------------------------------------------------


@app.command("cluster")
def cluster_cmd(
    by: str = typer.Option(
        "name,error,step,checks,label,findings",
        "--by",
        help="Signature components, comma-separated: name, error, step, checks, label, "
        "findings. Fewer components group more loosely.",
    ),
    name: str | None = typer.Option(None, "--name", help="Only runs of this agent name."),
    since: str | None = typer.Option(
        None, "--since", help="Started within 30m / 2h / 7d / 1w, or after an ISO date."
    ),
    run_prefix: str | None = typer.Option(None, "--run", help="Run id prefix."),
    flagged: bool = typer.Option(
        False,
        "--flagged",
        help="Also cluster runs that passed but have an error/critical runtime finding.",
    ),
    min_size: int = typer.Option(1, "--min-size", help="Hide clusters smaller than this."),
    limit: int = typer.Option(0, "--limit", "-n", help="Read at most this many runs (0 = all)."),
    show_runs: int = typer.Option(3, "--runs", help="Run ids to list per cluster."),
    as_json: bool = typer.Option(False, "--json", help="Print the clusters as JSON."),
) -> None:
    """Group failed runs by failure signature.

    A run counts as failed when it raised, its outcome is fail, or a @crashlab.check did
    not pass. Two runs share a cluster only when every --by component matches: the
    agent name, the normalized exception, the first failing step (operation, name and
    error, not the index), the failing check names, the diagnosis label and the
    error/critical finding kinds. Largest clusters first.
    """
    from .cluster import cluster_runs

    console = _console()
    parts = tuple(p.strip() for p in by.split(",") if p.strip())
    try:
        result = cluster_runs(
            store=Store(),
            by=parts,
            name=name,
            since=_since(since),
            run_prefix=run_prefix,
            flagged=flagged,
            min_size=max(min_size, 1),
            limit=limit or None,
        )
    except ValueError as exc:
        console.print(f"[red]{_e(exc)}[/]")
        raise typer.Exit(code=2) from None
    if as_json:
        typer.echo(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return
    if not result.clusters:
        console.print(
            f"No failed runs to cluster ({result.scanned} scanned"
            + (f", {result.failed} failed below --min-size" if result.failed else "")
            + ")."
        )
        _unreadable_note(console, result.unreadable)
        return
    id_len = _unique_prefix_len(Store().index.find_run_ids(""))  # unique store-wide
    table = Table(show_lines=True)
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("runs", justify="right", no_wrap=True, style="bold")
    table.add_column("cluster", no_wrap=True, style="dim")
    table.add_column("signature", overflow="fold")
    table.add_column("latest run", no_wrap=True, style="cyan")
    for i, c in enumerate(result.clusters, 1):
        sig = Text()
        if c.signature.get("name"):
            sig.append(c.signature["name"], style="bold").append("\n")
        sig.append(c.summary)
        if show_runs > 0 and c.size > 1:
            more = c.size - min(show_runs, c.size)
            sig.append(
                "\nruns: "
                + ", ".join(r[:id_len] for r in c.run_ids[:show_runs])
                + (f" +{more} more" if more else ""),
                style="dim",
            )
        latest = Text(c.representative[:id_len]).append(
            "\n" + _local_time(c.last_seen)[5:16], style="dim"
        )
        table.add_row(str(i), str(c.size), c.cluster_id, sig, latest)
    console.print(table)
    console.print(
        f"[dim]{result.failed} failed run(s) in {len(result.clusters)} cluster(s) · "
        f"{result.scanned} scanned · by {', '.join(result.by)}[/]"
    )
    console.print(
        f"[dim]inspect one: crashlab show {result.clusters[0].representative[:id_len]} --failing[/]"
    )
    _unreadable_note(console, result.unreadable)


# --- crashlab triage ----------------------------------------------------------------


@app.command("triage")
def triage_cmd(
    name: str | None = typer.Option(None, "--name", help="Only runs of this agent name."),
    since: str | None = typer.Option(
        None, "--since", help="Started within 30m / 2h / 7d / 1w, or after an ISO date."
    ),
    limit: int = typer.Option(
        10, "--limit", "-n", help="Diagnose and show the top N clusters (0 = all)."
    ),
    by: str = typer.Option(
        ",".join(_TRIAGE_BY),
        "--by",
        help="Signature components (see crashlab cluster); default leaves out the label.",
    ),
    flagged: bool = typer.Option(
        False, "--flagged", help="Also runs that passed with an error/critical finding."
    ),
    rediagnose: bool = typer.Option(
        False, "--rediagnose", help="Replace stored diagnoses with new heuristic ones."
    ),
    save: bool = typer.Option(
        True, "--save/--no-save", help="Store new diagnoses with the representative runs."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the triage as JSON."),
) -> None:
    """Triage the store's failures: cluster failed runs, diagnose each cluster's newest
    run offline (heuristic, no LLM or keys; a diagnosis already stored is reused) and
    print them ranked with the commands to go further."""
    from .triage import triage_runs

    console = _console()
    parts = tuple(p.strip() for p in by.split(",") if p.strip())
    try:
        r = triage_runs(
            store=Store(), by=parts, name=name, since=_since(since), flagged=flagged,
            limit=limit or None, rediagnose=rediagnose, save=save,
        )  # fmt: skip
    except ValueError as exc:
        console.print(f"[red]{_e(exc)}[/]")
        raise typer.Exit(code=2) from None
    if as_json:
        typer.echo(json.dumps(r.to_dict(), indent=2, ensure_ascii=False, default=str))
        return
    if not r.items:
        console.print(f"No failed runs to triage ({r.scanned} scanned).")
        _unreadable_note(console, r.unreadable)
        return
    id_len = _unique_prefix_len(Store().index.find_run_ids(""))
    table = Table(show_lines=True)
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("runs", justify="right", no_wrap=True, style="bold")
    table.add_column("failure", overflow="fold", ratio=2)
    table.add_column("diagnosis", overflow="fold", ratio=3)
    for it in r.items:
        fail = Text()
        if it.signature.get("name"):
            fail.append(it.signature["name"], style="bold").append("\n")
        fail.append(_clip(it.summary, 240))
        rep = it.representative[:id_len]
        fail.append(f"\nlatest {rep} · {_local_time(it.last_seen)[5:16]}", style="dim")
        diag = Text()
        if it.diagnosis_error:
            diag.append(f"diagnosis failed: {it.diagnosis_error}", style="red")
        elif it.label:
            diag.append(it.label, style="bold")
            src = "heuristic" if it.diagnoser == "heuristic" else f"model {it.diagnoser}"
            diag.append(f" ({src}, {it.diagnosis_source})", style="dim")
            if it.failing_step is not None:
                diag.append(f"\nstep {it.failing_step}")
                if it.step_description:
                    diag.append(f" {it.step_description.split(' ', 2)[-1]}")
                if it.agent:
                    diag.append(f" @{it.agent}", style="magenta")
            if it.root_cause:
                diag.append("\n" + _clip(it.root_cause, 200))
            if it.evidence:
                diag.append("\n" + it.evidence, style="italic")
        cmds = [c.replace(it.representative, rep) for c in it.commands]
        diag.append("\n" + "\n".join(cmds), style="cyan")
        table.add_row(str(it.rank), str(it.size), fail, diag)
    console.print(table)
    more = r.clusters_total - len(r.items)
    console.print(
        f"[dim]{r.failed} failed run(s) in {r.clusters_total} cluster(s)"
        + (f" ({more} more: --limit 0)" if more else "")
        + f" · {r.scanned} scanned · diagnosed {r.diagnosed} new, reused {r.reused} stored · "
        "heuristic labels are rule-based guesses; `crashlab diagnose RUN` asks an LLM[/]"
    )
    _unreadable_note(console, r.unreadable)


# --- crashlab watch -----------------------------------------------------------------


@app.command("watch")
def watch_cmd(
    interval: float = typer.Option(1.0, "--interval", help="Seconds between polls."),
    include_existing: bool = typer.Option(
        False, "--existing", help="Also print runs already in the store."
    ),
    failed_only: bool = typer.Option(False, "--failed", help="Only print failed runs."),
    name: str | None = typer.Option(None, "--name", help="Only runs of this agent name."),
    timeout: float = typer.Option(
        0, "--timeout", help="Stop after this many seconds (0 = until Ctrl-C)."
    ),
    as_json: bool = typer.Option(False, "--json", help="One JSON object per event."),
) -> None:
    """Print runs as they start and finish (polls the store; Ctrl-C to stop).

    For a local debug loop: run your agent in one terminal and `crashlab watch` in
    another. A finished run that failed gets a pointer to its first failure.
    """
    import time

    from .watch import watch_runs

    console = _console()
    if interval <= 0:
        console.print("[red]--interval must be > 0[/]")
        raise typer.Exit(code=2)
    polls = None if timeout <= 0 else max(int(timeout / interval), 0) + 1
    store = Store()
    if not as_json:
        console.print(f"[dim]watching {_e(store.root)} every {interval:g}s (Ctrl-C to stop)[/]")
    try:
        for ev in watch_runs(
            store,
            interval=interval,
            include_existing=include_existing,
            failed_only=failed_only,
            name=name,
            polls=polls,
            sleep=time.sleep,
        ):
            if as_json:
                typer.echo(json.dumps(ev.to_dict(), ensure_ascii=False))
                continue
            line = Text(_local_time(ev.started_at)[11:] + "  ", style="dim")
            line.append(ev.run_id, style="cyan").append(f"  {ev.name}  ")
            if ev.kind == "started":
                line.append("started", style="yellow")
            else:
                line.append(ev.status, style=_STATUS_STYLE.get(ev.status, "")).append(" ")
                line.append(ev.outcome, style=_OUTCOME_STYLE.get(ev.outcome, ""))
                line.append(f"  {ev.steps} steps")
                if ev.cost_usd is not None:
                    line.append(f"  {_cost_text(ev.cost_usd)}")
                if ev.failure:
                    line.append("\n  → " + " · ".join(ev.failure), style="red")
            console.print(line)
    except KeyboardInterrupt:
        pass


# --- crashlab show helpers ----------------------------------------------------------


def select_steps(
    run: Run,
    *,
    errors: bool,
    tools: bool,
    llm: bool,
    failing: bool,
    context: int = 2,
    agent: str | None = None,
) -> tuple[list[Step], str | None]:
    """Steps for `crashlab show` filters, plus a one-line note describing the filter
    (None when nothing is filtered). ``tools`` / ``llm`` combine as a union; ``errors``
    and ``failing`` narrow further."""
    steps = list(run.steps)
    notes = []
    if tools or llm:
        ops = set()
        if tools:
            ops |= {"execute_tool", "invoke_agent"}
        if llm:
            ops |= {"chat", "embeddings"}
        steps = [s for s in steps if s.operation in ops]
        notes.append(" + ".join(n for n, on in (("tool", tools), ("llm", llm)) if on) + " steps")
    if agent is not None:
        mine = {s.idx for s in run.chat_steps(agent=agent) + run.tool_calls(agent=agent)}
        steps = [s for s in steps if s.idx in mine or ("#" not in agent and s.agent == agent)]
        names = run.agent_labels
        known = ", ".join(names) or "none labelled"
        ok = agent in names or agent in run.agents
        notes.append(f"agent {agent}" + ("" if ok else f" (agents: {known})"))
    if errors:
        steps = [s for s in steps if s.error is not None]
        notes.append("steps with an error")
    if failing:
        from .cluster import failure_point

        fp = failure_point(run)
        centre = fp.step if fp.step is not None else fp.diagnosis_step
        if centre is None:
            notes.append("no failing step recorded")
        else:
            lo, hi = centre - context, centre + context
            steps = [s for s in steps if lo <= s.idx <= hi]
            notes.append(f"steps {max(lo, 0)}–{hi} around step {centre}")
    if not notes:
        return steps, None
    return steps, f"showing {len(steps)} of {len(run.steps)} steps: " + "; ".join(notes)


def failure_line(run: Run, ref: str) -> Text | None:
    """``→ first error: step 3 (...) · checks failed: x · diagnosis: step 2 LABEL`` plus
    the command to open that step, or None when nothing failed."""
    from .cluster import failure_point

    fp = failure_point(run)
    bits: list[str] = []
    if fp.step is not None:
        what = "first error" if fp.step_kind == "error" else f"first {fp.step_kind} finding"
        bits.append(f"{what}: {fp.step_description}")
    if run.error is not None:
        bits.append(f"raised {run.error.type.rsplit('.', 1)[-1]}")
    if fp.failing_checks:
        bits.append("checks failed: " + ", ".join(fp.failing_checks))
    if fp.diagnosis_label and fp.diagnosis_label != "NO_FAILURE":
        at = f"step {fp.diagnosis_step} " if fp.diagnosis_step is not None else ""
        bits.append(f"diagnosis: {at}{fp.diagnosis_label}")
    if not bits:
        return None
    line = Text("→ ", style="bold red").append(" · ".join(bits))
    jump = fp.step if fp.step is not None else fp.diagnosis_step
    if jump is not None:
        line.append(f"\n  open it: crashlab show {ref} --step {jump}", style="dim")
    return line


def _pretty(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def print_step_detail(console: Any, run: Run, idx: int, *, as_json: bool) -> None:
    """Everything recorded for one step: request, response, error with traceback,
    timing, tokens and cost."""
    from .pricing import step_cost

    step = next((s for s in run.steps if s.idx == idx), None)
    if step is None:
        last = run.steps[-1].idx if run.steps else None
        rng = f"0–{last}" if last is not None else "none recorded"
        console.print(f"[red]run {_e(run.run_id)} has no step {idx} (steps: {rng})[/]")
        raise typer.Exit(code=1)
    if as_json:
        typer.echo(step.model_dump_json(indent=2))
        return
    meta = Table.grid(padding=(0, 2))
    meta.add_column(style="bold")
    meta.add_column(overflow="fold")
    meta.add_row("step", run.describe_step(idx))
    meta.add_row("run", f"{run.run_id} ({run.name})")
    meta.add_row("source", step.source)
    meta.add_row("started", f"{_local_time(step.started_at)} (local)")
    meta.add_row("duration", f"{step.duration_ms:.1f} ms")
    attrs = step.attributes or {}
    if step.operation == "chat":
        models = [str(attrs[k]) for k in ("gen_ai.request.model", "gen_ai.response.model")
                  if attrs.get(k)]  # fmt: skip
        if models:
            meta.add_row("model", " → ".join(dict.fromkeys(models)))
        inp, out = attrs.get("gen_ai.usage.input_tokens"), attrs.get("gen_ai.usage.output_tokens")
        if inp is not None or out is not None:
            meta.add_row("tokens", f"{inp or 0} in · {out or 0} out")
        cost = attrs.get("crashlab.usage.cost_usd")
        cost = step_cost(step) if cost is None else cost
        meta.add_row("est. cost", _cost_text(cost))
    if step.error is not None:
        meta.add_row("error", Text(f"{step.error.type}: {step.error.detail}", style="red"))
    marks = [
        f for f in run.findings if idx in (f.steps or ([f.step] if f.step is not None else []))
    ]
    for f in marks:
        meta.add_row("finding", f"{f.kind} [{f.severity}]: {f.message}")
    from rich.panel import Panel

    console.print(Panel(meta, title=f"step {idx}", title_align="left"))
    if step.operation == "chat":
        req = step.request or {}
        if req.get("system"):
            console.print(Text("system", style="bold"))
            console.print(Text(_pretty(req["system"])))
        for msg in req.get("messages") or []:
            role = msg.get("role", "?") if isinstance(msg, dict) else "?"
            console.print(Text(f"{role}", style="bold"))
            console.print(Text(_pretty(msg.get("parts", msg) if isinstance(msg, dict) else msg)))
        if req.get("params"):
            console.print(Text("params", style="bold"))
            console.print(Text(_pretty(req["params"])))
        console.print(Text("response", style="bold green"))
        console.print(Text(_pretty((step.response or {}).get("messages", step.response))))
    else:
        console.print(Text("request", style="bold"))
        console.print(Text(_pretty(step.request)))
        console.print(Text("response", style="bold green"))
        console.print(Text(_pretty(step.response)))
    if step.error is not None and step.error.traceback:
        console.print(Text("traceback", style="bold red"))
        console.print(Text(step.error.traceback.rstrip()))


__all__ = [
    "cluster_cmd",
    "triage_cmd",
    "failure_line",
    "print_step_detail",
    "search_cmd",
    "select_steps",
]
