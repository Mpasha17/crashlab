"""``crashlab`` command line: ``ls``, ``show``, ``replay``, ``check``, ``fork``, ``diff``."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ._version import __version__
from .models import CheckResult, Run, Step
from .store import ENV_DIR, RunNotFoundError, Store

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
            r["name"] + _relation_suffix(r),
            Text(r["status"], style=_STATUS_STYLE.get(r["status"], "")),
            Text(r["outcome"], style=_OUTCOME_STYLE.get(r["outcome"], "")),
            str(r["steps"]),
            str(r["total_tokens"]),
            _local_time(r["started_at"]),
        )
    console.print(table)


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
    meta.add_row(
        "usage",
        f"{u.steps} steps · {u.llm_calls} llm · {u.tool_calls} tool · {u.errors} errors · "
        f"{u.input_tokens}+{u.output_tokens}={u.total_tokens} tokens"
        + (f" · {u.wall_time_ms / 1000:.2f}s" if u.wall_time_ms is not None else ""),
    )
    meta.add_row("code", run.code_version or "-")
    console.print(Panel(meta, title="crashlab run", title_align="left"))
    if run.checks:
        console.print(_checks_table(run.checks))
    for flag in run.flags:
        at = f" (step {flag.step_idx})" if flag.step_idx is not None else ""
        console.print(f"[yellow]flag[/] {flag.detector}{at}: {flag.message}")

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
        op = step.operation + _SOURCE_MARK.get(step.source, "")
        table.add_row(str(step.idx), op, step.name, inp, out, err)
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
        f"Replaying [cyan]{source.run_id}[/] ({source.name}, {len(source.steps)} steps)"
        + (" offline" if not allow_network else "")
        + " ..."
    )
    try:
        rep = replay(source, offline=not allow_network, store_dir=store.root)
    except DivergenceError as err:
        console.print(Panel(Text(str(err)), title="[red]replay diverged[/]", title_align="left"))
        if err.replay_run_id:
            console.print(f"Diverged replay stored as [cyan]{err.replay_run_id}[/] "
                          f"(crashlab show {err.replay_run_id})")  # fmt: skip
        raise typer.Exit(code=1) from err
    except ReplayError as err:
        console.print(f"[red]cannot replay:[/] {err}")
        raise typer.Exit(code=2) from err
    served = sum(1 for s in rep.steps if s.source == "cassette")
    console.print(
        f"[green]reproduced[/]: {served}/{len(rep.steps)} steps served from the cassette, "
        + (
            f"same error as recorded ({rep.error.type})."
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
    run_ref: str = typer.Argument("last", help="Run id, unique id prefix, or 'last'."),
    save: bool = typer.Option(True, "--save/--no-save", help="Store the new outcome."),
) -> None:
    """Run the registered @crashlab.check oracles on a stored run (exit 1 if any fails)."""
    from .replay.divergence import ReplayError
    from .replay.engine import resolve_entrypoint
    from .verify.oracles import apply_outcome, checks_for, run_checks

    console = _console()
    store = Store()
    run = _resolve(store, run_ref)
    if run.entrypoint:  # importing the agent's module registers its checks
        try:
            resolve_entrypoint(run.entrypoint)
        except ReplayError as err:
            console.print(f"[yellow]could not import {run.entrypoint}:[/] {err}")
    specs = checks_for(run.name)
    if not specs:
        console.print(f"No checks registered for runs named {run.name!r}.")
        raise typer.Exit(code=0)
    results = run_checks(run, specs)
    apply_outcome(run, results)
    console.print(_checks_table(results))
    console.print(
        f"outcome: [{_OUTCOME_STYLE.get(run.outcome, '')}]{run.outcome}[/]"
        + (f"  ({run.outcome_reason})" if run.outcome_reason else "")
    )
    if save:
        store.update_run(run)
    if run.outcome == "fail":
        raise typer.Exit(code=1)


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
        out.append(P.ToolDescriptionPatch(tool=tool_description[0],
                                          description=tool_description[1]))  # fmt: skip
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
    ),  # fmt: skip
    system_prepend: str | None = typer.Option(
        None, "--system-prepend", help="Prepend to the system prompt."
    ),  # fmt: skip
    system_replace: str | None = typer.Option(
        None, "--system-replace", help="Replace the system prompt."
    ),  # fmt: skip
    system_insert_after: tuple[str, str] = typer.Option(
        (None, None),
        "--system-insert-after",
        metavar="ANCHOR TEXT",
        help="Insert TEXT as a new line after the system-prompt line containing ANCHOR.",
    ),  # fmt: skip
    system_sub: tuple[str, str] = typer.Option(
        (None, None),
        "--system-sub",
        metavar="OLD NEW",
        help="Replace OLD with NEW in the system prompt.",
    ),  # fmt: skip
    system_delete_line: str | None = typer.Option(
        None,
        "--system-delete-line",
        metavar="MATCH",
        help="Delete the system-prompt line containing MATCH.",
    ),  # fmt: skip
    model: str | None = typer.Option(None, "--model", help="Use this model from step N on."),
    params: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--param",
        metavar="KEY=VALUE",
        help="Request parameter (JSON value), e.g. temperature=0.7. Repeatable.",
    ),  # fmt: skip
    tool_description: tuple[str, str] = typer.Option(
        (None, None),
        "--tool-description",
        metavar="TOOL TEXT",
        help="Replace a tool's description.",
    ),  # fmt: skip
    tool_result: tuple[int, str] = typer.Option(
        (None, None),
        "--tool-result",
        metavar="STEP JSON",
        help="Serve this result at tool step STEP instead of running the tool.",
    ),  # fmt: skip
    message: tuple[int, int, str] = typer.Option(
        (None, None, None),
        "--message",
        metavar="STEP INDEX TEXT",
        help="Replace message INDEX of the request at chat step STEP.",
    ),  # fmt: skip
    inject_user: tuple[int, str] = typer.Option(
        (None, None),
        "--inject-user",
        metavar="STEP TEXT",
        help="Append a user message to the request at chat step STEP.",
    ),  # fmt: skip
    tools: str = typer.Option(
        "cassette_if_match",
        "--tools",
        help="cassette_if_match (recorded result for identical calls, else live) or live.",
    ),  # fmt: skip
    allow_live_side_effects: bool = typer.Option(
        False,
        "--allow-live-side-effects",
        help="Run side_effects=True tools live after the fork step (sandboxes only).",
    ),  # fmt: skip
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
            system_append=system_append, system_prepend=system_prepend,
            system_replace=system_replace, system_insert_after=system_insert_after,
            system_sub=system_sub, system_delete_line=system_delete_line, model=model,
            params=params or [], tool_description=tool_description, tool_result=tool_result,
            message=message, inject_user=inject_user,
        )  # fmt: skip
        n = len(source.steps)
        fork_step = at + n if at < 0 else at
        rendered = patches.render_diff(source, fork_step=fork_step)
    except (P.PatchError, ValueError, OSError) as err:
        console.print(f"[red]bad patch:[/] {err}")
        raise typer.Exit(code=2) from err
    if save_patch:
        Path(save_patch).write_text(patches.to_json(indent=2) + "\n", encoding="utf-8")
        console.print(f"patch written to {save_patch}")
    title = f"patch for {source.run_id} at step {fork_step}"
    body = rendered if patches else f"no patch: the fork re-runs live from step {fork_step}"
    console.print(Panel(Text(body), title=title, title_align="left"))
    if dry_run:
        return
    prefix = f"steps 0..{fork_step - 1} from the cassette, " if fork_step else ""
    console.print(
        f"Forking [cyan]{source.run_id}[/] ({source.name}) at step {fork_step}: "
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
        console.print(Panel(Text(str(err)), title="[red]fork prefix diverged[/]",
                            title_align="left"))  # fmt: skip
        raise typer.Exit(code=1) from err
    except (ReplayError, P.PatchError) as err:
        console.print(f"[red]cannot fork:[/] {err}")
        raise typer.Exit(code=2) from err
    if branch.error:
        console.print(
            f"[red]the fork raised[/] {branch.error.type}: {_short(branch.error.message, 160)}"
        )
    for flag in branch.flags:
        console.print(f"[yellow]flag[/] {flag.detector}: {flag.message}")
    _print_outcomes(console, source, branch)
    console.print(
        f"fork run [cyan]{branch.run_id}[/] (fork of {source.run_id} at step {fork_step})  "
        f"compare: crashlab diff {source.run_id} {branch.run_id}"
    )


def _outcome_text(run: Run) -> Text:
    return Text(run.outcome, style=_OUTCOME_STYLE.get(run.outcome, ""))


def _print_outcomes(console: Console, a: Run, b: Run) -> None:
    from .compare import diff

    console.print(Text("outcome: ") + _outcome_text(a) + Text(" -> ") + _outcome_text(b)
                  + Text(f"   ({b.outcome_reason})" if b.outcome_reason else ""))  # fmt: skip
    rd = diff(a, b)
    if rd.checks:
        console.print(_checks_diff_table(rd))


# --- diff ------------------------------------------------------------------------------

_STEP_STYLE = {"same": "green", "response": "yellow", "request": "yellow",
               "different": "red", "only_a": "magenta", "only_b": "magenta"}  # fmt: skip
_STEP_LABEL = {"same": "same", "response": "≠ response", "request": "≠ request",
               "different": "≠ call", "only_a": "only A", "only_b": "only B"}  # fmt: skip


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
        ("tool calls", str(ua.tool_calls), str(ub.tool_calls),
         delta(ua.tool_calls, ub.tool_calls)),
        ("errors", str(ua.errors), str(ub.errors), delta(ua.errors, ub.errors)),
        ("tokens in", str(ua.input_tokens), str(ub.input_tokens),
         delta(ua.input_tokens, ub.input_tokens)),
        ("tokens out", str(ua.output_tokens), str(ub.output_tokens),
         delta(ua.output_tokens, ub.output_tokens)),
        ("tokens total", str(ua.total_tokens), str(ub.total_tokens),
         delta(ua.total_tokens, ub.total_tokens)),
    ]  # fmt: skip
    if ua.cost_usd is not None or ub.cost_usd is not None:
        rows.append(("cost", f"${ua.cost_usd or 0:.4f}", f"${ub.cost_usd or 0:.4f}",
                     delta(ua.cost_usd, ub.cost_usd, "{:+.4f}")))  # fmt: skip
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
        typer.echo(json.dumps({
            "a": a.run_id, "b": b.run_id, "relation": rd.relation,
            "outcome": [a.outcome, b.outcome], "status": [a.status, b.status],
            "first_divergence": rd.first_divergence, "summary": rd.summary(),
            "checks": {c.name: [c.a.status if c.a else None, c.b.status if c.b else None]
                       for c in rd.checks},
            "steps": [{"idx": s.idx, "status": s.status, "fields": s.fields} for s in rd.steps],
            "usage": {"a": a.usage.model_dump(), "b": b.usage.model_dump()},
        }, indent=2))  # fmt: skip
        return

    head = Table(title=f"A {a.run_id}  vs  B {b.run_id}", title_justify="left")
    head.add_column("", style="bold", no_wrap=True)
    head.add_column("A", overflow="fold")
    head.add_column("B", overflow="fold")
    head.add_column("Δ", justify="right", no_wrap=True)
    if rd.relation:
        head.add_row("relation", rd.relation, "", "")
    head.add_row("name", a.name, b.name, "")
    head.add_row("status", Text(a.status, style=_STATUS_STYLE.get(a.status, "")),
                 Text(b.status, style=_STATUS_STYLE.get(b.status, "")), "")  # fmt: skip
    head.add_row("outcome", _outcome_text(a), _outcome_text(b),
                 "changed" if a.outcome != b.outcome else "")  # fmt: skip
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

    steps = Table(title="steps", title_justify="left")
    steps.add_column("idx", justify="right", style="cyan")
    steps.add_column("A", overflow="fold")
    steps.add_column("B", overflow="fold")
    steps.add_column("", no_wrap=True)
    for sd in rd.steps:
        label = _STEP_LABEL[sd.status]
        if sd.fields and sd.status in ("request", "response"):
            label += " (" + ", ".join(sd.fields[:3]) + ")"
        steps.add_row(str(sd.idx), _step_cell(sd.a), _step_cell(sd.b),
                      Text(label, style=_STEP_STYLE[sd.status]))  # fmt: skip
    if rd.steps:
        console.print(steps)
    shown = [s for s in rd.steps if not s.same and s.detail()]
    for sd in shown if full else shown[:1]:
        console.print(Panel(Text(sd.detail()), title=f"step {sd.idx}: {_STEP_LABEL[sd.status]}",
                            title_align="left"))  # fmt: skip
    console.print(rd.summary())


if __name__ == "__main__":  # pragma: no cover
    app()
