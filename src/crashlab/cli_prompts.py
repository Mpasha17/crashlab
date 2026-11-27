"""``crashlab prompts``: prompt versions, their outcomes, and A/B comparison."""

from __future__ import annotations

import json
from typing import Any

import typer
from rich.text import Text

from .cli import Table, _console, _e, app
from .store import Store

_HELP = "Track prompt versions and A/B compare their outcomes."

prompts_app = typer.Typer(
    name="prompts",
    help=_HELP,
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
app.add_typer(
    prompts_app, name="prompts", help=_HELP, rich_help_panel="Evaluate", no_args_is_help=True
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


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v * 100:.1f}%"


def _usd(v: float | None) -> str:
    if v is None:
        return "-"
    return "$0" if v == 0 else f"${v:.4f}" if v >= 0.0001 else f"${v:.6f}"


def _secs(v: float | None) -> str:
    return "-" if v is None else f"{v * 1000:.0f}ms" if v < 1 else f"{v:.2f}s"


def _day(iso: str | None) -> str:
    return iso[:16].replace("T", " ") if iso else "-"


def _fail(msg: str, as_json: bool, code: int = 2) -> typer.Exit:
    if as_json:
        typer.echo(json.dumps({"ok": False, "error": msg}, indent=2))
    else:
        _console(stderr=True).print(f"[red]{_e(msg)}[/]")
    return typer.Exit(code=code)


def _scan(as_json: bool, since: str | None, agent: str | None):
    from .prompts import list_prompts

    try:
        return list_prompts(Store(), since=_since(since), agent=agent)
    except ValueError as exc:
        raise _fail(str(exc), as_json) from exc


@prompts_app.command("list")
def list_cmd(
    since: str | None = typer.Option(
        None, "--since", help="Started within 30m / 2h / 7d / 1w, or after an ISO date."
    ),
    agent: str | None = typer.Option(None, "--agent", help="Only this agent (run name or step)."),
    as_json: bool = typer.Option(False, "--json", help="Print JSON."),
) -> None:
    """List every prompt version with its runs, pass rate, error and finding rates, and cost.

    A prompt id (p-3f9a1c) is a hash of the normalized system prompt: whitespace,
    template placeholders, dates and UUIDs do not change it. Runs recorded before
    prompt tracking are fingerprinted from their stored requests.
    """
    versions, without = _scan(as_json, since, agent)
    rows = [v.summary() for v in versions]
    if as_json:
        typer.echo(
            json.dumps({"ok": True, "prompts": rows, "runs_without_prompt": without}, indent=2)
        )
        return
    console = _console()
    if not rows:
        console.print("no prompt versions found (no recorded chat step has a system prompt)")
        return
    table = Table(title="prompt versions", title_justify="left", box=None)
    for col, right in (
        ("id", False), ("label", False), ("first seen", False), ("last seen", False),
        ("runs", True), ("pass", True), ("error", True), ("findings", True),
        ("avg cost", True), ("avg latency", True), ("avg tokens", True),
    ):  # fmt: skip
        table.add_column(
            col, justify="right" if right else "left", style="bold" if col == "id" else ""
        )
    for r in rows:
        table.add_row(
            Text(r["id"]),
            Text(", ".join(r["labels"]) or "-"),
            _day(r["first_seen"]),
            _day(r["last_seen"]),
            str(r["runs"]),
            _pct(r["pass_rate"]),
            _pct(r["error_rate"]),
            _pct(r["finding_rate"]),
            _usd(r["avg_cost_usd"]),
            _secs(r["avg_latency_s"]),
            "-" if r["avg_tokens"] is None else f"{r['avg_tokens']:.0f}",
        )
    console.print(table)
    if without:
        console.print(f"[dim]{without} run(s) without a system prompt are not listed[/]")


def _one(versions, ref: str, as_json: bool):
    from .prompts import PromptError, resolve_prompt

    try:
        return resolve_prompt(versions, ref)
    except PromptError as exc:
        raise _fail(str(exc), as_json) from exc


@prompts_app.command("show")
def show_cmd(
    ref: str = typer.Argument(..., help="Prompt id (p-3f9a1c, any unique prefix) or label."),
    since: str | None = typer.Option(None, "--since", help="Only runs started since."),
    agent: str | None = typer.Option(None, "--agent", help="Only this agent."),
    as_json: bool = typer.Option(False, "--json", help="Print JSON."),
) -> None:
    """Print the full text of a prompt version and its summary."""
    versions, _ = _scan(as_json, since, agent)
    for v in _one(versions, ref, as_json):
        if as_json:
            typer.echo(json.dumps({"ok": True, **v.summary(), "text": v.text}, indent=2))
            continue
        s = v.summary()
        console = _console()
        console.print(
            f"[bold]{_e(v.id)}[/]"
            + (f"  label: {_e(', '.join(v.labels))}" if v.labels else "")
            + f"  runs: {s['runs']}  pass: {_pct(s['pass_rate'])}"
        )
        console.print(
            f"[dim]first seen {_day(s['first_seen'])}, last seen {_day(s['last_seen'])}[/]"
        )
        typer.echo("")
        typer.echo(v.text)


@prompts_app.command("diff")
def diff_cmd(
    a: str = typer.Argument(..., help="Baseline prompt id or label."),
    b: str = typer.Argument(..., help="Candidate prompt id or label."),
    since: str | None = typer.Option(None, "--since", help="Only runs started since."),
    agent: str | None = typer.Option(None, "--agent", help="Only this agent."),
    as_json: bool = typer.Option(False, "--json", help="Print JSON."),
) -> None:
    """Show a unified text diff of two prompt versions plus their metric deltas."""
    from .prompts import diff_prompts

    versions, _ = _scan(as_json, since, agent)
    va, vb = _one(versions, a, as_json)[0], _one(versions, b, as_json)[0]
    sa, sb = va.summary(), vb.summary()
    keys = ("runs", "pass_rate", "error_rate", "finding_rate", "avg_cost_usd",
            "avg_latency_s", "avg_tokens")  # fmt: skip
    deltas = {k: (None if sa[k] is None or sb[k] is None else sb[k] - sa[k]) for k in keys}
    text = diff_prompts(va, vb)
    if as_json:
        typer.echo(
            json.dumps({"ok": True, "a": sa, "b": sb, "delta": deltas, "diff": text}, indent=2)
        )
        return
    console = _console()
    typer.echo(text if text else "(the two prompt texts are identical as recorded)")
    table = Table(title=f"{va.id} -> {vb.id}", title_justify="left", box=None)
    for col in ("metric", va.id, vb.id, "delta"):
        table.add_column(col, justify="left" if col == "metric" else "right")
    fmt: dict[str, Any] = {
        "runs": lambda x: "-" if x is None else f"{x:.0f}",
        "pass_rate": _pct, "error_rate": _pct, "finding_rate": _pct,
        "avg_cost_usd": _usd, "avg_latency_s": _secs,
        "avg_tokens": lambda x: "-" if x is None else f"{x:.0f}",
    }  # fmt: skip
    for k in keys:
        d = deltas[k]
        if d is None:
            shown = "-"
        else:
            sign = "+" if d > 0 else "-" if d < 0 else ""
            shown = sign + (f"{abs(d) * 100:.1f}pp" if k.endswith("_rate") else fmt[k](abs(d)))
        table.add_row(k, fmt[k](sa[k]), fmt[k](sb[k]), shown)
    console.print(table)


@prompts_app.command("ab")
def ab_cmd(
    a: str = typer.Argument(..., help="Baseline prompt id or label."),
    b: str = typer.Argument(..., help="Candidate prompt id or label."),
    metric: str = typer.Option(
        "pass_rate", "--metric", "-m", help="pass_rate, cost, latency or judge:<name>."
    ),
    since: str | None = typer.Option(None, "--since", help="Only runs started since."),
    agent: str | None = typer.Option(None, "--agent", help="Only this agent."),
    alpha: float = typer.Option(0.05, "--alpha", help="Significance level (default 0.05)."),
    as_json: bool = typer.Option(False, "--json", help="Print JSON."),
    fail_on_regression: bool = typer.Option(
        False, "--fail-on-regression", help="Exit 1 when B is significantly worse than A."
    ),
    model: str | None = typer.Option(None, "--model", help="Judge model for judge:<name>."),
) -> None:
    """Compare two prompt versions on one metric with a significance test.

    pass_rate uses Fisher's exact test (two-proportion z when samples are large),
    cost, latency and judge scores use Mann-Whitney U. The verdict is about B against A:
    better, worse, or inconclusive. Exit 0 normally, 1 with --fail-on-regression when B
    is worse, 2 for an unknown version or metric, no data, or a judge failure.
    """
    from .judges import JudgeError
    from .prompts import PromptError, compare_prompts

    try:
        res = compare_prompts(
            Store(), a, b, metric=metric, since=_since(since), agent=agent,
            alpha=alpha, judge_model=model,
        )  # fmt: skip
    except (PromptError, JudgeError) as exc:
        raise _fail(str(exc), as_json) from exc
    code = 1 if fail_on_regression and res.verdict == "worse" else 0
    if as_json:
        typer.echo(json.dumps({"ok": code == 0, **res.to_dict()}, indent=2))
        raise typer.Exit(code=code)
    console = _console()
    show = {"pass_rate": _pct, "cost": _usd, "latency": _secs}.get(
        res.metric, lambda x: "-" if x is None else f"{x:.3f}"
    )
    table = Table(title=f"{res.metric}: {res.a} vs {res.b}", title_justify="left", box=None)
    for col in ("version", "ids", "runs", res.metric):
        table.add_column(col, justify="left" if col in ("version", "ids") else "right")
    table.add_row(res.a, ", ".join(res.a_ids), str(res.n_a), show(res.value_a))
    table.add_row(res.b, ", ".join(res.b_ids), str(res.n_b), show(res.value_b))
    console.print(table)
    color = {"better": "green", "worse": "red"}.get(res.verdict, "yellow")
    p = "-" if res.p_value is None else f"{res.p_value:.4f}"
    console.print(
        f"[{color}]{res.verdict.upper()}[/] ({res.b} vs {res.a})  "
        f"p={p} ({res.test}, alpha={res.alpha})"
    )
    if res.overlap_runs:
        console.print(f"[dim]{res.overlap_runs} run(s) used both versions and were left out[/]")
    if res.note:
        console.print(f"[dim]{_e(res.note)}[/]")
    raise typer.Exit(code=code)
