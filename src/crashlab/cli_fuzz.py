"""``crashlab dataset`` and ``crashlab fuzz`` registered on the main app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.table import Table

from .cli import _console, _e, app
from .dataset import DatasetReport, DatasetSummary, build_dataset, inspect_dataset
from .fuzz import MUTATIONS, FuzzReport, fuzz_agent, fuzz_run
from .store import Store

_OUTCOMES = ("pass", "fail", "unknown")
_MUTATION_HELP = ", ".join(MUTATIONS)

dataset_app = typer.Typer(
    name="dataset",
    help="Build and inspect redacted golden datasets from recorded runs.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
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


def _outcome(value: str | None) -> str | None:
    if value is None:
        return None
    if value not in _OUTCOMES:
        raise typer.BadParameter(f"must be one of {', '.join(_OUTCOMES)}", param_hint="--outcome")
    return value


def _select_runs(
    store: Store,
    refs: list[str] | None,
    *,
    name: str | None,
    tag: str | None,
    outcome: str | None,
    since: str | None,
    limit: int | None,
) -> list[Any]:
    from .store import RunNotFoundError
    from .suite import collect_runs

    if refs:
        from ._runscan import parse_since
        from .suite import _wanted

        after = parse_since(since) if since else None
        loaded = []
        for ref in refs:
            try:
                loaded.append(store.load_run(ref))
            except RunNotFoundError as exc:
                _console().print(f"[red]{_e(exc)}[/]")
                raise typer.Exit(code=1) from exc
        return [
            run for run in loaded if _wanted(run, name=name, tag=tag, outcome=outcome, since=after)
        ][: limit if limit is not None and limit >= 0 else None]
    return collect_runs(store, None, name=name, tag=tag, outcome=outcome, since=since, limit=limit)


@dataset_app.command("build")
def dataset_build_cmd(
    runs: list[str] | None = typer.Argument(  # noqa: B008
        None, help="Run ids, unique prefixes, or 'last'. Default: scan the store."
    ),
    name: str | None = typer.Option(None, "--name", help="Only runs of this agent name."),
    tag: str | None = typer.Option(None, "--tag", help="Only runs with this tag."),
    since: str | None = typer.Option(
        None, "--since", help="Started within 30m / 2h / 7d / 1w, or after an ISO date."
    ),
    outcome: str | None = typer.Option(None, "--outcome", help="pass, fail or unknown."),
    out: Path = typer.Option(  # noqa: B008
        Path("dataset.jsonl"), "--out", help="Write the JSONL dataset to this path.", metavar="PATH"
    ),
    redact: bool = typer.Option(
        True, "--redact/--no-redact", help="Scrub API keys, secrets and emails (default)."
    ),
    include_tools: bool = typer.Option(
        True, "--tools/--no-tools", help="Include expected_tools from recorded tool calls."
    ),
    limit: int | None = typer.Option(None, "--limit", "-n", help="Max runs to export."),
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
) -> None:
    """Turn recorded runs into a portable JSONL dataset of golden test cases."""
    console = _console(stderr=as_json)
    store = Store()
    since_val = _since(since)
    outcome_val = _outcome(outcome)
    try:
        selected = _select_runs(
            store,
            runs,
            name=name,
            tag=tag,
            outcome=outcome_val,
            since=since_val,
            limit=limit,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        console.print(f"[red]{_e(exc)}[/]")
        raise typer.Exit(code=2) from exc
    if not selected:
        what = "matching those filters " if (name or tag or outcome or since or runs) else ""
        console.print(f"[red]no runs {what}to export in {_e(store.root)}[/]")
        raise typer.Exit(code=2)
    report = build_dataset(
        selected, out_path=out, redact=redact, include_tools=include_tools, store=store
    )
    if as_json:
        typer.echo(json.dumps(report.as_dict(), indent=2, default=str))
        return
    _print_build(console, report)


@dataset_app.command("inspect")
def dataset_inspect_cmd(
    path: Path = typer.Argument(..., help="Path to a JSONL dataset.", metavar="PATH"),  # noqa: B008
    as_json: bool = typer.Option(False, "--json", help="Print the summary as JSON."),
) -> None:
    """Count cases, agents, tags, tool coverage and malformed lines in a dataset."""
    console = _console(stderr=as_json)
    try:
        summary = inspect_dataset(path)
    except FileNotFoundError as exc:
        console.print(f"[red]{_e(exc)}[/]")
        raise typer.Exit(code=2) from exc
    if as_json:
        typer.echo(json.dumps(summary.as_dict(), indent=2, default=str))
        return
    _print_summary(console, summary)


def _print_build(console: Any, report: DatasetReport) -> None:
    dest = report.path or "(in memory)"
    console.print(
        f"wrote {report.cases} case(s) to {_e(dest)} "
        f"({len(report.agents)} agent(s), {len(report.tools)} tool(s), "
        f"{len(report.tags)} tag(s))"
    )


def _print_summary(console: Any, summary: DatasetSummary) -> None:
    console.print(
        f"{_e(summary.path)}: {summary.cases} case(s), {summary.malformed} malformed line(s)"
    )
    table = Table(title="coverage", title_justify="left")
    table.add_column("kind", no_wrap=True, style="cyan")
    table.add_column("name", overflow="fold")
    table.add_column("count", no_wrap=True, justify="right")
    for kind, mapping in (
        ("agent", summary.agents),
        ("tag", summary.tags),
        ("tool", summary.tools),
    ):
        if not mapping:
            table.add_row(kind, "—", "0")
            continue
        for i, (name, count) in enumerate(mapping.items()):
            table.add_row(kind if i == 0 else "", name, str(count))
    console.print(table)


app.add_typer(
    dataset_app,
    name="dataset",
    help="Build and inspect redacted golden datasets from recorded runs.",
    rich_help_panel="Share and benchmark",
    no_args_is_help=True,
)


def _looks_like_entrypoint(value: str) -> bool:
    return ":" in value and not value.startswith(".")


@app.command("fuzz")
def fuzz_cmd(
    run_or_agent: str = typer.Argument(
        "last", help="Run id / 'last', or a live entry point as module:fn."
    ),
    mutation: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--mutation",
        help=f"Mutation libraries (repeatable). One of: {_MUTATION_HELP}.",
    ),
    max_cases: int = typer.Option(10, "--max-cases", help="Maximum cases to run (default 10)."),
    seed: int = typer.Option(0, "--seed", help="Deterministic seed (default 0)."),
    agent: str | None = typer.Option(
        None, "--agent", help="Live entry point as module:fn (for a recorded run)."
    ),
    checks: list[Path] | None = typer.Option(  # noqa: B008
        None,
        "--checks",
        help="Python file or module with @crashlab.check oracles. Repeatable.",
    ),
    at_step: int | None = typer.Option(
        None, "--at-step", help="Fork at this step when fuzzing a recorded run (default 0)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
    fail_on_finding: bool = typer.Option(
        False,
        "--fail-on-finding",
        help="Exit 1 if a case crashed, failed a check, or raised a runtime finding.",
    ),
) -> None:
    """Run deterministic adversarial mutations against a recorded run or a live agent."""
    from .replay.divergence import ReplayError
    from .replay.engine import resolve_entrypoint
    from .store import RunNotFoundError

    console = _console(stderr=as_json)
    store = Store()
    if max_cases < 0:
        console.print("[red]--max-cases must be >= 0[/]")
        raise typer.Exit(code=2)
    try:
        mutations = tuple(mutation) if mutation else None
        check_mods = list(checks) if checks else None
        fn = None
        if agent:
            fn = resolve_entrypoint(agent)
        is_entry = _looks_like_entrypoint(run_or_agent)
        report: FuzzReport
        if is_entry and not agent:
            target = resolve_entrypoint(run_or_agent)
            report = fuzz_agent(
                target,
                mutations=mutations or ("prompt_injection", "edge_inputs"),
                max_cases=max_cases,
                seed=seed,
                checks=check_mods,
                store=store,
            )
        else:
            try:
                store.load_run(run_or_agent)
            except RunNotFoundError as exc:
                if is_entry:
                    target = resolve_entrypoint(run_or_agent)
                    report = fuzz_agent(
                        target,
                        mutations=mutations or ("prompt_injection", "edge_inputs"),
                        max_cases=max_cases,
                        seed=seed,
                        checks=check_mods,
                        store=store,
                    )
                else:
                    console.print(f"[red]{_e(exc)}[/]")
                    raise typer.Exit(code=1) from exc
            else:
                report = fuzz_run(
                    run_or_agent,
                    agent_fn=fn,
                    mutations=mutations or ("tool_poison",),
                    at_step=at_step,
                    max_cases=max_cases,
                    seed=seed,
                    checks=check_mods,
                    store=store,
                )
    except (ValueError, FileNotFoundError, ImportError, ReplayError, OSError) as exc:
        console.print(f"[red]{_e(exc)}[/]")
        raise typer.Exit(code=2) from exc
    if as_json:
        typer.echo(json.dumps(report.as_dict(), indent=2, default=str))
    else:
        _print_fuzz(console, report)
    if fail_on_finding and any(
        r.outcome in ("crashed", "failed_checks", "finding") or r.findings for r in report.results
    ):
        raise typer.Exit(code=1)


def _print_fuzz(console: Any, report: FuzzReport) -> None:
    table = Table(title=f"crashlab fuzz  source {report.source}", title_justify="left")
    table.add_column("mutation", no_wrap=True, style="cyan")
    table.add_column("case", overflow="fold")
    table.add_column("outcome", no_wrap=True)
    table.add_column("survived", no_wrap=True)
    table.add_column("detail", overflow="fold")
    style = {
        "survived": "green",
        "crashed": "red",
        "failed_checks": "red",
        "finding": "red",
        "skipped": "yellow",
        "needs_live_agent": "yellow",
    }
    for r in report.results:
        table.add_row(
            r.mutation,
            r.case,
            f"[{style.get(r.outcome, 'white')}]{r.outcome}[/]",
            "yes" if r.survived else "no",
            r.detail or ("· ".join(r.findings[:2]) if r.findings else (r.preview or "")),
        )
    console.print(table)
    ran = [r for r in report.results if r.outcome not in ("skipped", "needs_live_agent")]
    if not ran:
        console.print("No case could be executed (needs a live agent / client).")
    elif report.survived:
        console.print(f"[green]survived {sum(r.survived for r in ran)}/{len(ran)} executed[/]")
    else:
        bad = sum(1 for r in ran if not r.survived)
        console.print(f"[red]{bad}/{len(ran)} executed case(s) did not survive[/]")


__all__ = ["dataset_app", "dataset_build_cmd", "dataset_inspect_cmd", "fuzz_cmd"]
