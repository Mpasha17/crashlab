"""CI evaluation commands registered on the main ``crashlab`` app: ``eval`` and ``sweep``."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import typer

from .cli import _console, _e, app
from .store import Store

_OUTCOMES = ("pass", "fail", "unknown")
_SEVERITIES = ("info", "warn", "error", "critical")


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


def _write_outputs(markdown: str, *, markdown_path: Path | None, junit: str | None = None,
                   junit_path: Path | None = None) -> None:  # fmt: skip
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown if markdown.endswith("\n") else markdown + "\n", "utf-8")
    if junit is not None and junit_path is not None:
        junit_path.parent.mkdir(parents=True, exist_ok=True)
        junit_path.write_text(junit if junit.endswith("\n") else junit + "\n", "utf-8")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        path = Path(summary)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(markdown if markdown.endswith("\n") else markdown + "\n")


def _print_report(report: Any, *, as_json: bool) -> None:
    if as_json:
        typer.echo(json.dumps(report.to_dict(), indent=2, default=str))
        return
    typer.echo(report.to_markdown())


@app.command("eval")
def eval_cmd(
    files: list[Path] | None = typer.Argument(  # noqa: B008
        None,
        help="Exported .jsonl cassettes, directories, or glob patterns. Default: the store.",
    ),
    name: str | None = typer.Option(None, "--name", help="Only runs of this agent name."),
    tag: str | None = typer.Option(None, "--tag", help="Only runs with this tag."),
    since: str | None = typer.Option(
        None, "--since", help="Started within 30m / 2h / 7d / 1w, or after an ISO date."
    ),
    outcome: str | None = typer.Option(None, "--outcome", help="pass, fail or unknown."),
    limit: int | None = typer.Option(None, "--limit", "-n", help="Max runs to evaluate."),
    checks: list[Path] | None = typer.Option(  # noqa: B008
        None,
        "--checks",
        help="Python file or module with @crashlab.check oracles. Repeatable.",
    ),
    replay: bool = typer.Option(
        False, "--replay", help="Replay each run offline before grading it."
    ),
    max_cost: float | None = typer.Option(None, "--max-cost", help="Budget: USD."),
    max_tokens: int | None = typer.Option(None, "--max-tokens", help="Budget: total tokens."),
    max_seconds: float | None = typer.Option(None, "--max-seconds", help="Budget: wall time."),
    max_steps: int | None = typer.Option(None, "--max-steps", help="Budget: steps."),
    min_finding_severity: str = typer.Option(
        "error",
        "--min-finding-severity",
        help="Fail a run if a runtime finding is at least this: info, warn, error, critical.",
    ),
    junit: Path | None = typer.Option(  # noqa: B008
        None, "--junit", help="Write JUnit XML to this file.", metavar="PATH"
    ),
    markdown: Path | None = typer.Option(  # noqa: B008
        None, "--markdown", help="Write a Markdown summary to this file.", metavar="PATH"
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
) -> None:
    """Grade many recorded runs for CI: oracles, runtime checks and optional budgets.

    Pass exported .jsonl cassettes (files, directories or globs) or, with no files, scan
    the store. Exit 0 if every run passes, 1 if any fail, 2 if there are no runs or the
    inputs are invalid. --junit and --markdown write CI reports; when GITHUB_STEP_SUMMARY
    is set the Markdown is also appended there.
    """
    from .suite import eval_runs

    console = _console(stderr=as_json)
    since_val = _since(since)
    outcome_val = _outcome(outcome)
    if min_finding_severity not in _SEVERITIES:
        console.print(f"[red]--min-finding-severity must be one of {', '.join(_SEVERITIES)}[/]")
        raise typer.Exit(code=2)
    store = Store()
    try:
        report = eval_runs(
            store,
            files=files or None,
            name=name,
            tag=tag,
            outcome=outcome_val,
            since=since_val,
            limit=limit,
            check_modules=checks or None,
            replay=replay,
            max_cost_usd=max_cost,
            max_tokens=max_tokens,
            max_seconds=max_seconds,
            max_steps=max_steps,
            min_finding_severity=min_finding_severity,
        )
    except (FileNotFoundError, ValueError, OSError, ImportError) as exc:
        console.print(f"[red]{_e(exc)}[/]")
        raise typer.Exit(code=2) from exc
    if report.total == 0:
        what = "matching those filters" if (name or tag or outcome or since or files) else ""
        console.print(f"[red]no runs {what}to evaluate in {_e(store.root)}[/]".replace("  ", " "))
        raise typer.Exit(code=2)
    _write_outputs(
        report.to_markdown(),
        markdown_path=markdown,
        junit=report.to_junit_xml(),
        junit_path=junit,
    )
    _print_report(report, as_json=as_json)
    if markdown is not None and not as_json:
        console.print(f"wrote markdown {_e(markdown)}")
    if junit is not None and not as_json:
        console.print(f"wrote JUnit {_e(junit)}")
    if report.failed or report.errored:
        raise typer.Exit(code=1)


@app.command("sweep")
def sweep_cmd(
    run_refs: list[str] | None = typer.Argument(  # noqa: B008
        None, help="Run ids / prefixes / 'last'. Default: store runs matching the filters."
    ),
    patch_file: Path | None = typer.Option(  # noqa: B008
        None,
        "--patch",
        "-p",
        help="Patch file (.json / .yaml): one patch, a list, or {patches: [...]}.",
    ),
    system_prompt: str | None = typer.Option(
        None,
        "--system-prompt",
        help="Append this text to the system prompt on every forked run.",
    ),
    at_step: int = typer.Option(
        0, "--at-step", help="Fork step: steps before this replay from the cassette."
    ),
    agent: str | None = typer.Option(
        None, "--agent", help="Apply request patches only to this agent's LLM calls."
    ),
    model: str | None = typer.Option(None, "--model", help="Use this model from the fork step on."),
    name: str | None = typer.Option(None, "--name", help="Only runs of this agent name."),
    since: str | None = typer.Option(
        None, "--since", help="Started within 30m / 2h / 7d / 1w, or after an ISO date."
    ),
    outcome: str | None = typer.Option(None, "--outcome", help="pass, fail or unknown."),
    limit: int | None = typer.Option(None, "--limit", "-n", help="Max runs to sweep."),
    max_cost: float | None = typer.Option(
        None, "--max-cost", help="Budget: USD (fail if exceeded)."
    ),
    max_tokens: int | None = typer.Option(
        None, "--max-tokens", help="Budget: total tokens (fail if exceeded)."
    ),
    markdown: Path | None = typer.Option(  # noqa: B008
        None, "--markdown", help="Write a Markdown summary to this file.", metavar="PATH"
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
    fail_on_regression: bool = typer.Option(
        False, "--fail-on-regression", help="Exit 1 if any previously passing run fails."
    ),
) -> None:
    """Fork many runs with one candidate patch and report fixes vs regressions.

    Give --patch FILE and / or --system-prompt TEXT (and / or --model). Exit 1 if
    --fail-on-regression and any run regressed, 2 if there are no runs or the patch is
    missing. When GITHUB_STEP_SUMMARY is set the Markdown is also appended there.
    """
    from .store import RunNotFoundError
    from .suite import collect_runs, patch_from_flags, sweep_runs

    console = _console(stderr=as_json)
    if patch_file is None and system_prompt is None and model is None:
        console.print("[red]give --patch FILE, --system-prompt TEXT, or --model[/]")
        raise typer.Exit(code=2)
    since_val = _since(since)
    outcome_val = _outcome(outcome)
    store = Store()
    try:
        patches = patch_from_flags(
            patch_file=patch_file, system_prompt=system_prompt, model=model, agent=agent
        )
    except (OSError, ValueError) as exc:
        console.print(f"[red]bad patch:[/] {_e(exc)}")
        raise typer.Exit(code=2) from exc
    if not patches:
        console.print("[red]the patch is empty[/]")
        raise typer.Exit(code=2)
    runs = []
    if run_refs:
        for ref in run_refs:
            try:
                runs.append(store.load_run(ref))
            except RunNotFoundError as exc:
                console.print(f"[red]{_e(exc)}[/]")
                raise typer.Exit(code=2) from exc
        if name or outcome_val or since_val or limit is not None:
            from .suite import _parse_since, _wanted

            after = _parse_since(since_val)
            runs = [
                r for r in runs if _wanted(r, name=name, tag=None, outcome=outcome_val, since=after)
            ]
            if limit is not None:
                runs = runs[:limit]
    else:
        try:
            runs = collect_runs(
                store, None, name=name, tag=None, outcome=outcome_val, since=since_val,
                limit=limit,
            )  # fmt: skip
        except ValueError as exc:
            console.print(f"[red]{_e(exc)}[/]")
            raise typer.Exit(code=2) from exc
    if not runs:
        console.print(f"[red]no runs to sweep in {_e(store.root)}[/]")
        raise typer.Exit(code=2)
    try:
        report = sweep_runs(
            runs,
            patches,
            store=store,
            model=None,  # already applied in patch_from_flags
            at_step=at_step,
            agent=None,
            max_cost_usd=max_cost,
            max_tokens=max_tokens,
        )
    except (OSError, ValueError) as exc:
        console.print(f"[red]{_e(exc)}[/]")
        raise typer.Exit(code=2) from exc
    _write_outputs(report.to_markdown(), markdown_path=markdown)
    if as_json:
        typer.echo(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        typer.echo(report.to_markdown())
        if markdown is not None:
            console.print(f"wrote markdown {_e(markdown)}")
    if fail_on_regression and report.regressions:
        raise typer.Exit(code=1)
