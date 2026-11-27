"""``crashlab judge``: LLM-as-judge semantic evaluation of recorded runs."""

from __future__ import annotations

import json
from typing import Any

import typer
from rich.text import Text

from .cli import Table, _console, _e, app
from .store import RunNotFoundError, Store


@app.command("judge")
def judge_cmd(
    run_refs: list[str] | None = typer.Argument(  # noqa: B008
        None, help="Run ids / prefixes / 'last'. Default: last."
    ),
    all_runs: bool = typer.Option(False, "--all", help="Judge every stored run (newest first)."),
    name: str | None = typer.Option(None, "--name", help="With --all: only this agent."),
    judge: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--judge",
        "-j",
        help="faithfulness, answer_relevance, task_completion, tool_choice or safety. "
        "Repeatable. Default: all five.",
    ),
    threshold: float | None = typer.Option(
        None, "--threshold", help="Pass threshold (0–1) for every judge. Default: each judge's own."
    ),
    model: str | None = typer.Option(
        None, "--model", help="Judge model. Default: $CRASHLAB_JUDGE_MODEL or the diagnoser's."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
    fail_under: float | None = typer.Option(
        None, "--fail-under", help="Exit 1 if any judge score is below this value (0–1)."
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Ignore and do not write the judge verdict cache."
    ),
) -> None:
    """Score recorded runs with LLM-as-judge evaluators (faithfulness, relevance, tools, safety).

    Each judge asks the model configured by $CRASHLAB_JUDGE_MODEL (or
    $CRASHLAB_DIAGNOSER_MODEL), with $CRASHLAB_JUDGE_BASE_URL / _API_KEY for any
    OpenAI-compatible endpoint, and returns a 0–1 score with reasoning and evidence.
    Verdicts are cached under the store so re-runs are free and offline. Run data is sent
    to the judge model: use `crashlab redact-run` first for sensitive runs. Exit 0 if
    every judge passes, 1 if any fail (or score under --fail-under), 2 if there are no
    runs, a judge name is unknown, no model is configured, or a judge call fails.
    """
    from .judges import JudgeError, get_judge, judge_run, resolve_judge_config

    console = _console(stderr=as_json)

    def fail(msg: str, code: int = 2) -> typer.Exit:
        if as_json:
            typer.echo(json.dumps({"ok": False, "error": msg, "results": []}, indent=2))
        else:
            console.print(f"[red]{_e(msg)}[/]")
        return typer.Exit(code=code)

    for label, value in (("--threshold", threshold), ("--fail-under", fail_under)):
        if value is not None and not 0.0 <= value <= 1.0:
            raise fail(f"{label} must be between 0 and 1")
    try:
        from .judges import BUILTIN_JUDGES

        judges = [get_judge(j, threshold) for j in (judge or list(BUILTIN_JUDGES))]
    except ValueError as exc:
        raise fail(str(exc)) from exc
    if not resolve_judge_config(model)[0]:
        from .judges import _no_model_error

        raise fail(str(_no_model_error()))

    store = Store()
    runs = []
    if all_runs:
        refs = [r["run_id"] for r in store.list_runs(limit=None, name=name)]
        if not refs:
            raise fail(f"no runs to judge in {store.root}")
    else:
        refs = list(run_refs or ["last"])
    for ref in refs:
        try:
            runs.append(store.load_run(ref))
        except RunNotFoundError as exc:
            raise fail(str(exc)) from exc

    rows: list[dict[str, Any]] = []
    try:
        for run in runs:
            results = judge_run(
                run,
                judges,
                model=model,
                store=store,
                cache=False if no_cache else None,
            )
            rows.append(
                {
                    "run_id": run.run_id,
                    "name": run.name,
                    "passed": all(r.passed for r in results),
                    "judges": [r.to_dict() for r in results],
                }
            )
    except JudgeError as exc:
        raise fail(str(exc)) from exc

    scores = [j["score"] for row in rows for j in row["judges"] if not j["skipped"]]
    under = fail_under is not None and any(s < fail_under for s in scores)
    failed = sum(1 for row in rows if not row["passed"])
    ok = failed == 0 and not under
    if as_json:
        typer.echo(
            json.dumps(
                {
                    "ok": ok,
                    "total": len(rows),
                    "passed": len(rows) - failed,
                    "failed": failed,
                    "judges": [j.name for j in judges],
                    "results": rows,
                },
                indent=2,
                default=str,
            )
        )
    else:
        _print_report(rows, failed)
    if not ok:
        raise typer.Exit(code=1)


def _print_report(rows: list[dict[str, Any]], failed: int) -> None:
    console = _console()
    table = Table(title="crashlab judge", title_justify="left")
    table.add_column("run", no_wrap=True, style="cyan")
    table.add_column("judge", no_wrap=True)
    table.add_column("score", justify="right", no_wrap=True)
    table.add_column("result", no_wrap=True)
    table.add_column("reasoning", overflow="fold")
    for row in rows:
        for j in row["judges"]:
            if j["skipped"]:
                result = Text("n/a", style="dim")
            else:
                result = Text(
                    "pass" if j["passed"] else "fail", style="green" if j["passed"] else "red"
                )
            note = j["reasoning"] + (" (cached)" if j["cached"] else "")
            table.add_row(row["run_id"], j["name"], f"{j['score']:.2f}", result, note)
    console.print(table)
    console.print(f"{len(rows) - failed} passed, {failed} failed of {len(rows)} runs")
