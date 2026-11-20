"""Store-management commands registered on the main ``crashlab`` app: ``tag``, ``rm``
and ``prune``."""

from __future__ import annotations

import json
import re
from typing import Any

import typer

from .cli import _console, _e, app
from .store import PruneReport, RunNotFoundError, Store, norm_tag

_BYTES = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|TB|KIB|MIB|GIB|TIB|K|M|G|T)?\s*$",
    re.IGNORECASE,
)
_BYTE_UNIT = {
    "B": 1,
    "K": 1024,
    "KB": 1024,
    "KIB": 1024,
    "M": 1024**2,
    "MB": 1024**2,
    "MIB": 1024**2,
    "G": 1024**3,
    "GB": 1024**3,
    "GIB": 1024**3,
    "T": 1024**4,
    "TB": 1024**4,
    "TIB": 1024**4,
}


def parse_bytes(value: str) -> int:
    """``500MB``, ``1GiB``, ``1024`` (bytes). Units are binary (1 MB = 1,048,576)."""
    m = _BYTES.match(value)
    if not m:
        raise ValueError(f"--max-bytes expects a size like 500MB, 1GiB or 1024, not {value!r}")
    amount = float(m.group(1))
    unit = (m.group(2) or "B").upper()
    return int(amount * _BYTE_UNIT[unit])


def _resolve_id(store: Store, ref: str) -> str:
    try:
        return store.resolve_run_id(ref)
    except RunNotFoundError as exc:
        _console().print(f"[red]{_e(exc)}[/]")
        raise typer.Exit(code=1) from None


def _parse_tag_args(values: list[str]) -> tuple[list[str], list[str]]:
    add, remove = [], []
    for raw in values:
        if raw.startswith("+") and len(raw) > 1:
            add.append(norm_tag(raw[1:]))
        elif raw.startswith("-") and len(raw) > 1:
            remove.append(norm_tag(raw[1:]))
        else:
            raise typer.BadParameter(
                f"tags must be +name or -name, not {raw!r}",
                param_hint="TAGS",
            )
    return add, remove


@app.command("tag", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def tag_cmd(
    ctx: typer.Context,
    run: str = typer.Argument(..., help="Run id, unique prefix, or 'last'."),
    tags: list[str] | None = typer.Argument(  # noqa: B008
        None, help="+name to add a tag, -name to remove one. Omit to print current tags."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON."),
) -> None:
    """Add or remove tags on a recorded run, or list the tags it already has."""
    console = _console()
    store = Store()
    run_id = _resolve_id(store, run)
    # -name starts with a dash, so Click would otherwise treat it as an unknown option
    values = list(tags or []) + list(ctx.args or [])
    try:
        if values:
            add, remove = _parse_tag_args(values)
            current = store.tag_run(run_id, add=add, remove=remove)
        else:
            current = list(store._load_meta(run_id).tags)
    except ValueError as exc:
        console.print(f"[red]{_e(exc)}[/]")
        raise typer.Exit(code=2) from None
    if as_json:
        typer.echo(json.dumps({"run_id": run_id, "tags": current}, ensure_ascii=False))
        return
    if not current:
        console.print(f"{_e(run_id)}: no tags")
        return
    console.print(f"{_e(run_id)}: {', '.join(_e(t) for t in current)}")


@app.command("rm")
def rm_cmd(
    runs: list[str] = typer.Argument(  # noqa: B008
        ..., help="Run ids, unique prefixes, or 'last'."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Actually delete the runs."),
) -> None:
    """Delete recorded runs from the store. Pass --yes to confirm."""
    console = _console()
    store = Store()
    ids: list[str] = []
    seen: set[str] = set()
    for ref in runs:
        rid = _resolve_id(store, ref)
        if rid not in seen:
            seen.add(rid)
            ids.append(rid)
    if not yes:
        for rid in ids:
            console.print(_e(rid))
        n = len(ids)
        what = "run" if n == 1 else "runs"
        console.print(f"Pass --yes to delete {n} {what}.")
        raise typer.Exit(code=1)
    for rid in ids:
        try:
            store.delete_run(rid)
        except (RunNotFoundError, ValueError) as exc:
            console.print(f"[red]{_e(exc)}[/]")
            raise typer.Exit(code=1) from None
        console.print(f"deleted {_e(rid)}")


def _prune_cutoff(value: str | None) -> Any:
    if not value:
        return None
    from ._runscan import parse_since

    try:
        return parse_since(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--older-than") from None


@app.command("prune")
def prune_cmd(
    older_than: str | None = typer.Option(
        None, "--older-than", help="Delete runs started before 30d / 2h / 7d, or an ISO date."
    ),
    max_runs: int | None = typer.Option(
        None, "--max-runs", help="Keep at most this many newest runs (plus protected ones)."
    ),
    max_bytes: str | None = typer.Option(
        None, "--max-bytes", help="Keep remaining run directories at or under this size (500MB)."
    ),
    include_failed: bool = typer.Option(
        False, "--include-failed", help="Allow deleting failed runs (status error or outcome fail)."
    ),
    include_tagged: bool = typer.Option(
        False, "--include-tagged", help="Allow deleting tagged runs."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be deleted without deleting (the default)."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Actually delete the selected runs."),
    as_json: bool = typer.Option(False, "--json", help="Print JSON."),
) -> None:
    """Delete old or excess recorded runs. Dry-run unless --yes is passed."""
    console = _console()
    if max_runs is not None and max_runs < 0:
        console.print("[red]--max-runs must be >= 0[/]")
        raise typer.Exit(code=2)
    cutoff = _prune_cutoff(older_than)
    budget = None
    if max_bytes is not None:
        try:
            budget = parse_bytes(max_bytes)
        except ValueError as exc:
            console.print(f"[red]{_e(exc)}[/]")
            raise typer.Exit(code=2) from None
    if cutoff is None and max_runs is None and budget is None:
        console.print("[red]prune needs --older-than, --max-runs or --max-bytes[/]")
        raise typer.Exit(code=2)
    if yes and dry_run:
        console.print("[red]pass either --yes or --dry-run, not both[/]")
        raise typer.Exit(code=2)
    store = Store()
    report: PruneReport = store.prune(
        older_than=cutoff,
        max_runs=max_runs,
        max_bytes=budget,
        keep_failed=not include_failed,
        keep_tagged=not include_tagged,
        dry_run=not yes,
    )
    if as_json:
        typer.echo(json.dumps(report.to_json(), ensure_ascii=False))
        return
    targets = report.would_delete if not yes else report.deleted
    action = "would delete" if not yes else "deleted"
    if not targets:
        console.print(
            f"Nothing to delete ({report.kept} kept, {report.kept_failed} failed, "
            f"{report.kept_tagged} tagged)."
        )
        return
    for rid in targets:
        console.print(f"{action} {_e(rid)}")
    freed = f"{report.freed_bytes:,} bytes"
    console.print(
        f"{action} {len(targets)} run(s), freeing {freed} "
        f"({report.kept} kept, {report.kept_failed} failed, {report.kept_tagged} tagged)."
    )
    if not yes:
        console.print("Pass --yes to delete.")
