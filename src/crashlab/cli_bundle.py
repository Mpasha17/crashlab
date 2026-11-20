"""Share and store commands: ``bundle``, ``unbundle``, ``audit-secrets``, ``redact-run``."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.text import Text

from .cli import Table, _console, _e, app
from .store import RunNotFoundError, Store

_SEV_STYLE = {"info": "dim", "warn": "yellow", "error": "red", "critical": "bold red"}


def _resolve_run(store: Store, ref: str, console: Console):
    try:
        return store.load_run(ref)
    except RunNotFoundError as exc:
        console.print(f"[red]{_e(exc)}[/]")
        raise typer.Exit(code=2) from exc


@app.command("bundle")
def bundle_cmd(
    run_ref: str = typer.Argument("last", help="Run id, unique prefix, or 'last'."),
    output: Path | None = typer.Option(  # noqa: B008
        None,
        "--output",
        "-o",
        help="Zip path to write (default: <run_id>.crashlab.zip).",
    ),
    no_redact: bool = typer.Option(
        False, "--no-redact", help="Leave secrets in the packed runs (not recommended)."
    ),
    no_html: bool = typer.Option(False, "--no-html", help="Omit report.html from the archive."),
) -> None:
    """Pack a recorded run into a shareable .crashlab.zip debug bundle.

    Includes the root run and linked child runs as JSONL, a diagnosis, a prompt
    / tool-schema summary, and an HTML trace report. Secrets are redacted by
    default.
    """
    from .bundle import BundleError, export_bundle

    console = _console()
    store = Store()
    try:
        run = store.load_run(run_ref)
    except RunNotFoundError as exc:
        console.print(f"[red]{_e(exc)}[/]")
        raise typer.Exit(code=2) from exc
    dest = output or Path(f"{run.run_id}.crashlab.zip")
    try:
        manifest = export_bundle(
            store,
            run.run_id,
            dest,
            redact=not no_redact,
            include_html=not no_html,
        )
    except (OSError, ValueError, BundleError) as exc:
        console.print(f"[red]{_e(exc)}[/]")
        raise typer.Exit(code=2) from exc
    n = len(manifest.run_ids)
    runs = "run" if n == 1 else "runs"
    redacted = "redacted" if manifest.redacted else "NOT redacted"
    hits = manifest.audit.get("count", 0)
    console.print(
        f"bundled {manifest.root_run_id} ({n} {runs}, {redacted}, "
        f"{hits} secret hit(s)) -> {_e(manifest.path or dest)}"
    )


@app.command("unbundle")
def unbundle_cmd(
    bundle_path: Path = typer.Argument(..., help="Path to a .crashlab.zip bundle."),  # noqa: B008
) -> None:
    """Import every run from a .crashlab.zip debug bundle into the store."""
    from .bundle import BundleError, import_bundle

    console = _console()
    store = Store()
    if not bundle_path.is_file():
        what = "is a directory" if bundle_path.is_dir() else "no such file"
        console.print(f"[red]{_e(bundle_path)}: {what}[/]")
        raise typer.Exit(code=2)
    try:
        ids = import_bundle(store, bundle_path)
    except (OSError, BundleError, ValueError) as exc:
        console.print(f"[red]{_e(bundle_path)}: {_e(exc)}[/]")
        raise typer.Exit(code=2) from exc
    for rid in ids:
        console.print(f"imported {rid}")
    console.print(f"store: {_e(store.root)}  ({len(ids)} run(s))")


@app.command("audit-secrets")
def audit_secrets_cmd(
    run_refs: list[str] | None = typer.Argument(  # noqa: B008
        None, help="Run ids, unique prefixes, or 'last'. Defaults to last."
    ),
    all_runs: bool = typer.Option(False, "--all", help="Scan every run in the store."),
    include_pii: bool = typer.Option(
        False, "--include-pii", help="Also flag email addresses and phone-like numbers."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
    fail_on_match: bool = typer.Option(
        False, "--fail-on-match", help="Exit 1 when any secret or PII hit is found (CI)."
    ),
) -> None:
    """Scan recorded runs for leaked API keys, tokens, private keys, or PII.

    Reports the field path and a redacted preview; raw secret values are never
    printed. --include-pii adds email and phone-like patterns. --fail-on-match
    exits 1 when a hit is found, for CI.
    """
    from .bundle import audit_secrets

    console = _console(stderr=as_json)
    store = Store()
    if all_runs and run_refs:
        console.print("[red]pass either --all or run ids, not both[/]")
        raise typer.Exit(code=2)
    refs: list[str]
    if all_runs:
        refs = [row["run_id"] for row in store.list_runs(limit=None)]
        if not refs:
            console.print(f"No runs recorded yet in [bold]{_e(store.root)}[/].")
            raise typer.Exit(code=1)
    else:
        refs = list(run_refs) if run_refs else ["last"]
    reports = []
    for ref in refs:
        run = _resolve_run(store, ref, console)
        reports.append(audit_secrets(run, include_pii=include_pii))
    if as_json:
        payload = reports[0].to_dict() if len(reports) == 1 else [r.to_dict() for r in reports]
        typer.echo(json.dumps(payload, indent=2, default=str))
    else:
        for report in reports:
            _print_audit(report)
    if fail_on_match and any(not report.ok for report in reports):
        raise typer.Exit(code=1)


@app.command("redact-run")
def redact_run_cmd(
    run_ref: str = typer.Argument(..., help="Run id, unique prefix, or 'last'."),
    include_pii: bool = typer.Option(
        False, "--include-pii", help="Also replace email addresses and phone-like numbers."
    ),
    in_place: bool = typer.Option(
        False,
        "--in-place",
        help="Write the redacted run back to the store (replaces the cassette).",
    ),
) -> None:
    """Replace leaked secrets in a recorded run with [REDACTED:<kind>] placeholders.

    Without --in-place this is a dry report: the store is left unchanged.
    """
    from .bundle import redact_run

    console = _console()
    store = Store()
    run = _resolve_run(store, run_ref, console)
    try:
        redacted, count = redact_run(
            run, include_pii=include_pii, store=store if in_place else None, in_place=in_place
        )
    except ValueError as exc:
        console.print(f"[red]{_e(exc)}[/]")
        raise typer.Exit(code=2) from exc
    action = "redacted in place" if in_place else "would redact"
    console.print(f"{action} {redacted.run_id}: {count} replacement(s)")


def _print_audit(report: object) -> None:
    from .bundle import SecretAuditReport

    assert isinstance(report, SecretAuditReport)
    console = _console()
    status = "ok" if report.ok else "HITS"
    pii = "  (including PII)" if report.include_pii else ""
    console.print(
        f"audit-secrets {_e(report.run_id)}  ({_e(report.name)})  "
        f"{report.count} hit(s)  {status}{pii}"
    )
    if not report.findings:
        console.print("no secrets found")
        return
    table = Table(title="findings", title_justify="left", box=None)
    table.add_column("severity", no_wrap=True)
    table.add_column("kind", style="bold", no_wrap=True)
    table.add_column("step", justify="right", no_wrap=True)
    table.add_column("field", overflow="fold")
    table.add_column("preview", overflow="fold")
    for item in report.findings:
        table.add_row(
            Text(item.severity, style=_SEV_STYLE.get(item.severity, "")),
            item.kind,
            "-" if item.step_idx is None else str(item.step_idx),
            item.field_path,
            item.redacted_preview,
        )
    console.print(table)
    kinds = ", ".join(f"{k} ×{n}" for k, n in report.counts_by_kind().items())
    console.print(f"by kind: {kinds}")
