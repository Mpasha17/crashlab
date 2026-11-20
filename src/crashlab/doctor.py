"""Operational diagnostics for a crashlab store: ``crashlab doctor`` / :func:`run_doctor`."""

from __future__ import annotations

import importlib.util
import os
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .pricing import price_for, provider_of
from .store import Store, crashlab_dir
from .store.redact import get_redactor
from .store.sqlite import SCHEMA_VERSION

__all__ = ["DoctorCheck", "DoctorReport", "OPTIONAL_PACKAGES", "run_doctor"]

CheckStatus = Literal["ok", "warn", "error"]

# Optional extras and nearby frameworks: reported if present, never imported (a broken
# install must not crash the doctor).
OPTIONAL_PACKAGES: tuple[tuple[str, str], ...] = (
    ("openai", "openai"),
    ("anthropic", "anthropic"),
    ("litellm", "litellm"),
    ("langchain", "langchain"),
    ("mcp", "mcp"),
    ("pydantic_ai", "pydantic_ai"),
    ("pytest", "pytest"),
    ("pyyaml", "yaml"),
)

_ENV_VARS = (
    "CRASHLAB_DIR",
    "CRASHLAB_MODE",
    "CRASHLAB_REDACT",
    "CRASHLAB_DIAGNOSER_MODEL",
    "CRASHLAB_DIAGNOSER_BASE_URL",
    "CRASHLAB_DIAGNOSER_API_KEY",
    "CRASHLAB_NO_PYPROJECT",
    "CRASHLAB_THREAD_FALLBACK",
    "CRASHLAB_EXPERIMENTAL_COMPACT",
    "CRASHLAB_PARENT_RUN",
    "CRASHLAB_TRACE_ID",
)

_SECRET_ENV = frozenset({"CRASHLAB_DIAGNOSER_API_KEY"})
_REDACT_OFF = frozenset({"0", "false", "off", "no", "disable", "disabled"})
_DB_SUFFIXES = ("", "-wal", "-shm")
_RECENT = 50


@dataclass
class DoctorCheck:
    """One diagnostic check in a :class:`DoctorReport`."""

    name: str
    status: CheckStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    fixed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DoctorReport:
    """Structured result of :func:`run_doctor`."""

    ok: bool
    checks: list[DoctorCheck] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    summary: str = ""
    fixes_applied: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["checks"] = [c.to_dict() if isinstance(c, DoctorCheck) else c for c in self.checks]
        return data


def run_doctor(
    store_dir: str | os.PathLike[str] | None = None, *, fix: bool = False
) -> DoctorReport:
    """Inspect the crashlab store, optional packages, redaction and pricing coverage.

    ``store_dir`` defaults to ``$CRASHLAB_DIR`` or ``./.crashlab``. ``fix=True`` runs
    only safe repairs: create an empty store if it is missing, and ``VACUUM`` /
    ``REINDEX`` an existing SQLite index. User runs are never deleted.
    """
    root = Path(store_dir).expanduser().resolve() if store_dir else crashlab_dir()
    checks: list[DoctorCheck] = []
    warnings: list[str] = []
    errors: list[str] = []
    fixes: list[str] = []

    def add(
        name: str,
        status: CheckStatus,
        message: str,
        details: dict[str, Any] | None = None,
        *,
        fixed: bool = False,
    ) -> DoctorCheck:
        item = DoctorCheck(name, status, message, details or {}, fixed=fixed)
        checks.append(item)
        if status == "warn":
            warnings.append(message)
        elif status == "error":
            errors.append(message)
        return item

    db_path = root / "index.sqlite"
    _check_store(root, db_path, fix=fix, add=add, fixes=fixes)
    _check_sqlite(root, db_path, fix=fix, add=add, fixes=fixes)
    _check_orphans_and_last(root, db_path, add=add)
    _check_packages(add=add)
    _check_redaction_and_env(add=add)
    _check_pricing(root, db_path, add=add)

    n_ok = sum(1 for c in checks if c.status == "ok")
    n_warn, n_err = len(warnings), len(errors)
    bits = [f"{n_ok} ok"]
    if n_warn:
        bits.append(f"{n_warn} warning{'s' if n_warn != 1 else ''}")
    if n_err:
        bits.append(f"{n_err} error{'s' if n_err != 1 else ''}")
    if fixes:
        bits.append(f"{len(fixes)} fix{'es' if len(fixes) != 1 else ''} applied")
    summary = "store check: " + ", ".join(bits)
    return DoctorReport(
        ok=n_err == 0,
        checks=checks,
        warnings=warnings,
        errors=errors,
        summary=summary,
        fixes_applied=fixes,
    )


def _check_store(
    root: Path,
    db_path: Path,
    *,
    fix: bool,
    add: Any,
    fixes: list[str],
) -> None:
    if root.exists() and not root.is_dir():
        add(
            "store",
            "error",
            f"{root} exists but is not a directory",
            {"path": str(root)},
        )
        return
    if not root.exists() or not db_path.exists():
        if not root.exists():
            msg = f"store directory {root} does not exist"
        elif not any(root.iterdir()) if root.is_dir() else True:
            msg = f"store directory {root} is empty (no index.sqlite)"
        else:
            msg = f"store index {db_path} is missing"
        if fix:
            try:
                _init_store(root)
            except OSError as exc:
                add(
                    "store",
                    "error",
                    f"could not initialize store {root}: {exc}",
                    {"path": str(root)},
                )
                return
            fixes.append(f"initialized empty store at {root}")
            add(
                "store",
                "ok",
                f"initialized empty store at {root}",
                {"path": str(root), "initialized": True},
                fixed=True,
            )
            return
        add("store", "warn", msg + " (pass --fix to initialize)", {"path": str(root)})
        return
    add("store", "ok", f"store directory {root}", {"path": str(root)})


def _init_store(root: Path) -> None:
    """Create the store directory and an empty SQLite index. Never deletes files."""
    root.mkdir(parents=True, exist_ok=True)
    # list_runs short-circuits when the file is missing; opening the index creates it.
    store = Store(root)
    with closing(store.index._connect()):
        pass


def _check_sqlite(
    root: Path,
    db_path: Path,
    *,
    fix: bool,
    add: Any,
    fixes: list[str],
) -> None:
    if not db_path.exists():
        add("sqlite", "ok", "no SQLite index yet (empty or missing store)", {"path": str(db_path)})
        add("schema", "ok", "schema not created yet", {"expected": SCHEMA_VERSION})
        return
    db_size = _db_size(db_path)
    blob_size = _blob_size(root, db_path)
    details: dict[str, Any] = {
        "path": str(db_path),
        "db_bytes": db_size,
        "blob_bytes": blob_size,
        "run_count": 0,
        "integrity": None,
        "schema_version": None,
    }
    try:
        integrity, schema, run_count = _inspect_db(db_path)
    except sqlite3.Error as exc:
        add(
            "sqlite",
            "error",
            f"index {db_path} is not a usable SQLite database ({exc})",
            {**details, "error": str(exc)},
        )
        add("schema", "error", "could not read schema version", {"expected": SCHEMA_VERSION})
        return
    details["integrity"] = integrity
    details["schema_version"] = schema
    details["run_count"] = run_count
    details["db_bytes_human"] = _human(db_size)
    details["blob_bytes_human"] = _human(blob_size)

    if integrity != "ok":
        add(
            "sqlite",
            "error",
            f"PRAGMA integrity_check failed: {integrity}",
            details,
        )
    else:
        msg = (
            f"index healthy: {run_count} run{'s' if run_count != 1 else ''}, "
            f"db {_human(db_size)}, cassettes {_human(blob_size)}"
        )
        if fix:
            try:
                _vacuum_reindex(db_path)
            except sqlite3.Error as exc:
                add("sqlite", "warn", f"{msg}; VACUUM/REINDEX failed ({exc})", details)
            else:
                fixes.append(f"VACUUM/REINDEX {db_path}")
                details = {**details, "db_bytes": _db_size(db_path)}
                details["db_bytes_human"] = _human(details["db_bytes"])
                add("sqlite", "ok", msg + "; VACUUM/REINDEX applied", details, fixed=True)
                _check_schema(schema, add=add)
                return
        else:
            add("sqlite", "ok", msg, details)

    _check_schema(schema, add=add)


def _check_schema(schema: int | None, *, add: Any) -> None:
    if schema is None:
        add(
            "schema",
            "warn",
            f"schema version is missing (expected {SCHEMA_VERSION})",
            {"expected": SCHEMA_VERSION, "found": None},
        )
        return
    if schema == SCHEMA_VERSION:
        add("schema", "ok", f"schema version {schema}", {"version": schema})
        return
    if schema < SCHEMA_VERSION:
        add(
            "schema",
            "warn",
            f"schema version {schema} is older than {SCHEMA_VERSION} "
            "(it will be migrated on the next write)",
            {"found": schema, "expected": SCHEMA_VERSION},
        )
        return
    add(
        "schema",
        "warn",
        f"schema version {schema} is newer than this crashlab ({SCHEMA_VERSION})",
        {"found": schema, "expected": SCHEMA_VERSION},
    )


def _inspect_db(db_path: Path) -> tuple[str, int | None, int]:
    """Read-only integrity / schema / run count. Does not create or migrate the file."""
    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30)
    try:
        row = conn.execute("PRAGMA integrity_check").fetchone()
        integrity = str(row[0]) if row else "failed"
        schema: int | None = None
        try:
            meta = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            if meta and str(meta[0]).strip().isdigit():
                schema = int(meta[0])
        except sqlite3.Error:
            schema = None
        try:
            run_count = int(conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0])
        except sqlite3.Error:
            run_count = 0
        return integrity, schema, run_count
    finally:
        conn.close()


def _vacuum_reindex(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.execute("VACUUM")
        conn.execute("REINDEX")
        conn.commit()
    finally:
        conn.close()


def _check_orphans_and_last(root: Path, db_path: Path, *, add: Any) -> None:
    if not db_path.exists():
        add("orphans", "ok", "no runs to check for orphaned parents", {"orphans": []})
        add("last_run", "ok", "no last run (empty store)", {})
        return
    try:
        orphans, last_id, last_issue = _orphans_and_last(root, db_path)
    except sqlite3.Error as exc:
        add("orphans", "error", f"could not scan runs for orphans ({exc})", {"error": str(exc)})
        add("last_run", "error", f"could not resolve the last run ({exc})", {"error": str(exc)})
        return
    if orphans:
        preview = ", ".join(f"{cid} -> {pid}" for cid, pid, _ in orphans[:5])
        extra = f" (+{len(orphans) - 5} more)" if len(orphans) > 5 else ""
        add(
            "orphans",
            "warn",
            f"{len(orphans)} orphaned child run(s): {preview}{extra}",
            {
                "orphans": [
                    {"run_id": cid, "missing_parent": pid, "via": via} for cid, pid, via in orphans
                ]
            },
        )
    else:
        add("orphans", "ok", "no orphaned child runs", {"orphans": []})

    if last_id is None:
        add("last_run", "ok", "no last run (empty store)", {})
    elif last_issue:
        add(
            "last_run",
            "warn",
            f"last run {last_id} is dangling: {last_issue}",
            {"run_id": last_id, "issue": last_issue},
        )
    else:
        add("last_run", "ok", f"last run {last_id} is present", {"run_id": last_id})


def _orphans_and_last(
    root: Path, db_path: Path
) -> tuple[list[tuple[str, str, str]], str | None, str | None]:
    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        rows = list(conn.execute("SELECT run_id, parent_run_id, replay_of, run_json FROM runs"))
        ids = {str(r["run_id"]) for r in rows}
        orphans: list[tuple[str, str, str]] = []
        for r in rows:
            rid = str(r["run_id"])
            for via, parent in _parent_refs(r):
                if parent and parent not in ids:
                    orphans.append((rid, parent, via))
        last = conn.execute("SELECT run_id FROM runs ORDER BY run_id DESC LIMIT 1").fetchone()
        last_id = str(last["run_id"]) if last else None
    finally:
        conn.close()

    last_issue: str | None = None
    if last_id:
        last_issue = _dangling_run(root, last_id)
    return orphans, last_id, last_issue


def _parent_refs(row: sqlite3.Row) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    parent = row["parent_run_id"]
    if parent:
        refs.append(("parent_run_id", str(parent)))
    cols = set(row.keys())
    replay = row["replay_of"] if "replay_of" in cols else None
    if replay:
        refs.append(("replay_of", str(replay)))
    raw = row["run_json"]
    if raw:
        import json

        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            data = None
        if isinstance(data, dict):
            forked = data.get("forked_from")
            if forked and ("parent_run_id", str(forked)) not in refs:
                refs.append(("forked_from", str(forked)))
            link = data.get("env") or {}
            if isinstance(link, dict):
                link = link.get("link") or {}
            if isinstance(link, dict):
                linked = link.get("parent_run")
                if linked and all(p != str(linked) for _, p in refs):
                    refs.append(("env.link.parent_run", str(linked)))
    return refs


def _dangling_run(root: Path, run_id: str) -> str | None:
    run_dir = root / "runs" / run_id
    if not run_dir.exists():
        return "run directory is missing"
    cassette = run_dir / "cassette.jsonl"
    meta = run_dir / "run.json"
    if not cassette.is_file() and not meta.is_file():
        return "cassette.jsonl and run.json are missing"
    if not cassette.is_file():
        return "cassette.jsonl is missing"
    return None


def _check_packages(*, add: Any) -> None:
    installed: list[str] = []
    missing: list[str] = []
    broken: list[dict[str, str]] = []
    for dist, module in OPTIONAL_PACKAGES:
        try:
            spec = importlib.util.find_spec(module)
        except Exception as exc:  # noqa: BLE001 - a broken install must not crash doctor
            broken.append({"package": dist, "module": module, "error": str(exc)})
            continue
        if spec is None:
            missing.append(dist)
        else:
            installed.append(dist)
    details = {"installed": installed, "missing": missing, "broken": broken}
    if broken:
        names = ", ".join(b["package"] for b in broken)
        add(
            "packages",
            "warn",
            f"optional package(s) failed to inspect: {names}",
            details,
        )
        return
    inst = ", ".join(installed) if installed else "none"
    miss = ", ".join(missing) if missing else "none"
    add(
        "packages",
        "ok",
        f"optional packages installed: {inst}; not installed: {miss}",
        details,
    )


def _check_redaction_and_env(*, add: Any) -> None:
    red = get_redactor()
    details = {
        "enabled": True,
        "value_patterns": len(red.value_patterns),
        "key_patterns": len(red.key_patterns),
        "hooks": len(red.hooks),
    }
    add(
        "redaction",
        "ok",
        (
            "redaction is enabled for every write "
            f"({details['value_patterns']} value patterns, "
            f"{details['key_patterns']} key patterns, {details['hooks']} hook(s))"
        ),
        details,
    )

    env: dict[str, str] = {}
    for name in _ENV_VARS:
        raw = os.environ.get(name)
        if raw is None:
            env[name] = "unset"
            continue
        env[name] = "set" if name in _SECRET_ENV else raw
    redact_val = os.environ.get("CRASHLAB_REDACT")
    if redact_val is not None and redact_val.strip().lower() in _REDACT_OFF:
        add(
            "env",
            "warn",
            "CRASHLAB_REDACT is set to disable redaction, but crashlab always redacts "
            "writes; this variable does not turn redaction off",
            {"env": env},
        )
        return
    set_names = [n for n, v in env.items() if v != "unset"]
    msg = (
        "env: " + ", ".join(f"{n}={env[n]}" for n in set_names)
        if set_names
        else "no crashlab environment variables set"
    )
    add("env", "ok", msg, {"env": env})


def _check_pricing(root: Path, db_path: Path, *, add: Any) -> None:
    if not db_path.exists():
        add("pricing", "ok", "no runs to check for unpriced models", {"unpriced": [], "models": []})
        return
    try:
        models, unpriced = _observed_models(root)
    except Exception as exc:  # noqa: BLE001 - a bad cassette must not crash doctor
        add(
            "pricing",
            "warn",
            f"could not scan recent runs for model prices ({exc})",
            {"error": str(exc)},
        )
        return
    if not models:
        add(
            "pricing",
            "ok",
            "no chat models observed in recent runs",
            {"unpriced": [], "models": []},
        )
        return
    if unpriced:
        names = ", ".join(unpriced)
        add(
            "pricing",
            "warn",
            f"unpriced model(s) in recent runs: {names} "
            "(set [tool.crashlab.prices] or crashlab.pricing.set_price)",
            {"unpriced": unpriced, "models": models},
        )
        return
    add(
        "pricing",
        "ok",
        f"pricing covers all {len(models)} observed model(s): {', '.join(models)}",
        {"unpriced": [], "models": models},
    )


def _observed_models(root: Path) -> tuple[list[str], list[str]]:
    store = Store(root)
    rows = store.list_runs(limit=_RECENT)
    seen: dict[str, None] = {}
    unpriced: dict[str, None] = {}
    for row in rows:
        try:
            run = store.load_run(row["run_id"])
        except Exception:  # noqa: BLE001 - skip unreadable runs
            continue
        for step in run.steps:
            if step.operation != "chat":
                continue
            attrs = step.attributes or {}
            model = str(
                attrs.get("gen_ai.response.model")
                or attrs.get("gen_ai.request.model")
                or step.name
                or ""
            ).strip()
            if not model:
                continue
            seen.setdefault(model, None)
            provider = provider_of(attrs, model)
            if price_for(model, provider, step.started_at) is None:
                req = attrs.get("gen_ai.request.model")
                if req and str(req) != model and price_for(str(req), provider, step.started_at):
                    continue
                unpriced.setdefault(model, None)
    return list(seen), list(unpriced)


def _db_size(db_path: Path) -> int:
    total = 0
    for suffix in _DB_SUFFIXES:
        p = Path(str(db_path) + suffix)
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def _blob_size(root: Path, db_path: Path) -> int:
    skip = {Path(str(db_path) + suffix).resolve() for suffix in _DB_SUFFIXES}
    total = 0
    if not root.exists():
        return 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in (".", "..")]
        for name in filenames:
            fp = Path(dirpath) / name
            try:
                if fp.resolve() in skip:
                    continue
                if fp.is_file() and not fp.is_symlink():
                    total += fp.stat().st_size
            except OSError:
                continue
    return total


def _human(n: int) -> str:
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            if unit == "B":
                return f"{int(size)}B"
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{n}B"
