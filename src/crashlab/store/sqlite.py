"""SQLite index at ``.crashlab/index.sqlite``: runs, step metadata and flags (no payloads)."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 4

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entrypoint TEXT,
    status TEXT NOT NULL,
    outcome TEXT NOT NULL,
    outcome_reason TEXT,
    parent_run_id TEXT,
    fork_at INTEGER,
    replay_of TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    code_version TEXT,
    steps INTEGER NOT NULL DEFAULT 0,
    llm_calls INTEGER NOT NULL DEFAULT 0,
    tool_calls INTEGER NOT NULL DEFAULT 0,
    errors INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    flag_count INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL,
    max_severity TEXT,
    run_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_name ON runs(name);
CREATE TABLE IF NOT EXISTS steps (
    run_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    span_id TEXT NOT NULL,
    parent_span_id TEXT,
    operation TEXT NOT NULL,
    name TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    tool_call_id TEXT,
    finish_reason TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    error_type TEXT,
    error_message TEXT,
    started_at TEXT NOT NULL,
    duration_ms REAL NOT NULL,
    source TEXT NOT NULL,
    PRIMARY KEY (run_id, idx)
);
CREATE TABLE IF NOT EXISTS flags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    step_idx INTEGER,
    detector TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS flags_run ON flags(run_id);
CREATE TABLE IF NOT EXISTS diagnoses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    diagnosis_json TEXT NOT NULL
);
"""


_RANK = {"info": 0, "warn": 1, "error": 2, "critical": 3}


def _flag_row(f: dict[str, Any]) -> tuple[Any, ...]:
    """A finding (or a pre-v3 flag) as a flags-table row (the column names predate
    findings: detector = kind, step_idx = step)."""
    step = f.get("step", f.get("step_idx"))
    return (step, f.get("kind") or f.get("detector") or "?", f.get("severity", "warn"),
            f.get("message", ""))  # fmt: skip


# Serialises schema setup across SQLiteIndex instances in this process; other processes
# are covered by _retry_locked and the idempotent migrations in _init_schema.
_INIT_LOCK = threading.Lock()


def _retry_locked(fn: Callable[[], None], timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    delay = 0.01
    while True:
        try:
            fn()
            return
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc) or time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.25)


def _add_column(conn: sqlite3.Connection, table: str, column: str, decl: str) -> None:
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    except sqlite3.OperationalError as exc:  # another process migrated first
        if "duplicate column" not in str(exc):
            raise


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(runs)")}
    if "replay_of" not in cols:  # schema v1 -> v2
        _add_column(conn, "runs", "replay_of", "TEXT")
    if "cost_usd" not in cols:  # v2 -> v3: cost and findings severity for `ls`
        _add_column(conn, "runs", "cost_usd", "REAL")
        _add_column(conn, "runs", "max_severity", "TEXT")
    dcols = {r["name"] for r in conn.execute("PRAGMA table_info(diagnoses)")}
    if "cache_key" not in dcols:  # v3 -> v4: `crashlab diagnose` cache
        _add_column(conn, "diagnoses", "cache_key", "TEXT")
        _add_column(conn, "diagnoses", "model", "TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS diagnoses_key ON diagnoses(run_id, cache_key)")
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()


class StoreError(OSError):
    """The store directory or its index cannot be used (not a directory, no permission,
    a corrupt ``index.sqlite`` ...)."""


class SQLiteIndex:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path, timeout=30)
        except (OSError, sqlite3.Error) as exc:
            raise StoreError(f"cannot use crashlab store {self.path.parent}: {exc}") from exc
        conn.row_factory = sqlite3.Row
        if not self._initialized:
            try:
                # Two runs starting at once on a fresh store (agents in threads) used to
                # race here: `PRAGMA journal_mode=WAL` does not honour the busy timeout,
                # so one of them failed with "database is locked" before recording.
                with _INIT_LOCK:
                    if not self._initialized:
                        _retry_locked(lambda: _init_schema(conn))
                        self._initialized = True
            except sqlite3.DatabaseError as exc:
                conn.close()
                if isinstance(exc, sqlite3.OperationalError) and "locked" in str(exc):
                    raise
                raise StoreError(
                    f"{self.path} is not a usable crashlab index ({exc}); move it away "
                    "and crashlab starts a new one"
                ) from exc
            except BaseException:
                conn.close()
                raise
        return conn

    def exists(self) -> bool:
        return self.path.exists()

    def upsert_run(self, run: dict[str, Any]) -> None:
        usage = run.get("usage") or {}
        findings = list(run.get("findings") or run.get("flags") or [])
        row = {
            "run_id": run["run_id"],
            "name": run["name"],
            "entrypoint": run.get("entrypoint"),
            "status": run["status"],
            "outcome": run["outcome"],
            "outcome_reason": run.get("outcome_reason"),
            # index columns predate the Run field names forked_from / fork_step
            "parent_run_id": run.get("forked_from"),
            "fork_at": run.get("fork_step"),
            "replay_of": run.get("replay_of"),
            "started_at": run["started_at"],
            "ended_at": run.get("ended_at"),
            "code_version": run.get("code_version"),
            "steps": usage.get("steps", 0),
            "llm_calls": usage.get("llm_calls", 0),
            "tool_calls": usage.get("tool_calls", 0),
            "errors": usage.get("errors", 0),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "flag_count": len(findings),
            "cost_usd": usage.get("cost_usd"),
            "max_severity": max(
                (f.get("severity", "info") for f in findings),
                key=lambda sev: _RANK.get(sev, 0),
                default=None,
            ),
            "run_json": json.dumps(run, ensure_ascii=False, default=str),
        }
        cols = ", ".join(row)
        params = ", ".join(f":{c}" for c in row)
        updates = ", ".join(f"{c}=excluded.{c}" for c in row if c != "run_id")
        with closing(self._connect()) as conn, conn:
            conn.execute(
                f"INSERT INTO runs ({cols}) VALUES ({params}) "
                f"ON CONFLICT(run_id) DO UPDATE SET {updates}",
                row,
            )
            conn.execute("DELETE FROM flags WHERE run_id = ?", (run["run_id"],))
            conn.executemany(
                "INSERT INTO flags (run_id, step_idx, detector, severity, message) "
                "VALUES (?, ?, ?, ?, ?)",
                [(run["run_id"], *_flag_row(f)) for f in findings],
            )

    def insert_step(self, step: dict[str, Any]) -> None:
        attrs = step.get("attributes") or {}
        err = step.get("error") or {}
        reasons = attrs.get("gen_ai.response.finish_reasons") or []
        row = {
            "run_id": step["run_id"],
            "idx": step["idx"],
            "span_id": step["span_id"],
            "parent_span_id": step.get("parent_span_id"),
            "operation": step["operation"],
            "name": step["name"],
            "request_hash": step["request_hash"],
            "tool_call_id": attrs.get("gen_ai.tool.call.id"),
            "finish_reason": reasons[0] if reasons else None,
            "input_tokens": attrs.get("gen_ai.usage.input_tokens"),
            "output_tokens": attrs.get("gen_ai.usage.output_tokens"),
            "error_type": err.get("type"),
            "error_message": err.get("message"),
            "started_at": step["started_at"],
            "duration_ms": step.get("duration_ms", 0.0),
            "source": step.get("source", "live"),
        }
        cols = ", ".join(row)
        params = ", ".join(f":{c}" for c in row)
        with closing(self._connect()) as conn, conn:
            conn.execute(f"INSERT OR REPLACE INTO steps ({cols}) VALUES ({params})", row)

    def insert_flag(self, run_id: str, flag: dict[str, Any]) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO flags (run_id, step_idx, detector, severity, message) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, *_flag_row(flag)),
            )

    def list_runs(
        self, limit: int | None = 20, name: str | None = None, *, flagged: str | None = None
    ) -> list[dict[str, Any]]:
        """Newest first. ``flagged``: only runs with a finding of at least that severity."""
        if not self.exists():
            return []
        sql = "SELECT * FROM runs"
        where, args = [], []
        if name:
            where.append("name = ?")
            args.append(name)
        if flagged:
            levels = [k for k, v in _RANK.items() if v >= _RANK.get(flagged, 0)]
            where.append(f"max_severity IN ({', '.join('?' for _ in levels)})")
            args += levels
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY run_id DESC"
        if limit:
            sql += " LIMIT ?"
            args.append(limit)
        with closing(self._connect()) as conn:
            return [dict(r) for r in conn.execute(sql, args)]

    def get_run_json(self, run_id: str) -> dict[str, Any] | None:
        if not self.exists():
            return None
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT run_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return json.loads(row["run_json"]) if row else None

    def find_run_ids(self, prefix: str) -> list[str]:
        if not self.exists():
            return []
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT run_id FROM runs WHERE run_id LIKE ? ORDER BY run_id DESC",
                (prefix.upper() + "%",),
            ).fetchall()
        return [r["run_id"] for r in rows]

    def latest_run_id(self, name: str | None = None) -> str | None:
        rows = self.list_runs(limit=1, name=name)
        return rows[0]["run_id"] if rows else None

    def steps(self, run_id: str) -> list[dict[str, Any]]:
        if not self.exists():
            return []
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM steps WHERE run_id = ? ORDER BY idx", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def flags(self, run_id: str) -> list[dict[str, Any]]:
        if not self.exists():
            return []
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM flags WHERE run_id = ? ORDER BY id", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # diagnoses -----------------------------------------------------------------------
    def insert_diagnosis(self, diagnosis: dict[str, Any], cache_key: str | None) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO diagnoses(run_id, created_at, diagnosis_json, cache_key, model) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    diagnosis["run_id"],
                    str(diagnosis.get("created_at") or ""),
                    json.dumps(diagnosis, ensure_ascii=False, default=str),
                    cache_key,
                    diagnosis.get("model"),
                ),
            )
            conn.commit()

    def find_diagnosis(self, run_id: str, cache_key: str) -> dict[str, Any] | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT diagnosis_json FROM diagnoses WHERE run_id = ? AND cache_key = ? "
                "ORDER BY id DESC LIMIT 1",
                (run_id, cache_key),
            ).fetchone()
        return json.loads(row["diagnosis_json"]) if row else None

    def diagnoses(self, run_id: str) -> list[dict[str, Any]]:
        """Every stored diagnosis of ``run_id``, oldest first."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT diagnosis_json FROM diagnoses WHERE run_id = ? ORDER BY id", (run_id,)
            ).fetchall()
        return [json.loads(r["diagnosis_json"]) for r in rows]
