"""SQLite index at ``.crashlab/index.sqlite``: runs, step metadata and flags (no payloads)."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

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


class SQLiteIndex:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        if not self._initialized:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            conn.commit()
            self._initialized = True
        return conn

    def exists(self) -> bool:
        return self.path.exists()

    def upsert_run(self, run: dict[str, Any]) -> None:
        usage = run.get("usage") or {}
        row = {
            "run_id": run["run_id"],
            "name": run["name"],
            "entrypoint": run.get("entrypoint"),
            "status": run["status"],
            "outcome": run["outcome"],
            "outcome_reason": run.get("outcome_reason"),
            "parent_run_id": run.get("parent_run_id"),
            "fork_at": run.get("fork_at"),
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
            "flag_count": len(run.get("flags") or []),
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
                (run_id, flag.get("step_idx"), flag["detector"], flag["severity"], flag["message"]),
            )

    def list_runs(self, limit: int | None = 20, name: str | None = None) -> list[dict[str, Any]]:
        if not self.exists():
            return []
        sql = "SELECT * FROM runs"
        args: list[Any] = []
        if name:
            sql += " WHERE name = ?"
            args.append(name)
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
