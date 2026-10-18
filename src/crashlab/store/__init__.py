"""Local run store: ``.crashlab/index.sqlite`` + ``.crashlab/runs/<run_id>/cassette.jsonl``.

The store directory defaults to ``./.crashlab`` and can be changed with ``CRASHLAB_DIR``.
Everything is redacted (see :mod:`crashlab.store.redact`) before it is written.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from ..models import Flag, Run, Step
from .cassette import CassetteWriter, read_cassette, step_to_line
from .redact import get_redactor
from .sqlite import SQLiteIndex

ENV_DIR = "CRASHLAB_DIR"


class RunNotFoundError(LookupError):
    pass


def crashlab_dir() -> Path:
    """Resolve the store directory (``$CRASHLAB_DIR`` or ``./.crashlab``)."""
    return Path(os.environ.get(ENV_DIR) or ".crashlab").expanduser().resolve()


class Store:
    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.root = Path(root).expanduser().resolve() if root else crashlab_dir()
        self.index = SQLiteIndex(self.root / "index.sqlite")
        self._writers: dict[str, CassetteWriter] = {}
        self._lock = threading.Lock()

    # paths ---------------------------------------------------------------------------
    def run_dir(self, run_id: str) -> Path:
        return self.root / "runs" / run_id

    def cassette_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "cassette.jsonl"

    # writing -------------------------------------------------------------------------
    def _writer(self, run_id: str) -> CassetteWriter:
        with self._lock:
            w = self._writers.get(run_id)
            if w is None:
                w = self._writers[run_id] = CassetteWriter(self.cassette_path(run_id))
            return w

    def _run_dict(self, run: Run) -> dict[str, Any]:
        return get_redactor().redact(run.model_dump(mode="json", exclude={"steps"}))

    def _write_run(self, run: Run) -> None:
        data = self._run_dict(run)
        self.index.upsert_run(data)
        path = self.run_dir(run.run_id) / "run.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), "utf-8")
        tmp.replace(path)

    def begin_run(self, run: Run) -> None:
        self._writer(run.run_id)
        self._write_run(run)

    def add_step(self, step: Step) -> None:
        data = get_redactor().redact(step.model_dump(mode="json"))
        self._writer(step.run_id).append(step_to_line(data))
        self.index.insert_step(data)

    def add_flag(self, run_id: str, flag: Flag) -> None:
        self.index.insert_flag(run_id, get_redactor().redact(flag.model_dump(mode="json")))

    def update_run(self, run: Run) -> None:
        """Rewrite a stored run's metadata (e.g. after re-running checks)."""
        self._write_run(run)

    def end_run(self, run: Run) -> None:
        self._write_run(run)
        with self._lock:
            self._writers.pop(run.run_id, None)

    # reading -------------------------------------------------------------------------
    def list_runs(self, limit: int | None = 20, name: str | None = None) -> list[dict[str, Any]]:
        return self.index.list_runs(limit=limit, name=name)

    def resolve_run_id(self, ref: str) -> str:
        """Resolve ``"last"``, a full run id or a unique id prefix to a run id."""
        if ref in ("last", "latest"):
            rid = self.index.latest_run_id()
            if rid is None:
                raise RunNotFoundError(f"no runs recorded yet in {self.root}")
            return rid
        matches = self.index.find_run_ids(ref)
        if not matches:
            raise RunNotFoundError(f"no run matching {ref!r} in {self.root}")
        if len(matches) > 1 and ref.upper() not in matches:
            raise RunNotFoundError(f"run id prefix {ref!r} is ambiguous ({len(matches)} matches)")
        return ref.upper() if ref.upper() in matches else matches[0]

    def load_run(self, ref: str) -> Run:
        run_id = self.resolve_run_id(ref)
        data = self.index.get_run_json(run_id)
        if data is None:  # fall back to the portable run.json
            data = json.loads((self.run_dir(run_id) / "run.json").read_text("utf-8"))
        run = Run.model_validate(data)
        run.steps = read_cassette(self.cassette_path(run_id))
        return run


def load(ref: str = "last", *, store_dir: str | os.PathLike[str] | None = None) -> Run:
    """Load a recorded run (with its steps) by id, unique id prefix, or ``"last"``."""
    return Store(store_dir).load_run(ref)


__all__ = ["ENV_DIR", "RunNotFoundError", "Store", "crashlab_dir", "load"]
