"""Local run store: ``.crashlab/index.sqlite`` + ``.crashlab/runs/<run_id>/cassette.jsonl``.

The store directory defaults to ``./.crashlab`` and can be changed with ``CRASHLAB_DIR``.
Everything is redacted (see :mod:`crashlab.store.redact`) before it is written.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..models import Diagnosis, Finding, Run, Step
from .cassette import CassetteWriter, read_cassette, step_to_line
from .redact import get_redactor
from .sqlite import SQLiteIndex, StoreError

ENV_DIR = "CRASHLAB_DIR"


class RunNotFoundError(LookupError):
    pass


@dataclass
class PruneReport:
    """What ``Store.prune`` deleted or would delete."""

    deleted: list[str] = field(default_factory=list)
    would_delete: list[str] = field(default_factory=list)
    freed_bytes: int = 0
    kept: int = 0
    kept_failed: int = 0
    kept_tagged: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "deleted": list(self.deleted),
            "would_delete": list(self.would_delete),
            "freed_bytes": self.freed_bytes,
            "kept": self.kept,
            "kept_failed": self.kept_failed,
            "kept_tagged": self.kept_tagged,
        }


_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}")


def norm_tag(value: str) -> str:
    """A single tag: strip, reject empty / whitespace / control characters."""
    if not isinstance(value, str):
        raise ValueError(f"not a valid tag: {value!r}")
    tag = value.strip()
    if not tag or not _TAG.fullmatch(tag):
        raise ValueError(
            f"not a valid tag: {value!r} (letters, digits, '.', '_', ':', '/', '-' "
            "and a letter or digit first)"
        )
    return tag


# generated ids are 26-char ULIDs; tests and older tools use short names. Letters, digits,
# ".", "_" and "-", starting with a letter or digit (so never "." / ".." / "-x").
_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


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
        """``<root>/runs/<run_id>``. A run id is one plain path component: an id taken
        from a file (``crashlab import`` of a shared export) can never point outside the
        store (``../x``, ``/abs``, ``C:x``, NUL)."""
        if not isinstance(run_id, str) or not _SAFE_RUN_ID.fullmatch(run_id):
            from .compact import safe_text

            raise ValueError(f"not a valid run id: {safe_text(repr(run_id), 80)}")
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

    def add_flag(self, run_id: str, flag: Finding) -> None:
        self.index.insert_flag(run_id, get_redactor().redact(flag.model_dump(mode="json")))

    def update_run(self, run: Run) -> None:
        """Rewrite a stored run's metadata (e.g. after re-running checks)."""
        self._write_run(run)

    def end_run(self, run: Run) -> None:
        self._write_run(run)
        with self._lock:
            self._writers.pop(run.run_id, None)

    def save_diagnosis(self, run: Run, diagnosis: Diagnosis, cache_key: str | None = None) -> None:
        """Attach ``diagnosis`` to the stored run (run.json and the index) and keep it in
        the diagnoses history, keyed for the `crashlab diagnose` cache."""
        run.diagnosis = diagnosis
        self._write_run(run)
        data = get_redactor().redact(diagnosis.model_dump(mode="json"))
        self.index.insert_diagnosis(data, cache_key)

    def cached_diagnosis(self, run_id: str, cache_key: str) -> Diagnosis | None:
        data = self.index.find_diagnosis(run_id, cache_key)
        return Diagnosis.model_validate(data) if data is not None else None

    # reading -------------------------------------------------------------------------
    def list_runs(
        self,
        limit: int | None = 20,
        name: str | None = None,
        *,
        flagged: str | None = None,
        trace: str | None = None,
        tag: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.index.list_runs(limit=limit, name=name, flagged=flagged, trace=trace, tag=tag)

    def child_runs(self, run_id: str) -> list[dict[str, Any]]:
        """Index rows of the runs linked to ``run_id`` as their parent, oldest first."""
        return self.index.list_runs(limit=None, parent=run_id)[::-1]

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

    def _load_meta(self, run_id: str) -> Run:
        """The run row without its cassette (tag / prune only need metadata)."""
        data = self.index.get_run_json(run_id)
        if data is None:
            path = self.run_dir(run_id) / "run.json"
            if not path.is_file():
                raise RunNotFoundError(f"no run matching {run_id!r} in {self.root}")
            data = json.loads(path.read_text("utf-8"))
        return Run.model_validate(data)

    def tag_run(
        self, run_id: str, add: Iterable[str] = (), remove: Iterable[str] = ()
    ) -> list[str]:
        """Add and/or remove tags on ``run_id``. Returns the updated, deduped list
        (existing order is preserved; new tags are appended)."""
        run_id = self.resolve_run_id(run_id)
        run = self._load_meta(run_id)
        tags = list(run.tags)
        for raw in add:
            tag = norm_tag(raw)
            if tag not in tags:
                tags.append(tag)
        if remove:
            drop = {norm_tag(raw) for raw in remove}
            tags = [t for t in tags if t not in drop]
        run.tags = tags
        self._write_run(run)
        return list(run.tags)

    def delete_run(self, run_id: str) -> None:
        """Remove ``run_id``'s directory and its SQLite rows. The id must be one plain
        path component (the same rule as ``run_dir``): ``../x`` is refused."""
        # validate before touching the index or the filesystem
        path = self.run_dir(run_id)
        root = (self.root / "runs").resolve()
        try:
            path.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError(f"not a valid run id: {run_id!r}") from exc
        in_index = self.index.get_run_json(run_id) is not None
        if not in_index and not path.exists():
            raise RunNotFoundError(f"no run matching {run_id!r} in {self.root}")
        with self._lock:
            self._writers.pop(run_id, None)
        self.index.delete_run(run_id)
        if path.exists():
            shutil.rmtree(path)

    def prune(
        self,
        *,
        older_than: datetime | None = None,
        max_runs: int | None = None,
        max_bytes: int | None = None,
        keep_failed: bool = True,
        keep_tagged: bool = True,
        dry_run: bool = True,
    ) -> PruneReport:
        """Drop old or excess runs. Default is a dry run. Failed and tagged runs are
        kept unless ``keep_failed`` / ``keep_tagged`` is false.

        ``older_than``: started before this instant. ``max_runs``: keep this many newest
        (plus any protected older ones). ``max_bytes``: keep the sum of remaining run
        directories at or under this size, deleting oldest unprotected first.
        """
        rows = self.index.list_runs(limit=None)
        # newest first already; deletion walks oldest first
        sizes = {r["run_id"]: _run_dir_size(self.run_dir(r["run_id"])) for r in rows}
        protected: set[str] = set()
        failed_ids: set[str] = set()
        tagged_ids: set[str] = set()
        for r in rows:
            rid = r["run_id"]
            tags = _row_tags(r)
            is_failed = r.get("status") == "error" or r.get("outcome") == "fail"
            if is_failed:
                failed_ids.add(rid)
                if keep_failed:
                    protected.add(rid)
            if tags:
                tagged_ids.add(rid)
                if keep_tagged:
                    protected.add(rid)

        def _started(row: dict[str, Any]) -> datetime | None:
            value = row.get("started_at")
            if isinstance(value, datetime):
                dt = value
            elif isinstance(value, str) and value:
                from .._isotime import parse_iso

                try:
                    dt = parse_iso(value)
                except ValueError:
                    return None
            else:
                return None
            if dt.tzinfo is None:
                from datetime import timezone

                dt = dt.replace(tzinfo=timezone.utc)
            return dt

        to_delete: list[str] = []
        remaining = [r["run_id"] for r in rows]  # newest first

        if older_than is not None:
            cutoff = older_than
            if cutoff.tzinfo is None:
                from datetime import timezone

                cutoff = cutoff.replace(tzinfo=timezone.utc)
            for r in reversed(rows):  # oldest first
                rid = r["run_id"]
                if rid in protected or rid in to_delete:
                    continue
                started = _started(r)
                if started is not None and started < cutoff:
                    to_delete.append(rid)

        remaining = [rid for rid in remaining if rid not in to_delete]
        if max_runs is not None and max_runs >= 0:
            extra = remaining[max_runs:]  # older than the newest max_runs
            for rid in reversed(extra):  # still delete oldest first
                if rid not in protected:
                    to_delete.append(rid)
            remaining = [rid for rid in remaining if rid not in to_delete]

        if max_bytes is not None and max_bytes >= 0:
            total = sum(sizes.get(rid, 0) for rid in remaining)
            if total > max_bytes:
                for rid in reversed(list(remaining)):  # oldest first
                    if total <= max_bytes:
                        break
                    if rid in protected:
                        continue
                    to_delete.append(rid)
                    remaining = [x for x in remaining if x != rid]
                    total -= sizes.get(rid, 0)

        # unique, oldest first (stable)
        seen: set[str] = set()
        ordered: list[str] = []
        for rid in to_delete:
            if rid not in seen:
                seen.add(rid)
                ordered.append(rid)
        freed = sum(sizes.get(rid, 0) for rid in ordered)
        remaining_ids = [r["run_id"] for r in rows if r["run_id"] not in seen]
        report = PruneReport(
            deleted=[] if dry_run else list(ordered),
            would_delete=list(ordered) if dry_run else [],
            freed_bytes=freed,
            kept=len(remaining_ids),
            kept_failed=sum(1 for rid in remaining_ids if rid in failed_ids),
            kept_tagged=sum(1 for rid in remaining_ids if rid in tagged_ids),
        )
        if not dry_run:
            for rid in ordered:
                self.delete_run(rid)
        return report


def _row_tags(row: dict[str, Any]) -> list[str]:
    raw = row.get("tags_json")
    if isinstance(raw, str) and raw:
        try:
            tags = json.loads(raw)
            if isinstance(tags, list):
                return [t for t in tags if isinstance(t, str)]
        except ValueError:
            pass
    data = row.get("run_json")
    if isinstance(data, str) and data:
        try:
            parsed = json.loads(data)
            tags = (parsed or {}).get("tags") if isinstance(parsed, dict) else None
            if isinstance(tags, list):
                return [t for t in tags if isinstance(t, str)]
        except ValueError:
            pass
    elif isinstance(row.get("tags"), list):
        return [t for t in row["tags"] if isinstance(t, str)]
    return []


def _run_dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in (".", "..")]
        for name in filenames:
            fp = Path(dirpath) / name
            try:
                if fp.is_file() and not fp.is_symlink():
                    total += fp.stat().st_size
            except OSError:
                continue
    return total


EXPORT_HEADER = "crashlab.run"
EXPORT_VERSION = 1


def export_run(store: Store, run_id: str, path: str | os.PathLike[str]) -> Path:
    """Write ``run_id`` as JSONL: a header line with the run, then its cassette lines."""
    run = store.load_run(run_id)
    red = get_redactor()
    head = {"type": EXPORT_HEADER, "version": EXPORT_VERSION, "run": store._run_dict(run)}
    lines = [head] + [step_to_line(red.redact(s.model_dump(mode="json"))) for s in run.steps]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(x, ensure_ascii=False, default=str) + "\n" for x in lines), "utf-8"
    )
    return out


def _line_hash(line: dict[str, Any]) -> str:
    from ..capture.normalize import request_hash

    request = line.get("request") or {}
    if line.get("op") == "execute_tool":  # capture/tools.py hash_input
        request = {"tool": line.get("name"), "args": request.get("args")}
    return request_hash(get_redactor().redact(request))


def rehash_line(old: dict[str, Any], new: dict[str, Any]) -> None:
    """``new`` is ``old`` with some text rewritten (export path tokens): give it the hash
    of its new request, if ``old``'s hash is the one its request produces (otherwise the
    hash came from elsewhere and is left alone)."""
    if old.get("request") == new.get("request") or not old.get("hash"):
        return
    if _line_hash(old) == old["hash"]:
        new["hash"] = _line_hash(new)


def import_run(
    store: Store, path: str | os.PathLike[str], *, replace: bool = False
) -> tuple[str, bool]:
    """Add an exported run to ``store``. Returns ``(run_id, added)``; an existing run is
    left alone unless ``replace``."""
    from .cassette import line_to_step, read_lines

    lines = read_lines(Path(path))
    if not lines or lines[0].get("type") != EXPORT_HEADER:
        raise ValueError("not a crashlab export (first line must be the run header)")
    from ..export import expand_local_paths

    lines = expand_local_paths(store, lines)
    if int(lines[0].get("version", 0)) > EXPORT_VERSION:
        raise ValueError(f"export version {lines[0].get('version')} is newer than supported")
    run = Run.model_validate(lines[0]["run"])
    unknown: list[str] = []
    from . import compact
    from .compact import safe_text

    body, problems = compact.expand(lines[1:])
    if problems:
        import warnings

        warnings.warn(
            f"crashlab: {len(problems)} compact step(s) in {path} could not be expanded: "
            + compact.safe_text(" | ".join(problems[:5])),
            RuntimeWarning,
            stacklevel=2,
        )
    store.run_dir(run.run_id)  # rejects an id that is not a plain path component
    steps = [line_to_step(line, unknown) for line in body]
    # every step is stored under the run's own directory, whatever its line claims
    steps = [s if s.run_id == run.run_id else s.model_copy(update={"run_id": run.run_id})
             for s in steps]  # fmt: skip
    if unknown:
        import warnings

        warnings.warn(
            f"crashlab: {path} was written by a newer crashlab; ignored field(s) it does "
            f"not know: {safe_text(', '.join(sorted(set(unknown))))}",
            RuntimeWarning,
            stacklevel=2,
        )
    if store.index.get_run_json(run.run_id) is not None and not replace:
        return run.run_id, False
    cassette = store.cassette_path(run.run_id)
    cassette.parent.mkdir(parents=True, exist_ok=True)
    cassette.write_text("", "utf-8")
    for step in steps:
        store.add_step(step)
    store.end_run(run)
    return run.run_id, True


def load(ref: str = "last", *, store_dir: str | os.PathLike[str] | None = None) -> Run:
    """Load a recorded run (with its steps) by id, unique id prefix, or ``"last"``."""
    return Store(store_dir).load_run(ref)


__all__ = [
    "ENV_DIR",
    "PruneReport",
    "RunNotFoundError",
    "StoreError",
    "Store",
    "crashlab_dir",
    "export_run",
    "import_run",
    "load",
    "norm_tag",
]
