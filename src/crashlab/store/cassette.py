"""JSONL cassettes: one line per recorded step at ``.crashlab/runs/<run_id>/cassette.jsonl``.

Line format (human-readable, diffable)::

    {"seq": 3, "op": "chat", "name": "gpt-4o-mini", "hash": "9f1c...", "request": {...},
     "response": {...}, ...}

``seq``/``op``/``hash`` are the step's ``idx``/``operation``/``request_hash``; every other
key is a :class:`crashlab.models.Step` field. Lines are appended when a step *finishes*,
so nested steps can appear out of order; :func:`read_cassette` sorts by ``seq``.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from ..models import Step

_RENAMES = {"idx": "seq", "operation": "op", "request_hash": "hash"}
_REVERSE = {v: k for k, v in _RENAMES.items()}
_FIRST = ("seq", "op", "name", "hash")


def step_to_line(step_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert a (JSON-mode) step dict into cassette line order/keys."""
    renamed = {_RENAMES.get(k, k): v for k, v in step_dict.items()}
    ordered = {k: renamed.pop(k) for k in _FIRST if k in renamed}
    ordered.update(renamed)
    return ordered


def line_to_step(line: dict[str, Any]) -> Step:
    return Step.model_validate({_REVERSE.get(k, k): v for k, v in line.items()})


class CassetteWriter:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append(self, line: dict[str, Any]) -> None:
        data = json.dumps(line, ensure_ascii=False, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(data + "\n")
            fh.flush()


def read_lines(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if raw:
                out.append(json.loads(raw))
    return out


def read_cassette(path: Path) -> list[Step]:
    """The steps of a cassette, sorted by ``seq``. A damaged line (a process killed
    mid-write leaves a truncated last line) is skipped with a warning instead of making
    the whole run unreadable; a replay then reports the step as missing."""
    path = Path(path)
    if not path.exists():
        return []
    steps: list[Step] = []
    bad: list[int] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for n, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                steps.append(line_to_step(json.loads(raw)))
            except (ValueError, TypeError):  # JSONDecodeError / pydantic ValidationError
                bad.append(n)
    if bad:
        import warnings

        shown = ", ".join(map(str, bad[:5])) + (" ..." if len(bad) > 5 else "")
        warnings.warn(
            f"crashlab: skipped {len(bad)} damaged line(s) in {path} (line {shown})",
            RuntimeWarning,
            stacklevel=2,
        )
    steps.sort(key=lambda s: s.idx)
    return steps
