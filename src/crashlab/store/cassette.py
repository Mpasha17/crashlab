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
from . import compact

_RENAMES = {"idx": "seq", "operation": "op", "request_hash": "hash"}
_REVERSE = {v: k for k, v in _RENAMES.items()}
_FIRST = ("seq", "op", "name", "hash")


STATE_ATTR = "crashlab.error.state"


def step_to_line(step_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert a (JSON-mode) step dict into cassette line order/keys.

    ``error.state`` is written as the attribute ``crashlab.error.state``: crashlab 0.0.2
    rejects unknown keys (and skipped such steps) but takes any attribute."""
    err = step_dict.get("error")
    if isinstance(err, dict) and "state" in err:
        err = dict(err)
        state = err.pop("state")
        step_dict = {**step_dict, "error": err}
        if state is not None:
            step_dict["attributes"] = {**(step_dict.get("attributes") or {}), STATE_ATTR: state}
    renamed = {_RENAMES.get(k, k): v for k, v in step_dict.items()}
    ordered = {k: renamed.pop(k) for k in _FIRST if k in renamed}
    ordered.update(renamed)
    return ordered


def line_to_step(line: dict[str, Any], unknown: list[str] | None = None) -> Step:
    """Parse a cassette line. Keys this version does not know (written by a newer
    crashlab) are left out and named in ``unknown`` rather than failing the step."""
    from ..models import ErrorInfo

    data = {_REVERSE.get(k, k): v for k, v in line.items()}
    extra = [k for k in data if k not in Step.model_fields]
    for k in extra:
        data.pop(k)
    err = data.get("error")
    if isinstance(err, dict):
        err = dict(err)
        bad = [k for k in err if k not in ErrorInfo.model_fields]
        for k in bad:
            err.pop(k)
        extra += [f"error.{k}" for k in bad]
        data["error"] = err
    attrs = data.get("attributes")
    if isinstance(attrs, dict) and STATE_ATTR in attrs:
        attrs = dict(attrs)
        state = attrs.pop(STATE_ATTR)
        data["attributes"] = attrs
        if isinstance(err, dict) and err.get("state") is None and isinstance(state, dict):
            err["state"] = state
    if unknown is not None:
        unknown.extend(extra)
    return Step.model_validate(data)


class CassetteWriter:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        # Experimental, opt-in only (see store/compact.py); the default is the full format.
        self._compactor = compact.Compactor() if compact.enabled() else None

    def append(self, line: dict[str, Any]) -> None:
        with self._lock:
            if self._compactor is not None:
                line = self._compactor.compact(line)
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
    unknown: list[str] = []
    parsed: list[tuple[int, dict[str, Any]]] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for n, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except ValueError:
                bad.append(n)
                continue
            if isinstance(obj, dict):
                parsed.append((n, obj))
            else:
                bad.append(n)
    expanded, problems = compact.expand([obj for _, obj in parsed])
    for (n, _), obj in zip(parsed, expanded, strict=True):
        try:
            steps.append(line_to_step(obj, unknown))
        except (ValueError, TypeError):  # pydantic ValidationError
            bad.append(n)
    bad.sort()
    import warnings

    if problems:
        warnings.warn(
            f"crashlab: {len(problems)} compact step(s) in {path} could not be expanded and "
            f"were kept with partial requests (attribute {compact.UNRESOLVED_ATTR}): "
            + " | ".join(problems[:5]),
            RuntimeWarning,
            stacklevel=2,
        )

    if unknown:
        names = ", ".join(sorted(set(unknown)))
        warnings.warn(
            f"crashlab: {path} was written by a newer crashlab; ignored field(s) it does "
            f"not know: {names}. Upgrade crashlab to replay it exactly.",
            RuntimeWarning,
            stacklevel=2,
        )
    if bad:
        shown = ", ".join(map(str, bad[:5])) + (" ..." if len(bad) > 5 else "")
        warnings.warn(
            f"crashlab: skipped {len(bad)} damaged line(s) in {path} (line {shown})",
            RuntimeWarning,
            stacklevel=2,
        )
    steps.sort(key=lambda s: s.idx)
    return steps
