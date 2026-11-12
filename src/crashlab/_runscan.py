"""Shared helpers for commands that scan many stored runs (``search``, ``cluster``):
time filters, run selection from the index, and tolerant loading."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from ._isotime import parse_iso
from .models import Run
from .store import Store

_REL = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhdw])\s*$", re.IGNORECASE)
_UNIT = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks"}


def parse_since(value: str, *, now: datetime | None = None) -> datetime:
    """``"30m"``, ``"2h"``, ``"7d"``, ``"1w"`` (relative to now) or an ISO date / datetime.
    A date or datetime without a timezone is read as local time. Returns an aware UTC
    datetime. Raises ``ValueError`` on anything else."""
    now = now or datetime.now(timezone.utc)
    m = _REL.match(value)
    if m:
        amount, unit = float(m.group(1)), m.group(2).lower()
        return now - timedelta(**{_UNIT[unit]: amount})
    try:
        dt = parse_iso(value)
    except ValueError:
        raise ValueError(
            f"--since expects 30m / 2h / 7d / 1w or an ISO date like 2026-11-12, not {value!r}"
        ) from None
    if dt.tzinfo is None:
        dt = dt.astimezone()  # local time
    return dt.astimezone(timezone.utc)


def _as_utc(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value:
        try:
            dt = parse_iso(value)
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)  # the store writes UTC
    return dt.astimezone(timezone.utc)


@dataclass
class ScanStats:
    """What a scan looked at, so callers can say "scanned 120 runs, 1 unreadable"."""

    scanned: int = 0
    unreadable: list[str] = field(default_factory=list)


def iter_runs(
    store: Store,
    *,
    since: datetime | None = None,
    run_prefix: str | None = None,
    name: str | None = None,
    stats: ScanStats | None = None,
) -> Iterator[Run]:
    """Stored runs, newest first, filtered on index columns before any cassette is read.
    A run whose files cannot be read is skipped and listed in ``stats.unreadable``."""
    prefix = run_prefix.upper() if run_prefix else None
    for row in store.list_runs(limit=None, name=name):
        rid = row["run_id"]
        if prefix and not rid.startswith(prefix):
            continue
        if since is not None:
            started = _as_utc(row.get("started_at"))
            if started is not None and started < since:
                continue
        try:
            run = store.load_run(rid)
        except Exception:  # a half-written or hand-edited run must not stop the scan
            if stats is not None:
                stats.unreadable.append(rid)
            continue
        if stats is not None:
            stats.scanned += 1
        yield run
