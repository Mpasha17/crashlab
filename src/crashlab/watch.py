"""``crashlab watch``: print runs as they start and finish, for a local debug loop.

Polls the store index (no file watching, no extra dependencies). Each poll reads the
newest ``window`` runs from the index; a run is reported once when it is first seen
running (``started``) and once when it has finished (``finished``). Finished runs are
loaded to point at the first failure.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .store import Store

__all__ = ["WatchEvent", "watch_runs"]

WINDOW = 200


@dataclass
class WatchEvent:
    kind: Literal["started", "finished"]
    run_id: str
    name: str
    status: str
    outcome: str
    steps: int
    cost_usd: float | None
    started_at: str
    failed: bool = False
    failure: list[str] = field(default_factory=list)  # first error, failing checks ...

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _event(store: Store, row: dict[str, Any], kind: str) -> WatchEvent:
    ev = WatchEvent(
        kind=kind,  # type: ignore[arg-type]
        run_id=row["run_id"],
        name=row.get("name") or "",
        status=row.get("status") or "",
        outcome=row.get("outcome") or "",
        steps=int(row.get("steps") or 0),
        cost_usd=row.get("cost_usd"),
        started_at=str(row.get("started_at") or ""),
    )
    if kind != "finished":
        return ev
    from .cluster import failure_point, is_failed

    try:
        run = store.load_run(ev.run_id)
    except Exception:  # still being written, or damaged: report what the index has
        ev.failed = ev.status == "error" or ev.outcome == "fail"
        return ev
    ev.steps = len(run.steps) or ev.steps
    ev.failed = is_failed(run)
    if ev.failed:
        fp = failure_point(run)
        if fp.step is not None:
            what = "first error" if fp.step_kind == "error" else f"first {fp.step_kind} finding"
            ev.failure.append(f"{what}: {fp.step_description}")
        if run.error is not None:
            ev.failure.append(f"raised {run.error.type.rsplit('.', 1)[-1]}")
        if fp.failing_checks:
            ev.failure.append("checks failed: " + ", ".join(fp.failing_checks))
    return ev


def watch_runs(
    store: Store | None = None,
    *,
    interval: float = 1.0,
    include_existing: bool = False,
    failed_only: bool = False,
    name: str | None = None,
    polls: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    window: int = WINDOW,
) -> Iterator[WatchEvent]:
    """Yield :class:`WatchEvent` s as runs start and finish, oldest first within a poll.

    Runs already in the store when watching starts are skipped unless
    ``include_existing`` (then finished ones are reported on the first poll). With
    ``failed_only`` only finished runs that failed are yielded. ``polls`` stops after
    that many polls (None = forever). Only the newest ``window`` runs are looked at on
    each poll, so if more runs than that start within one interval, some are missed.
    """
    store = store or Store()
    reported: dict[str, str] = {}  # run_id -> "started" | "finished"
    first = True
    n = 0
    while polls is None or n < polls:
        if n:
            sleep(interval)
        n += 1
        rows = store.list_runs(limit=window, name=name)
        for row in reversed(rows):  # oldest first
            rid, running = row["run_id"], row.get("status") == "running"
            state = reported.get(rid)
            if first and not include_existing:
                reported[rid] = "started" if running else "finished"
                continue
            if running:
                if state is None:
                    reported[rid] = "started"
                    if not failed_only:
                        yield _event(store, row, "started")
                continue
            if state == "finished":
                continue
            reported[rid] = "finished"
            ev = _event(store, row, "finished")
            if failed_only and not ev.failed:
                continue
            yield ev
        first = False
