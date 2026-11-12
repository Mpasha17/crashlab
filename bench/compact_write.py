"""Measure cassette write time and size: full format vs experimental compact format.

    PYTHONPATH=src python bench/compact_write.py            # writer only
    PYTHONPATH=src python bench/compact_write.py --store    # Store.add_step (redaction + index)

Each workload is a chat that grows by one user and one assistant message per step, the
way an agent loop resends its whole history. Numbers are wall-clock on this machine; run
it yourself before quoting them.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

WORKLOADS = [
    # (label, steps, chars per message, first message rewritten every step)
    ("5 steps, 40 B msgs", 5, 40, False),
    ("20 steps, 200 B msgs", 20, 200, False),
    ("50 steps, 1 KB msgs", 50, 1000, False),
    ("50 steps, 1 KB, system msg changes", 50, 1000, True),
    ("750 steps (1500 msgs), 40 B", 750, 40, False),
    ("100 steps, 50 KB msgs", 100, 50_000, False),
]


def lines(steps: int, size: int, rewrite: bool):
    msgs = [{"role": "system", "content": "s" * size}]
    for i in range(steps):
        if rewrite:  # e.g. a system prompt that carries the current time
            msgs[0] = {"role": "system", "content": f"t={i} " + "s" * size}
        msgs.append({"role": "user", "content": (f"q{i} " * size)[:size]})
        yield i, list(msgs)
        msgs.append({"role": "assistant", "content": (f"a{i} " * size)[:size]})


def run_writer(steps: int, size: int, rewrite: bool, compact_on: bool) -> tuple[float, int]:
    from crashlab.store import compact
    from crashlab.store.cassette import CassetteWriter

    os.environ.pop(compact.ENV, None)
    if compact_on:
        os.environ[compact.ENV] = "1"
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.jsonl"
        w = CassetteWriter(p)
        t = time.perf_counter()
        for i, msgs in lines(steps, size, rewrite):
            w.append({"seq": i, "op": "chat", "name": "m", "hash": f"h{i}", "run_id": "r",
                      "span_id": f"s{i}", "request": {"messages": msgs},
                      "raw_request": {"model": "m", "messages": msgs},
                      "response": {"content": "ok"}})  # fmt: skip
        dt = time.perf_counter() - t
        return dt, p.stat().st_size


def run_store(steps: int, size: int, rewrite: bool, compact_on: bool) -> tuple[float, int]:
    from crashlab.models import Step
    from crashlab.store import Store, compact

    os.environ.pop(compact.ENV, None)
    if compact_on:
        os.environ[compact.ENV] = "1"
    with tempfile.TemporaryDirectory() as d:
        s = Store(d)
        t = time.perf_counter()
        for i, msgs in lines(steps, size, rewrite):
            s.add_step(
                Step(
                    run_id="r",
                    idx=i,
                    span_id=f"s{i}",
                    operation="chat",
                    name="m",
                    request_hash=f"h{i}",
                    request={"messages": msgs},
                    raw_request={"model": "m", "messages": msgs},
                    response={"content": "ok"},
                    started_at="2026-11-13T00:00:00Z",
                )
            )
        dt = time.perf_counter() - t
        return dt, s.cassette_path("r").stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", action="store_true", help="time Store.add_step, not the writer")
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()
    fn = run_store if a.store else run_writer
    head = ("workload", "full s", "compact s", "ratio", "full MB", "compact MB")
    print(f"{head[0]:36} {head[1]:>8} {head[2]:>10} {head[3]:>6} {head[4]:>9} {head[5]:>11}")
    for label, steps, size, rewrite in WORKLOADS:
        reps = a.repeat if steps * size < 5_000_000 else 1
        f = [fn(steps, size, rewrite, False) for _ in range(reps)]
        c = [fn(steps, size, rewrite, True) for _ in range(reps)]
        ft = statistics.median(x[0] for x in f)
        ct = statistics.median(x[0] for x in c)
        print(f"{label:36} {ft:8.3f} {ct:10.3f} {ct / ft:6.2f} "
              f"{f[0][1] / 1e6:9.2f} {c[0][1] / 1e6:11.2f}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
