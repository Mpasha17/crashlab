"""Successive asyncio.run() calls must not leave a closed-loop bomb for the next trial.

crashlab verify forks an agent many times in one process; each trial typically does
asyncio.run(...). Closing the AsyncOpenAI client in a finally-block is the safe pattern
(see examples/integrations/openai-agents.md)."""

from __future__ import annotations

import asyncio

import pytest

import crashlab


def test_successive_asyncio_run_with_fresh_closed_clients(crashlab_dir):
    pytest.importorskip("openai")
    from openai import AsyncOpenAI

    @crashlab.agent(name="async-twice")
    def once(i: int) -> str:
        client = crashlab.wrap(AsyncOpenAI(api_key="sk-test", base_url="http://127.0.0.1:9"))

        async def go() -> str:
            try:
                return f"ok-{i}"
            finally:
                await client.close()

        return asyncio.run(go())

    assert once(1) == "ok-1"
    assert once(2) == "ok-2"
    assert once(3) == "ok-3"
    run = crashlab.load("last")
    assert run.outcome in ("pass", "fail", "unknown") and run.status == "ok"


def test_async_entry_points_share_one_loop_across_replays():
    """An async entry function replayed / forked many times (verify) runs on one loop per
    thread, so a client cached by the framework is not bound to a loop crashlab closed."""
    from crashlab.replay.engine import _run_callable

    loops = []

    async def entry(i: int) -> int:
        loops.append(asyncio.get_running_loop())
        asyncio.get_running_loop().create_task(asyncio.sleep(60))  # left behind: cancelled
        return i

    assert _run_callable(entry, [1], {}) == 1
    assert _run_callable(entry, [2], {}) == 2
    assert loops[0] is loops[1] and not loops[0].is_closed()
    assert not [t for t in asyncio.all_tasks(loops[0]) if not t.done()]
    assert asyncio.run(entry(3)) == 3  # the user's own asyncio.run still works afterwards
