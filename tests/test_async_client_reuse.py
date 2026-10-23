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
