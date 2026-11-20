"""crashlab.integrations.mcp: record, replay, side-effects, CallToolResult, runtime checks.

No module-level openai / anthropic / litellm / langchain / mcp / yaml imports.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

import crashlab
from crashlab.integrations.mcp import (
    McpCallToolResult,
    wrap_mcp_session,
)
from crashlab.replay.divergence import SideEffectBlockedError


@dataclass
class Text:
    text: str
    type: str = "text"


@dataclass
class CallToolResult:
    content: list[Any] = field(default_factory=list)
    isError: bool = False
    structuredContent: Any = None


class AsyncSession:
    def __init__(
        self, results: dict[str, Any] | None = None, error: Exception | None = None
    ) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.results = results or {}
        self.error = error
        self.listed = 0

    async def call_tool(self, name: str, arguments: Any = None, **kwargs: Any) -> Any:
        self.calls.append((name, arguments, kwargs))
        if self.error is not None:
            raise self.error
        if name in self.results:
            return self.results[name]
        return CallToolResult(content=[Text(f"{name}:{arguments}")], isError=False)

    async def list_tools(self) -> list[str]:
        self.listed += 1
        return ["read", "write"]


class SyncSession:
    def __init__(
        self, results: dict[str, Any] | None = None, error: Exception | None = None
    ) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.results = results or {}
        self.error = error

    def call_tool(self, name: str, arguments: Any = None, **kwargs: Any) -> Any:
        self.calls.append((name, arguments, kwargs))
        if self.error is not None:
            raise self.error
        if name in self.results:
            return self.results[name]
        return CallToolResult(content=[Text(f"{name}:{arguments}")], isError=False)


def test_import_does_not_need_mcp():
    import crashlab.integrations.mcp as mcp

    assert callable(mcp.wrap_mcp_session)
    assert mcp.wrap_mcp_session.__module__ == "crashlab.integrations.mcp"


def test_wrap_is_idempotent_and_requires_call_tool():
    session = wrap_mcp_session(SyncSession(), server_name="fs")
    assert wrap_mcp_session(session) is session
    with pytest.raises(TypeError, match="call_tool"):
        wrap_mcp_session(object())
    with pytest.raises(ValueError, match="server_name"):
        wrap_mcp_session(SyncSession(), server_name="  ")


def test_records_async_call_tool_and_replays_offline_without_contacting_the_server():
    raw = AsyncSession()
    session = wrap_mcp_session(raw, server_name="fs")

    @crashlab.agent(name="mcp-bot")
    def run_bot(path: str) -> str:
        result = asyncio.run(session.call_tool("read", arguments={"path": path}))
        assert result.content[0].text.startswith("read:")
        assert result.isError is False
        return result.content[0].text

    live = run_bot("a.txt")
    src = crashlab.load("last")
    step = src.steps[0]
    assert step.operation == "execute_tool"
    assert step.name == "mcp:fs:read"
    assert step.attributes["gen_ai.tool.name"] == "mcp:fs:read"
    assert step.attributes["crashlab.mcp.server"] == "fs"
    assert step.attributes["crashlab.mcp.tool"] == "read"
    assert step.request["args"] == {"path": "a.txt"}
    assert step.result["isError"] is False
    assert step.result["content"][0]["text"].startswith("read:")
    assert live.startswith("read:")
    assert len(raw.calls) == 1

    raw.calls.clear()
    replayed = crashlab.replay(src, fn=run_bot)
    assert raw.calls == []  # the MCP server was not contacted
    assert replayed.output == live
    served = replayed.steps[0].result
    assert served["isError"] is False
    # the agent saw a rebuilt CallToolResult (content[0].text) during replay


def test_sync_call_tool_and_positional_arguments():
    raw = SyncSession()
    session = wrap_mcp_session(raw, server_name="db")

    with crashlab.record("mcp-sync") as run:
        result = session.call_tool("query", {"sql": "select 1"})
        run.output = result.content[0].text
    src = crashlab.load(run.run_id)
    assert src.steps[0].name == "mcp:db:query"
    assert src.steps[0].request["args"] == {"sql": "select 1"}
    assert result.isError is False


def test_name_without_server_and_list_tools_is_forwarded():
    raw = AsyncSession()
    session = wrap_mcp_session(raw)

    async def _go():
        with crashlab.record("mcp-plain"):
            await session.call_tool("ping", arguments={})
            return await session.list_tools()

    tools = asyncio.run(_go())
    assert tools == ["read", "write"] and raw.listed == 1
    assert crashlab.load("last").steps[0].name == "ping"
    assert crashlab.load("last").steps[0].attributes["gen_ai.tool.name"] == "ping"


def test_recorded_exception_is_reraised_on_replay_without_calling_the_session():
    raw = SyncSession(error=RuntimeError("server down"))
    session = wrap_mcp_session(raw, server_name="fs")

    @crashlab.agent(name="mcp-err")
    def run_bot() -> None:
        session.call_tool("read", arguments={"path": "a.txt"})

    with pytest.raises(RuntimeError, match="server down"):
        run_bot()
    src = crashlab.load("last")
    assert src.steps[0].error is not None
    assert "server down" in src.steps[0].error.message
    raw.calls.clear()
    raw.error = RuntimeError("should not run")
    # replay reproduces an error run without re-raising the matching exception
    replayed = crashlab.replay(src, fn=run_bot)
    assert raw.calls == []
    assert replayed.status == "error"
    assert replayed.error is not None and "server down" in replayed.error.message


def test_iserror_result_is_rebuilt_and_trips_runtime_checks():
    raw = SyncSession(
        results={
            "lookup": CallToolResult(
                content=[Text("no such row")], isError=True, structuredContent={"ok": False}
            )
        }
    )
    session = wrap_mcp_session(raw)

    with crashlab.record("mcp-iserror") as run:
        for _ in range(3):
            result = session.call_tool("lookup", arguments={"id": "x"})
            assert result.isError is True
            assert result.content[0].text == "no such row"
        run.output = "done"
    src = crashlab.load(run.run_id)
    payload = src.steps[0].result
    assert payload["isError"] is True and payload["error"] == "no such row"
    kinds = {f.kind for f in src.findings}
    assert "suspicious_tool_result" in kinds
    assert "tool_error_streak" in kinds

    raw.calls.clear()

    def replay_fn():
        with crashlab.record("mcp-iserror") as again:
            for _ in range(3):
                result = session.call_tool("lookup", arguments={"id": "x"})
                assert isinstance(result, McpCallToolResult)
                assert result.isError is True
                assert result.content[0].text == "no such row"
                assert result.structuredContent == {"ok": False}
            again.output = "done"

    crashlab.replay(src, fn=replay_fn)
    assert raw.calls == []


def test_side_effects_block_a_new_call_on_fork():
    raw = SyncSession()
    session = wrap_mcp_session(raw, server_name="fs", side_effects=True)
    path = {"path": "a.txt"}

    @crashlab.agent(name="mcp-write")
    def run_bot() -> str:
        result = session.call_tool("write", arguments=dict(path))
        return result.content[0].text

    run_bot()
    src = crashlab.load("last")
    assert src.steps[0].attributes["crashlab.tool.side_effects"] is True
    raw.calls.clear()
    path["path"] = "b.txt"
    with pytest.raises(SideEffectBlockedError):
        crashlab.fork(src, at_step=0, fn=run_bot, raise_errors=True)
    # the underlying session was not asked to write b.txt
    assert all(c[1] != {"path": "b.txt"} for c in raw.calls)


def test_plain_json_results_round_trip():
    raw = SyncSession(results={"inc": {"n": 2}})
    session = wrap_mcp_session(raw)

    @crashlab.agent(name="mcp-json")
    def run_bot() -> dict:
        return session.call_tool("inc", arguments={"n": 1})

    assert run_bot() == {"n": 2}
    src = crashlab.load("last")
    raw.calls.clear()
    replayed = crashlab.replay(src, fn=run_bot)
    assert replayed.output == {"n": 2}
    assert raw.calls == []
