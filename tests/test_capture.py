from __future__ import annotations

import json
import sqlite3

import pytest
from examples.refund_bot import agent as refund

import crashlab
from crashlab.capture.normalize import request_hash
from crashlab.store import Store
from crashlab.store.cassette import read_lines
from tests.fakes import (
    AsyncFakeOpenAI,
    FakeAPIError,
    FakeOpenAI,
    refund_script,
    text_response,
    tool_call_response,
)


def _cassette(run_id: str) -> list[dict]:
    return read_lines(Store().cassette_path(run_id))


def test_full_refund_bot_run(crashlab_dir):
    fake = FakeOpenAI(refund_script())
    client = crashlab.wrap(fake)
    answer = refund.run_agent("Refund my order A17, it arrived broken", client=client, model="m1")
    assert answer.startswith("Your refund")
    assert refund.REFUNDS == [
        {"refund_id": "R0001", "order_id": "A17", "amount": 49.0, "status": "issued"}
    ]

    run = crashlab.load("last")
    assert run.name == "refund-bot"
    assert run.status == "ok"
    assert run.output == answer
    assert run.input == {"user_msg": "Refund my order A17, it arrived broken", "model": "m1",
                         "max_turns": 8}  # fmt: skip
    assert run.entrypoint == "examples.refund_bot.agent:run_agent"
    assert run.env["crashlab"] == crashlab.__version__
    # 3 chat calls + 2 tool calls, interleaved
    assert [(s.idx, s.operation, s.name) for s in run.steps] == [
        (0, "chat", "m1"),
        (1, "execute_tool", "get_order"),
        (2, "chat", "m1"),
        (3, "execute_tool", "issue_refund"),
        (4, "chat", "m1"),
    ]
    u = run.usage
    assert (u.steps, u.llm_calls, u.tool_calls, u.errors) == (5, 3, 2, 0)
    assert (u.input_tokens, u.output_tokens, u.total_tokens) == (101 + 102 + 200, 40, 443)

    chat0, tool1, chat2, tool3, chat4 = run.steps
    # chat request normalization
    assert chat0.request["system"] == refund.SYSTEM_PROMPT
    assert chat0.request["messages"] == [
        {"role": "user", "content": "Refund my order A17, it arrived broken"}
    ]
    assert [t["function"]["name"] for t in chat0.request["tools"]] == ["get_order", "issue_refund"]
    assert chat0.request["params"] == {"model": "m1", "temperature": 0}
    assert chat0.attributes["gen_ai.operation.name"] == "chat"
    assert chat0.attributes["gen_ai.request.model"] == "m1"
    assert chat0.attributes["gen_ai.response.finish_reasons"] == ["tool_calls"]
    assert chat0.attributes["gen_ai.usage.input_tokens"] == 101
    assert chat0.attributes["server.address"] == "fake.local"
    # chat response normalization incl. tool calls
    assert chat0.response["finish_reason"] == "tool_calls"
    assert chat0.response["tool_calls"] == [
        {"id": "call_1", "type": "function", "name": "get_order",
         "arguments": '{"order_id": "A17"}'}
    ]  # fmt: skip
    assert chat0.raw_response["id"] == "chatcmpl-1"
    assert chat4.response["finish_reason"] == "stop"
    assert chat4.response["messages"][0]["content"].startswith("Your refund")
    # later requests carry the tool results
    assert chat2.request["messages"][-1]["role"] == "tool"
    assert json.loads(chat2.request["messages"][-1]["content"])["total"] == 49.0

    # tool steps: args, result, linkage to the tool call id, side-effect metadata
    assert tool1.args == {"order_id": "A17"}
    assert tool1.result["total"] == 49.0
    assert tool1.attributes["gen_ai.tool.name"] == "get_order"
    assert tool1.attributes["gen_ai.tool.call.id"] == "call_1"
    assert tool1.attributes["crashlab.tool.side_effects"] is False
    assert tool1.attributes["gen_ai.tool.description"].startswith("Look up an order")
    assert tool3.args == {"order_id": "A17", "amount": 49.0}
    assert tool3.attributes["crashlab.tool.side_effects"] is True
    assert tool3.attributes["gen_ai.tool.call.id"] == "call_2"
    for s in run.steps:
        assert s.parent_span_id == run.span_id
        assert s.error is None and s.source == "live" and s.duration_ms >= 0

    # cassette lines
    lines = _cassette(run.run_id)
    assert [ln["seq"] for ln in lines] == [0, 1, 2, 3, 4]
    assert [ln["op"] for ln in lines] == ["chat", "execute_tool", "chat", "execute_tool", "chat"]
    assert all(len(ln["hash"]) == 64 for ln in lines)

    # sqlite rows
    db = sqlite3.connect(crashlab_dir / "index.sqlite")
    (n_runs,) = db.execute("SELECT count(*) FROM runs").fetchone()
    assert n_runs == 1
    rows = db.execute(
        "SELECT idx, operation, name, tool_call_id, finish_reason, request_hash FROM steps "
        "WHERE run_id=? ORDER BY idx",
        (run.run_id,),
    ).fetchall()
    assert [r[:5] for r in rows] == [
        (0, "chat", "m1", None, "tool_calls"),
        (1, "execute_tool", "get_order", "call_1", None),
        (2, "chat", "m1", None, "tool_calls"),
        (3, "execute_tool", "issue_refund", "call_2", None),
        (4, "chat", "m1", None, "stop"),
    ]
    assert [r[5] for r in rows] == [ln["hash"] for ln in lines]
    status, total = db.execute(
        "SELECT status, total_tokens FROM runs WHERE run_id=?", (run.run_id,)
    ).fetchone()
    assert (status, total) == ("ok", 443)


def test_request_hashes_are_stable_and_recomputable():
    hashes = []
    for _ in range(2):
        refund.reset_db()
        refund.run_agent("Refund A17", client=crashlab.wrap(FakeOpenAI(refund_script())), model="m")
        run = crashlab.load("last")
        # hash recomputes from the stored (redacted, normalized) request
        for s in run.steps:
            expected = (
                request_hash(s.request)
                if s.operation == "chat"
                else request_hash({"tool": s.name, "args": s.args})
            )
            assert s.request_hash == expected
        hashes.append([s.request_hash for s in run.steps])
    assert hashes[0] == hashes[1]  # identical inputs -> identical hashes across runs
    assert len(set(hashes[0])) == 5  # each request differs


def test_fault_html_502_is_recorded(monkeypatch):
    monkeypatch.setenv("FAULT_HTML_502", "1")
    refund.run_agent("Refund A17", client=crashlab.wrap(FakeOpenAI(refund_script())), model="m")
    run = crashlab.load("last")
    get_order = run.tool_steps("get_order")[0]
    assert isinstance(get_order.result, str)
    assert "<title>502 Bad Gateway</title>" in get_order.result
    assert get_order.error is None  # garbage result, not an exception
    # the HTML is what the model saw next
    assert "502 Bad Gateway" in run.steps[2].request["messages"][-1]["content"]


def test_tool_exception_recorded_and_agent_continues():
    script = [
        tool_call_response(1, "get_order", {"order_id": "A17"}),
        tool_call_response(2, "issue_refund", {"order_id": "A17", "amount": 500}),
        text_response(3, "Sorry, I can only refund up to the order total."),
    ]
    refund.run_agent("Refund 500 for A17", client=crashlab.wrap(FakeOpenAI(script)), model="m")
    run = crashlab.load("last")
    assert run.status == "ok"
    step = run.tool_steps("issue_refund")[0]
    assert step.response is None
    assert step.error.type == "ValueError"
    assert step.error.is_tool_error is True
    assert "must be > 0 and <= order total 49.0" in step.error.message
    assert "Traceback" in step.error.traceback
    assert run.usage.errors == 1
    assert step.attributes["error.type"] == "ValueError"
    assert refund.REFUNDS == []


def test_llm_error_recorded_and_run_marked_error():
    script = [FakeAPIError("upstream overloaded", status_code=529)]
    with pytest.raises(FakeAPIError):
        refund.run_agent("Refund A17", client=crashlab.wrap(FakeOpenAI(script)), model="m")
    run = crashlab.load("last")
    assert run.status == "error" and run.outcome == "fail"
    assert run.error.type.endswith("FakeAPIError")
    (chat,) = run.steps
    assert chat.operation == "chat" and chat.response is None
    assert chat.error.status_code == 529
    assert chat.error.is_tool_error is False


def test_secrets_are_redacted_everywhere(crashlab_dir):
    secret = "sk-proj-THISISAVERYSECRETKEY1234567890"
    crashlab.redaction.add_pattern(r"ACCT-\d{6}")
    fake = FakeOpenAI([text_response(1, f"echo {secret} for ACCT-123456")])
    client = crashlab.wrap(fake)
    with crashlab.record("redact-test", input={"api_key": secret, "msg": "hi"}):
        client.chat.completions.create(
            model="m",
            messages=[{"role": "user", "content": f"my key is {secret}, account ACCT-654321"}],
            extra_headers={"Authorization": f"Bearer {secret}", "X-Trace": "ok"},
            timeout=30,
        )
    # the real call still got the real values
    assert fake.calls[0]["extra_headers"]["Authorization"] == f"Bearer {secret}"

    on_disk = "".join(
        p.read_text() for p in crashlab_dir.rglob("*") if p.is_file() and p.suffix != ".sqlite"
    )
    assert "SECRETKEY" not in on_disk and "ACCT-123456" not in on_disk
    assert "ACCT-654321" not in on_disk
    db = sqlite3.connect(crashlab_dir / "index.sqlite")
    dumped = "\n".join(str(r) for r in db.iterdump())
    assert "SECRETKEY" not in dumped and "ACCT-" not in dumped

    run = crashlab.load("last")
    chat = run.steps[0]
    assert run.input["api_key"] == "[REDACTED]"
    assert chat.raw_request["extra_headers"] == {"Authorization": "[REDACTED]", "X-Trace": "ok"}
    assert "extra_headers" not in chat.request["params"]  # transport-only kwargs not hashed
    assert "timeout" not in chat.request["params"]
    assert chat.request["messages"][0]["content"] == "my key is [REDACTED], account [REDACTED]"
    assert chat.request_hash == request_hash(chat.request)


async def test_async_client_and_async_tools():
    calls = []

    @crashlab.tool(name="lookup")
    async def lookup(order_id: str) -> dict:
        calls.append(order_id)
        return {"order_id": order_id, "total": 10.0}

    @crashlab.agent(name="async-bot", ignore_args=("client",))
    async def run(msg: str, client) -> str:
        messages = [{"role": "user", "content": msg}]
        resp = await client.chat.completions.create(model="am", messages=messages)
        tc = resp.choices[0].message.tool_calls[0]
        result = await lookup(**json.loads(tc.function.arguments))
        resp = await client.chat.completions.create(
            model="am", messages=[{"role": "tool", "tool_call_id": tc.id, "content": str(result)}]
        )
        return resp.choices[0].message.content

    fake = AsyncFakeOpenAI(
        [tool_call_response(1, "lookup", {"order_id": "Z9"}), text_response(2, "done")]
    )
    out = await run("hello", crashlab.wrap(fake))
    assert out == "done" and calls == ["Z9"]
    rec = crashlab.load("last")
    assert rec.name == "async-bot" and rec.status == "ok" and rec.output == "done"
    assert rec.input == {"msg": "hello"}
    assert [(s.operation, s.name) for s in rec.steps] == [
        ("chat", "am"),
        ("execute_tool", "lookup"),
        ("chat", "am"),
    ]
    assert rec.steps[1].attributes["gen_ai.tool.call.id"] == "call_1"


async def test_async_tool_exception_recorded():
    @crashlab.tool()
    async def flaky() -> None:
        raise ConnectionError("db down")

    with pytest.raises(ConnectionError):
        async with crashlab.record("async-err"):
            await flaky()
    run = crashlab.load("last")
    assert run.status == "error"
    assert run.steps[0].error.type == "ConnectionError"


def test_nested_agent_and_tool_spans():
    @crashlab.tool()
    def leaf(x: int) -> int:
        return x + 1

    @crashlab.tool()
    def parent_tool(x: int) -> int:
        return leaf(x) * 2

    @crashlab.agent(name="sub")
    def sub_agent(x: int) -> int:
        return parent_tool(x)

    @crashlab.agent(name="outer")
    def outer(x: int) -> int:
        return sub_agent(x)

    assert outer(1) == 4
    run = crashlab.load("last")
    assert run.name == "outer"
    by_name = {s.name: s for s in run.steps}
    assert [s.idx for s in run.steps] == [0, 1, 2]
    assert by_name["sub"].operation == "invoke_agent"
    assert by_name["sub"].parent_span_id == run.span_id
    assert by_name["parent_tool"].parent_span_id == by_name["sub"].span_id
    assert by_name["leaf"].parent_span_id == by_name["parent_tool"].span_id
    # cassette lines are written on completion (leaf first) but load() sorts by seq
    assert [ln["seq"] for ln in _cassette(run.run_id)] == [2, 1, 0]


def test_wrap_detection():
    fake = FakeOpenAI([])
    wrapped = crashlab.wrap(fake)
    assert crashlab.wrap(wrapped) is wrapped
    assert wrapped.base_url == fake.base_url  # attribute passthrough
    assert wrapped.unwrapped is fake

    class Anthropicish:
        pass

    Anthropicish.__module__ = "anthropic._client"
    with pytest.raises(NotImplementedError, match="Anthropic"):
        crashlab.wrap(Anthropicish())
    with pytest.raises(TypeError):
        crashlab.wrap(object())


def test_call_outside_run_warns_and_passes_through(crashlab_dir):
    import crashlab.capture.openai as cap

    cap._warned_outside_run = False
    client = crashlab.wrap(FakeOpenAI([text_response(1, "hi")]))
    with pytest.warns(UserWarning, match="outside a recorded run"):
        resp = client.chat.completions.create(model="m", messages=[])
    assert resp.choices[0].message.content == "hi"
    assert Store().list_runs() == []
