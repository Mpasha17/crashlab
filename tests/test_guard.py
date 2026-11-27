"""crashlab.guard: live circuit breakers, loop breaker, tool policies, argument
validation / repair, retries with backoff, model fallback, secret redaction."""

from __future__ import annotations

import asyncio
import importlib
import json
import re
import threading

import pytest
from typer.testing import CliRunner

import crashlab
from crashlab.cli import app
from crashlab.faults import faults, llm_error
from crashlab.guard import (
    ATTEMPTS_ATTR,
    BLOCKED_ATTR,
    FALLBACK_ATTR,
    GUARD_ATTR,
    GuardConfig,
    GuardViolation,
    GuardWarning,
    RetryPolicy,
    active_scopes,
    is_transient,
    load_config,
    schema_from_signature,
    validate_args,
)
from tests.fakes import AsyncFakeOpenAI, FakeAPIError, FakeOpenAI, text_response, tool_call_response

G = importlib.import_module("crashlab.guard")  # the module (crashlab.guard is the function)
ANSI = re.compile(r"\x1b\[[0-9;]*m")
runner = CliRunner()

CALLS: list[tuple[str, dict]] = []


@crashlab.tool()
def lookup(key: str) -> dict:
    CALLS.append(("lookup", {"key": key}))
    return {"key": key, "value": 42}


@crashlab.tool()
def delete_user(user_id: str) -> str:
    CALLS.append(("delete_user", {"user_id": user_id}))
    return "deleted"


@crashlab.tool()
def leak(n: int) -> dict:
    CALLS.append(("leak", {"n": n}))
    return {"note": "use key " + "sk-" + "proj-" + "A" * 24 + " now", "n": n}


TOOLS = {"lookup": lookup, "delete_user": delete_user, "leak": leak}
SCHEMAS = [
    {"type": "function", "function": {"name": "lookup", "parameters": {
        "type": "object", "properties": {"key": {"type": "string", "minLength": 2}},
        "required": ["key"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "delete_user"}},
    {"type": "function", "function": {"name": "leak", "parameters": {
        "type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}}},
]  # fmt: skip


@pytest.fixture(autouse=True)
def _clear(monkeypatch):
    CALLS.clear()
    slept: list[float] = []
    monkeypatch.setattr(G, "_sleep", lambda s: slept.append(s))

    async def fake_async_sleep(s):
        slept.append(s)

    monkeypatch.setattr(G, "_async_sleep", fake_async_sleep)
    monkeypatch.setenv("CRASHLAB_NO_PYPROJECT", "1")
    yield slept


@crashlab.agent(name="guard-bot", ignore_args=("client",))
def run_agent(user: str, client, max_turns: int = 10) -> str:
    messages = [{"role": "user", "content": user}]
    for _ in range(max_turns):
        r = client.chat.completions.create(model="fake-model", messages=messages, tools=SCHEMAS)
        msg = r.choices[0].message
        if not msg.tool_calls:
            return msg.content or ""
        messages.append(
            {"role": "assistant", "tool_calls": [t.model_dump() for t in msg.tool_calls]}
        )
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            try:
                result = TOOLS[tc.function.name](**args)
            except GuardViolation as exc:  # the agent tells the model and carries on
                result = {"error": str(exc)}
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
    return "gave up"


def _client(*items):
    return crashlab.wrap(FakeOpenAI(list(items)))


def _loop_script(n: int, name: str = "lookup", args: dict | None = None):
    return [tool_call_response(i, name, args or {"key": "abc"}) for i in range(1, n + 1)]


def _guard_entries(run):
    return [(s.idx, e["policy"], e["action"]) for s in run.steps
            for e in s.attributes.get(GUARD_ATTR) or []]  # fmt: skip


# ---------------------------------------------------------------------------
# circuit breakers
# ---------------------------------------------------------------------------


def test_max_steps_blocks_the_next_call_and_records_a_finding():
    client = _client(*_loop_script(10))
    with pytest.raises(GuardViolation) as info, crashlab.guard(max_steps=3):
        run_agent("hi", client)
    exc = info.value
    assert exc.policy == "max_steps" and exc.detail == {"steps": 3, "limit": 3}
    run = crashlab.load("last")
    assert run.status == "error"
    blocked = run.steps[exc.step_idx]
    assert blocked.attributes[BLOCKED_ATTR] is True
    assert blocked.error is not None and blocked.error.type == "crashlab.guard.GuardViolation"
    # the blocked tool call (step 3) is caught by the agent; the next LLM call is blocked too
    found = [f for f in run.findings if f.kind == "guard.max_steps"]
    assert [f.step for f in found] == [3, 4] and exc.step_idx == 4
    assert all(f.severity == "error" for f in found)
    assert len([s for s in run.steps if not s.attributes.get(BLOCKED_ATTR)]) == 3


def test_max_llm_calls_and_max_tool_calls():
    with pytest.raises(GuardViolation) as info, crashlab.guard(max_llm_calls=2):
        run_agent("hi", _client(*_loop_script(5)))
    assert info.value.policy == "max_llm_calls"
    assert sum(1 for s in crashlab.load("last").steps if s.operation == "chat") == 3  # 1 blocked

    CALLS.clear()
    out = None
    with crashlab.guard(max_tool_calls=1):
        out = run_agent("hi", _client(*_loop_script(2), text_response(3, "done")))
    # the second tool call is blocked; the agent catches it and the model finishes
    assert out == "done" and len(CALLS) == 1
    run = crashlab.load("last")
    assert ("max_tool_calls", "blocked") in [(p, a) for _, p, a in _guard_entries(run)]


def test_max_tokens_and_max_cost(monkeypatch):
    with pytest.raises(GuardViolation) as info, crashlab.guard(max_tokens=200):
        run_agent("hi", _client(*_loop_script(5)))
    assert info.value.policy == "max_tokens"  # 111 + 112 tokens > 200 after 2 calls

    from crashlab import pricing

    monkeypatch.setattr(pricing, "step_cost", lambda step: 0.10)  # $0.10 per LLM call
    with pytest.raises(GuardViolation) as info, crashlab.guard(max_cost_usd=0.25):
        run_agent("hi", _client(*_loop_script(8)))
    assert info.value.policy == "max_cost_usd"
    assert info.value.detail["cost_usd"] == pytest.approx(0.30)


def test_max_seconds(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(G.time, "monotonic", lambda: clock[0])
    client = _client(*_loop_script(5))

    @crashlab.agent(name="slow", ignore_args=("c",))
    def slow(c):
        c.chat.completions.create(model="fake-model", messages=[])
        clock[0] += 31
        c.chat.completions.create(model="fake-model", messages=[])

    with pytest.raises(GuardViolation) as info, crashlab.guard(max_seconds=30):
        slow(client)
    assert info.value.policy == "max_seconds"


# ---------------------------------------------------------------------------
# loop breaker
# ---------------------------------------------------------------------------


def test_loop_breaker_on_repeated_tool_calls():
    with pytest.raises(GuardViolation) as info, crashlab.guard(loop_breaker=2):
        run_agent("hi", _client(*_loop_script(6, args={"key": "same"})))
    # third identical LLM reply: the loop is broken on the LLM side or the tool side
    assert info.value.policy == "loop_breaker"
    assert len(CALLS) == 2  # the identical tool call ran twice, then was stopped


def test_loop_breaker_tool_side_with_different_replies():
    @crashlab.agent(name="tool-loop")
    def tool_loop():
        for _ in range(5):
            lookup("same")

    with pytest.raises(GuardViolation) as info, crashlab.guard(loop_breaker=3):
        tool_loop()
    assert info.value.policy == "loop_breaker" and info.value.detail["kind"] == "tool_call"
    assert len(CALLS) == 3


def test_loop_breaker_on_identical_llm_replies():
    @crashlab.agent(name="echo", ignore_args=("c",))
    def echo(c):
        for _ in range(5):
            c.chat.completions.create(model="fake-model", messages=[])

    fake = FakeOpenAI([text_response(i, "same answer") for i in range(5)])
    with pytest.raises(GuardViolation) as info, crashlab.guard(loop_breaker=2):
        echo(crashlab.wrap(fake))
    assert info.value.detail["kind"] == "llm_reply"
    assert len(fake.calls) == 2  # the third call never reached the provider
    run = crashlab.load("last")
    assert sum(1 for s in run.steps if not s.attributes.get(BLOCKED_ATTR)) == 2


# ---------------------------------------------------------------------------
# tool policies and argument validation
# ---------------------------------------------------------------------------


def test_tool_denylist_and_allowlist():
    script = [tool_call_response(1, "delete_user", {"user_id": "u1"}), text_response(2, "ok")]
    with crashlab.guard(tool_denylist=["delete_*"]):
        assert run_agent("hi", _client(*script)) == "ok"
    assert CALLS == []
    run = crashlab.load("last")
    tool = next(s for s in run.steps if s.operation == "execute_tool")
    assert tool.error is not None and tool.attributes[BLOCKED_ATTR]
    assert any(f.kind == "guard.tool_denylist" for f in run.findings)

    with crashlab.guard(tool_allowlist=["lookup"]):
        assert run_agent("hi", _client(*script)) == "ok"
    assert CALLS == []
    assert any(f.kind == "guard.tool_allowlist" for f in crashlab.load("last").findings)


def test_validate_tool_args_raises_against_declared_schema():
    script = [tool_call_response(1, "lookup", {"key": 7}), text_response(2, "ok")]
    with crashlab.guard(validate_tool_args=True):
        assert run_agent("hi", _client(*script)) == "ok"
    assert CALLS == []
    f = next(f for f in crashlab.load("last").findings if f.kind == "guard.validate_tool_args")
    assert "should be string" in f.message


def test_validate_tool_args_repair_returns_error_to_the_model():
    client = FakeOpenAI([tool_call_response(1, "lookup", {"key": "x", "extra": 1}),
                         text_response(2, "fixed")])  # fmt: skip
    with crashlab.guard(validate_tool_args="repair"):
        assert run_agent("hi", crashlab.wrap(client)) == "fixed"
    assert CALLS == []
    tool_msg = client.calls[1]["messages"][-1]
    payload = json.loads(tool_msg["content"])
    assert payload["error"] == "invalid_tool_arguments"
    assert any("unexpected argument 'extra'" in p for p in payload["problems"])
    assert any("shorter than 2" in p for p in payload["problems"])
    run = crashlab.load("last")
    tool = next(s for s in run.steps if s.operation == "execute_tool")
    assert tool.error is None and tool.result["error"] == "invalid_tool_arguments"
    assert (tool.idx, "validate_tool_args", "repaired") in _guard_entries(run)
    assert run.status == "ok"


def test_validate_args_and_signature_schema():
    schema = {"type": "object", "required": ["a"], "additionalProperties": False,
              "properties": {"a": {"type": "integer", "minimum": 1},
                             "b": {"type": "array", "items": {"enum": ["x", "y"]}}}}  # fmt: skip
    assert validate_args({"a": 2, "b": ["x"]}, schema) == []
    problems = validate_args({"a": 0, "b": ["z"], "c": 1}, schema)
    assert len(problems) == 3
    assert validate_args({}, schema) == ["missing required argument 'a'"]

    def f(name: str, count: int = 1, **kw): ...

    sch = schema_from_signature(f)
    assert sch["required"] == ["name"] and "additionalProperties" not in sch
    assert validate_args({"name": 3}, sch) == ["name should be string, got int"]


def test_validate_falls_back_to_the_python_signature():
    @crashlab.agent(name="sig")
    def sig():
        return leak("three")  # type: ignore[arg-type]

    with pytest.raises(GuardViolation) as info, crashlab.guard(validate_tool_args=True):
        sig()
    assert info.value.policy == "validate_tool_args" and CALLS == []


# ---------------------------------------------------------------------------
# on_violation modes, nesting, config
# ---------------------------------------------------------------------------


def test_warn_mode_records_but_lets_the_call_run():
    with pytest.warns(GuardWarning), crashlab.guard(max_tool_calls=0, on_violation="warn"):
        out = run_agent("hi", _client(*_loop_script(1), text_response(2, "done")))
    assert out == "done" and len(CALLS) == 1
    run = crashlab.load("last")
    f = next(f for f in run.findings if f.kind == "guard.max_tool_calls")
    assert f.severity == "warn"
    assert ("max_tool_calls", "warned") in [(p, a) for _, p, a in _guard_entries(run)]


def test_stop_mode_stops_the_whole_run():
    script = [tool_call_response(1, "delete_user", {"user_id": "u1"}), text_response(2, "ok")]
    with (
        pytest.raises(crashlab.CircuitBreakerError),
        crashlab.guard(tool_denylist="delete_user", on_violation="stop"),
    ):
        run_agent("hi", _client(*script))  # agent catches the violation; next call stops
    run = crashlab.load("last")
    f = next(f for f in run.findings if f.kind == "guard.tool_denylist")
    assert f.stopped_run


def test_nested_guards_each_count_their_own_scope():
    @crashlab.agent(name="nest")
    def nest():
        lookup("a")
        with crashlab.guard(max_tool_calls=1) as inner:
            assert len(active_scopes()) == 2
            lookup("b")
            with pytest.raises(GuardViolation):
                lookup("c")
        assert inner.tool_calls == 1
        lookup("d")  # outer allows 4
        with pytest.raises(GuardViolation):
            lookup("e")

    with crashlab.guard(max_tool_calls=3) as outer:
        nest()
    assert active_scopes() == ()
    assert outer.tool_calls == 3
    assert [c[1]["key"] for c in CALLS] == ["a", "b", "d"]


def test_decorator_gives_each_call_a_fresh_scope():
    @crashlab.guard(max_tool_calls=1)
    @crashlab.agent(name="deco")
    def deco(key):
        lookup(key)
        return "ok"

    assert deco("a") == "ok" and deco("b") == "ok"


def test_threads_get_their_own_guard_context():
    errors: list[BaseException] = []

    def worker():
        try:
            assert active_scopes() == ()
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    with crashlab.guard(max_steps=1):
        t = threading.Thread(target=worker)
        t.start()
        t.join()
    assert not errors


def test_config_from_pyproject_and_dict(tmp_path, monkeypatch):
    monkeypatch.delenv("CRASHLAB_NO_PYPROJECT")
    (tmp_path / "pyproject.toml").write_text(
        "[tool.crashlab.guard]\nmax_steps = 7\nloop_breaker = true\n"
        'tool_denylist = ["rm_*"]\nretry_policy = { max_attempts = 4, base_delay = 0.1 }\n'
    )
    monkeypatch.chdir(tmp_path)
    cfg = load_config({"max_steps": 9}, max_tokens=100)
    assert cfg.max_steps == 9 and cfg.loop_breaker == 3 and cfg.max_tokens == 100
    assert cfg.tool_denylist == ("rm_*",) and cfg.retry_policy == RetryPolicy(4, 0.1)
    assert crashlab.guard().config.max_steps == 7
    assert crashlab.guard(pyproject=False).config.max_steps is None
    assert GuardConfig.from_mapping({"validate_tool_args": {"repair": True}}).repair_tool_args
    with pytest.raises(ValueError, match="unknown guard settings"):
        GuardConfig.from_mapping({"max_stepz": 1})
    with pytest.raises(ValueError):
        crashlab.guard(on_violation="explode")  # type: ignore[arg-type]
    assert json.dumps(cfg.as_dict())


def test_alert_sinks_receive_violations():
    events = []
    from crashlab.alerts import CallbackSink

    with crashlab.guard(tool_denylist="delete_user", alerts=[CallbackSink(events.append)]):
        run_agent("hi", _client(tool_call_response(1, "delete_user", {"user_id": "u"}),
                                text_response(2, "ok")))  # fmt: skip
    assert len(events) == 1
    assert events[0].finding_kind == "guard.tool_denylist" and events[0].run_id


# ---------------------------------------------------------------------------
# retries, backoff, fallback
# ---------------------------------------------------------------------------


def test_retry_with_backoff_on_transient_errors(_clear):
    slept = _clear
    client = FakeOpenAI([FakeAPIError("rate limited", 429), FakeAPIError("boom", 503),
                         text_response(1, "done")])  # fmt: skip
    policy = RetryPolicy(max_attempts=3, base_delay=1.0, jitter=False)
    with crashlab.guard(retry_policy=policy):
        assert run_agent("hi", crashlab.wrap(client)) == "done"
    assert slept == [1.0, 2.0]
    run = crashlab.load("last")
    assert run.status == "ok" and len(run.steps) == 1  # one step, three attempts
    step = run.steps[0]
    assert step.attributes[ATTEMPTS_ATTR] == 3
    entry = step.attributes[GUARD_ATTR][0]
    assert entry["policy"] == "retry_policy" and entry["action"] == "retried"
    assert [e["status"] for e in entry["detail"]["errors"]] == [429, 503]
    assert any(f.kind == "guard.retry_policy" and f.severity == "info" for f in run.findings)


def test_retry_jitter_is_seeded_and_bounded(_clear):
    p = RetryPolicy(max_attempts=5, base_delay=1.0, max_delay=3.0, seed=7)
    import random

    d = [p.delay(n, random.Random(7)) for n in (1, 2, 3, 4)]
    assert 0.5 <= d[0] <= 1.0 and 1.0 <= d[1] <= 2.0 and 1.5 <= d[3] <= 3.0
    assert d == [p.delay(n, random.Random(7)) for n in (1, 2, 3, 4)]


def test_retry_gives_up_and_does_not_retry_permanent_errors(_clear):
    slept = _clear
    client = FakeOpenAI([FakeAPIError("bad", 400), text_response(1, "never")])
    with pytest.raises(FakeAPIError), crashlab.guard(retry_policy=True):
        run_agent("hi", crashlab.wrap(client))
    assert slept == [] and len(client.script) == 1

    client = FakeOpenAI([FakeAPIError("x", 500)] * 3)
    with pytest.raises(FakeAPIError), crashlab.guard(retry_policy=RetryPolicy(2, jitter=False)):
        run_agent("hi", crashlab.wrap(client))
    step = crashlab.load("last").steps[0]
    assert step.error is not None and step.attributes[ATTEMPTS_ATTR] == 2
    assert step.attributes[GUARD_ATTR][0]["action"] == "gave_up"


def test_retry_recovers_from_injected_faults():
    with faults(llm_error(429, on_calls=[1])), crashlab.guard(retry_policy=2):
        assert run_agent("hi", _client(text_response(1, "done"))) == "done"
    step = crashlab.load("last").steps[0]
    assert step.error is None and step.attributes[ATTEMPTS_ATTR] == 2


def test_fallback_model_after_repeated_failures():
    client = FakeOpenAI([FakeAPIError("down", 503), FakeAPIError("down", 503),
                         text_response(1, "from fallback", model="small-model")])  # fmt: skip
    with crashlab.guard(retry_policy=RetryPolicy(2, jitter=False), fallback_model="small-model"):
        assert run_agent("hi", crashlab.wrap(client)) == "from fallback"
    assert [c["model"] for c in client.calls] == ["fake-model", "fake-model", "small-model"]
    run = crashlab.load("last")
    step = run.steps[0]
    assert step.attributes[FALLBACK_ATTR] == "small-model"
    assert step.request["params"]["model"] == "fake-model"  # the request as the agent sent it
    policies = [e["policy"] for e in step.attributes[GUARD_ATTR]]
    assert policies == ["retry_policy", "fallback_model"]


def test_is_transient():
    assert is_transient(FakeAPIError("x", 429)) and is_transient(TimeoutError())
    assert not is_transient(FakeAPIError("x", 401)) and not is_transient(ValueError())
    assert not is_transient(GuardViolation("x"))


# ---------------------------------------------------------------------------
# secret redaction
# ---------------------------------------------------------------------------


def test_block_secrets_in_llm_reply_and_tool_result():
    key = "sk-" + "proj-" + "B" * 30
    client = FakeOpenAI([tool_call_response(1, "leak", {"n": 1}),
                         text_response(2, f"your key is {key}")])  # fmt: skip
    with crashlab.guard(block_secrets_in_output=True):
        out = run_agent("hi", crashlab.wrap(client))
    assert key not in out and "[REDACTED:openai_api_key]" in out
    tool_msg = client.calls[1]["messages"][-1]["content"]
    assert "sk-proj-" not in tool_msg and "REDACTED" in tool_msg
    run = crashlab.load("last")
    assert "sk-proj-" not in json.dumps([s.response for s in run.steps], default=str)
    redactions = [e for _, p, e in _guard_entries(run) if p == "block_secrets_in_output"]
    assert redactions == ["redacted", "redacted"]


# ---------------------------------------------------------------------------
# async
# ---------------------------------------------------------------------------


def test_async_guard_retry_and_breaker(_clear):
    slept = _clear
    client = crashlab.wrap(AsyncFakeOpenAI([FakeAPIError("x", 429), text_response(1, "a"),
                                            text_response(2, "b")]))  # fmt: skip

    @crashlab.agent(name="async-bot", ignore_args=("c",))
    async def abot(c):
        r = await c.chat.completions.create(model="fake-model", messages=[])
        await c.chat.completions.create(model="fake-model", messages=[])
        return r.choices[0].message.content

    async def main():
        async with crashlab.guard(retry_policy=RetryPolicy(3, jitter=False), max_llm_calls=1):
            return await abot(client)

    with pytest.raises(GuardViolation) as info:
        asyncio.run(main())
    assert info.value.policy == "max_llm_calls" and slept == [0.5]

    @crashlab.guard(max_llm_calls=5)
    async def guarded():
        return await abot(crashlab.wrap(AsyncFakeOpenAI([text_response(1, "x"),
                                                         text_response(2, "y")])))  # fmt: skip

    assert asyncio.run(guarded()) == "x"


def test_concurrent_tasks_have_independent_guards():
    @crashlab.agent(name="tasks")
    async def tasks():
        async def one(key):
            async with crashlab.guard(max_tool_calls=1):
                lookup(key)
                await asyncio.sleep(0)
                with pytest.raises(GuardViolation):
                    lookup(key + "2")

        await asyncio.gather(one("a"), one("b"))

    asyncio.run(tasks())
    assert sorted(c[1]["key"] for c in CALLS) == ["a", "b"]


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


def test_strict_replay_does_not_retry_or_fall_back(_clear):
    slept = _clear
    client = FakeOpenAI([FakeAPIError("x", 503), text_response(1, "done")])
    with crashlab.guard(retry_policy=RetryPolicy(2, jitter=False), fallback_model="other"):
        run_agent("hi", crashlab.wrap(client))
    src = crashlab.load("last")
    slept.clear()
    with crashlab.guard(retry_policy=True, fallback_model="other", max_steps=100):
        replayed = crashlab.replay(src, overrides={"client": _client()})
    assert slept == []
    step = replayed.steps[0]
    assert step.source == "cassette" and step.attributes[ATTEMPTS_ATTR] == 2
    assert replayed.output == "done"


def test_strict_replay_reproduces_a_recorded_violation():
    with pytest.raises(GuardViolation), crashlab.guard(max_steps=2):
        run_agent("hi", _client(*_loop_script(5)))
    src = crashlab.load("last")
    CALLS.clear()
    # no guard active: the blocked calls are served (and raised) from the cassette
    replayed = crashlab.replay(src, overrides={"client": _client()})
    assert CALLS == [] and replayed.status == "error"
    last = replayed.steps[-1]
    assert last.source == "cassette" and last.error.type == "crashlab.guard.GuardViolation"
    assert last.attributes[BLOCKED_ATTR] is True
    assert [e["policy"] for e in last.attributes[GUARD_ATTR]] == ["max_steps"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_guard_report_cli():
    client = FakeOpenAI([FakeAPIError("x", 429), tool_call_response(1, "delete_user",
                         {"user_id": "u"}), text_response(2, "ok")])  # fmt: skip
    with crashlab.guard(retry_policy=RetryPolicy(2, jitter=False), tool_denylist="delete_*"):
        run_agent("hi", crashlab.wrap(client))
    res = runner.invoke(app, ["guard-report", "last"])
    assert res.exit_code == 0, res.output
    out = ANSI.sub("", res.output)
    assert "retry_policy" in out and "tool_denylist" in out and "blocked" in out
    res = runner.invoke(app, ["guard-report", "last", "--json"])
    data = json.loads(res.output)
    assert [i["policy"] for i in data["interventions"]] == ["retry_policy", "tool_denylist"]
    assert {f["kind"] for f in data["findings"]} == {"guard.retry_policy", "guard.tool_denylist"}

    res = runner.invoke(app, ["guard-report", "nope-not-a-run"])
    assert res.exit_code == 2


def test_guard_report_with_no_interventions():
    run_agent("hi", _client(text_response(1, "ok")))
    res = runner.invoke(app, ["guard-report"])
    assert res.exit_code == 0 and "no guard interventions" in ANSI.sub("", res.output)


def test_guard_report_help_and_panel():
    res = runner.invoke(app, ["guard-report", "--help"])
    out = ANSI.sub("", res.output)
    assert res.exit_code == 0 and "--json" in out and "interventions" in out
    top = ANSI.sub("", runner.invoke(app, ["--help"]).output)
    assert "guard-report" in top


def test_no_guard_no_overhead_path():
    assert active_scopes() == ()
    assert run_agent("hi", _client(text_response(1, "ok"))) == "ok"
    assert all(GUARD_ATTR not in s.attributes for s in crashlab.load("last").steps)
