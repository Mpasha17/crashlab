"""Fault injection: deterministic, recorded, not re-injected on strict replay."""

from __future__ import annotations

import contextlib
import json
import threading
import time

import pytest
from typer.testing import CliRunner

import crashlab
from crashlab.cli import app
from crashlab.faults import (
    FAULT_ATTR,
    HTML_502,
    current_policies,
    faults,
    llm_bad_tool_call,
    llm_empty,
    llm_error,
    llm_truncate,
    tool_delay,
    tool_error,
    tool_result,
)
from tests.fakes import FakeOpenAI, text_response, tool_call_response

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@crashlab.tool()
def lookup(key: str) -> dict:
    LOOKUP.append(key)
    return {"key": key, "value": 42, "total": 10}


LOOKUP: list[str] = []


def _script(tool: bool = True, text: str = "done"):
    if tool:
        return [tool_call_response(1, "lookup", {"key": "abc"}), text_response(2, text)]
    return [text_response(1, text)]


def _client(script=None):
    return crashlab.wrap(FakeOpenAI(list(script or _script())))


@crashlab.agent(name="fault-bot", ignore_args=("client",))
def run_agent(user: str, client, model: str = "fake-model") -> str:
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user}],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
    )
    msg = r.choices[0].message
    if not msg.tool_calls:
        return msg.content or ""
    tc = msg.tool_calls[0]
    args = json.loads(tc.function.arguments)
    try:
        result = lookup(**args)
    except Exception as exc:  # noqa: BLE001 - the agent under test recovers
        result = {"error": f"{type(exc).__name__}: {exc}"}
    r2 = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": user},
            {"role": "assistant", "tool_calls": [tc], "content": None},
            {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)},
        ],
    )
    return r2.choices[0].message.content or ""


@pytest.fixture(autouse=True)
def _clean():
    LOOKUP.clear()
    yield
    LOOKUP.clear()


def _fault_of(run, operation=None):
    steps = [s for s in run.steps if FAULT_ATTR in s.attributes]
    if operation:
        steps = [s for s in steps if s.operation == operation]
    return steps


# ---------------------------------------------------------------------------
# constructors / policy
# ---------------------------------------------------------------------------


def test_no_policy_is_inactive():
    assert current_policies() == ()


def test_faults_is_exported_and_constructors_hang_off_it():
    assert crashlab.faults is faults
    assert crashlab.faults.tool_error is tool_error
    assert callable(crashlab.faults)


def test_bad_constructor_arguments_raise():
    with pytest.raises(ValueError):
        tool_error(on_calls=[0])
    with pytest.raises(ValueError):
        tool_delay(seconds=-1)
    with pytest.raises(ValueError):
        tool_result(mode="replace")  # missing result
    with pytest.raises(ValueError):
        tool_result(mode="nope", result={})
    with pytest.raises(ValueError):
        llm_error(status=1)
    with pytest.raises(ValueError):
        llm_truncate(max_chars=-4)
    with pytest.raises(ValueError):
        tool_error(probability=1.5)
    with pytest.raises(ValueError):
        tool_error(agent="  ")


def test_rules_nest_innermost_wins():
    client = _client(_script())
    with faults(tool_result(tool="lookup", mode="empty", on_calls=[1])):
        with faults(tool_result(tool="lookup", mode="html_502", on_calls=[1])):
            out = run_agent("hi", client)
    run = crashlab.load("last")
    assert out == "done"
    tool = next(s for s in run.steps if s.operation == "execute_tool")
    assert tool.result == HTML_502
    assert tool.attributes[FAULT_ATTR]["mode"] == "html_502"
    assert not LOOKUP  # html_502 does not call the tool
    assert current_policies() == ()


# ---------------------------------------------------------------------------
# tool faults
# ---------------------------------------------------------------------------


def test_tool_error_is_recorded_and_agent_can_recover():
    client = _client(_script())
    with faults(tool_error(tool="lookup", on_calls=[1])):
        out = run_agent("hi", client)
    assert out == "done" and not LOOKUP
    run = crashlab.load("last")
    tool = _fault_of(run, "execute_tool")[0]
    assert tool.attributes[FAULT_ATTR]["kind"] == "tool_error"
    assert tool.error is not None and "injected fault" in (tool.error.message or "")
    assert run.status == "ok"


def test_tool_error_http_status_and_timeout():
    with crashlab.record("http"):
        with faults(tool_error(tool="lookup", status=502, on_calls=[1])):
            out = run_agent("hi", _client(_script()))
    tool = _fault_of(crashlab.load("last"), "execute_tool")[0]
    assert tool.attributes[FAULT_ATTR]["status"] == 502
    assert "502" in (tool.error.message or "")
    assert out == "done"

    LOOKUP.clear()
    with crashlab.record("to"):
        with faults(tool_error(tool="lookup", exc=TimeoutError, on_calls=[1])):
            run_agent("hi", _client(_script()))
    tool = _fault_of(crashlab.load("last"), "execute_tool")[0]
    assert tool.error.type.endswith("TimeoutError")


def test_tool_result_modes_replace_empty_html_truncated_drift():
    def once(mode, **kw):
        LOOKUP.clear()
        with crashlab.record(mode):
            with faults(tool_result(tool="lookup", mode=mode, on_calls=[1], **kw)):
                run_agent("hi", _client(_script()))
        return next(s for s in crashlab.load("last").steps if s.operation == "execute_tool")

    replaced = once("replace", result={"ok": False})
    assert replaced.result == {"ok": False} and not LOOKUP
    assert replaced.attributes[FAULT_ATTR]["kind"] == "tool_result"

    # empty does not call the tool, so there is no original value to type from → ""
    empty = once("empty")
    assert empty.result == "" and not LOOKUP

    html = once("html_502")
    assert (
        isinstance(html.result, str) and "502 Bad Gateway" in html.result and "nginx" in html.result
    )

    # truncated / schema_drift call the tool first
    cut = once("truncated")
    assert LOOKUP == ["abc"] and isinstance(cut.result, str) and not cut.result.endswith("}")
    LOOKUP.clear()
    drifted = once("schema_drift")
    assert LOOKUP == ["abc"]
    assert "total" not in drifted.result and drifted.result.get("amount_total") == 10
    assert "key_v2" in drifted.result


def test_tool_delay_sleeps_live_and_is_recorded(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    with crashlab.record("delay"):
        with faults(tool_delay(tool="lookup", seconds=0.05, on_calls=[1])):
            out = run_agent("hi", _client(_script()))
    assert out == "done" and slept == [0.05] and LOOKUP == ["abc"]
    tool = _fault_of(crashlab.load("last"), "execute_tool")[0]
    assert tool.attributes[FAULT_ATTR] == {
        "kind": "tool_delay",
        "call": 1,
        "triggers": 1,
        "tool": "lookup",
        "seconds": 0.05,
    }


def test_on_calls_after_calls_every_and_max_triggers():
    @crashlab.tool()
    def ping(n: int) -> int:
        return n

    def burst(rule, n=4):
        hits = []
        with crashlab.record("trig"):
            with faults(rule):
                for i in range(n):
                    try:
                        ping(i)
                        hits.append("ok")
                    except RuntimeError:
                        hits.append("err")
        return hits

    assert burst(tool_error(tool="ping", on_calls=[2, 4])) == ["ok", "err", "ok", "err"]
    assert burst(tool_error(tool="ping", after_calls=2, max_triggers=1)) == [
        "ok",
        "ok",
        "err",
        "ok",
    ]
    assert burst(tool_error(tool="ping", every=2)) == ["ok", "err", "ok", "err"]


def test_probability_with_seed_is_deterministic():
    @crashlab.tool()
    def ping(n: int) -> int:
        return n

    def run(seed):
        flags = []
        with crashlab.record("p"):
            with faults(tool_error(tool="ping", probability=0.5, seed=seed)):
                for i in range(20):
                    try:
                        ping(i)
                        flags.append(0)
                    except RuntimeError:
                        flags.append(1)
        return flags

    assert run(0) == run(0) and run(0) != run(1)
    assert 0 < sum(run(0)) < 20


def test_agent_filter_skips_other_agents():
    client = _client(_script())
    with crashlab.record("agents"):
        with faults(tool_error(tool="lookup", agent="other", on_calls=[1])):
            run_agent("hi", client)
    run = crashlab.load("last")
    tool = next(s for s in run.steps if s.operation == "execute_tool")
    assert FAULT_ATTR not in tool.attributes and LOOKUP == ["abc"]
    assert tool.agent == "fault-bot"

    LOOKUP.clear()
    client = _client(_script())
    # @agent only applies agent_scope when nested in an outer run; that labels the steps.
    with crashlab.record("agents2"):
        with faults(tool_error(tool="lookup", agent="fault-bot", on_calls=[1])):
            run_agent("hi", client)
    run = crashlab.load("last")
    tool = next(s for s in run.steps if s.operation == "execute_tool")
    assert tool.agent == "fault-bot"
    assert tool.attributes[FAULT_ATTR]["kind"] == "tool_error" and not LOOKUP


def test_star_tool_matches_any_name():
    with crashlab.record("star"):
        with faults(tool_result(mode="html_502", on_calls=[1])):
            run_agent("hi", _client(_script()))
    tool = next(s for s in crashlab.load("last").steps if s.operation == "execute_tool")
    assert "502" in tool.result


# ---------------------------------------------------------------------------
# LLM faults
# ---------------------------------------------------------------------------


def test_llm_error_is_recorded():
    with crashlab.record("llm-err"):
        client = _client(_script(tool=False))
        with (
            faults(llm_error(status=429, on_calls=[1])),
            pytest.raises(Exception, match="(?i)rate"),
        ):
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
    step = _fault_of(crashlab.load("last"), "chat")[0]
    assert step.attributes[FAULT_ATTR]["kind"] == "llm_error"
    assert step.attributes[FAULT_ATTR]["status"] == 429
    assert step.error is not None


def test_llm_empty_clears_text_and_tool_calls():
    with crashlab.record("empty"):
        client = _client([text_response(1, "hello there")])
        with faults(llm_empty(on_calls=[1])):
            r = client.chat.completions.create(
                model="m", messages=[{"role": "user", "content": "hi"}]
            )
    assert (r.choices[0].message.content or "") == ""
    assert not r.choices[0].message.tool_calls
    assert r.choices[0].finish_reason == "stop"
    step = _fault_of(crashlab.load("last"), "chat")[0]
    assert step.attributes[FAULT_ATTR]["kind"] == "llm_empty"


def test_llm_truncate_sets_finish_reason_length():
    with crashlab.record("trunc"):
        client = _client([text_response(1, "abcdefghij")])
        with faults(llm_truncate(max_chars=4, on_calls=[1])):
            r = client.chat.completions.create(
                model="m", messages=[{"role": "user", "content": "hi"}]
            )
    assert r.choices[0].message.content == "abcd"
    assert r.choices[0].finish_reason == "length"
    assert (
        _fault_of(crashlab.load("last"), "chat")[0].attributes[FAULT_ATTR]["kind"] == "llm_truncate"
    )


def test_llm_bad_tool_call_injects_malformed_args():
    with crashlab.record("bad"):
        client = _client([text_response(1, "hi")])
        with faults(llm_bad_tool_call(name="not_a_real_tool", arguments="{nope", on_calls=[1])):
            r = client.chat.completions.create(
                model="m", messages=[{"role": "user", "content": "hi"}]
            )
    tc = r.choices[0].message.tool_calls[0]
    assert tc.function.name == "not_a_real_tool"
    assert tc.function.arguments == "{nope"
    assert r.choices[0].finish_reason == "tool_calls"
    step = _fault_of(crashlab.load("last"), "chat")[0]
    assert step.attributes[FAULT_ATTR]["kind"] == "llm_bad_tool_call"


# ---------------------------------------------------------------------------
# replay does not re-inject; fork live steps do
# ---------------------------------------------------------------------------


def test_strict_replay_serves_a_faulted_cassette_without_reinjecting(monkeypatch):
    with faults(tool_error(tool="lookup", on_calls=[1])):
        run_agent("hi", _client(_script()))
    src = crashlab.load("last")
    assert _fault_of(src, "execute_tool")

    LOOKUP.clear()
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

    # A different policy would change the result if replay re-injected. It must not.
    with faults(tool_result(tool="lookup", mode="html_502"), tool_delay(tool="lookup", seconds=9)):
        replayed = crashlab.replay(src, overrides={"client": _client(_script())})
    assert slept == []
    tool = next(s for s in replayed.steps if s.operation == "execute_tool")
    assert tool.source == "cassette"
    assert tool.attributes[FAULT_ATTR]["kind"] == "tool_error"
    assert "html_502" not in str(tool.result)
    assert not LOOKUP  # the tool did not run; the recorded error was served


def test_fork_injects_on_live_steps_only():
    run_agent("hi", _client(_script()))
    src = crashlab.load("last")
    assert FAULT_ATTR not in src.steps[1].attributes  # clean recording

    live = _client(_script())
    LOOKUP.clear()
    with faults(tool_result(tool="lookup", mode="html_502", on_calls=[1])):
        branch = crashlab.fork(src, at_step=0, fn=run_agent, client=live, tools="live")
    tool = next(s for s in branch.steps if s.operation == "execute_tool")
    assert tool.source != "cassette"
    assert "502" in (tool.result or "")
    assert tool.attributes[FAULT_ATTR]["kind"] == "tool_result"
    # steps of a fork before at_step would be cassette; at 0 everything after can be live


def test_fork_does_not_inject_on_cassette_prefix():
    run_agent("hi", _client(_script()))
    src = crashlab.load("last")
    # fork at the last chat step: the tool step (idx 1) is before the fork and must
    # be served from the cassette, even with a tool fault active
    live = _client([text_response(9, "later")])
    with faults(tool_result(tool="lookup", mode="html_502")):
        branch = crashlab.fork(src, at_step=2, fn=run_agent, client=live, tools="live")
    tool = next(s for s in branch.steps if s.operation == "execute_tool")
    assert tool.source == "cassette"
    assert FAULT_ATTR not in tool.attributes
    assert tool.result == {"key": "abc", "value": 42, "total": 10}


def test_show_and_diff_surface_the_injected_fault():
    with faults(tool_result(tool="lookup", mode="html_502", on_calls=[1])):
        run_agent("hi", _client(_script()))
    faulted = crashlab.load("last")
    run_agent("hi", _client(_script()))
    clean = crashlab.load("last")

    runner = CliRunner()
    out = runner.invoke(app, ["show", faulted.run_id, "--step", str(_fault_of(faulted)[0].idx)])
    assert out.exit_code == 0, out.output
    assert "fault" in out.output.lower() and "html_502" in out.output

    listed = runner.invoke(app, ["show", faulted.run_id])
    assert listed.exit_code == 0 and "fault:tool_result" in listed.output

    d = crashlab.diff(clean, faulted)
    assert d.first_divergence is not None
    # the tool step's response (and the fault marker) differ
    sd = next(s for s in d.steps if s.idx == _fault_of(faulted)[0].idx)
    assert sd.status in ("response", "different")
    assert "fault" in (sd.fields or []) or "result" in (sd.fields or []) or sd.status == "response"


def test_policy_is_thread_safe():
    @crashlab.tool()
    def ping(n: int) -> int:
        return n

    errors = []

    def worker():
        try:
            for i in range(20):
                with contextlib.suppress(RuntimeError):
                    ping(i)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    with crashlab.record("threads"):
        with faults(tool_error(tool="ping", every=2)):
            threads = [threading.Thread(target=crashlab.bind_context(worker)) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
    assert errors == []
    run = crashlab.load("last")
    tool_steps = [s for s in run.steps if s.operation == "execute_tool"]
    assert len(tool_steps) == 80
    # every 2nd matching call across threads, counted on the shared rule
    injected = [s for s in tool_steps if FAULT_ATTR in s.attributes]
    assert len(injected) == 40
