"""Runtime checks (built-in detectors): findings, config, circuit breaker. Offline."""

from __future__ import annotations

import json
from typing import Any

import pytest

import crashlab
from crashlab.checks import (
    ChecksConfig,
    CircuitBreakerError,
    classify_result,
    evaluate,
    replace_findings,
    resolve_config,
)
from crashlab.checks.schema import validate
from crashlab.models import Finding
from crashlab.store import Store
from tests.fakes import FakeOpenAI, FakeStream, text_response, tool_call_response

HTML_502 = "<html><head><title>502 Bad Gateway</title></head><body><h1>502</h1></body></html>"
RESULTS: list[Any] = []
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look something up",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}, "n": {"type": "integer"}},
                "required": ["q"],
            },
        },
    }
]


@pytest.fixture(autouse=True)
def _no_pyproject(monkeypatch):
    monkeypatch.setenv("CRASHLAB_NO_PYPROJECT", "1")
    RESULTS.clear()


@crashlab.tool
def lookup(q: str, n: int = 1) -> Any:
    item = RESULTS.pop(0) if RESULTS else {"q": q, "value": 1}
    if isinstance(item, BaseException):
        raise item
    return item


def _loop(client: Any, max_turns: int = 8) -> str:
    messages: list[dict[str, Any]] = [{"role": "user", "content": "go"}]
    for _ in range(max_turns):
        resp = client.chat.completions.create(model="m1", messages=messages, tools=TOOLS)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or ""
        messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [tc.model_dump(exclude_none=True) for tc in msg.tool_calls],
            }
        )
        for tc in msg.tool_calls:
            try:
                out = lookup(**json.loads(tc.function.arguments))
            except Exception as exc:  # agents commonly do this; the breaker must survive it
                out = f"ERROR: {exc}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out)})
    return "gave up"


def _calls(n: int, args: dict[str, Any] | None = None) -> list[Any]:
    return [tool_call_response(i + 1, "lookup", args or {"q": "x"}) for i in range(n)]


def _run(script: list[Any], checks: Any = None, name: str = "t", **kw: Any) -> crashlab.Run:
    client = crashlab.wrap(FakeOpenAI(script))
    try:
        with crashlab.record(name, checks=checks):
            _loop(client, **kw)
    except CircuitBreakerError:
        pass
    return crashlab.load("last")


def _kinds(run: crashlab.Run) -> dict[str, str]:
    return {f.kind: f.severity for f in run.findings}


# --- detectors ---------------------------------------------------------------------------


def test_clean_run_has_no_findings():
    run = _run([*_calls(1), text_response(9, "done")])
    assert run.findings == []
    assert run.status == "ok"


def test_step_repetition_same_args_same_result():
    run = _run([*_calls(3), text_response(9, "done")])
    (f,) = run.findings_of("step_repetition")
    assert f.severity == "warn"
    assert f.steps == [1, 3, 5]
    assert "lookup was called 3 times with the same arguments" in f.message
    assert "same result" in f.message
    assert f.evidence[0].quote == '{"q": "x", "n": 1}'  # verbatim recorded args
    assert f.evidence[0].step_idx == 1


def test_step_repetition_critical_at_five_and_different_args_are_fine():
    run = _run([*_calls(5), text_response(9, "done")])
    assert run.findings_of("step_repetition")[0].severity == "critical"
    script = [tool_call_response(i, "lookup", {"q": f"x{i}"}) for i in range(5)]
    assert _run([*script, text_response(9, "done")]).findings_of("step_repetition") == []


def test_step_repetition_threshold_configurable_and_ignore_tools():
    script = [*_calls(3), text_response(9, "done")]
    assert _run(script, checks={"loop": {"threshold": 4}}).findings_of("step_repetition") == []
    off = {"step_repetition": {"ignore_tools": ["lookup"]}}
    assert _run(list(script), checks=off).findings_of("step_repetition") == []


def test_tool_error_streak_and_html_results():
    RESULTS.extend([HTML_502] * 3)
    run = _run([*_calls(3), text_response(9, "sorry")])
    kinds = _kinds(run)
    assert kinds["tool_error_streak"] == "error"
    assert kinds["suspicious_tool_result"] == "warn"
    assert kinds["step_repetition"] == "warn"
    (streak,) = run.findings_of("tool_error_streak")
    assert streak.steps == [1, 3, 5]
    assert "HTML page (502 Bad Gateway)" in streak.message
    sus = run.findings_of("suspicious_tool_result")[0]
    assert "<title>502 Bad Gateway</title>" in sus.evidence[0].quote


def test_tool_error_streak_exceptions_and_reset_on_success():
    RESULTS.extend([KeyError("a"), KeyError("b"), {"ok": 1}, KeyError("c")])
    run = _run([*_calls(4), text_response(9, "done")])
    assert run.findings_of("tool_error_streak") == []  # 2 then a success then 1
    RESULTS.extend([KeyError("a"), KeyError("b"), KeyError("c")])
    run = _run([tool_call_response(i, "lookup", {"q": str(i)}) for i in range(3)]
               + [text_response(9, "done")])  # fmt: skip
    (f,) = run.findings_of("tool_error_streak")
    assert f.severity == "error" and f.steps == [1, 3, 5]


def test_bad_tool_args_schema_unknown_tool():
    run = _run([tool_call_response(1, "lookup", {"n": "two"}), text_response(9, "done")])
    (f,) = run.findings_of("bad_tool_args")
    assert f.severity == "warn" and f.step == 0
    assert "missing required 'q'" in f.message
    assert "n: expected integer" in f.message


def test_bad_tool_args_unknown_tool_is_error():
    from crashlab.checks.engine import CheckEngine

    step = crashlab.Step(
        run_id="r", idx=0, span_id="s", operation="chat", name="m", request_hash="h",
        request={"tools": [{"name": "lookup", "parameters": {"type": "object"}}]},
        response={"tool_calls": [{"id": "1", "name": "delete_all", "arguments": {}}]},
    )  # fmt: skip
    run = crashlab.Run(run_id="r", name="t", steps=[step])
    (f,) = evaluate(run, ChecksConfig())
    assert f.kind == "bad_tool_args" and f.severity == "error"
    assert "unknown tool 'delete_all'" in f.message
    assert CheckEngine  # imported for the public API


@pytest.mark.parametrize(
    ("result", "reason"),
    [
        (HTML_502, "html"),
        ("Internal Server Error", "error_body"),
        ("", "empty"),
        ('{"items": [1, 2, 3', "truncated"),
        ("The list goes on...", "truncated"),
    ],
)
def test_classify_result(result, reason):
    assert [i.reason for i in classify_result(result)] == [reason]


@pytest.mark.parametrize("result", [{"a": 1}, "All good.", '{"a": 1}', [1, 2], 0])
def test_classify_result_clean(result):
    assert classify_result(result) == []


def test_schema_drift_within_run():
    RESULTS.extend([{"id": 1, "total": 5}, {"id": 2, "total": 6}, {"id": 3, "amount_total": 7}])
    run = _run([tool_call_response(i, "lookup", {"q": str(i)}) for i in range(3)]
               + [text_response(9, "done")])  # fmt: skip
    (f,) = run.findings_of("suspicious_tool_result")
    assert f.step == 5
    assert "missing 'total'" in f.message and "new 'amount_total'" in f.message


def test_schema_drift_against_earlier_runs():
    for i in range(2):
        RESULTS.append({"id": i, "total": 5})
        _run([tool_call_response(1, "lookup", {"q": "a"}), text_response(9, "ok")], name="b")
    RESULTS.append({"id": 9, "amount_total": 5})
    run = _run([tool_call_response(1, "lookup", {"q": "a"}), text_response(9, "ok")], name="b")
    (f,) = run.findings_of("suspicious_tool_result")
    assert "vs 2 earlier result(s)" in f.message
    other = _run([tool_call_response(1, "lookup", {"q": "a"}), text_response(9, "ok")], name="c")
    assert other.findings == []  # baselines are per agent name


def test_budget_warn_and_critical():
    run = _run([*_calls(1), text_response(9, "done")], checks={"max_steps": 3})
    # 3 steps >= 80% of 3: a warning at the step that reached it, no stop
    assert [(f.severity, f.step) for f in run.findings_of("budget")] == [("warn", 2)]
    run = _run([tool_call_response(i, "lookup", {"q": str(i)}) for i in range(3)]
               + [text_response(9, "done")], checks={"budget": {"max_tool_calls": 2}})  # fmt: skip
    crit = [f for f in run.findings_of("budget") if f.severity == "critical"]
    assert crit and "3 tool calls > max_tool_calls 2" in crit[0].message
    assert run.status == "ok"  # no breaker unless asked


def test_budget_cost_unknown_is_info_not_guessed():
    from crashlab.checks.engine import CheckEngine  # noqa: F401

    step = crashlab.Step(
        run_id="r", idx=0, span_id="s", operation="chat", name="mystery-model",
        request_hash="h", attributes={"server.address": "api.example.com",
        "gen_ai.request.model": "mystery-model", "gen_ai.usage.input_tokens": 10,
        "gen_ai.usage.output_tokens": 5}, response={"messages": []},
    )  # fmt: skip
    run = crashlab.Run(run_id="r", name="t", steps=[step])
    (f,) = evaluate(run, resolve_config({"max_cost_usd": 0.01}))
    assert f.kind == "budget" and f.severity == "info" and "unknown" in f.message


def test_length_cutoff():
    step = crashlab.Step(
        run_id="r", idx=0, span_id="s", operation="chat", name="m", request_hash="h",
        response={"finish_reason": "length", "messages": [{"role": "assistant",
                  "content": "The answer is"}]},
    )  # fmt: skip
    (f,) = evaluate(crashlab.Run(run_id="r", name="t", steps=[step]), ChecksConfig())
    assert f.kind == "length_cutoff" and "finish_reason=length" in f.message
    assert f.evidence[0].quote == "The answer is"


def test_identical_text_replies():
    run = _run([*_calls(1, {"q": "a"}), text_response(2, "hm")])
    assert run.findings_of("step_repetition") == []


# --- schema validator ----------------------------------------------------------------------


def test_schema_validator():
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "number", "minimum": 0},
            "b": {"type": "string", "enum": ["x", "y"]},
            "c": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["a"],
        "additionalProperties": False,
    }
    assert validate({"a": 1, "b": "x", "c": [1, 2]}, schema) == []
    problems = validate({"a": "1", "b": "z", "c": [1, "2"], "d": 0}, schema)
    text = "; ".join(problems)
    assert "a: expected number" in text and "b:" in text and "c[1]" in text and "'d'" in text
    assert ["missing required 'a'" in p for p in validate({}, schema)] == [True]
    assert validate(True, {"type": "integer"})  # bool is not an integer
    assert validate(3, {"type": ["integer", "null"]}) == []


# --- config ----------------------------------------------------------------------------------


def test_config_aliases_and_merge():
    cfg = resolve_config({"loop": {"threshold": 4}, "max_steps": 9, "length_cutoff": False})
    assert cfg.step_repetition.threshold == 4
    assert cfg.budget.max_steps == 9
    assert cfg.length_cutoff.enabled is False
    assert resolve_config(False).enabled is False
    given = ChecksConfig(circuit_breaker=True)
    assert resolve_config(given) is given


def test_pyproject_config_precedence(tmp_path, monkeypatch):
    monkeypatch.delenv("CRASHLAB_NO_PYPROJECT")
    (tmp_path / "pyproject.toml").write_text(
        "[tool.crashlab.checks]\nmax_steps = 7\ncircuit_breaker = true\n"
        "[tool.crashlab.checks.step_repetition]\nthreshold = 6\n"
    )
    sub = tmp_path / "pkg"
    sub.mkdir()
    monkeypatch.chdir(sub)  # found by walking up
    cfg = resolve_config(None)
    assert (cfg.budget.max_steps, cfg.circuit_breaker, cfg.step_repetition.threshold) == (
        7,
        True,
        6,
    )
    cfg = resolve_config({"max_steps": 3})  # record(checks=...) wins over pyproject
    assert (cfg.budget.max_steps, cfg.step_repetition.threshold) == (3, 6)
    monkeypatch.setenv("CRASHLAB_NO_PYPROJECT", "1")
    assert resolve_config(None).budget.max_steps is None


def test_bad_config_is_rejected():
    with pytest.raises(ValueError):
        resolve_config({"step_repetition": {"threshold": "lots"}})


# --- circuit breaker ----------------------------------------------------------------------


def test_breaker_off_by_default():
    run = _run([*_calls(6), text_response(9, "done")])
    assert run.findings_of("step_repetition")[0].severity == "critical"
    assert run.status == "ok"
    assert not any(f.stopped_run for f in run.findings)


def test_breaker_stops_at_critical_tool_step():
    fake = FakeOpenAI([*_calls(8), text_response(9, "done")])
    with pytest.raises(CircuitBreakerError) as err:
        with crashlab.record("t", checks={"circuit_breaker": True}):
            _loop(crashlab.wrap(fake))
    assert "[step_repetition]" in str(err.value)
    run = crashlab.load("last")
    assert run.status == "error" and run.outcome == "fail"
    assert run.outcome_reason.startswith("stopped by crashlab: [step_repetition]")
    assert run.steps[-1].idx == 9  # 5th identical lookup; nothing ran after it
    assert len(fake.calls) == 5
    (stopper,) = [f for f in run.findings if f.stopped_run]
    assert stopper.kind == "step_repetition" and stopper.severity == "critical"


def test_breaker_survives_except_exception():
    """Agents that catch Exception around tools must not swallow the stop."""
    with pytest.raises(CircuitBreakerError):
        with crashlab.record("t", checks={"circuit_breaker": True, "max_tool_calls": 1}):
            _loop(crashlab.wrap(FakeOpenAI([*_calls(3), text_response(9, "x")])))
    assert crashlab.load("last").status == "error"


def test_breaker_on_chat_step_budget():
    fake = FakeOpenAI([*_calls(3), text_response(9, "done")])
    with pytest.raises(CircuitBreakerError):
        with crashlab.record("t", checks={"circuit_breaker": True, "max_llm_calls": 1}):
            _loop(crashlab.wrap(fake))
    run = crashlab.load("last")
    assert [s.idx for s in run.steps] == [0, 1, 2]  # stopped right after the 2nd LLM call
    assert run.findings_of("budget", min_severity="critical")[0].stopped_run


def test_breaker_defers_on_streams():
    from tests.fakes import chunks_for_text

    client = crashlab.wrap(FakeOpenAI([FakeStream(chunks_for_text("hi"))] * 1))
    with pytest.raises(CircuitBreakerError):
        with crashlab.record("t", checks={"circuit_breaker": True, "max_llm_calls": 0}):
            stream = client.chat.completions.create(model="m1", messages=[], stream=True)
            assert len(list(stream)) > 1  # the stream is consumed in full, not cut
            client.chat.completions.create(model="m1", messages=[])  # next call is blocked
    run = crashlab.load("last")
    assert len(run.steps) == 1 and run.status == "error"


def test_end_of_run_findings_never_trip():
    with crashlab.record("t", checks={"circuit_breaker": True, "max_seconds": 0}):
        pass
    run = crashlab.load("last")
    assert run.status == "ok"


def test_breaker_replay_parity(crashlab_dir):
    from crashlab.replay import replay

    with pytest.raises(CircuitBreakerError):
        _breaker_agent(crashlab.wrap(FakeOpenAI([*_calls(8), text_response(9, "x")])))
    src = crashlab.load("last")
    rep = replay(src.run_id)
    assert rep.status == "error" and len(rep.steps) == len(src.steps)
    assert any(f.stopped_run for f in rep.findings)


@crashlab.agent(name="breaker-agent", ignore_args=("client",), checks={"circuit_breaker": True})
def _breaker_agent(client: Any = None) -> str:
    # in a replay the client is not recorded (ignore_args); responses come from the cassette
    return _loop(client or crashlab.wrap(FakeOpenAI([])))


def test_check_save_keeps_stopped_marker():
    with pytest.raises(CircuitBreakerError):
        with crashlab.record("t", checks={"circuit_breaker": True, "max_steps": 1}):
            _loop(crashlab.wrap(FakeOpenAI([*_calls(2), text_response(9, "x")])))
    run = crashlab.load("last")
    fresh = evaluate(run, resolve_config(run.check_config))
    assert not any(f.stopped_run for f in fresh)  # evaluation is opinion, not history
    replace_findings(run, fresh)
    assert [f.kind for f in run.findings if f.stopped_run] == ["budget"]
    replace_findings(run, [])  # config no longer produces it: the original is kept
    assert [f.kind for f in run.findings if f.stopped_run] == ["budget"]


# --- storage / legacy ---------------------------------------------------------------------


def test_findings_stored_indexed_and_legacy_flags():
    RESULTS.extend([HTML_502] * 3)
    run = _run([*_calls(3), text_response(9, "sorry")])
    rows = Store().list_runs(flagged="error")
    assert [r["run_id"] for r in rows] == [run.run_id]
    assert rows[0]["max_severity"] == "error"
    assert Store().list_runs(flagged="critical") == []
    legacy = crashlab.Run.model_validate(
        {"run_id": "r", "name": "n", "flags": [{"detector": "fork", "step_idx": 2,
                                                "message": "m"}]}
    )  # fmt: skip
    assert legacy.findings[0].kind == "fork" and legacy.findings[0].step == 2
    assert legacy.flags == legacy.findings
    assert Finding(kind="x", message="m").severity == "warn"


def test_checks_disabled():
    RESULTS.extend([HTML_502] * 3)
    assert _run([*_calls(3), text_response(9, "x")], checks=False).findings == []


def _tool_step(idx: int, name: str = "get_price") -> crashlab.Step:
    return crashlab.Step(
        run_id="r", idx=idx, span_id=f"s{idx}", operation="execute_tool", name=name,
        request_hash="h", request={"args": {"item": "widget"}}, response={"result": "4.5"},
    )  # fmt: skip


def test_no_llm_calls_flags_tools_without_a_recorded_model_call():
    # live finding: DSPy 3.4's default lm15 engine bypasses LiteLLM, so only the tool
    # calls were recorded and nothing said so
    run = crashlab.Run(run_id="r", name="t", steps=[_tool_step(0), _tool_step(1, "place_order")])
    (f,) = evaluate(run, ChecksConfig())
    assert f.kind == "no_llm_calls" and f.severity == "warn" and f.steps == [0, 1]
    assert "no LLM call was recorded" in f.message


def test_no_llm_calls_is_quiet_with_a_chat_step_or_when_disabled():
    chat = crashlab.Step(
        run_id="r", idx=0, span_id="c", operation="chat", name="m", request_hash="h",
        response={"messages": [{"role": "assistant", "content": "hi"}]},
    )  # fmt: skip
    run = crashlab.Run(run_id="r", name="t", steps=[chat, _tool_step(1)])
    assert [f for f in evaluate(run, ChecksConfig()) if f.kind == "no_llm_calls"] == []
    lone = crashlab.Run(run_id="r", name="t", steps=[_tool_step(0)])
    assert evaluate(lone, resolve_config({"no_llm_calls": False})) == []


@pytest.mark.parametrize(
    "result",
    [
        '{"status": "rejected", "error": "insufficient stock for gizmo: 2 available"}',
        '{"status": "failed", "detail": "card declined"}',
        {"status": "rejected"},
    ],
)
def test_json_text_and_failed_status_words_are_error_bodies(result):
    # live (PydanticAI, out-of-stock backend): place_order returned JSON *text* with
    # "status": "rejected" 6 times in a row (varied quantities) and nothing was flagged
    assert [i.reason for i in classify_result(result)] == ["error_body"]


@pytest.mark.parametrize("result", ['{"status": "ok", "order_id": "O-1"}', {"status": "done"}])
def test_ok_status_words_are_clean(result):
    assert classify_result(result) == []


def test_rejected_json_text_results_make_a_streak():
    rejected = '{"status": "rejected", "error": "insufficient stock: 2 available"}'
    RESULTS.extend([rejected] * 3)
    run = _run([tool_call_response(i, "lookup", {"q": str(i)}) for i in range(3)]
               + [text_response(9, "sorry")])  # fmt: skip
    (f,) = run.findings_of("tool_error_streak")
    assert f.steps == [1, 3, 5] and "an error object" in f.message
