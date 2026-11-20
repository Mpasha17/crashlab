"""crashlab.stress_run / crashlab stress: built-in chaos scenarios, offline."""

from __future__ import annotations

import json

from typer.testing import CliRunner

import crashlab
from crashlab.cli import app
from crashlab.stress import SCENARIOS, stress_run
from tests.fakes import FakeOpenAI, text_response
from tests.test_faults import LOOKUP, _script, run_agent

# reuse the recorded lookup tool and fault-bot agent from test_faults


def _client(script=None):
    return crashlab.wrap(FakeOpenAI(list(script or _script())))


def _record():
    LOOKUP.clear()
    run_agent("hi", _client(_script()))
    return crashlab.load("last")


def test_builtin_scenario_names():
    assert "tool_502" in SCENARIOS and "llm_rate_limit" in SCENARIOS


def test_stress_reports_survival_for_a_resilient_tool_fault():
    src = _record()
    # the agent catches tool errors / bad results and still answers from the next LLM turn
    report = stress_run(
        src,
        scenarios=["tool_502", "tool_empty", "tool_schema_drift"],
        fn=run_agent,
        client=_client(_script() * 3),
    )
    assert report.source_run_id == src.run_id
    assert [r.scenario for r in report.results] == [
        "tool_502",
        "tool_empty",
        "tool_schema_drift",
    ]
    assert all(r.target == "lookup" for r in report.results)
    assert all(r.outcome == "survived" and r.survived for r in report.results)
    assert report.survived
    for r in report.results:
        assert r.run_id
        trial = crashlab.load(r.run_id)
        assert any(s.attributes.get("crashlab.fault") for s in trial.steps)


def test_stress_marks_an_uncaught_llm_error_as_crashed():
    src = _record()
    report = stress_run(src, scenarios=["llm_rate_limit"], fn=run_agent, client=_client(_script()))
    assert len(report.results) == 1
    row = report.results[0]
    assert row.scenario == "llm_rate_limit" and row.outcome == "crashed"
    assert row.survived is False and report.survived is False
    assert row.run_id and crashlab.load(row.run_id).status == "error"


def test_stress_llm_empty_reply_completes():
    src = _record()
    # first live LLM call becomes an empty reply: the agent returns "" and ends ok
    report = stress_run(src, scenarios=["llm_empty_reply"], fn=run_agent, client=_client(_script()))
    row = report.results[0]
    assert row.outcome == "survived" and row.survived
    assert crashlab.load(row.run_id).output == ""


def test_needs_live_agent_when_there_is_no_entrypoint():
    with crashlab.record("no-ep"):

        @crashlab.tool()
        def ping() -> str:
            return "ok"

        ping()
    src = crashlab.load("last")
    report = stress_run(src, scenarios=["tool_empty"])
    assert report.results[0].outcome == "needs_live_agent"
    assert report.results[0].survived is False
    assert report.survived is False  # nothing executed


def test_skips_tool_scenarios_when_the_run_has_no_tools():
    with crashlab.record("no-tools"):
        client = _client([text_response(1, "hi")])
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
    src = crashlab.load("last")
    report = stress_run(src, scenarios=["tool_502", "llm_empty_reply"], fn=lambda: None, live=False)
    by = {r.scenario: r for r in report.results}
    assert by["tool_502"].outcome == "skipped"
    assert "no tool" in (by["tool_502"].detail or "")
    assert by["llm_empty_reply"].outcome == "needs_live_agent"


def test_unknown_scenario_raises():
    src = _record()
    try:
        stress_run(src, scenarios=["nope"])
    except ValueError as exc:
        assert "nope" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_cli_json_and_fail_on_crash():
    # a recorded block with no entry point: every scenario is needs_live_agent
    with crashlab.record("cli-ep"):

        @crashlab.tool()
        def ping() -> dict:
            return {"ok": True}

        ping()
    runner = CliRunner()
    res = runner.invoke(app, ["stress", "last", "--scenario", "tool_empty", "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["results"][0]["outcome"] == "needs_live_agent"
    assert data["results"][0]["scenario"] == "tool_empty"

    res = runner.invoke(app, ["stress", "last", "--scenario", "tool_empty", "--fail-on-crash"])
    # needs_live_agent is not a crash
    assert res.exit_code == 0, res.output
    assert "needs_live_agent" in res.output

    bad = runner.invoke(app, ["stress", "last", "--scenario", "nope"])
    assert bad.exit_code == 2 and "unknown" in bad.output

    missing = runner.invoke(app, ["stress", "does-not-exist"])
    assert missing.exit_code == 1


def test_cli_help_lists_stress():
    runner = CliRunner()
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0 and "stress" in res.output
    res = runner.invoke(app, ["stress", "--help"])
    assert res.exit_code == 0
    for flag in ("--scenario", "--at-step", "--agent", "--json", "--fail-on-crash"):
        assert flag in res.output
