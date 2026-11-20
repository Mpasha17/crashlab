"""crashlab.fuzz / crashlab fuzz: deterministic adversarial mutations."""

from __future__ import annotations

import json
import re

from typer.testing import CliRunner

import crashlab
from crashlab.cli import app
from crashlab.fuzz import MUTATIONS, fuzz_agent, fuzz_run, plan_cases
from tests.fakes import FakeOpenAI, text_response
from tests.test_faults import LOOKUP, _script, run_agent


def _ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _client(script=None):
    return crashlab.wrap(FakeOpenAI(list(script or _script())))


def _record():
    LOOKUP.clear()
    run_agent("hi", _client(_script()))
    return crashlab.load("last")


@crashlab.agent(name="echo-fuzz")
def echo(msg: str) -> str:
    return msg


@crashlab.agent(name="fragile-fuzz")
def fragile(msg: str) -> str:
    if "\x00" in msg or len(msg) > 4000:
        raise RuntimeError("boom")
    return "ok"


def test_builtin_mutation_names():
    assert MUTATIONS == ("prompt_injection", "tool_poison", "edge_inputs")


def test_plan_is_deterministic_for_a_seed():
    a = plan_cases(("prompt_injection", "edge_inputs"), max_cases=10, seed=0)
    b = plan_cases(("prompt_injection", "edge_inputs"), max_cases=10, seed=0)
    c = plan_cases(("prompt_injection", "edge_inputs"), max_cases=10, seed=1)
    assert [x["case"] for x in a] == [x["case"] for x in b]
    assert len(a) == 10
    kinds = {x["mutation"] for x in a}
    assert "prompt_injection" in kinds and "edge_inputs" in kinds
    assert [x["case"] for x in a] != [x["case"] for x in c] or [x["payload"] for x in a] != [
        x["payload"] for x in c
    ]


def test_unknown_mutation_raises():
    try:
        fuzz_agent(echo, mutations=("nope",), max_cases=1)
    except ValueError as exc:
        assert "nope" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_fuzz_agent_is_seeded_and_respects_max_cases():
    a = fuzz_agent(echo, inputs="hello", mutations=("prompt_injection", "edge_inputs"),
                   max_cases=6, seed=0)  # fmt: skip
    b = fuzz_agent(echo, inputs="hello", mutations=("prompt_injection", "edge_inputs"),
                   max_cases=6, seed=0)  # fmt: skip
    assert a.survived
    assert [r.case for r in a.results] == [r.case for r in b.results]
    assert len(a.results) == 6
    assert all(r.run_id and r.outcome == "survived" for r in a.results)
    # the original text is mutated (appended, prepended or replaced)
    assert any(
        r.preview
        and (
            "PWNED" in r.preview
            or "hello" not in r.preview
            or "SYSTEM" in r.preview
            or "Ignore" in r.preview
            or r.case == "empty"
        )
        for r in a.results
    )  # noqa: E501


def test_fuzz_agent_reports_crashes_on_edge_inputs():
    report = fuzz_agent(fragile, mutations=("edge_inputs",), max_cases=10, seed=0)
    outcomes = {r.outcome for r in report.results}
    assert "crashed" in outcomes
    crashed = [r for r in report.results if r.outcome == "crashed"]
    assert any("RuntimeError" in (r.detail or "") for r in crashed)
    assert report.survived is False


def test_fuzz_agent_failed_custom_oracle():
    def no_pwned(run):
        if run.output and "PWNED" in str(run.output):
            return "agent echoed the injection"
        return True

    report = fuzz_agent(
        echo,
        inputs="hello",
        mutations=("prompt_injection",),
        max_cases=8,
        seed=0,
        checks=[no_pwned],
    )
    failed = [r for r in report.results if r.outcome == "failed_checks"]
    # substitute / append of ignore_previous includes PWNED
    assert failed, [r.as_dict() for r in report.results]
    assert any("PWNED" in (r.detail or "") or "no_pwned" in (r.detail or "") for r in failed)
    assert report.survived is False


def test_fuzz_run_needs_live_agent_without_entrypoint():
    with crashlab.record("no-ep"):

        @crashlab.tool()
        def ping() -> dict:
            return {"ok": True}

        ping()
    src = crashlab.load("last")
    report = fuzz_run(src, mutations=("tool_poison",), max_cases=3, seed=0)
    assert report.source == src.run_id
    assert report.results
    assert all(r.outcome == "needs_live_agent" for r in report.results)
    assert report.survived is False


def test_fuzz_run_poisons_tool_results_with_a_live_agent():
    src = _record()
    report = fuzz_run(
        src,
        agent_fn=run_agent,
        mutations=("tool_poison",),
        max_cases=4,
        seed=0,
        client=_client(_script() * 8),
    )
    assert len(report.results) == 4
    executed = [r for r in report.results if r.outcome not in ("skipped", "needs_live_agent")]
    assert executed, [r.as_dict() for r in report.results]
    assert all(r.mutation == "tool_poison" and r.run_id for r in executed)
    trial = crashlab.load(executed[0].run_id)
    assert trial.forked_from == src.run_id
    tool = next(s for s in trial.steps if s.operation == "execute_tool")
    # the first planned poison payload was returned instead of {"key": ..., "value": 42}
    assert tool.result != {"key": "abc", "value": 42, "total": 10}


def test_fuzz_run_skips_tool_poison_when_the_run_has_no_tools():
    with crashlab.record("no-tools"):
        client = _client([text_response(1, "hi")])
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
    src = crashlab.load("last")
    report = fuzz_run(src, agent_fn=lambda: None, mutations=("tool_poison",), max_cases=2, seed=0)
    assert all(r.outcome == "skipped" for r in report.results)
    assert any("no tool" in (r.detail or "") for r in report.results)


def test_cli_json_and_fail_on_finding():
    with crashlab.record("cli-ep"):

        @crashlab.tool()
        def ping() -> dict:
            return {"ok": True}

        ping()
    runner = CliRunner()
    res = runner.invoke(
        app, ["fuzz", "last", "--mutation", "tool_poison", "--max-cases", "2", "--json"]
    )
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["results"][0]["outcome"] == "needs_live_agent"
    assert data["results"][0]["mutation"] == "tool_poison"
    assert data["seed"] == 0

    res = runner.invoke(
        app, ["fuzz", "last", "--mutation", "tool_poison", "--max-cases", "1", "--fail-on-finding"]
    )
    # needs_live_agent is not a finding
    assert res.exit_code == 0, res.output
    assert "needs_live_agent" in _ansi(res.output)

    bad = runner.invoke(app, ["fuzz", "last", "--mutation", "nope"])
    assert bad.exit_code == 2 and "unknown" in _ansi(bad.output)

    missing = runner.invoke(app, ["fuzz", "does-not-exist"])
    assert missing.exit_code == 1


def test_cli_help_lists_fuzz():
    runner = CliRunner()
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0 and "fuzz" in _ansi(res.output)
    res = runner.invoke(app, ["fuzz", "--help"])
    assert res.exit_code == 0
    clean = _ansi(res.output)
    for flag in (
        "--mutation",
        "--max-cases",
        "--seed",
        "--agent",
        "--checks",
        "--json",
        "--fail-on-finding",
    ):
        assert flag in clean
