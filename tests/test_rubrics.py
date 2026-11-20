"""Weighted rubric scoring (crashlab.rubrics / crashlab score / eval --rubric). Offline."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

import pytest
from typer.testing import CliRunner

import crashlab
from crashlab.cli import app
from crashlab.models import CheckResult, Finding, Run, Step, Usage
from crashlab.rubrics import (
    DEFAULT_RUBRIC,
    Rubric,
    RubricCriterion,
    RubricError,
    grade_for,
    load_rubric,
    parse_rubric,
    score_run,
    score_runs,
)
from crashlab.store import Store
from crashlab.suite import eval_runs

T0 = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
ENV = {"COLUMNS": "220", "TERM": "xterm-256color", "NO_COLOR": "1"}
runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


def chat(run_id: str, idx: int, *, tokens: int = 100, cost: float = 0.001) -> Step:
    return Step(
        run_id=run_id,
        idx=idx,
        span_id=f"s{idx}",
        operation="chat",
        name="fake-model",
        attributes={
            "gen_ai.request.model": "fake-model",
            "gen_ai.usage.input_tokens": tokens,
            "gen_ai.usage.output_tokens": 10,
            "crashlab.usage.cost_usd": cost,
        },
        request={"messages": [{"role": "user", "content": "hi"}]},
        response={"messages": [{"role": "assistant", "content": "ok"}]},
        request_hash=f"h{idx}",
        started_at=T0 + timedelta(seconds=idx),
        duration_ms=100.0,
    )


def tool(run_id: str, idx: int, name: str, *, args: dict | None = None) -> Step:
    return Step(
        run_id=run_id,
        idx=idx,
        span_id=f"t{idx}",
        operation="execute_tool",
        name=name,
        attributes={"gen_ai.tool.name": name},
        request={"args": args or {}},
        response={"result": "ok"},
        request_hash=f"th{idx}",
        started_at=T0 + timedelta(seconds=idx),
        duration_ms=20.0,
    )


def make_run(
    rid: str,
    *,
    name: str = "bot",
    outcome: str = "pass",
    status: str = "ok",
    findings: list[Finding] | None = None,
    checks: list[CheckResult] | None = None,
    tools: list[str] | None = None,
    cost: float = 0.002,
    wall_time_ms: float = 2000.0,
    n_chats: int = 1,
) -> Run:
    steps: list[Step] = [chat(rid, i, cost=cost / max(n_chats, 1)) for i in range(n_chats)]
    for i, tool_name in enumerate(tools or [], start=n_chats):
        steps.append(tool(rid, i, tool_name))
    started = T0
    ended = started + timedelta(milliseconds=wall_time_ms)
    return Run(
        run_id=rid,
        name=name,
        status=status,
        outcome=outcome,
        started_at=started,
        ended_at=ended,
        findings=list(findings or []),
        checks=list(checks or []),
        steps=steps,
        usage=Usage(
            input_tokens=100 * n_chats,
            output_tokens=10 * n_chats,
            total_tokens=110 * n_chats,
            cost_usd=cost,
            steps=len(steps),
            llm_calls=n_chats,
            tool_calls=len(tools or []),
            wall_time_ms=wall_time_ms,
        ),
    )


def put(store: Store, run: Run) -> Run:
    store.begin_run(run)
    for step in run.steps:
        store.add_step(step)
    store.end_run(run)
    return store.load_run(run.run_id)


def test_grade_for_cutoffs():
    assert grade_for(100) == "A" and grade_for(90) == "A"
    assert grade_for(89.9) == "B" and grade_for(80) == "B"
    assert grade_for(70) == "C" and grade_for(60) == "D"
    assert grade_for(59.9) == "F" and grade_for(0) == "F"


def test_default_rubric_weights_and_clean_run():
    weights = {c.name: c.weight for c in DEFAULT_RUBRIC.criteria}
    assert weights == {
        "outcome": 0.35,
        "safety": 0.25,
        "tool_efficiency": 0.15,
        "cost": 0.15,
        "latency": 0.10,
    }
    assert DEFAULT_RUBRIC.min_score == 70.0
    run = make_run("RUNRUB00000000000000000001", tools=["get_order"])
    result = score_run(run)
    assert result.score == pytest.approx(100.0)
    assert result.grade == "A" and result.passed is True
    assert {c.name for c in result.breakdown} == set(weights)
    assert result.run_id == run.run_id


def test_outcome_pass_and_error():
    ok = score_run(make_run("a", outcome="pass"))
    assert next(c.score for c in ok.breakdown if c.kind == "outcome_pass") == 1.0
    failed = score_run(make_run("b", outcome="fail"))
    assert next(c.score for c in failed.breakdown if c.kind == "outcome_pass") == 0.0
    errored = score_run(make_run("c", status="error", outcome="unknown"))
    assert next(c.score for c in errored.breakdown if c.kind == "outcome_pass") == 0.0
    checked = score_run(
        make_run("d", checks=[CheckResult(name="cap", status="fail", message="too much")])
    )
    assert next(c.score for c in checked.breakdown if c.kind == "outcome_pass") == 0.0


def test_no_findings_partial_credit_for_warn_only():
    clean = make_run("a")
    warned = make_run(
        "b", findings=[Finding(kind="loop", severity="warn", message="repeated step")]
    )
    broken = make_run(
        "c", findings=[Finding(kind="budget", severity="error", message="over budget")]
    )
    rubric = Rubric(
        criteria=[
            RubricCriterion(name="safety", kind="no_findings", target="error", partial_credit=0.5)
        ]
    )
    assert score_run(clean, rubric).score == pytest.approx(100.0)
    assert score_run(warned, rubric).score == pytest.approx(50.0)
    assert score_run(broken, rubric).score == pytest.approx(0.0)

    strict = Rubric(criteria=[RubricCriterion(name="safety", kind="no_findings", target="warn")])
    assert score_run(warned, strict).score == pytest.approx(0.0)


def test_cost_and_latency_linear_and_threshold():
    run = make_run("a", cost=0.05, wall_time_ms=5000.0)
    linear = Rubric(
        criteria=[
            RubricCriterion(name="cost", kind="cost_budget", target=0.01, max_value=0.09),
        ]
    )
    # 0.05 is halfway between 0.01 and 0.09
    assert score_run(run, linear).score == pytest.approx(50.0)
    under = make_run("b", cost=0.005)
    assert score_run(under, linear).score == pytest.approx(100.0)
    over = make_run("c", cost=0.20)
    assert score_run(over, linear).score == pytest.approx(0.0)

    threshold = Rubric(
        criteria=[RubricCriterion(name="lat", kind="latency_budget", max_value=3000)]
    )
    assert score_run(make_run("d", wall_time_ms=2000), threshold).score == pytest.approx(100.0)
    assert score_run(make_run("e", wall_time_ms=4000), threshold).score == pytest.approx(0.0)

    aliases = Rubric(
        criteria=[
            RubricCriterion(
                name="cost",
                kind="cost_budget",
                target_cost_usd=0.01,
                max_cost_usd=0.09,
            )
        ]
    )
    assert score_run(run, aliases).score == pytest.approx(50.0)


def test_tool_efficiency_expected_and_redundant():
    expected = ["get_order", "issue_refund"]
    rubric = Rubric(
        criteria=[RubricCriterion(name="tools", kind="tool_efficiency", target=expected)]
    )
    perfect = make_run("a", tools=["get_order", "issue_refund"])
    assert score_run(perfect, rubric).score == pytest.approx(100.0)

    missing = make_run("b", tools=["get_order"])
    # 1 hit / (2 expected + 0 extra + 0 redundant) = 0.5
    assert score_run(missing, rubric).score == pytest.approx(50.0)

    extra = make_run("c", tools=["get_order", "issue_refund", "delete_user"])
    # 2 / (2 + 1 extra + 0) = 2/3
    assert score_run(extra, rubric).score == pytest.approx(100.0 * 2 / 3)

    redundant = make_run("d", tools=["get_order", "issue_refund", "get_order"])
    # 2 / (2 + 0 + 1 redundant) = 2/3
    assert score_run(redundant, rubric).score == pytest.approx(100.0 * 2 / 3)

    # no expected list: unique / total
    repeats = Rubric(criteria=[RubricCriterion(name="tools", kind="tool_efficiency")])
    assert score_run(redundant, repeats).score == pytest.approx(100.0 * 2 / 3)
    assert score_run(make_run("e"), repeats).score == pytest.approx(100.0)


def test_rule_criterion_reuses_assertions():
    rubric = Rubric(
        criteria=[
            RubricCriterion(
                name="must_refund",
                kind="rule",
                rule={"type": "must_call", "tools": ["issue_refund"]},
            )
        ]
    )
    ok = make_run("a", tools=["get_order", "issue_refund"])
    bad = make_run("b", tools=["get_order"])
    assert score_run(ok, rubric).score == pytest.approx(100.0)
    assert score_run(bad, rubric).score == pytest.approx(0.0)

    many = Rubric(
        criteria=[
            RubricCriterion(
                name="policy",
                kind="rules",
                rule=[
                    {"type": "must_call", "tools": ["get_order"]},
                    {"type": "must_not_call", "tools": ["delete_user"]},
                ],
            )
        ]
    )
    assert score_run(ok, many).score == pytest.approx(100.0)
    assert score_run(make_run("c", tools=["get_order", "delete_user"]), many).score == 0.0


def test_custom_check_callable_forms():
    run = make_run("a")

    def as_bool(r):
        return True

    def as_frac(r):
        return 0.4

    def as_pct(r):
        return 80

    def as_result(r):
        return CheckResult(name="cap", status="fail", message="nope")

    def boom(r):
        raise RuntimeError("explode")

    assert (
        score_run(
            run, Rubric(criteria=[RubricCriterion(name="c", kind="check", check=as_bool)])
        ).score
        == 100
    )
    assert score_run(
        run, Rubric(criteria=[RubricCriterion(name="c", kind="check", check=as_frac)])
    ).score == pytest.approx(40)
    assert score_run(
        run, Rubric(criteria=[RubricCriterion(name="c", kind="check", check=as_pct)])
    ).score == pytest.approx(80)
    assert (
        score_run(
            run, Rubric(criteria=[RubricCriterion(name="c", kind="check", check=as_result)])
        ).score
        == 0
    )
    boomed = score_run(run, Rubric(criteria=[RubricCriterion(name="c", kind="check", check=boom)]))
    assert boomed.score == 0
    assert "RuntimeError" in (boomed.breakdown[0].message or "")


def test_score_run_checks_update_outcome():
    run = make_run("a", outcome="pass")

    def cap(r):
        return "refunded too much"

    result = score_run(run, checks=[cap])
    outcome = next(c for c in result.breakdown if c.kind == "outcome_pass")
    assert outcome.score == 0.0
    # the original run is left alone
    assert run.outcome == "pass"


def test_score_runs_fleet_stats_and_weakest():
    cheap = make_run("a", cost=0.01, wall_time_ms=1000)
    pricey = make_run("b", cost=0.10, wall_time_ms=1000, outcome="fail")
    rubric = Rubric(
        min_score=70,
        criteria=[
            RubricCriterion(name="outcome", kind="outcome_pass", weight=0.5),
            RubricCriterion(
                name="cost", kind="cost_budget", weight=0.5, target=0.02, max_value=0.08
            ),
        ],
    )
    report = score_runs([cheap, pricey], rubric)
    assert report.total == 2
    assert report.passed == 1 and report.failed == 1
    assert report.min <= report.median <= report.max
    assert report.mean == pytest.approx(sum(report.scores) / 2)
    assert (
        report.grades["A"]
        + report.grades["B"]
        + report.grades["C"]
        + report.grades["D"]
        + report.grades["F"]
        == 2
    )
    # pricey fails outcome (0) and is over max cost (0) -> 0; cheap is perfect
    assert report.weakest in {"outcome", "cost"}
    payload = report.to_dict()
    assert payload["total"] == 2 and "results" in payload


def test_load_rubric_yaml_json_toml(tmp_path):
    yaml_text = """
name: refund
min_score: 80
criteria:
  - name: outcome
    kind: outcome_pass
    weight: 0.5
  - name: cost
    kind: cost_budget
    weight: 0.5
    target: 0.01
    max_value: 0.10
"""
    yaml_path = tmp_path / "rubric.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    loaded = load_rubric(yaml_path)
    assert loaded.name == "refund" and loaded.min_score == 80
    assert [c.kind for c in loaded.criteria] == ["outcome_pass", "cost_budget"]

    json_path = tmp_path / "rubric.json"
    json_path.write_text(
        json.dumps(
            {
                "name": "refund-json",
                "min_score": 65,
                "criteria": [
                    {"name": "safety", "kind": "no_findings", "weight": 1, "target": "warn"}
                ],
            }
        ),
        encoding="utf-8",
    )
    js = load_rubric(json_path)
    assert js.name == "refund-json" and js.min_score == 65
    assert js.criteria[0].kind == "no_findings"

    toml_path = tmp_path / "rubric.toml"
    toml_path.write_text(
        'name = "refund-toml"\n'
        "min_score = 72.0\n\n"
        "[[criteria]]\n"
        'name = "latency"\n'
        'kind = "latency_budget"\n'
        "weight = 1.0\n"
        "target = 1000\n"
        "max_value = 8000\n",
        encoding="utf-8",
    )
    tm = load_rubric(toml_path)
    assert tm.name == "refund-toml" and tm.min_score == 72.0
    assert tm.criteria[0].kind == "latency_budget"
    assert tm.criteria[0].target == 1000

    with pytest.raises(RubricError, match="not found"):
        load_rubric(tmp_path / "missing.yaml")
    with pytest.raises(RubricError):
        parse_rubric({"criteria": [{"name": "x", "kind": "nope"}]})


def test_score_cli_json_and_fail_under(crashlab_dir):
    store = Store()
    put(store, make_run("RUNSCORE00000000000000001", name="bot", outcome="pass"))
    put(store, make_run("RUNSCORE00000000000000002", name="bot", outcome="fail"))

    res = runner.invoke(app, ["score", "RUNSCORE00000000000000001", "--json"], env=ENV)
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["ok"] is True
    assert payload["results"][0]["grade"]
    assert payload["results"][0]["score"] >= 70
    assert payload["results"][0]["passed"] is True

    failed = runner.invoke(app, ["score", "RUNSCORE00000000000000002", "--json"], env=ENV)
    assert failed.exit_code == 1, failed.output
    body = json.loads(failed.output)
    assert body["ok"] is False
    assert body["results"][0]["passed"] is False

    # high-scoring run still exits 1 when --fail-under is above its score
    under = runner.invoke(
        app,
        ["score", "RUNSCORE00000000000000001", "--json", "--fail-under", "100.1"],
        env=ENV,
    )
    assert under.exit_code == 1, under.output
    # but passes if the floor is below the score
    ok = runner.invoke(
        app,
        ["score", "RUNSCORE00000000000000001", "--fail-under", "10"],
        env=ENV,
    )
    assert ok.exit_code == 0, ok.output

    fleet = runner.invoke(app, ["score", "--all", "--name", "bot", "--json"], env=ENV)
    assert fleet.exit_code == 1, fleet.output
    report = json.loads(fleet.output)
    assert report["total"] == 2
    assert report["failed"] == 1

    missing = runner.invoke(app, ["score", "no-such-run"], env=ENV)
    assert missing.exit_code == 2
    empty = runner.invoke(app, ["score", "--all", "--name", "nope"], env=ENV)
    assert empty.exit_code == 2


def test_score_cli_rubric_file_and_min_score(crashlab_dir, tmp_path):
    store = Store()
    put(store, make_run("RUNSCORE00000000000000003", cost=0.05))
    path = tmp_path / "tight.yaml"
    path.write_text(
        "name: tight\nmin_score: 90\ncriteria:\n"
        "  - name: cost\n    kind: cost_budget\n    target: 0.01\n    max_value: 0.09\n",
        encoding="utf-8",
    )
    # halfway on cost -> 50, below min_score 90
    res = runner.invoke(app, ["score", "last", "--rubric", str(path), "--json"], env=ENV)
    assert res.exit_code == 1, res.output
    payload = json.loads(res.output)
    assert payload["results"][0]["score"] == pytest.approx(50.0)
    assert payload["min_score"] == 90
    assert payload["rubric"] == "tight"

    # override the passing floor so 50 still "passes"
    passed = runner.invoke(
        app,
        ["score", "last", "--rubric", str(path), "--min-score", "40", "--json"],
        env=ENV,
    )
    assert passed.exit_code == 0, passed.output
    body = json.loads(passed.output)
    assert body["results"][0]["passed"] is True
    assert body["min_score"] == 40

    bad = runner.invoke(app, ["score", "last", "--rubric", str(tmp_path / "missing.yaml")], env=ENV)
    assert bad.exit_code == 2


def test_eval_rubric_option_fails_under_min_score(crashlab_dir, tmp_path):
    store = Store()
    put(store, make_run("RUNEVALR00000000000000001", cost=0.05, outcome="pass"))
    path = tmp_path / "cost.yaml"
    path.write_text(
        "name: cost-only\nmin_score: 70\ncriteria:\n"
        "  - name: cost\n    kind: cost_budget\n    target: 0.01\n    max_value: 0.09\n",
        encoding="utf-8",
    )
    # default eval (no rubric) still passes: binary outcome is pass, no serious findings
    plain = eval_runs(store)
    assert plain.passed == 1 and plain.results[0].score is None

    graded = eval_runs(store, rubric=path)
    assert graded.failed == 1
    result = graded.results[0]
    assert result.score == pytest.approx(50.0)
    assert result.grade == "F"
    assert result.rubric_passed is False
    assert "min_score" in (result.reason or "")

    md = graded.to_markdown()
    assert "| score | grade |" in md
    assert "50.0" in md and "| F |" in md

    cli = runner.invoke(
        app, ["eval", "--rubric", str(path), "--min-score", "70", "--json"], env=ENV
    )
    assert cli.exit_code == 1, cli.output
    payload = json.loads(cli.output)
    assert payload["failed"] == 1
    assert payload["results"][0]["score"] == pytest.approx(50.0)
    assert payload["results"][0]["grade"] == "F"

    # a cheap run passes both eval and the rubric
    put(store, make_run("RUNEVALR00000000000000002", name="cheap", cost=0.005))
    ok = runner.invoke(
        app,
        ["eval", "--name", "cheap", "--rubric", str(path), "--json"],
        env=ENV,
    )
    assert ok.exit_code == 0, ok.output
    ok_payload = json.loads(ok.output)
    assert ok_payload["passed"] == 1
    assert ok_payload["results"][0]["score"] == pytest.approx(100.0)


def test_eval_without_rubric_markdown_unchanged():
    from crashlab.suite import EvalReport, EvalResult

    report = EvalReport(
        results=[
            EvalResult(
                run_id="OK1",
                name="bot",
                status="pass",
                cost_usd=0.001,
                tokens=20,
                duration_seconds=1.0,
            )
        ]
    )
    md = report.to_markdown()
    assert "| run | agent | result | checks | findings | cost | tokens | duration |" in md
    assert "| score |" not in md


def test_score_and_eval_help_strips_ansi():
    for args in (["score", "--help"], ["eval", "--help"], ["--help"]):
        res = runner.invoke(app, args, env=ENV)
        assert res.exit_code == 0, res.output
        out = _plain(res.output)
        assert "\x1b[" not in out
        if args[0] == "score":
            assert "--rubric" in out and "--min-score" in out
            assert "--fail-under" in out and "--json" in out
            assert "--all" in out
            assert "Grade recorded runs" in out
        if args[0] == "eval":
            assert "--rubric" in out and "--min-score" in out
        if args == ["--help"]:
            assert re.search(r"(?<![\w-])score\s{2,}\S", out)


def test_public_exports():
    assert crashlab.Rubric is Rubric
    assert crashlab.RubricCriterion is RubricCriterion
    assert crashlab.score_run is score_run
    assert crashlab.score_runs is score_runs
