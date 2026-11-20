"""Batch evaluation suites (crashlab.suite.eval_runs / crashlab eval). Offline only."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from xml.etree.ElementTree import fromstring

import pytest
from typer.testing import CliRunner

import crashlab
from crashlab.cli import app
from crashlab.models import CheckResult, Finding, Run, Step, Usage
from crashlab.store import Store, export_run
from crashlab.suite import (
    EvalReport,
    EvalResult,
    eval_runs,
    expand_cassette_paths,
    load_check_modules,
    load_export,
)

T0 = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
ENV = {"COLUMNS": "220", "TERM": "xterm-256color", "NO_COLOR": "1"}
runner = CliRunner()


@pytest.fixture(autouse=True)
def _clean_oracles():
    """Restore the process-wide oracle registry: unregister() would drop the
    refund-bot checks imported by other tests in this session."""
    from crashlab.verify.oracles import _registry

    before = dict(_registry)
    yield
    _registry.clear()
    _registry.update(before)


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


def make_run(
    rid: str,
    *,
    name: str = "bot",
    outcome: str = "pass",
    status: str = "ok",
    tags: list[str] | None = None,
    checks: list[CheckResult] | None = None,
    findings: list[Finding] | None = None,
    n_steps: int = 2,
    tokens: int = 100,
    cost: float = 0.001,
    started_at: datetime | None = None,
    check_config: dict | None = None,
    outcome_reason: str | None = None,
) -> Run:
    steps = [chat(rid, i, tokens=tokens, cost=cost) for i in range(n_steps)]
    started = started_at or T0
    ended = started + timedelta(seconds=2)
    total = (tokens + 10) * n_steps
    return Run(
        run_id=rid,
        name=name,
        status=status,
        outcome=outcome,
        outcome_reason=outcome_reason,
        started_at=started,
        ended_at=ended,
        tags=list(tags or []),
        checks=list(checks or []),
        findings=list(findings or []),
        check_config=check_config,
        steps=steps,
        usage=Usage(
            input_tokens=tokens * n_steps,
            output_tokens=10 * n_steps,
            total_tokens=total,
            cost_usd=cost * n_steps,
            steps=n_steps,
            llm_calls=n_steps,
            wall_time_ms=2000.0,
        ),
    )


def put(store: Store, run: Run) -> Run:
    store.begin_run(run)
    for step in run.steps:
        store.add_step(step)
    store.end_run(run)
    return store.load_run(run.run_id)


def test_eval_passes_clean_store_runs():
    store = Store()
    put(store, make_run("RUNPASS000000000000000001"))
    put(store, make_run("RUNPASS000000000000000002", name="other"))
    report = eval_runs(store, name="bot")
    assert report.total == 1 and report.passed == 1 and report.failed == 0
    assert report.pass_rate == 1.0
    assert report.total_tokens > 0
    assert report.total_cost_usd == pytest.approx(0.002)


def test_eval_fails_on_oracle_and_records_the_name():
    store = Store()
    put(store, make_run("RUNFAIL000000000000000001"))

    @crashlab.check("bot")
    def refund_cap(run):
        return "refunded too much"

    report = eval_runs(store)
    assert report.passed == 0 and report.failed == 1
    result = report.results[0]
    assert result.status == "fail"
    assert result.failing_checks == ["refund_cap"]
    assert "refunded too much" in (result.reason or "")


def test_eval_oracle_exception_is_error_status():
    store = Store()
    put(store, make_run("RUNERR0000000000000000001"))

    @crashlab.check("bot")
    def broken(run):
        raise RuntimeError("boom")

    report = eval_runs(store)
    assert report.errored == 1 and report.failed == 0 and report.passed == 0
    assert report.results[0].status == "error"
    assert "RuntimeError" in (report.results[0].reason or "")


def test_eval_fails_on_stored_outcome_when_no_oracles():
    store = Store()
    put(store, make_run("RUNSTORED0000000000000001", outcome="fail",
                        outcome_reason="check refund_cap failed"))  # fmt: skip
    report = eval_runs(store)
    assert report.failed == 1
    assert "refund_cap" in (report.results[0].reason or "")


def test_eval_budget_gate():
    store = Store()
    put(store, make_run("RUNBUDGET0000000000000001", tokens=500, cost=0.02))
    ok = eval_runs(store, max_cost_usd=1.0, max_tokens=10_000)
    assert ok.passed == 1
    over = eval_runs(store, max_cost_usd=0.0001)
    assert over.failed == 1 and "max_cost_usd" in (over.results[0].reason or "")
    over_tok = eval_runs(store, max_tokens=10)
    assert over_tok.failed == 1 and "max_tokens" in (over_tok.results[0].reason or "")


def test_eval_runtime_findings_at_default_error_severity():
    store = Store()
    # built-in budget detector: exceeding max_steps is critical
    put(store, make_run("RUNFIND0000000000000000001", n_steps=3,
                        check_config={"budget": {"max_steps": 1}}))  # fmt: skip
    report = eval_runs(store)
    assert report.failed == 1
    kinds = [f["kind"] for f in report.results[0].findings]
    assert any(k == "budget" for k in kinds)
    # warnings / info would not fail at the default floor; a clean run still passes
    put(store, make_run("RUNFIND0000000000000000002", n_steps=1))
    only_clean = eval_runs(store, name="bot", limit=1)  # newest first: ...0002
    # list_runs is newest first; RUNFIND...0002 was added last
    assert only_clean.results[0].run_id.endswith("0002")
    assert only_clean.passed == 1


def test_eval_filters_tag_outcome_since_limit():
    store = Store()
    now = datetime.now(timezone.utc)
    put(store, make_run("RUNFILT0000000000000000001", tags=["ci"], outcome="fail",
                        outcome_reason="nope", started_at=now - timedelta(days=10)))  # fmt: skip
    put(store, make_run("RUNFILT0000000000000000002", tags=["ci"], outcome="pass",
                        started_at=now))  # fmt: skip
    put(store, make_run("RUNFILT0000000000000000003", tags=["other"], outcome="pass",
                        started_at=now))  # fmt: skip
    tagged = eval_runs(store, tag="ci")
    assert {r.run_id for r in tagged.results} == {
        "RUNFILT0000000000000000001",
        "RUNFILT0000000000000000002",
    }
    recent = eval_runs(store, since="2d")
    assert {r.run_id for r in recent.results} == {
        "RUNFILT0000000000000000002",
        "RUNFILT0000000000000000003",
    }
    failing = eval_runs(store, outcome="fail")
    assert [r.run_id for r in failing.results] == ["RUNFILT0000000000000000001"]
    limited = eval_runs(store, limit=1)
    assert limited.total == 1


def test_eval_loads_jsonl_files_directories_and_globs(tmp_path):
    store = Store()
    a = put(store, make_run("RUNJSON0000000000000000001", name="alpha"))
    b = put(store, make_run("RUNJSON0000000000000000002", name="beta"))
    folder = tmp_path / "cassettes"
    folder.mkdir()
    export_run(store, a.run_id, folder / "a.jsonl")
    export_run(store, b.run_id, folder / "b.jsonl")
    (folder / "notes.txt").write_text("ignore", "utf-8")

    loaded = load_export(folder / "a.jsonl")
    assert loaded.run_id == a.run_id and loaded.steps and loaded.name == "alpha"

    paths = expand_cassette_paths([folder])
    assert {p.name for p in paths} == {"a.jsonl", "b.jsonl"}
    globbed = expand_cassette_paths([str(folder / "*.jsonl")])
    assert len(globbed) == 2

    # a fresh store: grade from the files themselves
    other = Store(tmp_path / "other-store")
    report = eval_runs(other, files=folder)
    assert report.total == 2
    assert {r.name for r in report.results} == {"alpha", "beta"}

    one = eval_runs(files=str(folder / "a.jsonl"), name="alpha")
    assert one.total == 1 and one.results[0].run_id == a.run_id

    with pytest.raises(FileNotFoundError):
        expand_cassette_paths([tmp_path / "missing"])


def test_eval_check_modules_register_oracles(tmp_path):
    store = Store()
    put(store, make_run("RUNMOD00000000000000000001"))
    mod = tmp_path / "oracles.py"
    mod.write_text(
        "import crashlab\n"
        "@crashlab.check('bot')\n"
        "def from_file(run):\n"
        "    return 'loaded from file'\n",
        encoding="utf-8",
    )
    names = load_check_modules([mod])
    assert names and "from_file" in dir(__import__("sys").modules[names[0]])
    report = eval_runs(store, check_modules=[mod])
    assert report.failed == 1
    assert report.results[0].failing_checks == ["from_file"]


def test_eval_replay_divergence_and_replay_error_are_failures(monkeypatch):
    store = Store()
    put(store, make_run("RUNREPLAY0000000000000001"))
    from crashlab.replay import DivergenceError, ReplayError

    def diverge(run, **_kw):
        raise DivergenceError("request", "messages differed", source_run_id=run.run_id)

    monkeypatch.setattr("crashlab.suite.replay_run", diverge)
    report = eval_runs(store, replay=True)
    assert report.failed == 1 and report.errored == 0
    assert "DivergenceError" in (report.results[0].reason or "")

    def cannot(run, **_kw):
        raise ReplayError("no entry point")

    monkeypatch.setattr("crashlab.suite.replay_run", cannot)
    report = eval_runs(store, replay=True)
    assert report.failed == 1
    assert "ReplayError" in (report.results[0].reason or "")


def test_eval_unexpected_exception_is_error(monkeypatch):
    store = Store()
    put(store, make_run("RUNBOOM0000000000000000001"))

    def boom(*_a, **_k):
        raise ValueError("x")

    monkeypatch.setattr("crashlab.suite._judge", boom)
    report = eval_runs(store)
    assert report.errored == 1
    assert "ValueError" in (report.results[0].reason or "")


def test_junit_xml_is_valid_and_includes_failures():
    report = EvalReport(
        results=[
            EvalResult(
                run_id="OK1",
                name="bot",
                status="pass",
                tokens=10,
                duration_seconds=1.25,
                cost_usd=0.001,
            ),  # fmt: skip
            EvalResult(
                run_id="BAD1",
                name="bot",
                status="fail",
                reason="check cap failed",
                failing_checks=["cap"],
                duration_seconds=0.5,
            ),  # fmt: skip
            EvalResult(run_id="ERR1", name="bot", status="error", reason="RuntimeError: boom"),
        ]
    )
    xml = report.to_junit_xml()
    root = fromstring(xml)
    assert root.tag == "testsuites"
    suite = root.find("testsuite")
    assert suite is not None
    cases = list(suite.findall("testcase"))
    assert len(cases) == 3
    assert suite.attrib["tests"] == "3"
    assert suite.attrib["failures"] == "1"
    assert suite.attrib["errors"] == "1"
    by_name = {c.attrib["name"]: c for c in cases}
    assert by_name["OK1"].find("failure") is None
    fail = by_name["BAD1"].find("failure")
    assert fail is not None and fail.attrib["message"] == "check cap failed"
    err = by_name["ERR1"].find("error")
    assert err is not None and "RuntimeError" in err.attrib["message"]
    assert by_name["OK1"].attrib["classname"] == "bot"
    assert by_name["OK1"].attrib["time"] == "1.250"


def test_markdown_is_github_flavored_table():
    report = EvalReport(
        results=[
            EvalResult(
                run_id="OK1",
                name="bot",
                status="pass",
                cost_usd=0.001,
                tokens=20,
                duration_seconds=1.0,
            ),  # fmt: skip
            EvalResult(
                run_id="BAD1",
                name="bot",
                status="fail",
                reason="cap failed",
                failing_checks=["cap"],
                findings=[{"kind": "budget", "severity": "error"}],
                cost_usd=0.002,
                tokens=40,
            ),  # fmt: skip
        ]
    )
    md = report.to_markdown()
    assert md.startswith("# crashlab eval")
    assert "| run | agent | result | checks | findings | cost | tokens | duration |" in md
    assert "| --- |" in md
    assert "| OK1 | bot | pass |" in md
    assert "| BAD1 | bot | fail | cap | budget:error |" in md
    assert "## Failures" in md
    assert "`BAD1`" in md
    assert "1 passed" in md and "1 failed" in md


def test_eval_cli_exit_codes_and_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = Store()
    put(store, make_run("RUNCLI00000000000000000001"))
    junit = tmp_path / "out.xml"
    md = tmp_path / "out.md"
    summary = tmp_path / "step-summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    result = runner.invoke(app, ["eval", "--junit", str(junit), "--markdown", str(md)], env=ENV)
    assert result.exit_code == 0, result.output
    assert "passed" in result.output and "failed" in result.output
    assert junit.is_file() and "<testsuites" in junit.read_text("utf-8")
    assert md.is_file() and "# crashlab eval" in md.read_text("utf-8")
    assert "# crashlab eval" in summary.read_text("utf-8")

    @crashlab.check("bot")
    def always_fail(run):
        return "nope"

    result = runner.invoke(app, ["eval", "--json"], env=ENV)
    assert result.exit_code == 1, result.output
    payload = result.output
    assert '"failed": 1' in payload or '"status": "fail"' in payload

    result = runner.invoke(app, ["eval", "--name", "no-such-agent"], env=ENV)
    assert result.exit_code == 2
    result = runner.invoke(app, ["eval", "missing.jsonl"], env=ENV)
    assert result.exit_code == 2
    result = runner.invoke(app, ["eval", "--since", "yesterday"], env=ENV)
    assert result.exit_code == 2
    result = runner.invoke(app, ["eval", "--outcome", "maybe"], env=ENV)
    assert result.exit_code == 2


def test_eval_cli_files_and_checks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = Store()
    run = put(store, make_run("RUNCLIF000000000000000001"))
    cassette = tmp_path / "run.jsonl"
    export_run(store, run.run_id, cassette)
    # grade from the file against a different empty store dir
    other = tmp_path / "empty-store"
    mod = tmp_path / "checks.py"
    mod.write_text(
        "import crashlab\n@crashlab.check('bot')\ndef ok(run):\n    return True\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        ["--dir", str(other), "eval", str(cassette), "--checks", str(mod), "--json"],
        env=ENV,
    )
    assert result.exit_code == 0, result.output
    assert '"passed": 1' in result.output


def test_eval_bad_min_finding_severity():
    with pytest.raises(ValueError, match="min_finding_severity"):
        eval_runs(Store(), min_finding_severity="urgent")
    result = runner.invoke(app, ["eval", "--min-finding-severity", "urgent"], env=ENV)
    assert result.exit_code == 2
