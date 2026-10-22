"""crashlab bench: catalog, scoring, the live loop (fake model) and offline rescoring."""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from examples.sql_analyst import agent as sql
from typer.testing import CliRunner

import crashlab
from crashlab.bench import CONDITIONS, FAULTS, TASKS, harness, score_runs
from crashlab.cli import app
from crashlab.diagnose import diagnoser
from crashlab.diagnose import fixes as fixes_mod
from tests.fakes import FakeOpenAI, text_response, tool_call_response

runner = CliRunner()
RULE = "Revenue counts paid orders only."


class _Analyst:
    """Follows the prompt: filters on status = 'paid' only when a revenue rule is there."""

    def __init__(self, owner: SqlModel) -> None:
        self.owner = owner

    def create(self, **kw: Any) -> Any:
        o = self.owner
        o.n += 1
        msgs = kw["messages"]
        system = msgs[0]["content"]
        tools = [m for m in msgs if m["role"] == "tool"]
        if not tools:
            where = " AND status = 'paid'" if ("Revenue" in system) else ""
            q = f"SELECT SUM(total) FROM orders WHERE order_date LIKE '2026-03-%'{where}"
            return tool_call_response(o.n, "run_query", {"sql": q})
        if len(tools) == 1:
            res = json.loads(tools[0]["content"])
            return tool_call_response(o.n, "final_answer", {"answer": f"{res['rows'][0][0]:.2f}"})
        return text_response(o.n, "done")


class SqlModel(FakeOpenAI):
    def __init__(self) -> None:
        super().__init__([])
        self.n = 0
        self.chat.completions = _Analyst(self)


def _fake_diagnoser(run, msgs, model, **kw):
    text = json.dumps(msgs)
    if "status = 'paid'" in text:
        return json.dumps({"label": "NO_FAILURE", "failing_step": None, "evidence": []}), {}
    return json.dumps(
        {
            "label": "TASK_SPEC_VIOLATION",
            "failing_step": 1,
            "confidence": 0.8,
            "root_cause": "Revenue included cancelled and refunded orders.",
            "evidence": [{"step": 1, "quote": "SUM(total)"}],
        }
    ), {"cost_usd": 0.001}


def _fake_fixer(run, msgs, model, **kw):
    patch = {"kind": "system_prompt", "op": "append", "text": RULE}
    return json.dumps({"fixes": [{"patch": patch, "rationale": "rule missing"}]}), {}


@pytest.fixture
def fake_models(monkeypatch):
    pytest.importorskip("openai")
    model = crashlab.wrap(SqlModel())
    monkeypatch.setattr(sql, "make_client", lambda: model)
    monkeypatch.setattr(diagnoser, "_ask", _fake_diagnoser)
    monkeypatch.setattr(fixes_mod, "_ask", _fake_fixer)
    monkeypatch.setattr(harness.time, "sleep", lambda s: None)
    return model


def test_catalog_is_consistent():
    assert set(CONDITIONS) == {"clean", *FAULTS} and len(FAULTS) >= 4
    for task in TASKS.values():
        mod = task.load()
        assert task.split in ("dev", "test") and callable(mod.ground_truth)
        for cond in FAULTS:
            assert task.agent_faults(cond)[0] in mod.FAULTS and task.prompts_for(cond)
    assert TASKS["refund-bot"].agent_faults("tool_5xx") == ["html_502"]
    assert {t.split for t in TASKS.values()} == {"dev", "test"}


def test_live_loop_scores_and_rescores_offline(fake_models, tmp_path, crashlab_dir):
    bench_id, ids, stopped = harness.run_live(
        "fake-model",
        base_url="http://fake",
        api_key="x",
        tasks=["sql-analyst"],
        conditions=["clean", "spec_drop"],
        episodes=2,
        k=2,
        pace_s=0,
        log=lambda m: None,
    )
    assert stopped is None and len(ids) == 4
    store = crashlab.store.Store()
    res = score_runs([store.load_run(i) for i in ids], label="fake")
    eps = {(r["condition"], r["episode"]): r for r in res["episodes"]}
    drop = eps[("spec_drop", 0)]
    assert drop["triggered"] and drop["expected_step"] == 1 and drop["window"] == [0, 3]
    assert drop["llm"]["label_ok"] and drop["llm"]["step_exact"]
    assert drop["fix"]["verdict"] == "LIKELY"  # 2/2 vs 0/2: below the CI bar at k=2
    assert drop["fix"]["patched"] == "2/2" and drop["fix"]["control"] == "0/2"
    assert eps[("spec_drop", 1)]["fix"]["verdict"] is None  # one verify per cell
    s = res["summary"]["test"]["llm"]
    assert s["triggered"]["k"] == 2 and s["label_ok"]["k"] == 2
    # the fake always answers the March question, so clean ep1 (Germany) fails: only
    # passing clean runs count for false alarms
    assert eps[("clean", 1)]["outcome"] == "fail"
    assert s["clean_false_positive"] == {"k": 0, "n": 1, "rate": 0.0, "ci95": [0.0, 0.793]}
    assert "dev" not in res["summary"]
    # offline: export, re-import into an empty store, same numbers
    harness.export_runs(store, ids, tmp_path / "runs")
    again = score_runs(harness.load_runs_dir(tmp_path / "runs", tmp_path / "s2"), label="fake")
    strip = lambda r: {k: v for k, v in r.items() if k != "scored_at"}  # noqa: E731
    assert strip(again)["episodes"] == strip(res)["episodes"]
    assert again["summary"] == res["summary"]
    md = harness.render_md(res)
    assert "| test | fake |" in md and "| sql-analyst | spec_drop | 2 | 2/2 |" in md


def test_daily_quota_stops_the_benchmark(fake_models, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("429 GenerateRequestsPerDayPerProjectPerModel exceeded")

    monkeypatch.setattr(TASKS["sql-analyst"].__class__, "run", lambda self, *a: boom())
    _, ids, stopped = harness.run_live(
        "m",
        base_url="u",
        api_key="k",
        tasks=["sql-analyst"],
        conditions=["clean"],
        episodes=1,
        pace_s=0,
        log=lambda m: None,
    )
    assert ids == [] and stopped.startswith("daily quota exhausted")


def test_rate_limits_are_retried(monkeypatch):
    monkeypatch.setattr(harness.time, "sleep", lambda s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("Error code: 429 rate limit")
        return "ok"

    assert harness._retrying(flaky, lambda m: None) == "ok" and len(calls) == 3
    with pytest.raises(ValueError):
        harness._retrying(lambda: (_ for _ in ()).throw(ValueError("bug")), lambda m: None)


def test_cli_bench_offline(fake_models, tmp_path):
    _, ids, _ = harness.run_live(
        "fake-model",
        base_url="u",
        api_key="k",
        tasks=["sql-analyst"],
        conditions=["clean", "spec_drop"],
        episodes=1,
        k=1,
        pace_s=0,
        log=lambda m: None,
    )
    harness.export_runs(crashlab.store.Store(), ids, tmp_path / "runs")
    res = runner.invoke(
        app,
        [
            "bench",
            "--offline",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--label",
            "fake",
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )
    assert res.exit_code == 0, res.output
    data = json.loads((tmp_path / "out" / "seed_fake.json").read_text())
    assert len(data["episodes"]) == 2 and re.search(r"\| test \| heuristic \|", res.output)
    res = runner.invoke(app, ["bench", "--offline", "--runs-dir", str(tmp_path / "nope")])
    assert res.exit_code == 2


def test_a_model_side_400_is_kept_as_an_errored_episode(fake_models, monkeypatch):
    class Bad(SqlModel):
        def __init__(self) -> None:
            super().__init__()
            self.chat.completions = self

        def create(self, **kwargs):
            raise RuntimeError("Error code: 400 tool_use_failed: name not in request.tools")

    monkeypatch.setattr(sql, "make_client", lambda: crashlab.wrap(Bad()))
    _, ids, stopped = harness.run_live(
        "m",
        base_url="u",
        api_key="k",
        tasks=["sql-analyst"],
        conditions=["clean"],
        episodes=1,
        pace_s=0,
        log=lambda m: None,
    )
    assert stopped is None and len(ids) == 1
    assert crashlab.store.Store().load_run(ids[0]).status == "error"
