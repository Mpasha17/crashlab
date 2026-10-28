"""Findings in the CLI (show / ls / check / diff / export / import), the shipped demo
cassettes, and refund_bot's LiteLLM provider. Offline."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest
from examples.refund_bot import agent as refund
from typer.testing import CliRunner

import crashlab
from crashlab.cli import _unique_prefix_len, app
from crashlab.replay import replay
from tests.fakes import FakeOpenAI, refund_script, text_response, tool_call_response

runner = CliRunner()
ENV = {"COLUMNS": "220", "TERM": "xterm-256color", "NO_COLOR": "1"}
DEMO = Path(__file__).resolve().parents[1] / "examples" / "refund_bot" / "demo"


@pytest.fixture(autouse=True)
def _no_pyproject(monkeypatch):
    monkeypatch.setenv("CRASHLAB_NO_PYPROJECT", "1")


def _needs(module: str) -> None:
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"needs {module}")


def _cli(*args: str, env: dict | None = None):
    return runner.invoke(app, list(args), env=env or ENV)


def _html_run() -> str:
    script = [tool_call_response(i, "get_order", {"order_id": "A17"}) for i in (1, 2, 3)]
    script.append(text_response(4, "Sorry, the order system is down."))
    refund.run_agent("Refund A17", client=crashlab.wrap(FakeOpenAI(script)), model="m1",
                     faults=["html_502"])  # fmt: skip
    return crashlab.load("last").run_id


def _clean_run() -> str:
    refund.run_agent("Refund A17", client=crashlab.wrap(FakeOpenAI(refund_script())), model="m1")
    return crashlab.load("last").run_id


def test_show_lists_findings_with_evidence():
    run_id = _html_run()
    out = _cli("show", run_id).output
    assert "runtime findings" in out
    assert "tool_error_streak" in out and "step_repetition" in out
    assert "get_order failed 3 times in a row" in out
    assert "502 Bad Gateway" in out
    assert "⚑" in out  # step-table marker


def test_ls_flagged_and_min_severity():
    bad, good = _html_run(), _clean_run()
    out = _cli("ls", "--flagged").output
    assert bad in out and good not in out
    assert "error" in out
    assert "No flagged runs" in _cli("ls", "--flagged", "--min-severity", "critical").output
    assert _cli("ls", "--flagged", "--min-severity", "loud").exit_code == 2


def test_ls_narrow_terminal_short_unique_ids():
    run_id = _clean_run()
    out = _cli("ls", env={**ENV, "COLUMNS": "80"}).output
    assert run_id not in out and run_id[:10] in out
    assert _cli("show", run_id[:10]).exit_code == 0  # the shown prefix works everywhere
    assert _unique_prefix_len(["01ABCDEFGHIJ", "01ABCDEFGHXX"]) == 11
    assert _unique_prefix_len(["01A"]) == 10


def test_check_json_fail_on_and_save():
    bad, good = _html_run(), _clean_run()
    res = _cli("check", "--all", "--json", "--findings-only")
    assert res.exit_code == 0, res.output
    data = {r["run_id"]: r for r in json.loads(res.output)}
    assert {f["kind"] for f in data[bad]["findings"]} >= {"tool_error_streak", "step_repetition"}
    assert data[good]["findings"] == []
    assert _cli("check", bad, "--findings-only", "--fail-on", "error").exit_code == 1
    assert _cli("check", bad, "--findings-only", "--fail-on", "critical").exit_code == 0
    assert _cli("check", good, "--findings-only", "--fail-on", "info").exit_code == 0
    # stricter limits after the fact, saved with the run
    res = _cli("check", good, "--findings-only", "--max-steps", "3", "--save")
    out = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", res.output).split())
    assert "budget exceeded: 4 steps > max_steps 3" in out  # first step over the limit
    assert crashlab.load(good).findings_of("budget", min_severity="critical")


def test_check_runs_oracles_too():
    run_id = _clean_run()
    out = _cli("check", run_id).output
    assert "refund_within_policy" in out and "runtime checks: no findings" in out
    assert "refund_within_policy" not in _cli("check", run_id, "--findings-only").output


def test_diff_shows_findings_change():
    a, b = _clean_run(), _html_run()
    res = _cli("diff", a, b)
    assert res.exit_code in (0, 1), res.output
    assert "runtime findings" in res.output and "tool_error_streak" in res.output
    data = json.loads(_cli("diff", a, b, "--json").output)
    kinds = {f["kind"]: f for f in data["findings"]}
    assert kinds["tool_error_streak"]["changed"] is True
    assert kinds["tool_error_streak"]["worst_a"] is None
    assert kinds["tool_error_streak"]["worst_b"] == "error"
    assert kinds["tool_error_streak"]["b"][0]["steps"] == [1, 3, 5]


def test_recorded_text_is_not_rich_markup():
    run_id = _clean_run()
    run = crashlab.load(run_id)
    run.outcome_reason = "stopped by crashlab: [budget] over [/] [bold]x"
    crashlab.store.Store().update_run(run)
    out = _cli("show", run_id).output
    assert "[budget] over [/] [bold]x" in re.sub(r"\s+", " ", out)


def test_export_import_roundtrip(tmp_path, monkeypatch):
    run_id = _html_run()
    path = tmp_path / "run.jsonl"
    assert _cli("export", run_id, "-o", str(path)).exit_code == 0
    head = json.loads(path.read_text().splitlines()[0])
    assert head["type"] == "crashlab.run" and head["run"]["run_id"] == run_id
    monkeypatch.setenv("CRASHLAB_DIR", str(tmp_path / "other"))
    res = _cli("import", str(path))
    assert res.exit_code == 0 and "imported" in res.output
    assert "already there" in _cli("import", str(path)).output
    assert crashlab.load(run_id).findings_of("tool_error_streak")
    _needs("openai")  # the replay re-runs the agent, which builds an OpenAI client
    rep = replay(run_id)
    assert rep.status == "ok" and len(rep.steps) == len(crashlab.load(run_id).steps)


# --- the shipped no-API-key demo -------------------------------------------------------


def _demo_files() -> list[Path]:
    return sorted(DEMO.glob("*.jsonl"))


def test_demo_cassettes_have_no_secrets_or_local_paths():
    files = _demo_files()
    assert len(files) >= 3
    for f in files:
        text = f.read_text()
        # key prefixes are built, so a grep of the repo for them stays at zero hits
        for bad in ("AQ." + "Ab8", "gsk" + "_", "sk-", "/workspace/", "/home/", "Bearer "):
            assert bad not in text, (f.name, bad)


def test_demo_imports_checks_and_replays_offline():
    for f in _demo_files():
        assert _cli("import", str(f)).exit_code == 0
    ids = {f.stem: json.loads(f.read_text().splitlines()[0])["run"]["run_id"]
           for f in _demo_files()}  # fmt: skip
    html = crashlab.load(ids["html_502_loop_a17"])
    assert {f.kind for f in html.findings} >= {"tool_error_streak", "step_repetition"}
    assert crashlab.load(ids["policy_drop_c03"]).outcome == "fail"
    stopped = crashlab.load(ids["local_breaker_qwen"])
    assert stopped.status == "error" and any(f.stopped_run for f in stopped.findings)
    out = _cli("fork", ids["policy_drop_c03"], "--at", "0", "--dry-run",
               "--system-append", "Never refund orders older than 30 days.").output  # fmt: skip
    assert "+Never refund orders older than 30 days." in out
    _needs("openai")
    _needs("litellm")  # local_breaker_qwen was recorded through LiteLLM
    for run_id in ids.values():
        src = crashlab.load(run_id)
        rep = replay(run_id)  # strict replay: no network, no keys
        assert rep.status == src.status, run_id
        assert len(rep.steps) == len(src.steps)


# --- refund_bot on LiteLLM (e.g. ollama_chat/qwen2.5:3b) --------------------------------


@pytest.fixture
def fake_litellm(monkeypatch):
    mod = types.ModuleType("litellm")
    mod.calls = []
    script = refund_script()

    def completion(model, messages, **kwargs):
        mod.calls.append(model)
        item = script.pop(0)
        item.model = model
        return item

    async def acompletion(*a, **kw):  # pragma: no cover - not used
        raise NotImplementedError

    mod.completion, mod.acompletion = completion, acompletion
    monkeypatch.setitem(sys.modules, "litellm", mod)
    return mod


def test_refund_bot_litellm_provider_records_and_replays(fake_litellm):
    answer = refund.run_agent("Refund A17", provider="litellm", model="ollama_chat/qwen2.5:3b")
    assert answer.startswith("Your refund")
    assert fake_litellm.calls == ["ollama_chat/qwen2.5:3b"] * 3
    run = crashlab.load("last")
    assert run.input["provider"] == "litellm"
    assert run.steps[0].attributes["crashlab.api"] == "litellm.completion"
    assert run.usage.cost_usd == 0.0  # local model
    assert run.outcome == "pass" and run.findings == []
    refund.reset_db()
    rep = replay(run.run_id)
    assert rep.status == "ok" and rep.output == run.output
    assert fake_litellm.calls == ["ollama_chat/qwen2.5:3b"] * 3  # replay made no calls


def test_refund_bot_providers(monkeypatch):
    assert refund.DEFAULT_MODELS["litellm"].startswith("ollama_chat/")
    monkeypatch.setenv("PROVIDER", "litellm")
    assert refund.default_provider() == "litellm"
    monkeypatch.setenv("PROVIDER", "nope")
    with pytest.raises(SystemExit):
        refund.default_provider()
