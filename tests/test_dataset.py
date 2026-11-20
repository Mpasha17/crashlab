"""crashlab.dataset / crashlab dataset: portable golden JSONL test cases."""

from __future__ import annotations

import json
import re

from typer.testing import CliRunner

import crashlab
from crashlab.cli import app
from crashlab.dataset import CASE_FIELDS, build_dataset, inspect_dataset
from tests.fakes import FakeOpenAI, text_response, tool_call_response


def _ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _client(script=None):
    return crashlab.wrap(FakeOpenAI(list(script or [text_response(1, "Refunded $49.")])))


@crashlab.tool()
def get_order(order_id: str) -> dict:
    return {"order_id": order_id, "total": 49}


@crashlab.agent(name="dataset-refund", ignore_args=("client",))
def run_refund(user: str, client, model: str = "fake-model") -> str:
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a refund assistant. Email support@shop.test."},
            {"role": "user", "content": user},
        ],
        tools=[{"type": "function", "function": {"name": "get_order"}}],
    )
    msg = r.choices[0].message
    if not msg.tool_calls:
        return msg.content or ""
    args = json.loads(msg.tool_calls[0].function.arguments)
    get_order(**args)
    r2 = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user}],
    )
    return r2.choices[0].message.content or ""


@crashlab.agent(name="dataset-echo")
def run_echo(msg: str) -> str:
    return f"echo:{msg}"


_SECRET_USER = "Refund order A17, email alice@example.com, key sk-proj-ABCDEFGHIJKLMNOPQRSTUV"


def _record_refund(user: str = _SECRET_USER):
    script = [
        tool_call_response(1, "get_order", {"order_id": "A17"}),
        text_response(2, "Refunded $49 to alice@example.com"),
    ]
    run_refund(user, _client(script))
    return crashlab.load("last")


def test_build_dataset_writes_jsonl_cases(tmp_path):
    src = _record_refund()
    crashlab.store.Store().tag_run(src.run_id, add=["golden"])
    src = crashlab.load(src.run_id)
    out = tmp_path / "golden.jsonl"
    report = build_dataset([src], out_path=out, redact=True, include_tools=True)
    assert report.cases == 1 and report.path == str(out)
    assert "dataset-refund" in report.agents
    assert "get_order" in report.tools
    assert "golden" in report.tags
    line = out.read_text("utf-8").strip()
    rec = json.loads(line)
    assert tuple(rec) == CASE_FIELDS or set(CASE_FIELDS) <= set(rec)
    for key in CASE_FIELDS:
        assert key in rec
    assert rec["id"] == src.run_id
    assert rec["name"] == "dataset-refund"
    assert rec["expected_tools"] == ["get_order"]
    assert rec["expected_outcome"] in ("pass", "fail", "unknown")
    assert rec["system_prompt"] and "refund assistant" in rec["system_prompt"]
    assert rec["reference_reply"]
    assert "golden" in rec["tags"]
    assert rec["metadata"]["run_id"] == src.run_id


def test_redact_scrubs_keys_and_emails(tmp_path):
    src = _record_refund()
    # the store already redacts API keys on write; put a fresh key on the in-memory
    # run so we can see redact=False keep it and redact=True strip it (plus emails).
    src.input = {
        "user": "Refund order A17, email alice@example.com, key sk-proj-ABCDEFGHIJKLMNOPQRSTUV"
    }
    out = tmp_path / "redacted.jsonl"
    rec = build_dataset(src, out_path=out, redact=True).records[0]
    blob = json.dumps(rec)
    assert "sk-proj-" not in blob
    assert "alice@example.com" not in blob
    assert "support@shop.test" not in blob
    assert "[REDACTED]" in blob
    raw = build_dataset(src, redact=False).records[0]
    raw_blob = json.dumps(raw)
    assert "alice@example.com" in raw_blob
    assert "sk-proj-" in raw_blob


def test_include_tools_false_leaves_expected_tools_empty():
    src = _record_refund()
    rec = build_dataset(src, include_tools=False).records[0]
    assert rec["expected_tools"] == []


def test_inspect_counts_cases_and_malformed_lines(tmp_path):
    a = _record_refund()
    run_echo("hi")
    b = crashlab.load("last")
    out = tmp_path / "set.jsonl"
    build_dataset([a, b], out_path=out)
    # trailing garbage + a blank line + a line missing id
    with out.open("a", encoding="utf-8") as fh:
        fh.write("\n")
        fh.write("{not json\n")
        fh.write(json.dumps({"name": "no-id"}) + "\n")
    summary = inspect_dataset(out)
    assert summary.cases == 2
    assert summary.malformed == 2
    assert summary.agents["dataset-refund"] == 1
    assert summary.agents["dataset-echo"] == 1
    assert summary.tools["get_order"] == 1
    assert summary.tool_coverage == summary.tools
    data = summary.as_dict()
    assert data["malformed"] == 2 and data["cases"] == 2


def test_inspect_missing_file_raises(tmp_path):
    try:
        inspect_dataset(tmp_path / "missing.jsonl")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")


def test_cli_build_filters_and_inspect_json(tmp_path):
    src = _record_refund()
    crashlab.store.Store().tag_run(src.run_id, add=["golden"])
    run_echo("hi")
    runner = CliRunner()
    out = tmp_path / "ds.jsonl"
    res = runner.invoke(
        app,
        [
            "dataset",
            "build",
            "--name",
            "dataset-refund",
            "--tag",
            "golden",
            "--out",
            str(out),
            "--json",
        ],
    )
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["cases"] == 1
    assert data["agents"] == ["dataset-refund"]
    recs = [json.loads(line) for line in out.read_text("utf-8").splitlines() if line.strip()]
    assert len(recs) == 1 and recs[0]["name"] == "dataset-refund"

    insp = runner.invoke(app, ["dataset", "inspect", str(out), "--json"])
    assert insp.exit_code == 0, insp.output
    summary = json.loads(insp.output)
    assert summary["cases"] == 1 and summary["malformed"] == 0
    assert summary["tools"]["get_order"] == 1


def test_cli_build_by_run_id_and_no_matches(tmp_path):
    src = _record_refund()
    runner = CliRunner()
    out = tmp_path / "from-id.jsonl"
    res = runner.invoke(app, ["dataset", "build", src.run_id, "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert "1 case" in _ansi(res.output)
    missing = runner.invoke(app, ["dataset", "build", "--name", "no-such-agent"])
    assert missing.exit_code == 2
    assert "no runs" in _ansi(missing.output).lower()
    absent = runner.invoke(app, ["dataset", "inspect", "does-not-exist.jsonl"])
    assert absent.exit_code == 2


def test_cli_help_lists_dataset():
    runner = CliRunner()
    res = runner.invoke(app, ["--help"])
    clean = _ansi(res.output)
    assert res.exit_code == 0 and "dataset" in clean
    res = runner.invoke(app, ["dataset", "--help"])
    clean = _ansi(res.output)
    assert res.exit_code == 0
    assert "build" in clean and "inspect" in clean
    res = runner.invoke(app, ["dataset", "build", "--help"])
    clean = _ansi(res.output)
    assert res.exit_code == 0
    for flag in ("--name", "--tag", "--outcome", "--since", "--out", "--json", "--no-redact"):
        assert flag in clean
    res = runner.invoke(app, ["dataset", "inspect", "--help"])
    clean = _ansi(res.output)
    assert res.exit_code == 0 and "--json" in clean
