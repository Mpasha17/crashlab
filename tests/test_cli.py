from __future__ import annotations

from examples.refund_bot import agent as refund
from typer.testing import CliRunner

import crashlab
from crashlab.cli import app
from tests.fakes import FakeOpenAI, refund_script

runner = CliRunner()
ENV = {"COLUMNS": "220", "TERM": "xterm-256color", "NO_COLOR": "1"}


def _record_one() -> str:
    refund.run_agent("Refund my order A17", client=crashlab.wrap(FakeOpenAI(refund_script())))
    return crashlab.load("last").run_id


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"crashlab {crashlab.__version__}"


def test_ls_empty():
    result = runner.invoke(app, ["ls"], env=ENV)
    assert result.exit_code == 0
    assert "No runs recorded yet" in result.output


def test_ls_and_show():
    run_id = _record_one()
    result = runner.invoke(app, ["ls"], env=ENV)
    assert result.exit_code == 0, result.output
    for text in (run_id, "refund-bot", "ok", "unknown", " 5 ", "443"):
        assert text in result.output

    result = runner.invoke(app, ["show", "last"], env=ENV)
    assert result.exit_code == 0, result.output
    out = result.output
    assert run_id in out
    assert "examples.refund_bot.agent:run_agent" in out
    assert "execute_tool" in out and "get_order" in out and "issue_refund" in out
    assert '→ get_order({"order_id": "A17"})' in out
    assert "Your refund of $49.00" in out

    # prefix lookup and JSON output
    result = runner.invoke(app, ["show", run_id[:22], "--json"], env=ENV)
    assert result.exit_code == 0
    assert '"run_id": "' + run_id in result.output


def test_show_error_column():
    from tests.fakes import text_response, tool_call_response

    script = [
        tool_call_response(1, "issue_refund", {"order_id": "A17", "amount": 999}),
        text_response(2, "no"),
    ]
    refund.run_agent("x", client=crashlab.wrap(FakeOpenAI(script)))
    result = runner.invoke(app, ["show", "last"], env=ENV)
    assert result.exit_code == 0
    assert "ValueError" in result.output


def test_show_missing_run():
    result = runner.invoke(app, ["show", "NOPE"], env=ENV)
    assert result.exit_code == 1
    assert "no run" in result.output


def test_dir_option(tmp_path):
    other = tmp_path / "elsewhere"
    result = runner.invoke(app, ["--dir", str(other), "ls"], env=ENV)
    assert result.exit_code == 0
    assert "No runs recorded yet" in result.output
