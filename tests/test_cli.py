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


def _table_rows(output: str) -> list[str]:
    """Step-table body rows with the model name blanked (it differs per provider)."""
    rows = [ln for ln in output.splitlines() if "│ chat " in ln or "│ execute_tool " in ln]
    cells = [[c.strip() for c in ln.strip("│").split("│")] for ln in rows]
    return [" | ".join("MODEL" if c in ("claude-fake", "fake-model") else c for c in row)
            for row in cells]  # fmt: skip


def test_show_identical_for_openai_and_anthropic():
    from tests.fakes import FakeAnthropic, anthropic_refund_script

    refund.run_agent("Refund my order A17", client=crashlab.wrap(FakeOpenAI(refund_script())))
    o = runner.invoke(app, ["show", "last"], env=ENV).output
    refund.reset_db()
    refund.run_agent(
        "Refund my order A17", client=crashlab.wrap(FakeAnthropic(anthropic_refund_script()))
    )
    a = runner.invoke(app, ["show", "last"], env=ENV).output
    o_rows, a_rows = _table_rows(o), _table_rows(a)
    assert len(o_rows) == 5
    # the Anthropic fake also says "Let me check." before each tool call; compare tool rows
    assert [r for r in o_rows if "execute_tool" in r] == [r for r in a_rows if "execute_tool" in r]
    assert "→ get_order" in a and "Your refund of $49.00" in a


def test_show_marks_streamed_steps():
    from tests.fakes import chunks_for_text

    client = crashlab.wrap(FakeOpenAI([chunks_for_text("streamed!")]))
    with crashlab.record("s"):
        list(client.chat.completions.create(model="m", messages=[], stream=True))
    out = runner.invoke(app, ["show", "last"], env=ENV).output
    assert "streamed! (stream)" in out
