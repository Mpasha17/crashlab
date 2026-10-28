from __future__ import annotations

import re

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
    for text in (run_id, "refund-bot", "ok", "pass", " 5 ", "443"):
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
    output = re.sub(r"\x1b\[[0-9;]*m", "", output)
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


def test_command_help_is_unwrapped_and_the_list_fits_80_columns():
    # live: help kept the docstrings' hard line breaks, so `crashlab --help` and
    # `crashlab fix --help` wrapped raggedly at 80 columns
    from crashlab.cli import app

    for info in app.registered_commands:
        paras = info.help.split("\n\n")
        assert all("\n" not in p for p in paras), info.name
        assert len(info.short_help) <= 60, (info.name, info.short_help)
    res = CliRunner().invoke(app, ["--help"], terminal_width=80)
    # the terminal width (and rich vs ascii boxes) differs on CI: the limits above are
    # what keeps the list on one line per command at 80 columns
    assert "Verify a fix" in res.output


def test_show_fits_80_columns_with_a_narrow_layout(crashlab_dir, monkeypatch):
    # live: at 80 columns the 8-column step table squeezed args / results to a few
    # characters per line
    from rich.console import Console

    import crashlab.cli as cli

    refund.run_agent("Refund my order A17", client=crashlab.wrap(FakeOpenAI(refund_script())),
                     model="fake-model", faults=[])  # fmt: skip
    monkeypatch.setenv("TERM", "xterm")  # rich ignores width= on a dumb terminal

    def show(width: int) -> str:
        monkeypatch.setattr(
            cli, "_console", lambda stderr=False: Console(width=width, color_system=None)
        )
        res = CliRunner().invoke(app, ["show", "last"])
        assert res.exit_code == 0, res.output
        return res.output

    lines = show(80).splitlines()
    assert all(len(ln) <= 80 for ln in lines)
    assert any("args → result / error" in ln for ln in lines)
    assert any("tool " in ln and "get_order" in ln for ln in lines)
    wide = show(160)
    assert "result / output" in wide and "args → result" not in wide


def test_a_tool_culprit_names_the_llm_call_that_made_it(crashlab_dir):
    # live: diagnoses blame the tool step (where the damage shows, e.g. place_order with
    # quantity 10) and users read it as "the tool is broken"; name the deciding LLM call
    refund.run_agent("Refund my order A17", client=crashlab.wrap(FakeOpenAI(refund_script())),
                     model="fake-model", faults=[])  # fmt: skip
    run = crashlab.load("last")
    tool = next(s for s in run.steps if s.operation == "execute_tool")
    chat = max(s.idx for s in run.steps if s.operation == "chat" and s.idx < tool.idx)
    assert run.describe_step(tool.idx) == (
        f"step {tool.idx} (tool {tool.name}, called by chat step {chat})"
    )
    assert run.describe_step(0) == f"step 0 (chat {run.steps[0].name})"
    assert run.describe_step(99) == "step 99"


def test_show_names_the_unpriced_model():
    # live: runs on qwen/qwen3.8-27b showed "est. ?" with no hint why or how to fix it
    from tests.fakes import text_response

    @crashlab.agent(name="unpriced")
    def go() -> str:
        fake = FakeOpenAI([text_response(1, "hi", model="acme/unpriced-1")])
        fake.base_url = "https://api.acme.example/v1"  # not a local server ($0)
        client = crashlab.wrap(fake)
        r = client.chat.completions.create(
            model="acme/unpriced-1", messages=[{"role": "user", "content": "hi"}]
        )
        return r.choices[0].message.content

    go()
    out = runner.invoke(app, ["show", "last"], env=ENV).output
    assert "est. ?" in out and "no price for acme/unpriced-1" in out
    assert "[tool.crashlab.prices]" in out
    _record_one()  # a priced model: no hint
    assert "no price for" not in runner.invoke(app, ["show", "last"], env=ENV).output
