"""CLI on a fresh store and with bad inputs: clear one-line errors, no tracebacks."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from crashlab.cli import app

runner = CliRunner()
ENV = {"COLUMNS": "200", "NO_COLOR": "1"}


def _out(result) -> str:
    return " ".join(result.output.split())


def test_export_with_no_runs_is_a_clean_error(tmp_path):
    result = runner.invoke(app, ["export", "last", "-o", str(tmp_path / "x.jsonl")], env=ENV)
    assert result.exit_code == 1
    assert "no runs recorded yet" in _out(result)
    assert isinstance(result.exception, SystemExit)  # not a RunNotFoundError traceback


def test_export_unknown_run(tmp_path):
    result = runner.invoke(app, ["export", "NOPE"], env=ENV)
    assert result.exit_code == 1
    assert "no run matching 'NOPE'" in _out(result)


def test_import_missing_file_says_so(tmp_path):
    result = runner.invoke(app, ["import", str(tmp_path / "missing.jsonl")], env=ENV)
    assert result.exit_code == 2
    assert "no such file" in _out(result)
    assert "not a crashlab export" not in _out(result)


def test_import_directory(tmp_path):
    result = runner.invoke(app, ["import", str(tmp_path)], env=ENV)
    assert result.exit_code == 2
    assert "is a directory" in _out(result)


def test_check_all_on_empty_store():
    result = runner.invoke(app, ["check", "--all"], env=ENV)
    assert result.exit_code == 0
    assert "No runs to check" in _out(result)
    result = runner.invoke(app, ["check", "--all", "--json"], env=ENV)
    assert result.exit_code == 0
    assert json.loads(result.output) == []


def test_check_all_json_is_a_list_even_with_one_run(crashlab_dir):
    import crashlab

    @crashlab.agent(name="solo")
    def solo() -> str:
        return "ok"

    solo()
    result = runner.invoke(app, ["check", "--all", "--json"], env=ENV)
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert isinstance(data, list) and len(data) == 1 and data[0]["name"] == "solo"
    one = json.loads(runner.invoke(app, ["check", "last", "--json"], env=ENV).output)
    assert isinstance(one, dict)  # a single explicit ref stays an object
