"""export-test tells you how to install pytest when it is missing. Offline, no SDKs."""

from __future__ import annotations

from crashlab.store import Store
from tests.debug_fixtures import make_run


def test_hint_is_none_when_pytest_is_importable():
    from crashlab.cli import _export_test_pytest_hint

    assert _export_test_pytest_hint() is None


def test_hint_names_both_install_commands_when_pytest_is_missing(monkeypatch):
    from crashlab import cli

    monkeypatch.setattr(cli, "_pytest_is_importable", lambda: False)
    hint = cli._export_test_pytest_hint()
    assert hint is not None
    assert "pytest is needed to run the exported test" in hint
    assert "pip install pytest" in hint
    assert 'pip install "crashlab[pytest]"' in hint


def test_export_test_still_writes_when_pytest_is_missing(tmp_path, monkeypatch, capsys):
    store_dir = tmp_path / "store"
    run = make_run(Store(store_dir), 1)
    monkeypatch.setenv("CRASHLAB_DIR", str(store_dir))
    monkeypatch.setattr("crashlab.cli._pytest_is_importable", lambda: False)
    from crashlab.cli import export_test_cmd

    out = tmp_path / "exported"
    export_test_cmd(run_ref=run.run_id, fix=None, out=out)
    written = list(out.rglob("test_*.py"))
    assert written, "export must still write the test file"
    captured = capsys.readouterr().out
    assert "wrote" in captured
    assert "run it: pytest" in captured
    assert "pytest is needed to run the exported test" in captured
    assert "pip install pytest" in captured
    assert "crashlab[pytest]" in captured


def test_export_test_omits_install_hint_when_pytest_is_present(tmp_path, monkeypatch, capsys):
    store_dir = tmp_path / "store"
    run = make_run(Store(store_dir), 1)
    monkeypatch.setenv("CRASHLAB_DIR", str(store_dir))
    from crashlab.cli import export_test_cmd

    export_test_cmd(run_ref=run.run_id, fix=None, out=tmp_path / "exported")
    captured = capsys.readouterr().out
    assert "run it: pytest" in captured
    assert "pytest is needed" not in captured
