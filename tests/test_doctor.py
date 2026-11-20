"""Operational diagnostics (``crashlab doctor``) and project scaffolding (``init``)."""

from __future__ import annotations

import json
import re

from typer.testing import CliRunner

from crashlab.cli import app
from crashlab.doctor import run_doctor
from crashlab.models import Run
from crashlab.scaffold import init_project
from crashlab.store import Store
from tests.debug_fixtures import chat, make_run, save

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


def _names(report, status: str | None = None) -> list[str]:
    return [c.name for c in report.checks if status is None or c.status == status]


def test_doctor_on_missing_store_warns_and_fix_initializes(tmp_path):
    missing = tmp_path / "no-such-store"
    report = run_doctor(missing, fix=False)
    assert report.ok
    assert "store" in _names(report, "warn")
    assert not (missing / "index.sqlite").exists()

    fixed = run_doctor(missing, fix=True)
    assert fixed.ok
    assert (missing / "index.sqlite").is_file()
    store_check = next(c for c in fixed.checks if c.name == "store")
    assert store_check.fixed
    assert "initialized" in store_check.message
    # a second --fix on the empty store must not wipe anything
    again = run_doctor(missing, fix=True)
    assert again.ok
    assert (missing / "index.sqlite").is_file()


def test_doctor_healthy_store_and_fix_keeps_runs(crashlab_dir):
    store = Store(crashlab_dir)
    make_run(store, 1)
    rid = store.resolve_run_id("last")
    report = run_doctor(crashlab_dir, fix=False)
    assert report.ok
    assert "sqlite" in _names(report, "ok")
    assert "schema" in _names(report, "ok")
    sqlite_check = next(c for c in report.checks if c.name == "sqlite")
    assert sqlite_check.details["run_count"] == 1
    assert sqlite_check.details["integrity"] == "ok"
    assert sqlite_check.details["db_bytes"] > 0

    fixed = run_doctor(crashlab_dir, fix=True)
    assert fixed.ok
    assert any("VACUUM" in f or "REINDEX" in f for f in fixed.fixes_applied)
    # --fix must never delete recorded runs
    loaded = store.load_run(rid)
    assert loaded.run_id == rid
    assert loaded.steps


def test_doctor_flags_orphaned_child_and_dangling_last(crashlab_dir):
    store = Store(crashlab_dir)
    parent = make_run(store, 1)
    child = Run(
        run_id="01FIXTURE00000000000000002",
        name="forked",
        status="ok",
        outcome="pass",
        forked_from="01MISSING00000000000000001",
        started_at=parent.started_at,
    )
    save(store, child, [chat(child.run_id, 0, "hi", "ok")])

    report = run_doctor(crashlab_dir)
    assert "orphans" in _names(report, "warn")
    orphan_check = next(c for c in report.checks if c.name == "orphans")
    parents = {o["missing_parent"] for o in orphan_check.details["orphans"]}
    assert "01MISSING00000000000000001" in parents
    assert report.ok  # warnings do not fail the report

    # dangling last: newest id, but its run directory is gone
    last = make_run(store, 9)
    import shutil

    shutil.rmtree(store.run_dir(last.run_id))
    dangling = run_doctor(crashlab_dir)
    last_check = next(c for c in dangling.checks if c.name == "last_run")
    assert last_check.status == "warn"
    assert last.run_id in last_check.message
    assert "dangling" in last_check.message


def test_doctor_flags_unpriced_models_and_reports_packages(crashlab_dir):
    store = Store(crashlab_dir)
    rid = "01FIXTURE00000000000000003"
    step = chat(rid, 0, "hello", "hi", model="acme/unpriced-1", tokens=(20, 5), cost=None)
    step.attributes["server.address"] = "api.acme.example"
    save(
        store,
        Run(run_id=rid, name="unpriced", status="ok", outcome="pass"),
        [step],
    )
    report = run_doctor(crashlab_dir)
    pricing = next(c for c in report.checks if c.name == "pricing")
    assert pricing.status == "warn"
    assert "acme/unpriced-1" in pricing.message
    assert "acme/unpriced-1" in pricing.details["unpriced"]

    packages = next(c for c in report.checks if c.name == "packages")
    assert packages.status == "ok"
    # pytest is required to run this file; a missing optional SDK must not crash
    assert "pytest" in packages.details["installed"]
    assert "missing" in packages.details
    assert "openai" in packages.details["installed"] + packages.details["missing"]


def test_doctor_reports_redaction_and_env(crashlab_dir, monkeypatch):
    monkeypatch.setenv("CRASHLAB_MODE", "record")
    monkeypatch.setenv("CRASHLAB_DIAGNOSER_API_KEY", "sk-secret-must-not-leak")
    report = run_doctor(crashlab_dir)
    redaction = next(c for c in report.checks if c.name == "redaction")
    assert redaction.status == "ok"
    assert redaction.details["enabled"] is True
    env = next(c for c in report.checks if c.name == "env")
    assert env.status == "ok"
    assert env.details["env"]["CRASHLAB_MODE"] == "record"
    assert env.details["env"]["CRASHLAB_DIAGNOSER_API_KEY"] == "set"
    blob = json.dumps(report.to_dict())
    assert "sk-secret-must-not-leak" not in blob
    assert "sk-secret" not in env.message

    monkeypatch.setenv("CRASHLAB_REDACT", "0")
    warned = run_doctor(crashlab_dir)
    env = next(c for c in warned.checks if c.name == "env")
    assert env.status == "warn"
    assert "CRASHLAB_REDACT" in env.message


def test_doctor_cli_help_strips_ansi_and_json_roundtrip(crashlab_dir):
    make_run(Store(crashlab_dir), 1)
    help_out = _plain(runner.invoke(app, ["doctor", "--help"]).output)
    assert "Diagnose the local crashlab store" in help_out
    assert "--fix" in help_out
    assert "\x1b[" not in help_out

    res = runner.invoke(app, ["doctor", "--json"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["ok"] is True
    assert payload["summary"]
    assert {c["name"] for c in payload["checks"]} >= {
        "store",
        "sqlite",
        "schema",
        "orphans",
        "last_run",
        "packages",
        "redaction",
        "env",
        "pricing",
    }


def test_init_writes_workflow_checks_and_gitignore(tmp_path):
    target = tmp_path / "proj"
    result = init_project(target, ci="github", checks=True, force=False)
    workflow = target / ".github" / "workflows" / "crashlab-eval.yml"
    checks = target / "crashlab_checks.py"
    gitignore = target / ".gitignore"
    assert workflow.is_file()
    text = workflow.read_text("utf-8")
    assert "crashlab eval" in text
    assert "--junit" in text
    assert "GITHUB_STEP_SUMMARY" in text
    assert checks.is_file()
    src = checks.read_text("utf-8")
    assert "@crashlab.check" in src
    assert "crashlab_checks.py" in result.created
    assert ".gitignore" in result.created
    assert ".crashlab/" in gitignore.read_text("utf-8")

    # existing files are left alone unless force=True
    workflow.write_text("# keep me\n", "utf-8")
    skipped = init_project(target, ci="github", checks=True, force=False)
    assert "crashlab_checks.py" in skipped.skipped
    assert workflow.read_text("utf-8") == "# keep me\n"
    forced = init_project(target, ci="github", checks=True, force=True)
    assert any(p.endswith("crashlab-eval.yml") for p in forced.updated + forced.created)
    assert "crashlab eval" in workflow.read_text("utf-8")

    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / ".gitignore").write_text("venv/\n", "utf-8")
    noted = init_project(bare, ci="none", checks=False, force=False)
    assert "skipped CI workflow" in " ".join(noted.notes)
    assert "crashlab_checks.py" not in (noted.created + noted.updated)
    assert ".crashlab/" in (bare / ".gitignore").read_text("utf-8")
    assert ".gitignore" in noted.updated


def test_init_cli_help_and_invalid_ci(tmp_path):
    help_out = _plain(runner.invoke(app, ["init", "--help"]).output)
    assert "Scaffold a crashlab eval" in help_out
    assert "--ci" in help_out
    assert "--force" in help_out
    assert "\x1b[" not in help_out

    res = runner.invoke(app, ["init", str(tmp_path / "x"), "--ci", "gitlab"])
    assert res.exit_code == 2
    assert "unsupported ci" in _plain(res.output).lower() or "gitlab" in _plain(res.output)

    res = runner.invoke(app, ["init", str(tmp_path / "ok"), "--ci", "none", "--no-checks"])
    assert res.exit_code == 0, res.output
    assert not (tmp_path / "ok" / ".github" / "workflows" / "crashlab-eval.yml").exists()
    assert not (tmp_path / "ok" / "crashlab_checks.py").exists()
    assert (tmp_path / "ok" / ".gitignore").is_file()


def test_doctor_does_not_create_store_without_fix(tmp_path):
    missing = tmp_path / "untouched"
    run_doctor(missing, fix=False)
    assert not missing.exists()


def test_init_project_rejects_unknown_ci(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="unsupported ci"):
        init_project(tmp_path, ci="bitbucket")
