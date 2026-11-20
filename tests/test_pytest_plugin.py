"""Native pytest plugin: fixtures, marker constraints, failure saving, import safety."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

from crashlab.store import Store

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.endswith("API_KEY")}
    env.pop("CRASHLAB_DIR", None)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + existing if existing else "")
    return env


def _pytest(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:crashlab",
            "-p",
            "crashlab.pytest_plugin",
            "-o",
            "addopts=",
            "--rootdir",
            str(path),
            str(path),
            *args,
        ],
        cwd=path,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )


def _write(path: Path, name: str, source: str) -> Path:
    target = path / name
    target.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# import safety
# ---------------------------------------------------------------------------


def test_import_crashlab_without_pytest():
    script = textwrap.dedent(
        """
        import builtins
        import sys

        real = builtins.__import__

        def blocked(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "pytest" or name.startswith("pytest."):
                raise ImportError("No module named pytest")
            return real(name, globals, locals, fromlist, level)

        builtins.__import__ = blocked
        sys.modules.pop("pytest", None)
        for key in [k for k in sys.modules if k == "pytest" or k.startswith("pytest.")]:
            sys.modules.pop(key, None)

        import crashlab
        assert crashlab.__version__
        import crashlab.pytest_plugin as plugin
        assert plugin.pytest is None
        print("ok")
        """
    )
    res = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "ok" in res.stdout


# ---------------------------------------------------------------------------
# in-process fixtures (this repo's tests opt in)
# ---------------------------------------------------------------------------


def test_crashlab_store_is_isolated_store(crashlab_store, tmp_path):
    assert isinstance(crashlab_store, Store)
    assert crashlab_store.root == (tmp_path / ".crashlab").resolve()
    assert os.environ.get("CRASHLAB_DIR") == str(crashlab_store.root)


def test_crashlab_record_records_the_test_body(crashlab_record, crashlab_store):
    with crashlab_record as run:
        run.output = "hello"
    assert crashlab_record.run is run
    assert run.name == "test_crashlab_record_records_the_test_body"
    assert run.output == "hello"


# ---------------------------------------------------------------------------
# opt-in only: unmarked tests and --crashlab-save-failures
# ---------------------------------------------------------------------------


def test_plugin_does_not_interfere_with_unmarked_tests(tmp_path):
    dest = tmp_path / "persistent"
    _write(
        tmp_path,
        "test_plain.py",
        """
        def test_ok():
            assert 1 + 1 == 2

        def test_fail():
            assert False
        """,
    )
    res = _pytest(tmp_path, "--crashlab-save-failures", "--crashlab-store", str(dest))
    out = res.stdout + res.stderr
    assert res.returncode == 1 and "1 passed" in res.stdout and "1 failed" in res.stdout, out
    assert "crashlab show" not in out
    assert not dest.exists() or not Store(dest).list_runs(limit=10)


def test_crashlab_flag_still_selects_marked_tests(tmp_path):
    _write(
        tmp_path,
        "test_mixed.py",
        """
        import pytest

        @pytest.mark.crashlab
        def test_a():
            pass

        def test_b():
            pass
        """,
    )
    res = _pytest(tmp_path, "--crashlab")
    assert res.returncode == 0 and "1 passed, 1 deselected" in res.stdout, res.stdout
    res = _pytest(tmp_path)
    assert "2 passed" in res.stdout, res.stdout


# ---------------------------------------------------------------------------
# marker constraints
# ---------------------------------------------------------------------------


def test_marker_budget_and_findings_and_rules_enforced(tmp_path):
    _write(
        tmp_path,
        "rules.yaml",
        """
        rules:
          - type: must_call
            tools: [issue_refund]
        """,
    )
    _write(
        tmp_path,
        "test_constraints.py",
        """
        import pytest
        from crashlab.models import Finding

        @pytest.mark.crashlab(agent="budget-bot", max_tokens=10, max_cost_usd=0.01)
        def test_budget_ok(crashlab_record):
            crashlab_record.run.usage.total_tokens = 3
            crashlab_record.run.usage.cost_usd = 0.0
            crashlab_record.run.output = "ok"

        @pytest.mark.crashlab(max_tokens=10)
        def test_budget_over(crashlab_record):
            crashlab_record.run.usage.total_tokens = 9999
            crashlab_record.run.output = "over"

        @pytest.mark.crashlab(no_findings_above="error")
        def test_findings_ok(crashlab_record):
            crashlab_record.run.findings.append(
                Finding(kind="tool_error_streak", severity="info", message="noise")
            )

        @pytest.mark.crashlab(no_findings_above="error")
        def test_findings_over(crashlab_record):
            crashlab_record.run.findings.append(
                Finding(kind="step_repetition", severity="error", message="looped")
            )

        @pytest.mark.crashlab(rules="rules.yaml")
        def test_rules_fail(crashlab_record):
            crashlab_record.run.output = "no tools called"
        """,
    )
    res = _pytest(tmp_path)
    out = res.stdout + res.stderr
    assert res.returncode != 0, out
    assert "2 passed" in res.stdout and "3 failed" in res.stdout, out
    assert "max_tokens" in out
    assert "no_findings_above" in out
    assert "issue_refund" in out or "must_call" in out


def test_marker_agent_names_the_run(tmp_path):
    _write(
        tmp_path,
        "test_agent.py",
        """
        import pytest

        @pytest.mark.crashlab(agent="refund-bot")
        def test_named(crashlab_record):
            assert crashlab_record.run.name == "refund-bot"
            crashlab_record.run.output = "ok"
        """,
    )
    res = _pytest(tmp_path)
    assert res.returncode == 0 and "1 passed" in res.stdout, res.stdout + res.stderr


# ---------------------------------------------------------------------------
# --crashlab-save-failures
# ---------------------------------------------------------------------------


def test_save_failures_copies_record_and_prints_summary(tmp_path):
    dest = tmp_path / "persistent"
    testdir = tmp_path / "suite"
    testdir.mkdir()
    _write(
        testdir,
        "test_boom.py",
        """
        def test_boom(crashlab_record):
            crashlab_record.run.output = "x"
            raise AssertionError("boom")
        """,
    )
    res = _pytest(testdir, "--crashlab-save-failures", "--crashlab-store", str(dest))
    out = res.stdout + res.stderr
    assert res.returncode == 1 and "1 failed" in res.stdout, out
    assert "crashlab show " in out and "crashlab diagnose " in out
    runs = Store(dest).list_runs(limit=10)
    assert len(runs) == 1
    run_id = runs[0]["run_id"]
    assert f"crashlab show {run_id}" in out
    assert f"crashlab diagnose {run_id}" in out
    saved = Store(dest).load_run(run_id)
    assert saved.name == "test_boom" and saved.output == "x"
    assert saved.status == "error" and saved.outcome == "fail"


def test_save_failures_also_copies_constraint_violations(tmp_path):
    dest = tmp_path / "persistent"
    testdir = tmp_path / "suite"
    testdir.mkdir()
    _write(
        testdir,
        "test_over.py",
        """
        import pytest

        @pytest.mark.crashlab(max_tokens=1)
        def test_over(crashlab_record):
            crashlab_record.run.usage.total_tokens = 100
            crashlab_record.run.output = "too many"
        """,
    )
    res = _pytest(testdir, "--crashlab-save-failures", "--crashlab-store", str(dest))
    out = res.stdout + res.stderr
    assert res.returncode == 1 and "1 failed" in res.stdout, out
    assert "crashlab show " in out and "crashlab diagnose " in out
    runs = Store(dest).list_runs(limit=10)
    assert len(runs) == 1
    saved = Store(dest).load_run(runs[0]["run_id"])
    assert saved.output == "too many" and saved.usage.total_tokens == 100


# ---------------------------------------------------------------------------
# crashlab_replay factory
# ---------------------------------------------------------------------------


def test_crashlab_replay_from_run_and_path(tmp_path):
    _write(
        tmp_path,
        "test_replay.py",
        """
        from pathlib import Path
        import crashlab

        def toy() -> str:
            with crashlab.record("toy") as run:
                run.output = "hello"
            return "hello"

        def test_replay(crashlab_replay, crashlab_store, tmp_path):
            toy()
            src = crashlab_store.load_run("last")
            assert src.output == "hello"
            rep = crashlab_replay(src, fn=toy)
            assert rep.output == "hello" and rep.replay_of == src.run_id
            cassette = tmp_path / "toy.jsonl"
            crashlab.store.export_run(crashlab_store, src.run_id, cassette)
            rep2 = crashlab_replay(cassette, fn=toy)
            assert rep2.output == "hello" and rep2.replay_of == src.run_id
        """,
    )
    res = _pytest(tmp_path)
    assert res.returncode == 0 and "1 passed" in res.stdout, res.stdout + res.stderr
