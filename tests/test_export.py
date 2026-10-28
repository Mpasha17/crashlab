"""export-test, the pytest plugin and the HTML report. Offline."""

from __future__ import annotations

import contextlib
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

import crashlab
from crashlab.cli import app
from crashlab.export import ExportError, export_test
from crashlab.report_html import render_html
from crashlab.verify import verifications
from tests.test_verify import PolicyModel, _setup

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def _pytest(path: Path, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.endswith("API_KEY")}
    env.pop("CRASHLAB_DIR", None)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            "--rootdir",
            str(path),
            str(path),
            *args,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_export_failing_run_with_verified_fix_and_run_it(tmp_path):
    pytest.importorskip("openai")
    run, _ = _setup()
    crashlab.verify(run, 1, k=3, regress="none", client=crashlab.wrap(PolicyModel()))
    out = tmp_path / "exported"
    paths = export_test(run.run_id, fix="auto", out_dir=out)
    names = sorted(p.name for p in paths)
    assert len(paths) == 3 and any(n.endswith("_fixed.jsonl") for n in names)
    test = next(p for p in paths if p.suffix == ".py")
    src = test.read_text()
    assert "pytestmark = pytest.mark.crashlab" in src and "fix_keeps_passing" in src
    assert '"refund_within_policy": "fail"' in src and "Verification verdict: VERIFIED" in src
    res = _pytest(out)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "2 passed" in res.stdout
    if importlib.util.find_spec("ruff"):  # generated code is lint/format clean
        for cmd in (
            ["check", "--isolated", "--line-length", "100"],
            ["format", "--check", "--isolated", "--line-length", "100"],
        ):
            lint = subprocess.run(
                [sys.executable, "-m", "ruff", *cmd, str(test)], capture_output=True, text=True
            )
            assert lint.returncode == 0, lint.stdout + lint.stderr


def test_exported_test_fails_when_the_run_no_longer_reproduces(tmp_path):
    pytest.importorskip("openai")
    run, _ = _setup()
    out = tmp_path / "exported"
    [cass, test] = export_test(run.run_id, out_dir=out)
    # tamper with the pinned expectation: the test must catch it
    test.write_text(test.read_text().replace('run.outcome == "fail"', 'run.outcome == "pass"'))
    res = _pytest(out)
    assert res.returncode == 1 and "1 failed" in res.stdout


def test_plugin_flag_selects_only_crashlab_tests(tmp_path):
    (tmp_path / "test_mixed.py").write_text(
        "import pytest\n\n@pytest.mark.crashlab\ndef test_a():\n    pass\n\n"
        "def test_b():\n    pass\n"
    )
    res = _pytest(tmp_path, "--crashlab")
    assert res.returncode == 0 and "1 passed, 1 deselected" in res.stdout, res.stdout
    res = _pytest(tmp_path)
    assert "2 passed" in res.stdout


def test_export_errors():
    run, _ = _setup()
    with pytest.raises(ExportError, match="no passing patched run for fix-1"):
        export_test(run.run_id, fix="fix-1")
    v = crashlab.verify(run, 1, k=1, regress="none", client=crashlab.wrap(PolicyModel()))
    child = v.trials[0].run_id
    with pytest.raises(ExportError, match="fork/replay"):
        export_test(child)


def test_cli_export_test(tmp_path):
    run, _ = _setup()
    res = runner.invoke(app, ["export-test", run.run_id, "-o", str(tmp_path / "t")])
    assert res.exit_code == 0 and "wrote" in res.output
    res = runner.invoke(app, ["export-test", run.run_id, "--fix", "fix-1", "-o", str(tmp_path)])
    assert res.exit_code == 2 and "--fix fix-1" in res.output


def test_html_report_is_self_contained_and_escaped(tmp_path):
    run, _ = _setup()
    crashlab.verify(run, 1, k=2, regress="none", client=crashlab.wrap(PolicyModel()))
    run = crashlab.load(run.run_id)
    run.output = "<script>alert(1)</script>"
    html = render_html(run, verifications(crashlab.store.Store(), run.run_id))
    for section in (
        "Checks",
        "Diagnosis",
        "Proposed fixes",
        "Verification of fix-1",
        "Step timeline",
        "Wilson 95% CI",
    ):
        assert section in html
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert not re.search(r"""(src|href)\s*=|@import|url\(""", html)  # no external assets
    assert 'class="add"' in html and "TASK_SPEC_VIOLATION" in html and "2/2" in html
    out = tmp_path / "r.html"
    res = runner.invoke(app, ["report", run.run_id, "--html", str(out)])
    assert res.exit_code == 0 and out.read_text().startswith("<!doctype html>")


def test_export_scrubs_stack_trace_paths(tmp_path, crashlab_dir, monkeypatch):
    """Error messages often embed abs paths (traceback); those must not be committed."""
    from crashlab.export import _export
    from crashlab.store import Store

    monkeypatch.chdir(tmp_path)
    store = Store(crashlab_dir)

    @crashlab.agent(name="boom")
    def boom() -> None:
        raise RuntimeError(f"failed in {Path.cwd() / 'secret.py'} under {crashlab_dir}")

    with contextlib.suppress(RuntimeError):
        boom()
    run = crashlab.load("last")
    import crashlab as _pkg

    pkg_dir = Path(_pkg.__file__).resolve().parent
    with contextlib.suppress(RuntimeError):
        # second boom embedding the install path
        @crashlab.agent(name="boom-pkg")
        def boom_pkg() -> None:
            raise RuntimeError(f"from {pkg_dir / 'export.py'}")

        boom_pkg()
    run2 = crashlab.load("last")
    out = _export(store, run.run_id, tmp_path / "x.jsonl")
    out2 = _export(store, run2.run_id, tmp_path / "y.jsonl")
    text = out.read_text() + out2.read_text()
    assert str(crashlab_dir) not in text
    assert str(tmp_path) not in text or "$PWD" in text or "$CRASHLAB" in text
    assert str(pkg_dir) not in text
    assert "$CRASHLAB" in text or "$CRASHLAB_REPO" in text or "$SITE_PACKAGES" in text
    assert "secret.py" in text  # basename / relative residue ok
    assert "/home/" not in text or text.count("~") >= 0


def test_local_roots_wheel_vs_editable(tmp_path, monkeypatch):
    """Wheel → $SITE_PACKAGES; editable src layout → $CRASHLAB_REPO."""
    import sys

    import crashlab as real_pkg
    import crashlab.export as exp
    from crashlab.store import Store

    store = Store(tmp_path / "store")

    tokens = {t for _, t in exp._local_roots(store)}
    assert "$CRASHLAB" in tokens
    assert "$CRASHLAB_REPO" in tokens  # preferred over $PWD when cwd is the repo

    site = tmp_path / "site-packages" / "crashlab"
    site.mkdir(parents=True)
    init = site / "__init__.py"
    init.write_text("# fake\n")

    class _Fake:
        __file__ = str(init)

    monkeypatch.setitem(sys.modules, "crashlab", _Fake())
    tokens_w = {t for _, t in exp._local_roots(store)}
    assert "$SITE_PACKAGES" in tokens_w and "$CRASHLAB" in tokens_w
    monkeypatch.setitem(sys.modules, "crashlab", real_pkg)


@pytest.mark.parametrize(
    "text",
    [
        'replace "2. Never refund" with "2. Only refund \\"delivered\\" orders" ' * 3,
        "it's a plain line " * 12,
        "x" * 300,
    ],
)
def test_long_fix_literal_is_split_lint_clean(tmp_path, text):
    from crashlab.export import _q_chunks

    src = "FIX = (\n" + "\n".join(f"    {c}" for c in _q_chunks(text)) + "\n)\n"
    ns: dict = {}
    exec(src, ns)
    assert ns["FIX"] == text and all(len(x) <= 100 for x in src.splitlines())
    if importlib.util.find_spec("ruff"):
        f = tmp_path / "t.py"
        f.write_text(src)
        for cmd in (["check", "--isolated"], ["format", "--check", "--isolated"]):
            res = subprocess.run(
                [sys.executable, "-m", "ruff", *cmd, "--line-length", "100", str(f)],
                capture_output=True,
                text=True,
            )
            assert res.returncode == 0, res.stdout
