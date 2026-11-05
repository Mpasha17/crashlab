"""Regression tests for the entry point import guard (``crashlab._import_guard``): every
CLI command that may import a stored run's entry point, async and generator entry points,
packages and installed (non-cwd) modules. Importing never records a run, never recurses,
and the guard is reset after any exception. Offline, subprocesses."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import crashlab
from crashlab import _import_guard as guard
from crashlab.replay.divergence import ReplayError
from crashlab.replay.engine import resolve_entrypoint

ROOT = Path(__file__).resolve().parents[1]

BODY = {
    "sync": """
@crashlab.agent(name="mini")
def go():
    search("fee")
    return "{answer}"

go()
""",
    "async": """
import asyncio

@crashlab.agent(name="mini")
async def go():
    search("fee")
    return "{answer}"

asyncio.run(go())
""",
    "gen": """
@crashlab.agent(name="mini")
def go():
    search("fee")
    yield "{answer}"

list(go())
""",
    "agen": """
import asyncio

@crashlab.agent(name="mini")
async def go():
    search("fee")
    yield "{answer}"

async def _drain():
    return [x async for x in go()]

asyncio.run(_drain())
""",
}

HEAD = """
import crashlab

@crashlab.tool()
def search(q: str) -> str:
    return "Fee is 5 EUR."

@crashlab.check("mini")
def says_five(run):
    assert "5" in str(run.output), "answer lost the fee"
"""


def src(kind="sync", answer="Fee is 5 EUR."):
    return HEAD + BODY[kind].format(answer=answer)


def _env(store, extra_path=()):
    env = {k: v for k, v in os.environ.items() if not k.endswith("API_KEY")}
    env.pop("CRASHLAB_DIAGNOSER_MODEL", None)
    env["CRASHLAB_DIR"] = str(store)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), *map(str, extra_path)])
    return env


def _py(cwd, store, *args, extra_path=(), timeout=90):
    return subprocess.run([sys.executable, *args], cwd=cwd, env=_env(store, extra_path),
                          capture_output=True, text=True, timeout=timeout)  # fmt: skip


def _cli(cwd, store, *args, extra_path=()):
    return _py(cwd, store, "-m", "crashlab", *args, extra_path=extra_path)


def _runs(cwd, store):
    res = _py(cwd, store, "-c", "import crashlab.store as s; print(len(s.Store().list_runs()))")
    assert res.returncode == 0, res.stderr
    return int(res.stdout.strip())


def _clean(res):
    out = res.stdout + res.stderr
    assert "RecursionError" not in out, out
    assert "maximum recursion" not in out, out
    return out


def _record_script(tmp_path, kind="sync", answer="Fee is 5 EUR."):
    (tmp_path / "agent_script.py").write_text(src(kind, answer))
    store = tmp_path / "store"
    rec = _py(tmp_path, store, "agent_script.py")
    assert rec.returncode == 0, rec.stderr
    assert _runs(tmp_path, store) == 1
    return tmp_path, store


# -- every command on an unguarded script ------------------------------------------------

COMMANDS = {
    "show": ["show", "last"],
    "check": ["check", "last"],
    "check-json": ["check", "last", "--json"],
    "diagnose": ["diagnose", "last", "--heuristic"],
    "fix": ["fix", "last"],
    "replay": ["replay", "last"],
    "export": ["export", "last", "-o", "out.jsonl"],
    "export-test": ["export-test", "last", "-o", "exported"],
}


@pytest.mark.parametrize("cmd", list(COMMANDS), ids=list(COMMANDS))
def test_cli_commands_never_record_or_recurse(tmp_path, cmd):
    cwd, store = _record_script(tmp_path)
    res = _cli(cwd, store, *COMMANDS[cmd])
    out = _clean(res)
    assert _runs(cwd, store) == 1, out  # importing the entry point recorded nothing
    if cmd == "check":
        assert "import time" in out and "says_five" in out, out
    if cmd == "check-json":
        assert json.loads(res.stdout)["checks"][0]["name"] == "says_five", out
    if cmd == "replay":
        assert res.returncode != 0 and "import time" in out, out
        assert "Traceback" not in out, out


def test_exported_test_of_unguarded_script_does_not_record(tmp_path):
    cwd, store = _record_script(tmp_path)
    assert _cli(cwd, store, "export-test", "last", "-o", "exported").returncode == 0
    res = _py(cwd, store, "-m", "pytest", "-q", "-p", "no:cacheprovider", "exported")
    _clean(res)
    assert _runs(cwd, store) == 1, res.stdout + res.stderr


# -- async / generator entry points ---------------------------------------------------------


@pytest.mark.parametrize("kind", ["sync", "async", "gen", "agen"])
def test_entrypoint_kinds_check_and_replay(tmp_path, kind):
    cwd, store = _record_script(tmp_path, kind)
    res = _cli(cwd, store, "check", "last")
    out = _clean(res)
    assert "import time" in out and "says_five" in out, out
    assert _runs(cwd, store) == 1
    code = "import crashlab; crashlab.replay(crashlab.load('last').run_id)"
    res = _py(cwd, store, "-c", code)
    out = _clean(res)
    assert res.returncode != 0 and "ReplayError" in out and "import time" in out, out
    assert _runs(cwd, store) == 1


def test_failed_check_creates_no_recording(tmp_path):
    cwd, store = _record_script(tmp_path, answer="no idea")
    res = _cli(cwd, store, "check", "last")
    out = _clean(res)
    assert res.returncode != 0, out  # the check fails
    assert "answer lost the fee" in out, out
    assert _runs(cwd, store) == 1


# -- packages and installed modules ------------------------------------------------------


def _make_pkg(root: Path, kind="sync"):
    pkg = root / "mypkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "sub" / "__init__.py").write_text("from . import agent  # noqa: F401\n")
    (pkg / "sub" / "agent.py").write_text(src(kind))


@pytest.mark.parametrize("installed", [False, True], ids=["cwd-package", "installed"])
def test_package_and_installed_module(tmp_path, installed):
    site = tmp_path / ("site" if installed else "proj")
    proj = tmp_path / "proj"
    proj.mkdir()
    _make_pkg(site)
    extra = (site,) if installed else ()
    store = tmp_path / "store"
    rec = _py(proj, store, "-c", "import mypkg.sub.agent", extra_path=extra)
    assert rec.returncode == 0, rec.stderr
    assert _runs(proj, store) == 1
    meta = _py(proj, store, "-c", "import crashlab; print(crashlab.load('last').entrypoint)")
    assert meta.stdout.strip() == "mypkg.sub.agent:go", meta.stdout + meta.stderr
    other = tmp_path / "elsewhere"
    other.mkdir()
    cwd = other if installed else proj  # installed modules resolve from any directory
    res = _cli(cwd, store, "check", "last", extra_path=extra)
    out = _clean(res)
    assert "import time" in out and "says_five" in out, out
    assert _runs(proj, store) == 1


# -- guard state resets ------------------------------------------------------------------


def test_guard_resets_after_exception_in_context():
    with pytest.raises(ValueError), guard.importing_entrypoint("x"):
        assert guard.importing() == "x"
        raise ValueError
    assert guard.importing() is None
    guard.refuse_run_at_import("ok")  # no error outside


def test_guard_nested_restores_outer():
    with guard.importing_entrypoint("outer"):
        with pytest.raises(KeyError), guard.importing_entrypoint("inner"):
            raise KeyError
        assert guard.importing() == "outer"
    assert guard.importing() is None


@pytest.mark.parametrize(
    "body,exc",
    [
        ("raise ValueError('boom at import')", ReplayError),
        ("import crashlab\nwith crashlab.record('m'):\n    pass", ReplayError),
        ("import crashlab\ncrashlab.replay('last')", ReplayError),
        ("raise SystemExit(3)", SystemExit),
        ("raise KeyboardInterrupt", KeyboardInterrupt),
    ],
    ids=["error", "record", "replay", "systemexit", "kbint"],
)
def test_guard_resets_after_failed_import(tmp_path, monkeypatch, body, exc):
    name = f"guardmod_{abs(hash(body)) % 10**8}"
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(body) + "\ndef go():\n    return 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("CRASHLAB_DIR", str(tmp_path / "store"))
    with crashlab.record("before") as run:  # so replay('last') has a run to load
        run.output = "x"
    with pytest.raises(exc):
        resolve_entrypoint(f"{name}:go")
    sys.modules.pop(name, None)
    assert guard.importing() is None
    with crashlab.record("after") as run:  # recording works again in this process
        run.output = "ok"
    assert run.output == "ok"


def test_guard_resets_after_script_file_failure(tmp_path, monkeypatch):
    p = tmp_path / "bad_script.py"
    p.write_text("import crashlab\nwith crashlab.record('m'):\n    pass\ndef go():\n    pass\n")
    monkeypatch.setenv("CRASHLAB_DIR", str(tmp_path / "store"))
    with pytest.raises(ReplayError, match="import time"):
        resolve_entrypoint(f"{p}:go")
    assert guard.importing() is None


def test_module_raising_at_import_gives_clear_error(tmp_path):
    # a stored run whose module now fails to import: check/replay report it, no traceback
    script = tmp_path / "agent_script.py"
    script.write_text(src().replace("go()\n", "if __name__ == '__main__':\n    go()\n"))
    store = tmp_path / "store"
    assert _py(tmp_path, store, "agent_script.py").returncode == 0
    script.write_text(script.read_text() + "\nraise ValueError('boom at import')\n")
    res = _cli(tmp_path, store, "check", "last")
    out = _clean(res)
    assert "Traceback" not in out and "boom at import" in out, out
    assert "could not import" in out, out
    res = _cli(tmp_path, store, "replay", "last")
    out = _clean(res)
    assert res.returncode != 0 and "Traceback" not in out and "boom at import" in out, out
    assert _runs(tmp_path, store) == 1
    assert guard.importing() is None
