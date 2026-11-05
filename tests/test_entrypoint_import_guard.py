"""`crashlab check` / replay import the run's entry point module. A script that starts its
agent at import time (no ``if __name__ == "__main__":``) must not record new runs or
recurse while crashlab imports it; it gets a clear error instead. Offline, subprocesses."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

HEAD = """
import crashlab

@crashlab.tool()
def search(q: str) -> str:
    return "Fee is 5 EUR."

@crashlab.check("mini")
def says_five(run):
    assert "5" in str(run.output), "answer lost the fee"

@crashlab.agent(name="mini")
def go():
    search("fee")
    return "Fee is 5 EUR."
"""

UNGUARDED = HEAD + "\ngo()\n"
UNGUARDED_REPLAY = HEAD + "\ngo()\ncrashlab.replay(crashlab.load('last').run_id)\n"
GUARDED = HEAD + '\nif __name__ == "__main__":\n    go()\n'


def _env(store):
    env = {k: v for k, v in os.environ.items() if not k.endswith("API_KEY")}
    env["CRASHLAB_DIR"] = str(store)
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


def _py(cwd, store, *args, timeout=60):
    return subprocess.run([sys.executable, *args], cwd=cwd, env=_env(store),
                          capture_output=True, text=True, timeout=timeout)  # fmt: skip


def _cli(cwd, store, *args):
    return _py(cwd, store, "-m", "crashlab", *args)


def _runs(cwd, store):
    code = "import crashlab.store as s; print(len(s.Store().list_runs()))"
    res = _py(cwd, store, "-c", code)
    assert res.returncode == 0, res.stderr
    return int(res.stdout.strip())


@pytest.fixture
def script(tmp_path):
    def make(src):
        (tmp_path / "agent_script.py").write_text(src)
        return tmp_path

    return make


@pytest.mark.parametrize("src", [UNGUARDED, UNGUARDED_REPLAY], ids=["call", "call+replay"])
def test_check_does_not_rerecord_or_recurse(script, tmp_path, src):
    cwd, store = script(src), tmp_path / "store"
    rec = _py(cwd, store, "agent_script.py")
    assert rec.returncode == 0, rec.stderr
    before = _runs(cwd, store)
    res = _cli(cwd, store, "check", "last")
    out = res.stdout + res.stderr
    assert "RecursionError" not in out and "Traceback" not in out, out
    assert "__main__" in out and "import time" in out, out
    assert _runs(cwd, store) == before  # importing for checks recorded nothing new
    # checks defined before the call still ran
    assert "says_five" in out, out


def test_replay_of_unguarded_script_gives_clear_error(script, tmp_path):
    cwd, store = script(UNGUARDED), tmp_path / "store"
    assert _py(cwd, store, "agent_script.py").returncode == 0
    before = _runs(cwd, store)
    code = "import crashlab; crashlab.replay(crashlab.load('last').run_id)"
    res = _py(cwd, store, "-c", code)
    assert res.returncode != 0
    assert "ReplayError" in res.stderr and "__main__" in res.stderr, res.stderr
    assert "RecursionError" not in res.stderr
    assert _runs(cwd, store) == before


def test_guarded_script_check_and_replay_work(script, tmp_path):
    cwd, store = script(GUARDED), tmp_path / "store"
    assert _py(cwd, store, "agent_script.py").returncode == 0
    res = _cli(cwd, store, "check", "last")
    assert res.returncode == 0 and "import time" not in res.stdout + res.stderr, res.stdout
    code = "import crashlab; print(crashlab.replay(crashlab.load('last').run_id).outcome)"
    res = _py(cwd, store, "-c", code)
    assert res.returncode == 0 and res.stdout.strip().endswith("pass"), res.stderr
    assert _runs(cwd, store) == 2  # the recording + its replay


def test_plain_import_by_user_code_still_records(script, tmp_path):
    # module-level recording is only refused while crashlab imports an entry point
    cwd, store = script(UNGUARDED), tmp_path / "store"
    res = _py(cwd, store, "-c", "import agent_script")
    assert res.returncode == 0, res.stderr
    assert _runs(cwd, store) == 1
