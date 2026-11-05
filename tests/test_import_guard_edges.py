"""Entry point import edge cases: secret-bearing import errors are redacted (type and the
rest of the message kept), SystemExit / KeyboardInterrupt at import through the CLI,
symlinked / path / Windows-style entry points, other import-time side effects, and
export-test output that does not import the agent. Offline, subprocesses."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from crashlab import _import_guard as guard
from crashlab.replay.divergence import ReplayError
from crashlab.replay.engine import resolve_entrypoint

ROOT = Path(__file__).resolve().parents[1]
# built at runtime so no key-shaped literal sits in the repo
FAKE = "sk-" + "proj-" + "T3stOnlyNotAKey" * 3

AGENT = """
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

if __name__ == "__main__":
    go()
"""


def _env(store):
    env = {k: v for k, v in os.environ.items() if not k.endswith("API_KEY")}
    env["CRASHLAB_DIR"] = str(store)
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


def _py(cwd, store, *args, timeout=90):
    return subprocess.run([sys.executable, *args], cwd=cwd, env=_env(store),
                          capture_output=True, text=True, timeout=timeout)  # fmt: skip


def _cli(cwd, store, *args):
    return _py(cwd, store, "-m", "crashlab", *args)


def _runs(cwd, store):
    res = _py(cwd, store, "-c", "import crashlab.store as s; print(len(s.Store().list_runs()))")
    assert res.returncode == 0, res.stderr
    return int(res.stdout.strip())


def _recorded(tmp_path, tail="", name="agent_script.py"):
    """Record a guarded agent, then append ``tail`` (runs at the next import)."""
    script = tmp_path / name
    script.write_text(AGENT)
    store = tmp_path / "store"
    rec = _py(tmp_path, store, name)
    assert rec.returncode == 0, rec.stderr
    if tail:
        script.write_text(AGENT + "\n" + tail + "\n")
    return script, store


def _out(res):
    out = res.stdout + res.stderr
    assert "RecursionError" not in out, out
    return out


# -- secrets in import errors --------------------------------------------------------------

SECRET_TAILS = {
    "value": f"raise ValueError('bad key {FAKE} for tenant acme')",
    "env-style": f"raise RuntimeError('OPENAI_API_KEY={FAKE} rejected by tenant acme')",
    "url": f"raise ConnectionError('https://u:{FAKE}@db.example/x tenant acme')",
}


@pytest.mark.parametrize("kind", list(SECRET_TAILS))
def test_import_error_secret_redacted_in_replay_error(tmp_path, monkeypatch, kind):
    mod = f"secretmod_{kind.replace('-', '_')}"
    (tmp_path / f"{mod}.py").write_text(SECRET_TAILS[kind] + "\ndef go():\n    pass\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ReplayError) as info:
        resolve_entrypoint(f"{mod}:go")
    sys.modules.pop(mod, None)
    msg = str(info.value)
    assert FAKE not in msg, "secret leaked into ReplayError"
    assert "tenant acme" in msg and type(info.value.__cause__).__name__ in msg, msg
    assert guard.importing() is None


@pytest.mark.parametrize("cmd", ["check", "replay", "diagnose", "fix"])
def test_import_error_secret_redacted_in_cli(tmp_path, cmd):
    script, store = _recorded(tmp_path, SECRET_TAILS["value"])
    args = [cmd, "last"] + (["--heuristic"] if cmd == "diagnose" else [])
    res = _cli(tmp_path, store, *args)
    out = _out(res)
    assert FAKE not in out, f"secret leaked by `crashlab {cmd}`"
    if cmd in ("check", "replay"):
        assert "ValueError" in out and "tenant acme" in out, out
    assert _runs(tmp_path, store) == 1


# -- SystemExit / KeyboardInterrupt --------------------------------------------------------


@pytest.mark.parametrize("tail", ["import sys\nsys.exit(3)", "raise SystemExit('usage: x')"])
def test_system_exit_at_import_is_a_clear_error(tmp_path, tail):
    script, store = _recorded(tmp_path, tail)
    res = _cli(tmp_path, store, "check", "last")
    out = _out(res)
    assert "SystemExit" in out and "could not import" in out, out
    assert "says_five" in out and "Traceback" not in out, out  # check still ran
    res = _cli(tmp_path, store, "replay", "last")
    out = _out(res)
    assert res.returncode != 0 and "SystemExit" in out and "Traceback" not in out, out
    assert _runs(tmp_path, store) == 1


def test_system_exit_in_process_resets_guard(tmp_path, monkeypatch):
    (tmp_path / "exitmod.py").write_text("raise SystemExit(4)\ndef go():\n    pass\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ReplayError, match="SystemExit"):
        resolve_entrypoint("exitmod:go")
    sys.modules.pop("exitmod", None)
    assert guard.importing() is None


def test_keyboard_interrupt_at_import_aborts(tmp_path):
    script, store = _recorded(tmp_path, "raise KeyboardInterrupt")
    for cmd in ("check", "replay"):
        res = _cli(tmp_path, store, cmd, "last")
        out = _out(res)
        assert res.returncode != 0, out  # an interrupt is never swallowed as success
        assert "says_five" not in out or cmd == "check", out
    assert _runs(tmp_path, store) == 1


# -- symlinks, path and Windows-style entry points -----------------------------------------


def test_symlinked_script_entrypoint(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "agent_impl.py").write_text(AGENT.replace('if __name__ == "__main__":\n    ', ""))
    link = tmp_path / "agent_link.py"
    try:
        link.symlink_to(real / "agent_impl.py")
    except OSError:
        pytest.skip("symlinks not supported")
    store = tmp_path / "store"
    assert _py(tmp_path, store, "agent_link.py").returncode == 0
    assert _runs(tmp_path, store) == 1
    res = _cli(tmp_path, store, "check", "last")
    out = _out(res)
    assert "import time" in out and "says_five" in out, out
    assert _runs(tmp_path, store) == 1


def test_path_with_spaces_and_subdir(tmp_path):
    sub = tmp_path / "my agents"
    sub.mkdir()
    (sub / "bot.py").write_text(AGENT)
    store = tmp_path / "store"
    assert _py(tmp_path, store, str(Path("my agents") / "bot.py")).returncode == 0
    res = _cli(tmp_path, store, "check", "last")
    assert res.returncode == 0, _out(res)
    code = "import crashlab; print(crashlab.replay(crashlab.load('last').run_id).outcome)"
    res = _py(tmp_path, store, "-c", code)
    assert res.returncode == 0 and res.stdout.strip().endswith("pass"), res.stderr


@pytest.mark.skipif(os.name == "nt", reason="POSIX: a Windows path names no file here")
@pytest.mark.parametrize(
    "ep", [r"C:\agents\bot.py:go", r"C:/agents/bot.py:go", r"\\srv\share\bot.py:go"]
)
def test_windows_style_path_gives_replay_error(ep):
    with pytest.raises(ReplayError):
        resolve_entrypoint(ep)
    assert guard.importing() is None


# -- other import-time side effects --------------------------------------------------------


def test_tool_call_at_import_records_nothing(tmp_path):
    script, store = _recorded(tmp_path, 'search("at import")')
    res = _cli(tmp_path, store, "check", "last")
    assert res.returncode == 0, _out(res)
    assert _runs(tmp_path, store) == 1


def test_fork_at_import_is_refused(tmp_path):
    tail = "crashlab.fork(crashlab.load('last').run_id, at_step=0)"
    script, store = _recorded(tmp_path, tail)
    res = _cli(tmp_path, store, "check", "last")
    out = _out(res)
    assert "import time" in out and "Traceback" not in out, out
    assert _runs(tmp_path, store) == 1


def test_thread_started_at_import_cannot_record(tmp_path):
    tail = "import threading\nt = threading.Thread(target=go)\nt.start()\nt.join()"
    script, store = _recorded(tmp_path, tail)
    out = _out(_cli(tmp_path, store, "check", "last"))
    # threads do not inherit the guard's contextvar; the CLI guards the whole process
    assert _runs(tmp_path, store) == 1, out
    assert "import time" in out, out


# -- export-test --------------------------------------------------------------------------


def test_exported_test_does_not_import_agent_at_collection(tmp_path):
    marker = tmp_path / "IMPORTED"
    script, store = _recorded(tmp_path)
    script.write_text(f"open({str(marker)!r}, 'w').close()\n" + AGENT + "\ngo()\n")
    marker.unlink(missing_ok=True)
    assert _cli(tmp_path, store, "export-test", "last", "-o", "exported").returncode == 0
    gen = "\n".join(p.read_text() for p in (tmp_path / "exported").rglob("*.py"))
    assert "agent_script" not in gen.split("CASSETTES")[0], gen  # no import of the agent
    res = _py(tmp_path, store, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
              "exported")  # fmt: skip
    assert res.returncode == 0, _out(res)
    assert not marker.exists(), "collecting the exported test imported the agent"


def test_process_wide_guard_resets_after_exception():
    import threading

    errs = []

    def child():
        try:
            guard.refuse_run_at_import("m")
        except guard.RunAtImportError as exc:
            errs.append(exc)

    with pytest.raises(ValueError), guard.process_wide("x"):
        t = threading.Thread(target=child)
        t.start()
        t.join()
        raise ValueError
    assert len(errs) == 1  # a thread started inside the window is refused
    guard.refuse_run_at_import("m")  # the importing thread itself is free again
    t = threading.Thread(target=child)
    t.start()
    t.join()
    assert len(errs) == 1 and guard.importing() is None  # new threads are not tainted
