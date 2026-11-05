"""The CLI's thread-aware import guard (``_import_guard.process_wide``): threads the entry
point module starts while being imported (directly, via pools, nested imports; also after
the import returns) never record, while threads that existed before keep recording."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import crashlab
import crashlab.store as cstore
from crashlab import _import_guard as guard
from crashlab.replay.engine import resolve_entrypoint

# refused threads raise RunAtImportError in the thread, by design
pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("CRASHLAB_DIR", str(tmp_path / "store"))
    monkeypatch.syspath_prepend(str(tmp_path))
    yield tmp_path
    for m in [m for m in sys.modules if m.startswith("thrmod_")]:
        sys.modules.pop(m, None)


def _count():
    return len(cstore.Store().list_runs())


def _record(name):
    with crashlab.record(name) as run:
        run.output = name


def _write(tmp, name, body):
    (tmp / f"{name}.py").write_text(
        "import crashlab, threading, time\n"
        "def rec(n='m'):\n"
        "    with crashlab.record(n) as r:\n"
        "        r.output = 1\n" + body + "\ndef go():\n    pass\n"
    )


def _import(name):
    with guard.process_wide(f"{name}:go"):
        resolve_entrypoint(f"{name}:go")


def test_preexisting_thread_keeps_recording_during_import(store):
    started, release, errors = threading.Event(), threading.Event(), []

    def user_thread():
        started.set()
        release.wait(10)
        try:
            _record("user")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=user_thread)
    t.start()
    started.wait(10)
    _write(store, "thrmod_pre", "import __main__\n_ev = __main__  # placeholder\n")
    with guard.process_wide("thrmod_pre:go"):
        release.set()
        t.join(10)
        resolve_entrypoint("thrmod_pre:go")
    assert not errors, errors
    assert _count() == 1


def test_thread_started_at_import_and_recording_later_is_refused(store):
    _write(
        store,
        "thrmod_late",
        "T = threading.Thread(target=lambda: (time.sleep(0.3), rec()))\nT.start()\n",
    )
    _import("thrmod_late")
    sys.modules["thrmod_late"].T.join(10)
    assert _count() == 0


def test_thread_pool_created_at_import_is_refused(store):
    _write(store, "thrmod_pool", (
        "from concurrent.futures import ThreadPoolExecutor\n"
        "P = ThreadPoolExecutor(2)\n"
        "F = [P.submit(rec) for _ in range(3)]\n"
        "for f in F:\n"
        "    try:\n        f.result()\n    except Exception:\n        pass\n"
    ))  # fmt: skip
    _import("thrmod_pool")
    # the pool's workers stay tainted after the import too
    late = sys.modules["thrmod_pool"].P.submit(sys.modules["thrmod_pool"].rec)
    with pytest.raises(guard.RunAtImportError):
        late.result(10)
    sys.modules["thrmod_pool"].P.shutdown()
    assert _count() == 0


def test_nested_import_thread_is_refused(store):
    _write(store, "thrmod_inner", "T = threading.Thread(target=rec)\nT.start()\nT.join()\n")
    _write(store, "thrmod_outer", "import thrmod_inner\n")
    _import("thrmod_outer")
    assert _count() == 0


def test_grandchild_thread_is_refused(store):
    _write(store, "thrmod_gc", (
        "def child():\n"
        "    time.sleep(0.2)\n"
        "    g = threading.Thread(target=rec)\n    g.start()\n    g.join()\n"
        "T = threading.Thread(target=child)\nT.start()\n"
    ))  # fmt: skip
    _import("thrmod_gc")
    sys.modules["thrmod_gc"].T.join(10)
    assert _count() == 0


def test_async_task_at_import_is_refused(store):
    _write(store, "thrmod_async", (
        "import asyncio\n"
        "async def a():\n    rec()\n"
        "try:\n    asyncio.run(a())\nexcept Exception as e:\n    ERR = e\n"
    ))  # fmt: skip
    _import("thrmod_async")
    assert isinstance(sys.modules["thrmod_async"].ERR, guard.RunAtImportError)
    assert _count() == 0


def test_main_thread_records_normally_after_import_and_after_failure(store):
    _write(store, "thrmod_boom", "raise ValueError('x')\n")
    with pytest.raises(Exception):  # noqa: B017
        _import("thrmod_boom")
    assert guard.importing() is None
    _record("after")
    t = threading.Thread(target=_record, args=("after-thread",))
    t.start()
    t.join()
    assert _count() == 2


def test_cli_check_with_lingering_nondaemon_thread_saves_nothing(tmp_path):
    agent = (
        "import crashlab, threading, time\n"
        "@crashlab.agent(name='mini')\ndef go():\n    return 'ok'\n"
        "if __name__ == '__main__':\n    go()\n"
    )
    (tmp_path / "a.py").write_text(agent)
    env = {k: v for k, v in os.environ.items() if not k.endswith("API_KEY")}
    env.update(CRASHLAB_DIR=str(tmp_path / "st"), PYTHONPATH=str(ROOT / "src"))
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=tmp_path, env=env,  # noqa: E731
                                    capture_output=True, text=True, timeout=60)  # fmt: skip
    assert run("a.py").returncode == 0
    (tmp_path / "a.py").write_text(
        agent + "threading.Thread(target=lambda: (time.sleep(0.5), go())).start()\n"
    )
    res = run("-m", "crashlab", "check", "last")
    assert "RecursionError" not in res.stdout + res.stderr
    n = run("-c", "import crashlab.store as s; print(len(s.Store().list_runs()))")
    assert n.stdout.strip() == "1", res.stdout + res.stderr
