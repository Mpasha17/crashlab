"""Executors created *before* the CLI imports an entry point: work the module submits to
them while importing (or from its own threads later) must not record; work unrelated
pre-existing threads submit to the same executors keeps recording."""

from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import crashlab
import crashlab.store as cstore
from crashlab import _import_guard as guard
from crashlab.replay.engine import resolve_entrypoint

pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")

SHARED: dict = {}


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("CRASHLAB_DIR", str(tmp_path / "store"))
    monkeypatch.syspath_prepend(str(tmp_path))
    pool = ThreadPoolExecutor(2)
    pool.submit(lambda: None).result()  # workers exist before the import
    SHARED["pool"] = pool
    yield tmp_path
    pool.shutdown()
    SHARED.clear()
    for m in [m for m in sys.modules if m.startswith("poolmod_")]:
        sys.modules.pop(m, None)


def _count():
    return len(cstore.Store().list_runs())


def rec(name="m"):
    with crashlab.record(name) as r:
        r.output = name
    return name


def _write(tmp, name, body):
    (tmp / f"{name}.py").write_text(
        "import tests.test_import_guard_pools as T\n" + body + "\ndef go():\n    pass\n"
    )


def _import(name):
    with guard.process_wide(f"{name}:go"):
        resolve_entrypoint(f"{name}:go")


def _swallow(fut):
    try:
        return fut.result(10)
    except guard.RunAtImportError as exc:
        return exc


def test_preexisting_pool_reused_at_import_is_refused(store):
    _write(store, "poolmod_a", "R = [T._swallow(T.SHARED['pool'].submit(T.rec)) for _ in range(3)]")
    _import("poolmod_a")
    assert all(isinstance(r, guard.RunAtImportError) for r in sys.modules["poolmod_a"].R)
    assert _count() == 0


def test_unrelated_thread_uses_same_pool_during_import(store):
    go_, done, out = threading.Event(), threading.Event(), []

    def user():
        go_.wait(10)
        out.append(SHARED["pool"].submit(rec, "user").result(10))
        done.set()

    t = threading.Thread(target=user)
    t.start()
    _write(store, "poolmod_b", "T.SHARED['go'].set()\nT.SHARED['done'].wait(10)\n")
    SHARED.update(go=go_, done=done)
    _import("poolmod_b")
    t.join(10)
    assert out == ["user"] and _count() == 1


def test_pool_work_after_import_records(store):
    _write(store, "poolmod_c", "X = 1")
    _import("poolmod_c")
    assert SHARED["pool"].submit(rec, "after").result(10) == "after"
    assert _count() == 1


def test_worker_not_tainted_after_refused_task(store):
    _write(store, "poolmod_d", "R = T._swallow(T.SHARED['pool'].submit(T.rec))")
    _import("poolmod_d")
    # the same workers, now running user work, record again
    assert [SHARED["pool"].submit(rec, f"u{i}").result(10) for i in range(4)]
    assert _count() == 4


def test_nested_pool_from_preexisting_pool(store):
    inner = ThreadPoolExecutor(1)
    inner.submit(lambda: None).result()
    SHARED["inner"] = inner
    _write(store, "poolmod_e", (
        "def hop():\n    return T._swallow(T.SHARED['inner'].submit(T.rec))\n"
        "R = T.SHARED['pool'].submit(hop).result(10)\n"
    ))  # fmt: skip
    _import("poolmod_e")
    inner.shutdown()
    assert isinstance(sys.modules["poolmod_e"].R, guard.RunAtImportError)
    assert _count() == 0


def test_thread_started_from_preexisting_pool_task(store):
    _write(store, "poolmod_f", (
        "import threading\n"
        "def spawn():\n"
        "    t = threading.Thread(target=T.rec)\n    t.start()\n    return t\n"
        "TH = T.SHARED['pool'].submit(spawn).result(10)\n"
    ))  # fmt: skip
    _import("poolmod_f")
    sys.modules["poolmod_f"].TH.join(10)
    assert _count() == 0


def test_tainted_thread_submits_after_import(store):
    _write(store, "poolmod_g", (
        "import threading, time\n"
        "OUT = []\n"
        "def later():\n    time.sleep(0.3)\n"
        "    OUT.append(T._swallow(T.SHARED['pool'].submit(T.rec)))\n"
        "TH = threading.Thread(target=later)\nTH.start()\n"
    ))  # fmt: skip
    _import("poolmod_g")
    sys.modules["poolmod_g"].TH.join(10)
    assert isinstance(sys.modules["poolmod_g"].OUT[0], guard.RunAtImportError)
    assert _count() == 0


def test_asyncio_to_thread_at_import(store):
    _write(store, "poolmod_h", (
        "import asyncio\n"
        "async def main():\n"
        "    try:\n        await asyncio.to_thread(T.rec)\n"
        "    except Exception as e:\n        return e\n"
        "R = asyncio.run(main())\n"
    ))  # fmt: skip
    _import("poolmod_h")
    assert isinstance(sys.modules["poolmod_h"].R, guard.RunAtImportError)
    assert _count() == 0


def test_task_exception_clears_marker(store):
    _write(store, "poolmod_i", (
        "def boom():\n    raise ValueError('x')\n"
        "try:\n    T.SHARED['pool'].submit(boom).result(10)\nexcept ValueError:\n    pass\n"
    ))  # fmt: skip
    _import("poolmod_i")
    assert [SHARED["pool"].submit(rec, f"u{i}").result(10) for i in range(4)]
    assert _count() == 4


def test_library_use_never_patches_threading(store):
    # only the CLI's process_wide import installs hooks; library replay does not
    import subprocess
    from pathlib import Path

    (store / "poolmod_lib.py").write_text("def go():\n    pass\n")
    code = (
        "import threading, concurrent.futures as cf, crashlab\n"
        "from crashlab.replay.engine import resolve_entrypoint\n"
        "resolve_entrypoint('poolmod_lib:go')\n"
        "print(threading.Thread.start.__module__, cf.ThreadPoolExecutor.submit.__module__)\n"
    )
    src = str(Path(__file__).resolve().parents[1] / "src")
    res = subprocess.run([sys.executable, "-c", code], cwd=store, capture_output=True,
                         text=True, env={"PYTHONPATH": src, "PATH": ""}, timeout=60)  # fmt: skip
    assert res.stdout.split() == ["threading", "concurrent.futures.thread"], res.stderr


def test_installed_wrappers_are_transparent_and_idempotent(store):
    import inspect

    with guard.process_wide("x:go"):
        pass
    first = (ThreadPoolExecutor.submit, threading.Thread.start)
    with guard.process_wide("x:go"):
        pass
    assert (ThreadPoolExecutor.submit, threading.Thread.start) == first  # installed once
    for wrapped in first:
        orig = wrapped.__wrapped__  # metadata of the stdlib method is kept
        assert wrapped.__doc__ == orig.__doc__ and wrapped.__name__ == orig.__name__
        assert inspect.signature(wrapped) == inspect.signature(orig)
    pool = SHARED["pool"]
    assert pool.submit(lambda a, *, k: a + k, 1, k=2).result(10) == 3
    assert guard._process_target is None and guard._importer is None


def test_marked_task_keeps_args_and_exceptions(store):
    _write(store, "poolmod_j", (
        "def f(a, *, k):\n    if k == 'boom':\n        raise KeyError(a)\n    return a + k\n"
        "OK = T.SHARED['pool'].submit(f, 'x', k='y').result(10)\n"
        "try:\n    T.SHARED['pool'].submit(f, 'z', k='boom').result(10)\n"
        "except KeyError as e:\n    ERR = e\n"
    ))  # fmt: skip
    _import("poolmod_j")
    mod = sys.modules["poolmod_j"]
    assert mod.OK == "xy" and mod.ERR.args == ("z",)
    assert getattr(guard._task, "target", None) is None
