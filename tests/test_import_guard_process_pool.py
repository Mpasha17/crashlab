"""``ProcessPoolExecutor`` work submitted by an entry point while the CLI imports it (or
later, by threads it started) must not record a run in the worker process, with any
start method. Work submitted by untainted code, before, during or after the import,
keeps recording, including in workers that were forked by a tainted ``submit``.

The last tests pin what is still *not* covered, so the docs cannot drift from the code:
``multiprocessing.Process`` with spawn/forkserver and a custom queue feeding threads
that already existed."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import sys
import threading
from concurrent.futures import ProcessPoolExecutor

import pytest

import crashlab
import crashlab.store as cstore
from crashlab import _import_guard as guard
from crashlab.replay.engine import resolve_entrypoint

METHODS = [m for m in ("fork", "spawn", "forkserver") if m in mp.get_all_start_methods()]
pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX start methods"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),  # fork with threads (3.12+)
]

SHARED: dict = {}


def rec(name: str = "child", store_dir: str | None = None) -> str:
    # a forkserver keeps the environment it started with, so tests that count runs
    # pass the store explicitly
    if store_dir is not None:
        os.environ["CRASHLAB_DIR"] = store_dir
    with crashlab.record(name) as r:
        r.output = name
    return name


def plain(x: int) -> int:
    return x * 2


def outcome(fut):
    try:
        return fut.result(60)
    except guard.RunAtImportError:
        return "refused"


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("CRASHLAB_DIR", str(tmp_path / "store"))
    monkeypatch.syspath_prepend(str(tmp_path))
    yield tmp_path
    for p in SHARED.pop("pools", []):
        p.shutdown(cancel_futures=True)
    SHARED.clear()
    for m in [m for m in sys.modules if m.startswith("ppmod_")]:
        sys.modules.pop(m, None)


def _count() -> int:
    return len(cstore.Store().list_runs())


def _write(tmp, name, body):
    (tmp / f"{name}.py").write_text(
        "import tests.test_import_guard_process_pool as T\n" + body + "\ndef go():\n    pass\n"
    )


def _import(name):
    with guard.process_wide(f"{name}:go"):
        resolve_entrypoint(f"{name}:go")
    return sys.modules[name]


def _pool(method: str, workers: int = 1) -> ProcessPoolExecutor:
    p = ProcessPoolExecutor(workers, mp_context=mp.get_context(method))
    SHARED.setdefault("pools", []).append(p)
    return p


@pytest.mark.parametrize("method", METHODS)
def test_pool_created_at_import_refuses_runs_in_the_worker(store, method):
    _write(store, f"ppmod_a_{method}", (
        "import multiprocessing as mp\nfrom concurrent.futures import ProcessPoolExecutor\n"
        f"with ProcessPoolExecutor(1, mp_context=mp.get_context({method!r})) as p:\n"
        "    R = [T.outcome(p.submit(T.rec)), T.outcome(p.submit(T.plain, 21)),\n"
        "         list(p.map(T.plain, [1, 2]))]\n"
    ))  # fmt: skip
    mod = _import(f"ppmod_a_{method}")
    assert mod.R == ["refused", 42, [2, 4]]
    assert _count() == 0


@pytest.mark.parametrize("method", METHODS)
def test_map_at_import_is_refused(store, method):
    SHARED["pool"] = _pool(method)
    _write(store, f"ppmod_b_{method}", (
        "try:\n    R = list(T.SHARED['pool'].map(T.rec, ['x', 'y']))\n"
        "except T.guard.RunAtImportError:\n    R = 'refused'\n"
    ))  # fmt: skip
    assert _import(f"ppmod_b_{method}").R == "refused"
    assert _count() == 0


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("warm", [True, False], ids=["warm", "cold"])
def test_preexisting_pool_records_again_after_the_import(store, method, warm):
    """A cold fork pool forks its workers inside the tainted submit; those workers must
    not stay tainted for the rest of their life."""
    pool = SHARED["pool"] = _pool(method, 2)
    if warm:
        assert pool.submit(plain, 1).result(60) == 2
    _write(store, f"ppmod_c_{method}_{warm}", "R = T.outcome(T.SHARED['pool'].submit(T.rec))")
    assert _import(f"ppmod_c_{method}_{warm}").R == "refused"
    assert _count() == 0
    d = os.environ["CRASHLAB_DIR"]
    assert sorted(f.result(60) for f in [pool.submit(rec, f"u{i}", d) for i in range(4)]) == [
        "u0", "u1", "u2", "u3"
    ]  # fmt: skip
    assert _count() == 4


@pytest.mark.parametrize("method", METHODS)
def test_unrelated_thread_submits_during_the_import_and_records(store, method):
    pool = SHARED["pool"] = _pool(method)
    d = os.environ["CRASHLAB_DIR"]
    go_, done, out = threading.Event(), threading.Event(), []

    def user():
        go_.wait(30)
        out.append(pool.submit(rec, "user", d).result(60))
        done.set()

    t = threading.Thread(target=user)
    t.start()
    SHARED.update(go=go_, done=done)
    _write(store, f"ppmod_d_{method}", "T.SHARED['go'].set()\nT.SHARED['done'].wait(60)\n")
    _import(f"ppmod_d_{method}")
    t.join(60)
    assert out == ["user"] and _count() == 1


def test_thread_started_at_import_submits_later_and_is_refused(store):
    SHARED["pool"] = _pool(METHODS[-1])
    _write(store, "ppmod_e", (
        "import threading, time\nOUT = []\n"
        "def later():\n    time.sleep(0.3)\n"
        "    OUT.append(T.outcome(T.SHARED['pool'].submit(T.rec)))\n"
        "TH = threading.Thread(target=later)\nTH.start()\n"
    ))  # fmt: skip
    mod = _import("ppmod_e")
    mod.TH.join(60)
    assert mod.OUT == ["refused"] and _count() == 0


def test_unpicklable_task_fails_the_same_way_as_without_the_guard(store):
    plain_err = None
    try:
        _pool(METHODS[-1]).submit(lambda: 1).result(60)
    except Exception as e:  # noqa: BLE001
        plain_err = type(e)
    SHARED["pool"] = _pool(METHODS[-1])
    # a local function cannot be pickled; the error must reach the caller unchanged
    _write(store, "ppmod_f", (
        "def make():\n    return lambda: 1\n"
        "try:\n    T.SHARED['pool'].submit(make()).result(60)\n    R = None\n"
        "except Exception as e:\n    R = type(e)\n"
    ))  # fmt: skip
    assert plain_err is not None and _import("ppmod_f").R is plain_err


def test_marked_call_restores_the_marker():
    call = guard._MarkedCall(plain, "m:go")
    assert call(4) == 8 and getattr(guard._task, "target", None) is None
    with pytest.raises(guard.RunAtImportError):
        guard._MarkedCall(rec, "m:go")()
    assert getattr(guard._task, "target", None) is None
    assert rec("after") == "after"


# --- documented limits (see docs/rag-testing.md); update the docs if these change ---


@pytest.mark.parametrize("method", [m for m in METHODS if m != "fork"])
def test_limit_multiprocessing_process_spawn_is_not_covered(store, method):
    _write(store, f"ppmod_g_{method}", (
        "import multiprocessing as mp\n"
        "D = T.os.environ['CRASHLAB_DIR']\n"
        f"P = mp.get_context({method!r}).Process(target=T.rec, args=('proc', D))\n"
        "P.start()\nP.join(60)\n"
    ))  # fmt: skip
    assert _import(f"ppmod_g_{method}").P.exitcode == 0
    assert _count() == 1  # LIMIT: a Process started at import still records


def test_limit_custom_queue_into_existing_thread_is_not_covered(store):
    q: queue.Queue = queue.Queue()
    out: list = []

    def worker():
        fn = q.get(timeout=30)
        out.append(fn())

    t = threading.Thread(target=worker)
    t.start()  # exists before the import
    SHARED["q"] = q
    _write(store, "ppmod_h", "T.SHARED['q'].put(T.rec)")
    _import("ppmod_h")
    t.join(60)
    assert out == ["child"] and _count() == 1  # LIMIT: recorded


@pytest.mark.skipif("fork" not in METHODS, reason="needs fork")
def test_process_forked_at_import_inherits_the_refusal(store):
    """A plain fork copies the importing thread's state, so a run started there is
    refused; only pool workers forked inside a tainted ``submit`` start clean."""
    _write(store, "ppmod_i", (
        "import multiprocessing as mp\n"
        "def child():\n"
        "    try:\n        T.rec('forked')\n    except T.guard.RunAtImportError:\n"
        "        raise SystemExit(3)\n"
        "P = mp.get_context('fork').Process(target=child)\nP.start()\nP.join(60)\n"
    ))  # fmt: skip
    assert _import("ppmod_i").P.exitcode == 3
    assert _count() == 0


def test_run_in_executor_with_a_process_pool_at_import_is_refused(store):
    SHARED["pool"] = _pool(METHODS[-1])
    _write(store, "ppmod_j", (
        "import asyncio\n"
        "async def call():\n"
        "    loop = asyncio.get_running_loop()\n"
        "    try:\n        return await loop.run_in_executor(T.SHARED['pool'], T.rec)\n"
        "    except T.guard.RunAtImportError:\n        return 'refused'\n"
        "R = asyncio.run(call())\n"
    ))  # fmt: skip
    assert _import("ppmod_j").R == "refused"
    assert _count() == 0
