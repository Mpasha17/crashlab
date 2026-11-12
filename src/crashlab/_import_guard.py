"""Guard for entry point imports: while crashlab imports a run's entry point module (for
``crashlab check``, replay, fork, exported tests), a module that starts a recording at
import time must not record a new run or re-enter the import (recursion)."""

from __future__ import annotations

import contextvars
import functools
import os
import threading
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_importing: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "crashlab_importing_entrypoint", default=None
)

HINT = (
    "it starts an agent run at import time; put that code under "
    '`if __name__ == "__main__":` so crashlab can import the module without running it'
)


class RunAtImportError(RuntimeError):
    """A recording was started while crashlab was importing a run's entry point."""


# The CLI's import of an entry point also refuses runs from threads that the module starts
# (contextvars are not inherited by threads): every thread started by the importing thread
# while it imports, and every thread those start later, is tainted for its lifetime.
# Threads that existed before the import are unaffected. Only the CLI uses this.
_process_target: str | None = None
_importer: threading.Thread | None = None
_tainted: weakref.WeakKeyDictionary[threading.Thread, str] = weakref.WeakKeyDictionary()
_hook_lock = threading.Lock()
_hooked = False


_task = threading.local()  # .target: a pool task submitted by tainted code is running


def _taint_of_caller() -> str | None:
    """The entry point whose import (transitively) started the current code, if any."""
    target = _tainted.get(threading.current_thread()) or getattr(_task, "target", None)
    if target is None and _process_target is not None and threading.current_thread() is _importer:
        target = _process_target
    return target


class _MarkedCall:
    """A task submitted to a ``ProcessPoolExecutor`` by tainted code. It is pickled to the
    worker process (spawn, forkserver or fork) and marks only that task there, so a run it
    starts is refused in the worker exactly like one in a thread; later tasks from
    untainted callers in the same worker record normally."""

    __slots__ = ("fn", "target")

    def __init__(self, fn: Any, target: str) -> None:
        self.fn = fn
        self.target = target

    def __call__(self, *a: Any, **kw: Any) -> Any:
        prev = getattr(_task, "target", None)
        _task.target = self.target
        try:
            return self.fn(*a, **kw)
        finally:
            _task.target = prev


def _forget_import_in_pool_worker() -> None:
    """``os.fork`` child hook. A ``ProcessPoolExecutor`` with the fork start method forks
    its workers inside ``submit``; when tainted code made that call, the worker would
    otherwise inherit the import state for its whole life and refuse every later task.
    Such a worker starts clean; the tainted task is still refused via ``_MarkedCall``.
    Other forks (the module calling ``os.fork`` or ``multiprocessing.Process``) keep it."""
    global _process_target, _importer
    if not getattr(_task, "launching_pool", False):
        return
    _task.launching_pool = False
    _task.target = None
    _process_target, _importer = None, None
    _tainted.clear()
    _importing.set(None)


def _install_start_hook() -> None:
    """Install once, from the CLI only: ``Thread.start`` taints threads started by tainted
    code; ``ThreadPoolExecutor.submit`` marks tasks submitted by tainted code for the time
    they run (the worker itself stays clean), so executors created before the import are
    covered; ``ProcessPoolExecutor.submit`` does the same in the worker process. Untainted
    callers go straight to the original methods."""
    global _hooked
    with _hook_lock:
        if _hooked:
            return
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

        orig_start = threading.Thread.start
        orig_submit = ThreadPoolExecutor.submit
        orig_psubmit = ProcessPoolExecutor.submit

        @functools.wraps(orig_start)
        def start(self: threading.Thread) -> None:
            target = _taint_of_caller()
            if target is not None:
                _tainted[self] = target
            orig_start(self)

        @functools.wraps(orig_submit)
        def submit(self: Any, fn: Any, /, *args: Any, **kwargs: Any) -> Any:
            target = _taint_of_caller()
            if target is None:
                return orig_submit(self, fn, *args, **kwargs)

            def marked(*a: Any, **kw: Any) -> Any:
                prev = getattr(_task, "target", None)
                _task.target = target
                try:
                    return fn(*a, **kw)
                finally:
                    _task.target = prev

            return orig_submit(self, marked, *args, **kwargs)

        @functools.wraps(orig_psubmit)
        def psubmit(self: Any, fn: Any, /, *args: Any, **kwargs: Any) -> Any:
            target = _taint_of_caller()
            if target is None:
                return orig_psubmit(self, fn, *args, **kwargs)
            _task.launching_pool = True  # a worker forked in here starts clean
            try:
                return orig_psubmit(self, _MarkedCall(fn, target), *args, **kwargs)
            finally:
                _task.launching_pool = False

        threading.Thread.start = start  # type: ignore[method-assign]
        ThreadPoolExecutor.submit = submit  # type: ignore[method-assign]
        ProcessPoolExecutor.submit = psubmit  # type: ignore[method-assign]
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=_forget_import_in_pool_worker)
        _hooked = True


def importing() -> str | None:
    """The entry point this context is importing (replay/fork re-entry check)."""
    return _importing.get()


@contextmanager
def process_wide(target: str) -> Iterator[None]:
    global _process_target, _importer
    _install_start_hook()
    prev = (_process_target, _importer)
    _process_target, _importer = target, threading.current_thread()
    try:
        yield
    finally:
        _process_target, _importer = prev


@contextmanager
def importing_entrypoint(target: str) -> Iterator[None]:
    token = _importing.set(target)
    try:
        yield
    finally:
        _importing.reset(token)


def _taint_of_caller_after_import() -> str | None:
    # the importing thread itself is covered by the contextvar while it imports
    return _tainted.get(threading.current_thread()) or getattr(_task, "target", None)


def refuse_run_at_import(name: str) -> None:
    target = _importing.get() or _taint_of_caller_after_import()
    if target is not None:
        raise RunAtImportError(
            f"crashlab was importing {target!r} (the entry point of a stored run) and the "
            f"module started a new {name!r} run: {HINT}"
        )
