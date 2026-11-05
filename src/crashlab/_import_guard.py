"""Guard for entry point imports: while crashlab imports a run's entry point module (for
``crashlab check``, replay, fork, exported tests), a module that starts a recording at
import time must not record a new run or re-enter the import (recursion)."""

from __future__ import annotations

import contextvars
import threading
import weakref
from collections.abc import Iterator
from contextlib import contextmanager

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


def _install_start_hook() -> None:
    global _hooked
    with _hook_lock:
        if _hooked:
            return
        orig = threading.Thread.start

        def start(self: threading.Thread) -> None:
            parent = threading.current_thread()
            target = _tainted.get(parent)
            if target is None and _process_target is not None and parent is _importer:
                target = _process_target
            if target is not None:
                _tainted[self] = target
            orig(self)

        threading.Thread.start = start  # type: ignore[method-assign]
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


def refuse_run_at_import(name: str) -> None:
    target = _importing.get() or _tainted.get(threading.current_thread())
    if target is not None:
        raise RunAtImportError(
            f"crashlab was importing {target!r} (the entry point of a stored run) and the "
            f"module started a new {name!r} run: {HINT}"
        )
