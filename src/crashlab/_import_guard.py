"""Guard for entry point imports: while crashlab imports a run's entry point module (for
``crashlab check``, replay, fork, exported tests), a module that starts a recording at
import time must not record a new run or re-enter the import (recursion)."""

from __future__ import annotations

import contextvars
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


# set by the CLI around its own entry point imports: there, no user threads exist yet, so a
# thread the module starts at import time (which does not inherit the contextvar) is refused
# too. Library callers keep the per-context guard so their other threads keep recording.
_process_target: str | None = None


def importing() -> str | None:
    """The entry point this context is importing (replay/fork re-entry check)."""
    return _importing.get()


@contextmanager
def process_wide(target: str) -> Iterator[None]:
    global _process_target
    prev, _process_target = _process_target, target
    try:
        yield
    finally:
        _process_target = prev


@contextmanager
def importing_entrypoint(target: str) -> Iterator[None]:
    token = _importing.set(target)
    try:
        yield
    finally:
        _importing.reset(token)


def refuse_run_at_import(name: str) -> None:
    target = _importing.get() or _process_target
    if target is not None:
        raise RunAtImportError(
            f"crashlab was importing {target!r} (the entry point of a stored run) and the "
            f"module started a new {name!r} run: {HINT}"
        )
