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


def importing() -> str | None:
    return _importing.get()


@contextmanager
def importing_entrypoint(target: str) -> Iterator[None]:
    token = _importing.set(target)
    try:
        yield
    finally:
        _importing.reset(token)


def refuse_run_at_import(name: str) -> None:
    target = _importing.get()
    if target is not None:
        raise RunAtImportError(
            f"crashlab was importing {target!r} (the entry point of a stored run) and the "
            f"module started a new {name!r} run: {HINT}"
        )
