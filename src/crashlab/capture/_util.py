from __future__ import annotations

import inspect
import os
import sys
from collections.abc import Callable
from typing import Any

from .normalize import to_jsonable


def bind_arguments(
    fn: Callable[..., Any], args: tuple, kwargs: dict, ignore: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Map a call's positional/keyword args to parameter names (JSON-able)."""
    try:
        sig = inspect.signature(fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
    except (TypeError, ValueError):
        return {"args": to_jsonable(list(args)), "kwargs": to_jsonable(kwargs)}
    out: dict[str, Any] = {}
    for name, value in bound.arguments.items():
        if name in ("self", "cls") or name in ignore:
            continue
        param = sig.parameters[name]
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            out.update({k: to_jsonable(v) for k, v in value.items() if k not in ignore})
        elif param.kind is inspect.Parameter.VAR_POSITIONAL:
            out[name] = to_jsonable(list(value))
        else:
            out[name] = to_jsonable(value)
    return out


def entrypoint_of(fn: Callable[..., Any]) -> str:
    """``module:qualname``; for scripts run as __main__ use the file path relative to cwd."""
    module = getattr(fn, "__module__", None) or "?"
    qualname = getattr(fn, "__qualname__", getattr(fn, "__name__", "?"))
    if module == "__main__":
        main_file = getattr(sys.modules.get("__main__"), "__file__", None)
        if main_file:
            path = os.path.abspath(main_file)
            try:
                rel = os.path.relpath(path)
                path = rel if not rel.startswith("..") else path
            except ValueError:
                pass
            return f"{path}:{qualname}"
    return f"{module}:{qualname}"


def first_doc_line(fn: Callable[..., Any]) -> str | None:
    doc = inspect.getdoc(fn)
    if not doc:
        return None
    return doc.strip().splitlines()[0].strip() or None
