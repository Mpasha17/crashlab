from __future__ import annotations

import inspect
import os
import sys
from collections.abc import Callable
from typing import Any

from .normalize import to_jsonable


def _safe_jsonable(value: Any) -> Any:
    """``to_jsonable`` that never raises: recording an argument must not crash the call."""
    try:
        return to_jsonable(value)
    except Exception as exc:  # a hostile __dict__ / property, a broken model_dump ...
        return f"<unrecordable {type(value).__qualname__}: {type(exc).__name__}>"


def bind_arguments(
    fn: Callable[..., Any],
    args: tuple,
    kwargs: dict,
    ignore: tuple[str, ...] = (),
    skip_value: Callable[[Any], bool] | None = None,
) -> dict[str, Any]:
    """Map a call's positional/keyword args to parameter names (JSON-able). Parameters
    named in ``ignore``, or whose value ``skip_value`` accepts, are left out."""
    try:
        sig = inspect.signature(fn)
        try:
            bound = sig.bind(*args, **kwargs)
        except TypeError:
            # a staticmethod whose __signature__ still lists `self` (smolagents' @tool
            # sets one on SimpleTool.forward): bind without it
            params = list(sig.parameters.values())
            if not params or params[0].name not in ("self", "cls"):
                raise
            sig = sig.replace(parameters=params[1:])
            bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
    except (TypeError, ValueError):
        return {"args": _safe_jsonable(list(args)), "kwargs": _safe_jsonable(kwargs)}
    out: dict[str, Any] = {}
    for name, value in bound.arguments.items():
        if name in ("self", "cls") or name in ignore:
            continue
        param = sig.parameters[name]
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            out.update({k: _safe_jsonable(v) for k, v in value.items()
                        if k not in ignore and not (skip_value and skip_value(v))})  # fmt: skip
        elif param.kind is inspect.Parameter.VAR_POSITIONAL:
            out[name] = _safe_jsonable([v for v in value if not (skip_value and skip_value(v))])
        elif not (skip_value and skip_value(value)):
            out[name] = _safe_jsonable(value)
    return out


REPLAY_MODULE_PREFIX = "crashlab_replay_"


def entrypoint_of(fn: Callable[..., Any]) -> str:
    """``module:qualname``; for scripts run as __main__ use the file path relative to cwd."""
    module = getattr(fn, "__module__", None) or "?"
    qualname = getattr(fn, "__qualname__", getattr(fn, "__name__", "?"))
    # a script (``__main__``) or a script file that replay imported by path
    # (``crashlab_replay_<stem>``, see ``resolve_entrypoint``): record the file path, so
    # a replay of a replay resolves too
    if module == "__main__" or module.startswith(REPLAY_MODULE_PREFIX):
        main_file = getattr(sys.modules.get(module), "__file__", None)
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
