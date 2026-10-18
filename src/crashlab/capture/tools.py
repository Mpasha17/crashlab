"""``@crashlab.tool`` records each tool execution as an ``execute_tool`` step."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar, overload

from ._steps import begin_step
from ._util import bind_arguments, first_doc_line
from .normalize import to_jsonable

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True)
class ToolSpec:
    name: str
    side_effects: bool
    description: str | None


def _begin(spec: ToolSpec, fn: Callable[..., Any], args: tuple, kwargs: dict):
    call_args = bind_arguments(fn, args, kwargs)
    attrs: dict[str, Any] = {
        "gen_ai.tool.name": spec.name,
        "gen_ai.tool.type": "function",
        "crashlab.tool.side_effects": spec.side_effects,
    }
    if spec.description:
        attrs["gen_ai.tool.description"] = spec.description
    pending = begin_step(
        "execute_tool",
        spec.name,
        {"args": call_args},
        attributes=attrs,
        hash_input={"tool": spec.name, "args": call_args},
    )
    if pending is not None:
        call_id = pending.ctx.claim_tool_call_id(spec.name, call_args)
        if call_id:
            pending.attributes["gen_ai.tool.call.id"] = call_id
    return pending


def _wrap(fn: Callable[..., Any], spec: ToolSpec) -> Callable[..., Any]:
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            pending = _begin(spec, fn, args, kwargs)
            if pending is None:
                return await fn(*args, **kwargs)
            try:
                with pending.activate():
                    result = await fn(*args, **kwargs)
            except BaseException as exc:
                pending.fail(exc, is_tool_error=True)
                raise
            pending.finish({"result": to_jsonable(result)})
            return result

        wrapper: Callable[..., Any] = async_wrapper
    else:

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            pending = _begin(spec, fn, args, kwargs)
            if pending is None:
                return fn(*args, **kwargs)
            try:
                with pending.activate():
                    result = fn(*args, **kwargs)
            except BaseException as exc:
                pending.fail(exc, is_tool_error=True)
                raise
            pending.finish({"result": to_jsonable(result)})
            return result

        wrapper = sync_wrapper
    wrapper.__crashlab_tool__ = spec  # type: ignore[attr-defined]
    return wrapper


@overload
def tool(fn: F, /) -> F: ...
@overload
def tool(
    *, name: str | None = None, side_effects: bool = False, description: str | None = None
) -> Callable[[F], F]: ...


def tool(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    side_effects: bool = False,
    description: str | None = None,
) -> Any:
    """Record a tool function's arguments, result and exceptions.

    Use ``side_effects=True`` for tools that change the world (payments, emails, writes).
    In v0.0.1 this is recorded as metadata; replay/fork (coming) will never re-run such
    tools live and will serve the recorded result instead.

    Works as ``@crashlab.tool`` or ``@crashlab.tool(...)`` on sync and async functions.
    """

    def decorate(f: Callable[..., Any]) -> Callable[..., Any]:
        spec = ToolSpec(
            name=name or f.__name__,
            side_effects=side_effects,
            description=description or first_doc_line(f),
        )
        return _wrap(f, spec)

    if fn is not None:
        return decorate(fn)
    return decorate
