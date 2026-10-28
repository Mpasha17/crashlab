"""``@crashlab.tool`` records each tool execution as an ``execute_tool`` step."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar, overload

from ..replay.divergence import PatchedToolError, SideEffectBlockedError
from ..replay.rebuild import rebuild_exception
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


def _serve_replay(pending: Any, spec: ToolSpec) -> tuple[bool, Any]:
    """Outside record mode: ``(True, value)`` when the call is served without running the
    tool, ``(False, None)`` when it should run live; raises the served exception.

    Strict replay serves every call from the cassette. A fork past its fork step serves a
    ``ToolResultPatch``, an identical recorded call, or runs the tool live; tools with
    ``side_effects=True`` are never run live there (``SideEffectBlockedError``) unless
    the fork allows it.
    """
    rp = pending.ctx.replayer
    if rp is None:
        if spec.side_effects and pending.ctx.mode != "record":
            _block(pending, f"tool {spec.name!r} has side effects and is never run live "
                            f"during {pending.ctx.mode}")  # fmt: skip
        return False, None
    kind, payload = rp.tool_decision(pending, spec.side_effects)
    if kind == "cassette":
        src = payload
        # steps recorded inside this tool (a sub-agent, an LLM call ...) are not re-made
        skipped = rp.served_from_cassette(src) if pending.idx == src.idx else []
        if skipped:
            pending.attributes["crashlab.replay.skipped_steps"] = skipped
        pending.finish_from(src)
        if skipped:
            pending.ctx.skip_idx(src.idx, skipped)
        if src.error is not None:
            raise rebuild_exception(src.error)
        return True, (src.response or {}).get("result")
    if kind == "patched":
        attrs = {"crashlab.fork.patches": [payload.describe()]}
        if payload.error is not None:
            err = PatchedToolError(payload.error)
            pending.fail(err, is_tool_error=True, attributes=attrs, source="patched")
            raise err
        pending.finish({"result": to_jsonable(payload.result)}, attributes=attrs,
                       source="patched")  # fmt: skip
        return True, payload.result
    if kind == "blocked":
        _block(pending, payload)
    return False, None


def _block(pending: Any, message: str) -> None:
    err = SideEffectBlockedError(message)
    pending.fail(err, is_tool_error=True, source="blocked")
    raise err


def _wrap(fn: Callable[..., Any], spec: ToolSpec) -> Callable[..., Any]:
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            pending = _begin(spec, fn, args, kwargs)
            if pending is None:
                return await fn(*args, **kwargs)
            served, value = _serve_replay(pending, spec)
            if served:
                return value
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
            served, value = _serve_replay(pending, spec)
            if served:
                return value
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
    *,
    name: str | None = None,
    side_effects: bool = False,
    side_effect: bool | None = None,
    description: str | None = None,
) -> Callable[[F], F]: ...


def tool(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    side_effects: bool = False,
    side_effect: bool | None = None,
    description: str | None = None,
) -> Any:
    """Record a tool function's arguments, result and exceptions.

    Use ``side_effects=True`` (alias ``side_effect=True``) for tools that change the world
    (payments, emails, writes). Such tools never run live outside record mode. In a strict
    replay *no* tool runs: every result (or exception) is served from the cassette. In a
    fork, a side-effect tool gets the result of an identical recorded call or raises
    ``SideEffectBlockedError`` (unless the fork passes ``allow_live_side_effects=True``).

    Works as ``@crashlab.tool`` or ``@crashlab.tool(...)`` on sync and async functions.
    """

    def decorate(f: Callable[..., Any]) -> Callable[..., Any]:
        spec = ToolSpec(
            name=name or f.__name__,
            side_effects=bool(side_effects or side_effect),
            description=description or first_doc_line(f),
        )
        return _wrap(f, spec)

    if fn is not None:
        return decorate(fn)
    return decorate
