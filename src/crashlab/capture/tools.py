"""``@crashlab.tool`` records each tool execution as an ``execute_tool`` step."""

from __future__ import annotations

import dataclasses
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
    fake: Callable[..., Any] | None = None


# tool name -> fake implementation, see :func:`fake_tool`
_FAKES: dict[str, Callable[..., Any]] = {}


def fake_tool(name: str, fn: Callable[..., Any] | None = None) -> Any:
    """Register a fake for the ``side_effects=True`` tool ``name``: forks (``fork``,
    ``verify``, ``fix --verify``) call it instead of blocking when the agent makes a
    side-effect call the recording never made, so those trials are graded without touching
    the real world. Same signature as the tool; record mode always runs the real tool.

    ``crashlab.fake_tool("place_order", fake_order)`` or as a decorator
    ``@crashlab.fake_tool("place_order")``. :func:`clear_fakes` removes them."""
    if fn is not None:
        _FAKES[name] = fn
        return fn

    def register(f: Callable[..., Any]) -> Callable[..., Any]:
        _FAKES[name] = f
        return f

    return register


def clear_fakes() -> None:
    """Remove every fake registered with :func:`fake_tool`."""
    _FAKES.clear()


def _fake_for(spec: ToolSpec) -> Callable[..., Any] | None:
    return spec.fake or _FAKES.get(spec.name)


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


def _serve_replay(
    pending: Any, spec: ToolSpec, fn: Callable[..., Any]
) -> tuple[bool, Any, Callable[..., Any] | None]:
    """Outside record mode: ``(True, value, None)`` when the call is served without running
    the tool, ``(False, None, impl)`` when ``impl`` should run: the tool itself, or its
    fake (:func:`fake_tool`) for a side-effect call that would otherwise be blocked; raises
    the served exception.

    Strict replay serves every call from the cassette. A fork past its fork step serves a
    ``ToolResultPatch``, an identical recorded call, or runs the tool live; tools with
    ``side_effects=True`` are never run live there (``SideEffectBlockedError``) unless
    the fork allows it, or run their registered fake.
    """
    rp = pending.ctx.replayer
    if rp is None:
        if spec.side_effects and pending.ctx.mode != "record":
            fake = _fake_for(spec)
            if fake is not None:
                return False, None, fake
            _block(pending, f"tool {spec.name!r} has side effects and is never run live "
                            f"during {pending.ctx.mode}")  # fmt: skip
        return False, None, fn
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
        return True, _revive((src.response or {}).get("result"), src, fn), None
    if kind == "patched":
        attrs = {"crashlab.fork.patches": [payload.describe()]}
        if payload.error is not None:
            err = PatchedToolError(payload.error)
            pending.fail(err, is_tool_error=True, attributes=attrs, source="patched")
            raise err
        pending.finish({"result": to_jsonable(payload.result)}, attributes=attrs,
                       source="patched")  # fmt: skip
        return True, payload.result, None
    if kind == "blocked":
        fake = _fake_for(spec)
        if fake is not None:
            return False, None, fake
        _block(pending, payload)
    return False, None, fn


# --- typed results ------------------------------------------------------------------
# Results are stored as JSON. A tool that returns a pydantic model or a dataclass (or a
# list of them) gets the same type back on replay, so ``order.total`` still works; so do
# tuples, sets, datetimes, Decimals, UUIDs, paths and enums (by the returned value's type,
# or nested ones by the return annotation, ``-> dict[str, date]``).

RESULT_TYPE = "crashlab.result.type"


def _is_rich(obj: Any) -> bool:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return True
    try:
        from pydantic import BaseModel
    except ImportError:  # pragma: no cover - pydantic is a dependency
        return False
    return isinstance(obj, BaseModel)


def _is_typed_scalar(obj: Any) -> bool:
    """Values stored as JSON that come back as another type unless their type is kept:
    tuples and sets become lists, datetimes / Decimals / UUIDs / paths become strings,
    enums their value."""
    import datetime
    import decimal
    import enum
    import pathlib
    import uuid

    kinds = (tuple, set, frozenset, datetime.date, datetime.time, datetime.timedelta,
             decimal.Decimal, uuid.UUID, pathlib.PurePath, enum.Enum)  # fmt: skip
    return isinstance(obj, kinds) and not hasattr(obj, "_fields")  # not namedtuples


def _result_type(result: Any) -> dict[str, Any] | None:
    attrs: dict[str, Any] = {}
    special = _special_numbers(result)
    if special:
        attrs[RESULT_NUMBERS] = special
    if _is_rich(result) or _is_typed_scalar(result):
        cls = type(result)
        attrs[RESULT_TYPE] = f"{cls.__module__}:{cls.__qualname__}"
    return attrs or None


# Numbers strict JSON cannot hold: NaN / inf are stored as "nan" / "inf" strings and
# ints too long for str() as hex. Their places in the result are kept, so replay gives
# the numbers back (``mean([])`` -> nan, then ``math.isnan(x)`` still works).
RESULT_NUMBERS = "crashlab.result.numbers"
_MAX_SPECIAL = 1000


def _special_numbers(obj: Any) -> list[list[Any]]:
    import math

    from .normalize import huge_int

    out: list[list[Any]] = []
    path: list[Any] = []  # shared; copied only for a hit
    isfinite = math.isfinite

    def walk(o: Any) -> None:
        if len(out) >= _MAX_SPECIAL or len(path) > 50:  # to_jsonable stops at depth 50 too
            return
        cls = type(o)
        if cls is float or (cls is not bool and isinstance(o, float)):
            if not isfinite(o):
                out.append([list(path), "float"])
        elif cls is str or cls is bool or o is None:
            return
        elif isinstance(o, int):
            if not isinstance(o, bool) and huge_int(o):
                out.append([list(path), "int"])
        elif isinstance(o, dict):
            for k, v in o.items():
                path.append(str(k))
                walk(v)
                path.pop()
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                path.append(i)
                walk(v)
                path.pop()

    walk(obj)
    return out


def _restore_numbers(value: Any, special: Any) -> Any:
    if not isinstance(special, list) or not special:
        return value
    import copy

    value = copy.deepcopy(value)  # never change the source run's recorded step

    def convert(s: Any, kind: Any) -> Any:
        if not isinstance(s, str):
            return s
        try:
            return float(s) if kind == "float" else int(s, 16)
        except ValueError:
            return s

    for entry in special:
        try:
            path, kind = entry
            if not path:
                value = convert(value, kind)
                continue
            parent = value
            for key in path[:-1]:
                parent = parent[key]
            parent[path[-1]] = convert(parent[path[-1]], kind)
        except (KeyError, IndexError, TypeError, ValueError):
            continue  # a result shape this does not match: leave it as recorded
    return value


def _import_type(ref: str) -> Any:
    import importlib

    module, _, qualname = ref.partition(":")
    obj: Any = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _mentions_rich_type(ann: Any) -> bool:
    """Whether a return annotation names a type JSON does not round-trip (a model, a
    dataclass, a tuple / set, a datetime ...), so the recorded JSON is validated into it."""
    import typing

    origin = typing.get_origin(ann)
    if origin is not None:  # list[Model], dict[str, date], Model | None ...
        if origin in (tuple, set, frozenset):
            return True
        return any(_mentions_rich_type(a) for a in typing.get_args(ann))
    if isinstance(ann, type):
        try:
            from pydantic import BaseModel
        except ImportError:  # pragma: no cover
            return False
        if ann in (tuple, set, frozenset) or (
            issubclass(ann, tuple) and not hasattr(ann, "_fields")
        ):
            return True
        return (dataclasses.is_dataclass(ann) or issubclass(ann, BaseModel)
                or _is_typed_type(ann))  # fmt: skip
    return any(_mentions_rich_type(a) for a in typing.get_args(ann))


def _is_typed_type(ann: type) -> bool:
    import datetime
    import decimal
    import enum
    import pathlib
    import uuid

    return issubclass(ann, (datetime.date, datetime.time, datetime.timedelta, decimal.Decimal,
                            uuid.UUID, pathlib.PurePath, enum.Enum))  # fmt: skip


def _revive(value: Any, src: Any, fn: Callable[..., Any]) -> Any:
    """The recorded JSON result as the type the tool returned, when that is knowable."""
    if value is None:
        return value
    value = _restore_numbers(value, src.attributes.get(RESULT_NUMBERS))
    target: Any = None
    ref = src.attributes.get(RESULT_TYPE)
    if isinstance(ref, str) and ":" in ref and "<locals>" not in ref:
        try:
            target = _import_type(ref)
        except Exception:
            target = None
    if target is None or target in (tuple, set, frozenset):  # -> tuple[Order, ...] is finer
        try:
            import typing

            ann = typing.get_type_hints(fn).get("return")
        except Exception:
            ann = None
        if ann is not None and _mentions_rich_type(ann):
            target = ann
    if target is None:
        return value
    import datetime

    if target in (datetime.datetime, datetime.date, datetime.time) and isinstance(value, str):
        try:  # exact inverse of isoformat() (pydantic would swap timezone.utc for its TzInfo)
            return target.fromisoformat(value)
        except ValueError:
            return value
    try:
        from pydantic import TypeAdapter

        return TypeAdapter(target).validate_python(value)
    except Exception:  # not rebuildable: serve the recorded JSON as before
        return value


def _block(pending: Any, message: str) -> None:
    err = SideEffectBlockedError(message)
    pending.fail(err, is_tool_error=True, source="blocked")
    raise err


FAKE_ATTR = {"crashlab.tool.fake": True}


def _done(pending: Any, result: Any, faked: bool) -> None:
    attrs = _result_type(result)
    if faked:
        attrs = {**(attrs or {}), **FAKE_ATTR}
    pending.finish(
        {"result": to_jsonable(result)}, attributes=attrs, source="fake" if faked else None
    )


def _wrap(fn: Callable[..., Any], spec: ToolSpec) -> Callable[..., Any]:
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            pending = _begin(spec, fn, args, kwargs)
            if pending is None:
                return await fn(*args, **kwargs)
            served, value, impl = _serve_replay(pending, spec, fn)
            if served:
                return value
            faked = impl is not fn
            try:
                with pending.activate():
                    result = impl(*args, **kwargs)  # type: ignore[misc]
                    if inspect.isawaitable(result):
                        result = await result
            except BaseException as exc:
                pending.fail(exc, is_tool_error=True, attributes=FAKE_ATTR if faked else None,
                             source="fake" if faked else None)  # fmt: skip
                raise
            _done(pending, result, faked)
            return result

        wrapper: Callable[..., Any] = async_wrapper
    else:

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            pending = _begin(spec, fn, args, kwargs)
            if pending is None:
                return fn(*args, **kwargs)
            served, value, impl = _serve_replay(pending, spec, fn)
            if served:
                return value
            faked = impl is not fn
            try:
                with pending.activate():
                    result = impl(*args, **kwargs)  # type: ignore[misc]
            except BaseException as exc:
                pending.fail(exc, is_tool_error=True, attributes=FAKE_ATTR if faked else None,
                             source="fake" if faked else None)  # fmt: skip
                raise
            _done(pending, result, faked)
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
    fake: Callable[..., Any] | None = None,
) -> Callable[[F], F]: ...


def tool(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    side_effects: bool = False,
    side_effect: bool | None = None,
    description: str | None = None,
    fake: Callable[..., Any] | None = None,
) -> Any:
    """Record a tool function's arguments, result and exceptions.

    Use ``side_effects=True`` (alias ``side_effect=True``) for tools that change the world
    (payments, emails, writes). Such tools never run live outside record mode. In a strict
    replay *no* tool runs: every result (or exception) is served from the cassette. In a
    fork, a side-effect tool gets the result of an identical recorded call or raises
    ``SideEffectBlockedError`` (unless the fork passes ``allow_live_side_effects=True``),
    or calls ``fake`` (same signature, e.g. returns a fake order id) when one is given here
    or registered with :func:`fake_tool`, so ``verify`` can grade those trials safely.

    Works as ``@crashlab.tool`` or ``@crashlab.tool(...)`` on sync and async functions.
    """

    def decorate(f: Callable[..., Any]) -> Callable[..., Any]:
        spec = ToolSpec(
            name=name or f.__name__,
            side_effects=bool(side_effects or side_effect),
            description=description or first_doc_line(f),
            fake=fake,
        )
        return _wrap(f, spec)

    if fn is not None:
        return decorate(fn)
    return decorate
