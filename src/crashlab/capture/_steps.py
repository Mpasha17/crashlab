"""Shared step-recording machinery used by the LLM proxies, tools and agents."""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from ..context import RunContext, active_span, current, current_span_id
from ..models import ErrorInfo, Operation, Step, StepSource, utcnow
from ..store.redact import get_redactor
from .normalize import request_hash, to_jsonable


def error_info(exc: BaseException, *, is_tool_error: bool = False) -> ErrorInfo:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    body = getattr(exc, "body", None)
    if body is not None:
        body = to_jsonable(body)
        if len(json.dumps(body, default=str)) > 4000:
            body = json.dumps(body, default=str)[:4000]
    return ErrorInfo(
        state=_exc_state(exc),
        body=body,
        type=f"{type(exc).__module__}.{type(exc).__qualname__}".removeprefix("builtins."),
        message=str(exc)[:4000],
        is_tool_error=is_tool_error,
        status_code=status if isinstance(status, int) else None,
        traceback=tb[-8000:],
    )


_SCALARS = (str, int, float, bool, type(None))
# set by the SDKs / rebuilt from other fields; never copied
_NOT_STATE = {"body", "response", "request", "status_code", "message", "recorded", "args"}


def _small_json(value: Any) -> bool:
    if isinstance(value, _SCALARS):
        return not isinstance(value, str) or len(value) <= 1000
    if isinstance(value, (list, tuple)) and len(value) <= 20:
        return all(isinstance(v, _SCALARS) for v in value)
    if isinstance(value, dict) and len(value) <= 20:
        return all(isinstance(k, str) and isinstance(v, _SCALARS) for k, v in value.items())
    return False


def _exc_state(exc: BaseException) -> dict[str, Any] | None:
    """``args`` when they are not just ``(str(exc),)``, and the exception's public
    JSON-able attributes, so a replayed ``except E as e: e.sku`` sees the same values."""
    state: dict[str, Any] = {}
    args = exc.args
    if args != (str(exc),) and all(_small_json(a) for a in args) and len(repr(args)) < 2000:
        state["args"] = list(args)
    try:
        own = vars(exc)
    except TypeError:
        own = {}
    attrs = {
        k: list(v) if isinstance(v, tuple) else v
        for k, v in own.items()
        if isinstance(k, str) and not k.startswith("_") and k not in _NOT_STATE
        and _small_json(v)
    }  # fmt: skip
    if attrs:
        state["attrs"] = dict(list(attrs.items())[:30])
    return state or None


def _cancelled_from_outside(exc: BaseException) -> bool:
    import asyncio

    if not isinstance(exc, asyncio.CancelledError):
        return False
    try:
        task = asyncio.current_task()
    except RuntimeError:  # not under asyncio (trio ...)
        return False
    cancelling = getattr(task, "cancelling", None)  # Python 3.11+
    return bool(task is not None and cancelling is not None and cancelling())


@dataclass
class PendingStep:
    ctx: RunContext
    idx: int
    span_id: str
    parent_span_id: str | None
    operation: Operation
    name: str
    request: dict[str, Any]
    request_hash: str
    attributes: dict[str, Any] = field(default_factory=dict)
    raw_request: dict[str, Any] | None = None
    started_at: Any = field(default_factory=utcnow)
    t0: float = field(default_factory=time.perf_counter)
    done: bool = False
    source: StepSource = "live"  # "patched" when a fork patch changed this call
    # streams finish inside iteration / close(); the circuit breaker then fires at the
    # next step instead of raising out of a stream
    defer_breaker: bool = False

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Make this step the parent span for anything recorded inside it."""
        with active_span(self.span_id):
            yield

    def _emit(self, response, raw_response, error, extra_attrs, source=None) -> Step:
        if self.done:
            raise RuntimeError("step already finished")
        self.done = True
        attrs = dict(self.attributes)
        attrs.update(extra_attrs or {})
        step = Step(
            run_id=self.ctx.run.run_id,
            idx=self.idx,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            operation=self.operation,
            name=self.name,
            attributes=attrs,
            request=self.request,
            response=response,
            raw_request=self.raw_request,
            raw_response=raw_response,
            error=error,
            request_hash=self.request_hash,
            started_at=self.started_at,
            duration_ms=round((time.perf_counter() - self.t0) * 1000, 3),
            source=source or self.source,
        )
        self.ctx.add_step(step)
        if not self.defer_breaker:
            self.ctx.raise_if_tripped()
        return step

    # replay ----------------------------------------------------------------------------
    def replay_source(self) -> Step | None:
        """In replay / fork mode, the recorded step this call must reproduce; None when it
        runs live (record mode, or a fork past its fork step).

        A mismatch is recorded as this step's error and raised as ``DivergenceError``.
        """
        replayer = self.ctx.replayer
        if replayer is None:
            return None
        try:
            return replayer.source_for(self)
        except BaseException as exc:
            if not self.done:
                self.fail(exc)
            raise

    def finish_from(self, src: Step, source: StepSource = "cassette") -> Step:
        """Finish with the recorded step's response / error / response attributes."""
        attrs = dict(src.attributes)
        attrs.update(self.attributes)  # this call's request attributes (server, tool call id)
        attrs["crashlab.replay.source_step"] = src.idx
        self.attributes = attrs
        return self._emit(src.response, src.raw_response, src.error, None, source)

    def finish(
        self,
        response: dict[str, Any] | None,
        *,
        raw_response: dict[str, Any] | None = None,
        attributes: dict[str, Any] | None = None,
        source: StepSource | None = None,
    ) -> Step:
        return self._emit(response, raw_response, None, attributes, source)

    def fail(
        self,
        exc: BaseException,
        *,
        is_tool_error: bool = False,
        response: dict[str, Any] | None = None,
        raw_response: dict[str, Any] | None = None,
        attributes: dict[str, Any] | None = None,
        source: StepSource | None = None,
    ) -> Step:
        attrs = {"error.type": type(exc).__qualname__}
        attrs.update(attributes or {})
        err = error_info(exc, is_tool_error=is_tool_error)
        if _cancelled_from_outside(exc):
            # the caller cancelled this call (wait_for timeout, a failing TaskGroup
            # sibling): replay waits to be cancelled again (rebuild.recancel)
            ms = round((time.perf_counter() - self.t0) * 1000, 1)
            err.state = {**(err.state or {}), "cancelled_outside_ms": ms}
        return self._emit(response, raw_response, err, attrs, source)


def begin_step(
    operation: Operation,
    name: str,
    request: dict[str, Any],
    *,
    attributes: dict[str, Any] | None = None,
    raw_request: dict[str, Any] | None = None,
    hash_input: Any = None,
    idx: int | None = None,
) -> PendingStep | None:
    """Start a step in the active run, or return None when nothing is being recorded.

    The request hash is computed over the redacted request (or ``hash_input``) so it can
    be recomputed from the stored cassette. ``idx`` is a step index already allocated
    with ``ctx.next_idx()`` (a fork patches the request before the step starts).
    """
    ctx = current()
    if ctx is None:
        return None
    ctx.raise_if_tripped()  # the circuit breaker stopped this run: no more calls
    from .._ids import new_span_id

    redactor = get_redactor()
    attrs = {"gen_ai.operation.name": operation}
    attrs.update(attributes or {})
    return PendingStep(
        ctx=ctx,
        idx=ctx.next_idx() if idx is None else idx,
        span_id=new_span_id(),
        parent_span_id=current_span_id(),
        operation=operation,
        name=name,
        request=request,
        request_hash=request_hash(
            redactor.redact(hash_input if hash_input is not None else request)
        ),
        attributes=attrs,
        raw_request=raw_request,
    )
