"""Run context: which run is active, the monotonic step counter, and the capture mode.

Uses ``contextvars`` so capture is thread- and asyncio-safe without changing user code.
Note: plain ``threading.Thread`` / ``ThreadPoolExecutor`` workers do not inherit context
vars automatically; use ``contextvars.copy_context().run`` if tools run in worker threads.
"""

from __future__ import annotations

import contextvars
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from .models import Flag, Run, Step

if TYPE_CHECKING:
    from .store import Store

Mode = Literal["record", "replay", "fork"]
SUPPORTED_MODES: tuple[Mode, ...] = ("record",)


@dataclass
class RunContext:
    run: Run
    store: Store
    mode: Mode = "record"
    _counter: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    # Tool calls requested by the most recent chat response that no tool step has
    # consumed yet; used to link execute_tool steps to gen_ai.tool.call.id.
    pending_tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def next_idx(self) -> int:
        with self._lock:
            idx = self._counter
            self._counter += 1
            return idx

    @property
    def steps_started(self) -> int:
        return self._counter

    def add_step(self, step: Step) -> None:
        with self._lock:
            u = self.run.usage
            u.steps += 1
            if step.operation == "chat":
                u.llm_calls += 1
                u.input_tokens += int(step.attributes.get("gen_ai.usage.input_tokens") or 0)
                u.output_tokens += int(step.attributes.get("gen_ai.usage.output_tokens") or 0)
                u.total_tokens = u.input_tokens + u.output_tokens
                if step.response:
                    self.pending_tool_calls = list(step.response.get("tool_calls") or [])
            elif step.operation == "execute_tool":
                u.tool_calls += 1
            if step.error is not None:
                u.errors += 1
        self.store.add_step(step)

    def add_flag(self, flag: Flag) -> None:
        self.run.flags.append(flag)
        self.store.add_flag(self.run.run_id, flag)

    def claim_tool_call_id(self, tool_name: str, args: dict[str, Any]) -> str | None:
        """Pop the pending tool call that best matches this tool execution."""
        with self._lock:
            fallback = None
            for i, call in enumerate(self.pending_tool_calls):
                if call.get("name") != tool_name:
                    continue
                if fallback is None:
                    fallback = i
                try:
                    parsed = json.loads(call.get("arguments") or "{}")
                except (TypeError, ValueError):
                    continue
                if isinstance(parsed, dict) and all(args.get(k) == v for k, v in parsed.items()):
                    return self.pending_tool_calls.pop(i).get("id")
            if fallback is not None:
                return self.pending_tool_calls.pop(fallback).get("id")
        return None


_current_run: contextvars.ContextVar[RunContext | None] = contextvars.ContextVar(
    "crashlab_current_run", default=None
)
_current_span: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "crashlab_current_span", default=None
)


def current() -> RunContext | None:
    """Return the active run context, or None when no run is being recorded."""
    return _current_run.get()


def current_run() -> Run | None:
    ctx = _current_run.get()
    return ctx.run if ctx else None


def current_mode() -> Mode:
    ctx = _current_run.get()
    return ctx.mode if ctx else "record"


def check_mode(mode: str) -> Mode:
    if mode not in SUPPORTED_MODES:
        raise NotImplementedError(
            f"crashlab mode {mode!r} is not implemented yet; only 'record' is available "
            "in v0.0.1 (replay and fork are on the roadmap)."
        )
    return mode  # type: ignore[return-value]


def set_current(ctx: RunContext | None) -> contextvars.Token:
    return _current_run.set(ctx)


def reset_current(token: contextvars.Token) -> None:
    _current_run.reset(token)


def current_span_id() -> str | None:
    span = _current_span.get()
    if span is not None:
        return span
    ctx = _current_run.get()
    return ctx.run.span_id if ctx else None


@contextmanager
def active_span(span_id: str) -> Iterator[None]:
    token = _current_span.set(span_id)
    try:
        yield
    finally:
        _current_span.reset(token)
