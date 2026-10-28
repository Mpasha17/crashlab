"""Run context: which run is active, the monotonic step counter, and the capture mode.

Uses ``contextvars`` so capture is thread- and asyncio-safe without changing user code.
asyncio tasks and ``asyncio.to_thread`` inherit the context automatically. Plain
``threading.Thread`` and ``concurrent.futures`` workers do not; use
:func:`copy_context_to_thread` (alias ``crashlab.bind_context``) or
:class:`ContextThreadPoolExecutor` so tool / LLM calls in worker threads are recorded.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import json
import os
import sqlite3
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from .models import Finding, Run, Step

if TYPE_CHECKING:
    from .checks.engine import CheckEngine
    from .replay.engine import Replayer
    from .store import Store

Mode = Literal["record", "replay", "fork"]
SUPPORTED_MODES: tuple[Mode, ...] = ("record", "replay", "fork")


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
    # Set in replay / fork mode: serves LLM/tool calls from the source run's cassette.
    replayer: Replayer | None = None
    # Finished steps, kept in memory so @crashlab.check oracles can run at run end.
    steps: list[Step] = field(default_factory=list)
    # Runtime checks (detectors) and the opt-in circuit breaker.
    checks: CheckEngine | None = None
    circuit_breaker: bool = False
    # replay: (step idx, finding) where the recording's circuit breaker stopped it
    breaker_replay: tuple[int, Finding] | None = None
    tripped: Finding | None = None
    _cost_unknown: bool = False
    store_failed: bool = False  # a write failed: the rest of the run is not stored
    # weakrefs to LLM streams opened in this run; any still unfinished at run end (the
    # consumer raised mid-stream and its traceback keeps the stream alive) are recorded
    # as partial then, not lost after the run is saved
    open_streams: list[Any] = field(default_factory=list)
    # the process recording this run; a forked child inherits the context but not the
    # step counter's future, so its steps would collide with the parent's (see current())
    pid: int = field(default_factory=os.getpid)

    def finish_open_streams(self) -> None:
        streams, self.open_streams = self.open_streams, []
        for ref in streams:
            stream = ref()
            if stream is not None:
                with contextlib.suppress(Exception):  # best effort, like __del__
                    stream._finish(complete=False)

    def write(self, fn: Callable[..., Any], *args: Any) -> None:
        """Write to the store; a failing disk (full, read-only, removed) stops recording
        this run with one warning instead of crashing the agent mid-run."""
        if self.store_failed:
            return
        try:
            fn(*args)
        except (OSError, sqlite3.Error) as err:
            self.store_failed = True
            import warnings

            warnings.warn(
                f"crashlab: stopped recording run {self.run.run_id} ({self.run.name!r}): "
                f"cannot write to {self.store.root}: {type(err).__name__}: {err}; "
                "the agent keeps running",
                RuntimeWarning,
                stacklevel=3,
            )

    def next_idx(self) -> int:
        with self._lock:
            idx = self._counter
            self._counter += 1
            return idx

    def skip_idx(self, after: int, idxs: list[int]) -> None:
        """Replay served step ``after`` whole: the recorded steps ``idxs`` nested inside it
        are not re-made, so the next step gets the index it had in the recording."""
        with self._lock:
            if (
                idxs
                and self._counter == after + 1
                and idxs == list(range(after + 1, after + 1 + len(idxs)))
            ):
                self._counter += len(idxs)

    @property
    def steps_started(self) -> int:
        return self._counter

    def add_step(self, step: Step) -> None:
        if step.operation == "chat":
            from .pricing import step_cost

            # a step served from a cassette keeps the cost of the recorded call (the
            # replay client points at a placeholder endpoint)
            if not (step.source == "cassette" and "crashlab.usage.cost_usd" in step.attributes):
                step.attributes["crashlab.usage.cost_usd"] = step_cost(step)
        with self._lock:
            u = self.run.usage
            u.steps += 1
            if step.operation == "chat":
                u.llm_calls += 1
                cost = step.attributes.get("crashlab.usage.cost_usd")
                if cost is None:  # unknown model, or no usage reported: never guessed
                    self._cost_unknown = True
                else:
                    u.cost_usd = round((u.cost_usd or 0.0) + cost, 8)
                if self._cost_unknown:
                    u.cost_usd = None
                inp = int(step.attributes.get("gen_ai.usage.input_tokens") or 0)
                out = int(step.attributes.get("gen_ai.usage.output_tokens") or 0)
                total = step.attributes.get("crashlab.usage.total_tokens")
                u.input_tokens += inp
                u.output_tokens += out
                u.total_tokens += int(total) if total is not None else inp + out
                if step.response:
                    self.pending_tool_calls = list(step.response.get("tool_calls") or [])
            elif step.operation == "embeddings":
                # tokens count toward the run; cost stays chat-only (no embedding prices)
                inp = int(step.attributes.get("gen_ai.usage.input_tokens") or 0)
                u.input_tokens += inp
                u.total_tokens += int(step.attributes.get("crashlab.usage.total_tokens") or inp)
            elif step.operation == "execute_tool":
                u.tool_calls += 1
            if step.error is not None:
                u.errors += 1
            self.steps.append(step)
        self.write(self.store.add_step, step)
        if self.checks is not None:
            self.add_findings(self.checks.on_step(step))
        if self.breaker_replay is not None and step.idx == self.breaker_replay[0]:
            self._replay_trip(self.breaker_replay[1])

    def _replay_trip(self, recorded: Finding) -> None:
        """A strict replay stops where the recording's circuit breaker stopped it."""
        mine = next(
            (f for f in self.run.findings if (f.kind, f.key) == (recorded.kind, recorded.key)),
            None,
        )
        if mine is None:
            mine = recorded.model_copy(deep=True)
            self.run.findings.append(mine)
        mine.stopped_run = True
        self._trip(mine)

    def add_findings(self, findings: list[Finding]) -> None:
        """Record new / upgraded findings; a critical one trips the circuit breaker."""
        for f in findings:
            if not any(f is g for g in self.run.findings):
                self.run.findings.append(f)
                self.write(self.store.add_flag, self.run.run_id, f)
            if self.circuit_breaker and f.severity == "critical" and self.tripped is None:
                f.stopped_run = True
                self._trip(f)

    def _trip(self, finding: Finding) -> None:
        if self.tripped is None:
            self.tripped = finding

    def raise_if_tripped(self) -> None:
        """Stop the agent: called after a step is recorded and before a new one starts."""
        if self.tripped is not None:
            from .checks.engine import CircuitBreakerError

            raise CircuitBreakerError(self.tripped)

    def add_flag(self, flag: Finding) -> None:  # older name
        self.add_findings([flag])

    def claim_tool_call_id(self, tool_name: str, args: dict[str, Any]) -> str | None:
        """Pop the pending tool call that best matches this tool execution."""
        with self._lock:
            fallback = None
            for i, call in enumerate(self.pending_tool_calls):
                if call.get("name") != tool_name:
                    continue
                if fallback is None:
                    fallback = i
                parsed = call.get("arguments")
                if isinstance(parsed, str):  # day-1 cassettes / invalid JSON
                    try:
                        parsed = json.loads(parsed or "{}")
                    except ValueError:
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


# Runs being recorded in this process, with the thread that started each. Frameworks that
# run tools in a fresh worker thread without copying context vars (smolagents' CodeAgent
# executes every code block in a new ThreadPoolExecutor) would otherwise drop those tool
# calls silently. A worker thread with no run of its own falls back to the only live run,
# never when two runs are live. CRASHLAB_THREAD_FALLBACK=0 turns this off.
_live: dict[int, tuple[RunContext, int]] = {}
_live_lock = threading.Lock()


def register_live(ctx: RunContext) -> None:
    with _live_lock:
        _live[id(ctx)] = (ctx, threading.get_ident())


def unregister_live(ctx: RunContext) -> None:
    with _live_lock:
        _live.pop(id(ctx), None)


def _thread_fallback() -> RunContext | None:
    if os.environ.get("CRASHLAB_THREAD_FALLBACK", "1") == "0":
        return None
    with _live_lock:
        if len(_live) != 1:
            return None
        ctx, owner = next(iter(_live.values()))
    return ctx if owner != threading.get_ident() else None


_forked_warned: int | None = None  # pid of the child process already warned


def _own(ctx: RunContext | None) -> RunContext | None:
    """``ctx`` unless it belongs to another process. A child made by ``fork``
    (multiprocessing's "fork" start method, ``os.fork``) inherits the parent's run but
    its steps cannot reach the parent's counter or cassette: they were stored with
    duplicate step indexes and a replay of the parent diverged (and hung waiting for the
    child). Calls in a child are not part of the parent's run, as with "spawn" and
    "forkserver"; an agent called there records a run of its own."""
    global _forked_warned
    if ctx is None or ctx.pid == os.getpid():
        return ctx
    if _forked_warned != os.getpid():
        _forked_warned = os.getpid()
        import warnings

        warnings.warn(
            f"crashlab: calls in child process {os.getpid()} are not recorded in run "
            f"{ctx.run.run_id} ({ctx.run.name!r}) of parent process {ctx.pid}; record "
            "them in the parent, or call a @crashlab.agent in the child for a run of its own",
            RuntimeWarning,
            stacklevel=4,
        )
    return None


def current() -> RunContext | None:
    """Return the active run context, or None when no run is being recorded."""
    ctx = _current_run.get()
    return _own(ctx if ctx is not None else _thread_fallback())


def bound_current() -> RunContext | None:
    """The run bound to this context only (no worker-thread fallback). Used to decide
    whether an agent call is nested: a top-level call in another thread (a web server
    handling two requests) is its own run, not a step of whatever run is live."""
    return _own(_current_run.get())


def current_run() -> Run | None:
    ctx = current()
    return ctx.run if ctx else None


def current_mode() -> Mode:
    ctx = current()
    return ctx.mode if ctx else "record"


def check_mode(mode: str) -> Mode:
    if mode not in SUPPORTED_MODES:
        raise NotImplementedError(
            f"crashlab mode {mode!r} is not implemented yet; available: "
            f"{', '.join(SUPPORTED_MODES)}."
        )
    return mode  # type: ignore[return-value]


# A replay prepared by crashlab.replay(); the next record() in this context picks it up.
_pending_replay: contextvars.ContextVar[Replayer | None] = contextvars.ContextVar(
    "crashlab_pending_replay", default=None
)


def take_pending_replay() -> Replayer | None:
    rp = _pending_replay.get()
    if rp is not None:
        _pending_replay.set(None)
    return rp


@contextmanager
def pending_replay(rp: Replayer) -> Iterator[None]:
    token = _pending_replay.set(rp)
    try:
        yield
    finally:
        _pending_replay.reset(token)


def is_replaying() -> bool:
    """True inside a run that is being replayed from a cassette (strict replay or fork)."""
    ctx = current()
    return ctx is not None and ctx.mode in ("replay", "fork")


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


# --- threads ---------------------------------------------------------------------------

T = TypeVar("T")


def copy_context_to_thread(fn: Callable[..., T]) -> Callable[..., T]:
    """Bind ``fn`` to a snapshot of the *current* context (active run, current span).

    Call this in the thread that owns the run, then run the returned callable anywhere::

        t = threading.Thread(target=crashlab.bind_context(worker), args=(order_id,))

    Each call runs in its own copy of the snapshot, so the same bound callable can run
    concurrently in several threads.
    """
    snapshot = contextvars.copy_context()

    @functools.wraps(fn)
    def bound(*args: Any, **kwargs: Any) -> T:
        return snapshot.copy().run(fn, *args, **kwargs)

    return bound


bind_context = copy_context_to_thread


class ContextThreadPoolExecutor(ThreadPoolExecutor):
    """A ``ThreadPoolExecutor`` whose tasks run in the submitter's context.

    Drop-in replacement: ``with crashlab.ContextThreadPoolExecutor() as pool:
    pool.map(get_order, ids)`` records every ``get_order`` call in the active run.
    """

    def submit(self, fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> Future[T]:
        ctx = contextvars.copy_context()
        return super().submit(ctx.run, fn, *args, **kwargs)


def run_in_thread(fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """Run ``fn`` in a new thread (in the current context) and return its result."""
    result: dict[str, Any] = {}

    def target() -> None:
        try:
            result["value"] = fn(*args, **kwargs)
        except BaseException as exc:
            result["error"] = exc

    t = threading.Thread(target=copy_context_to_thread(target), daemon=True)
    t.start()
    t.join()
    if "error" in result:
        raise result["error"]
    return result["value"]
