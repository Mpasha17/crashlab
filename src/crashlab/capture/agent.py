"""Run boundaries: ``@crashlab.agent`` and ``crashlab.record(...)``."""

from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
import os
import platform
import sqlite3
import subprocess
import sys
import time
import warnings
from collections.abc import Callable
from typing import Any, TypeVar, overload

from .._ids import new_span_id, new_ulid
from .._version import __version__
from ..context import (
    RunContext,
    bound_current,
    check_mode,
    register_live,
    reset_current,
    set_current,
    take_pending_replay,
    unregister_live,
)
from ..models import Run, utcnow
from ..store import Store
from ._steps import begin_step, error_info
from ._util import bind_arguments, entrypoint_of
from .normalize import to_jsonable

F = TypeVar("F", bound=Callable[..., Any])

_git_cache: dict[str, str | None] = {}


def git_version(cwd: str | None = None) -> str | None:
    """``<sha>`` or ``<sha>-dirty`` for the git repo containing ``cwd``; None if unavailable."""
    cwd = os.path.abspath(cwd or os.getcwd())
    if cwd in _git_cache:
        return _git_cache[cwd]
    value: str | None = None
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=3
        )
        if sha.returncode == 0 and sha.stdout.strip():
            value = sha.stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=3,
            )
            if dirty.returncode == 0 and dirty.stdout.strip():
                value += "-dirty"
    except (OSError, subprocess.SubprocessError):
        value = None
    _git_cache[cwd] = value
    return value


_SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH")


def _env_vars(names: Any) -> dict[str, str | None]:
    """Values of the environment variables an agent reads (``env=``), recorded so replays
    and forks run with the same model name / streaming flag. Names that look like secrets
    are refused: their values would land in the store and in exported cassettes."""
    out: dict[str, str | None] = {}
    for name in names or ():
        if any(h in name.upper() for h in _SECRET_HINTS):
            raise ValueError(
                f"crashlab: env={name!r} looks like a secret; only list non-secret settings "
                "(model name, base URL, streaming flag). Replays never need real keys."
            )
        out[name] = os.environ.get(name)
    return out


def _env() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": sys.platform,
        "crashlab": __version__,
    }
    for pkg in ("openai", "anthropic", "litellm", "pydantic"):
        mod = sys.modules.get(pkg)
        if mod is not None and getattr(mod, "__version__", None):
            env[pkg] = mod.__version__
    return env


class record:
    """Record everything inside the block as one run.

    ``with crashlab.record("refund-bot", input={"msg": msg}) as run: ...``  (also ``async with``)

    ``checks`` configures the built-in runtime checks (see :mod:`crashlab.checks`):
    ``None`` uses the defaults and ``[tool.crashlab.checks]`` from pyproject.toml, a dict
    overrides parts of that (``checks={"max_steps": 20, "circuit_breaker": True}``),
    ``False`` turns them off.

    ``env`` lists environment variables the agent reads (``["AGENT_MODEL", "STREAM"]``):
    their values are recorded and set again while the run is replayed or forked, so an
    exported test does not depend on the shell it runs in. Secret-looking names are refused.

    The yielded :class:`~crashlab.models.Run` can be mutated, e.g. ``run.output = answer``
    or ``run.outcome = "pass"``; it is persisted when the block exits. An exception marks
    the run ``status="error"`` and ``outcome="fail"`` and is re-raised.
    """

    def __init__(
        self,
        name: str,
        input: dict[str, Any] | None = None,
        *,
        entrypoint: str | None = None,
        tags: list[str] | None = None,
        store_dir: str | os.PathLike[str] | None = None,
        mode: str = "record",
        checks: Any = None,
        env: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.name = name
        self.checks = checks
        self.env_vars = _env_vars(env)
        self.input = to_jsonable(input or {})
        if not isinstance(self.input, dict):
            self.input = {"input": self.input}
        self.entrypoint = entrypoint
        self.tags = list(tags or [])
        self.store_dir = store_dir
        self.mode = check_mode(mode)
        self.ctx: RunContext | None = None
        self._token = None
        self._t0 = 0.0

    # sync ------------------------------------------------------------------------------
    def __enter__(self) -> Run:
        if bound_current() is not None:
            raise RuntimeError(
                "crashlab.record(): a run is already active in this context; nested runs are "
                "not supported (nested @crashlab.agent calls are recorded as invoke_agent steps)"
            )
        replayer = take_pending_replay()
        if self.mode in ("replay", "fork") and replayer is None:
            raise RuntimeError(
                f"{self.mode} runs are started with crashlab.{self.mode}(run, ...), not record()"
            )
        run = Run(
            run_id=new_ulid(),
            name=self.name,
            entrypoint=self.entrypoint,
            input=self.input,
            status="running",
            started_at=utcnow(),
            code_version=git_version(),
            env={**_env(), **({"vars": self.env_vars} if self.env_vars else {})},
            tags=list(self.tags),
            span_id=new_span_id(),
        )
        store_dir = self.store_dir
        if replayer is not None:
            replayer.prepare_run(run)  # replay_of / forked_from, fork_step, patch, tags
            replayer.replay_run_id = run.run_id
            store_dir = store_dir or replayer.store_dir
        try:
            store = Store(store_dir)
            store.begin_run(run)
        except (OSError, sqlite3.Error) as err:
            if replayer is not None:
                raise
            # recording must never take the agent down: an unusable store (a file, a
            # read-only dir, a corrupt index) means this run is simply not recorded
            warnings.warn(
                f"crashlab: cannot record run {self.name!r} in "
                f"{store_dir or os.environ.get('CRASHLAB_DIR') or '.crashlab'}: "
                f"{type(err).__name__}: {err}; the agent runs unrecorded",
                RuntimeWarning,
                stacklevel=3,
            )
            self.ctx = None
            return run
        self.ctx = RunContext(
            run=run,
            store=store,
            mode=replayer.mode if replayer else self.mode,  # type: ignore[arg-type]
            replayer=replayer,
        )
        self._setup_checks(run, store, replayer)
        self._token = set_current(self.ctx)
        register_live(self.ctx)
        self._t0 = time.perf_counter()
        return run

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.ctx is None and self._token is None:  # store unusable: not recorded
            return False
        assert self.ctx is not None and self._token is not None
        run = self.ctx.run
        try:
            self.ctx.finish_open_streams()
            run.ended_at = utcnow()
            run.usage.wall_time_ms = round((time.perf_counter() - self._t0) * 1000, 3)
            user_outcome = run.outcome != "unknown"
            tripped = self.ctx.tripped
            if self.ctx.checks is not None:  # end-of-run checks can no longer stop anything
                self.ctx.circuit_breaker = False
                self.ctx.add_findings(self.ctx.checks.on_end(run))
            if tripped is not None and exc is None:  # the agent swallowed the stop
                from ..checks.engine import CircuitBreakerError

                exc = CircuitBreakerError(tripped)
                exc_type = type(exc)
            if self.ctx.replayer is not None:
                rp = self.ctx.replayer
                divergence = rp.finalize(run, exc)
                if divergence is not None:  # also when user code swallowed the error
                    exc = divergence if exc is None else exc
                    exc_type = type(exc)
                    run.outcome_reason = f"{rp.mode} diverged: {divergence.summary}"[:500]
                for flag in rp.flags:
                    self.ctx.add_flag(flag)
            if exc is not None:
                run.status = "error"
                run.error = error_info(exc)
                if tripped is not None and run.outcome == "unknown":
                    run.outcome = "fail"
                    run.outcome_reason = f"stopped by crashlab: [{tripped.kind}] {tripped.message}"[
                        :500
                    ]
                if run.outcome == "unknown":
                    run.outcome = "fail"
                    run.outcome_reason = run.outcome_reason or f"exception: {exc_type.__name__}"
            else:
                run.status = "ok"
            self._run_checks(run, keep_outcome=user_outcome)
            if self.ctx.store_failed:  # still try to close the run, marked incomplete
                run.tags.append("recording-incomplete")
                with contextlib.suppress(OSError, sqlite3.Error):
                    self.ctx.store.end_run(run)
            else:
                self.ctx.write(self.ctx.store.end_run, run)
        finally:
            unregister_live(self.ctx)
            reset_current(self._token)
            self._token = None
        return False

    def _setup_checks(self, run: Run, store: Store, replayer: Any) -> None:
        from ..checks.config import resolve_config
        from ..checks.engine import CheckEngine, store_baseline

        source = getattr(replayer, "source", None)
        base = source.check_config if source is not None else None
        try:
            cfg = resolve_config(self.checks, base=base)
        except Exception as exc:
            raise ValueError(f"crashlab.record(): bad checks= config: {exc}") from exc
        run.check_config = cfg.model_dump(mode="json")
        assert self.ctx is not None
        if not cfg.enabled:
            return
        limit = cfg.suspicious_tool_result.baseline_runs
        baseline = store_baseline(store, run.name, run.run_id, limit) if limit else None
        self.ctx.checks = CheckEngine(cfg, run, baseline=baseline)
        if replayer is not None and replayer.mode == "replay":
            # a strict replay reproduces the recording: stop exactly where it was stopped
            stopped = next((f for f in source.findings if f.stopped_run), None)
            if stopped is not None and stopped.step is not None:
                self.ctx.breaker_replay = (max(stopped.steps or [stopped.step]), stopped)
        else:
            self.ctx.circuit_breaker = cfg.circuit_breaker

    def _run_checks(self, run: Run, *, keep_outcome: bool) -> None:
        from ..verify.oracles import apply_outcome, checks_for, run_checks

        specs = checks_for(run.name)
        if not specs:
            return
        assert self.ctx is not None
        run.steps = sorted(self.ctx.steps, key=lambda st: st.idx)
        results = run_checks(run, specs)
        if keep_outcome:
            run.checks = results
        else:
            apply_outcome(run, results)

    # async -----------------------------------------------------------------------------
    async def __aenter__(self) -> Run:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return self.__exit__(exc_type, exc, tb)


GENERATOR_CLOSED_TAG = "generator-closed"


def _generator_agent(
    f: Callable[..., Any],
    agent_name: str,
    ep: str,
    ignore_args: tuple[str, ...],
    record_kw: dict[str, Any],
) -> Callable[..., Any]:
    """A generator / async generator entry function (streaming UIs: FastAPI, Gradio ...):
    the run spans the whole iteration and records the yielded items as ``output``.

    The run is bound only while the generator body executes (each step runs in a private
    copy of the caller's context), so code the consumer runs between items is not
    attributed to it. A consumer that stops early closes the run as ``ok`` with the items
    it took and the tag ``generator-closed``; replay then takes the same number.
    """
    is_async = inspect.isasyncgenfunction(f)

    def start(args: tuple, kwargs: dict) -> tuple[Any, Any, contextvars.Context | None]:
        if bound_current() is not None:  # nested: steps go to the outer run directly
            return None, f(*args, **kwargs), None
        cv = contextvars.copy_context()
        inp = bind_arguments(f, args, kwargs, ignore_args)
        rec = record(agent_name, input=inp, entrypoint=ep, **record_kw)
        cv.run(rec.__enter__)
        return rec, cv.run(f, *args, **kwargs), cv

    def finish(rec: Any, cv: Any, items: list, exc: BaseException | None) -> None:
        run = rec.ctx.run if rec.ctx is not None else None
        if run is not None:
            run.output = to_jsonable(items)
        if isinstance(exc, GeneratorExit):
            if run is not None:
                run.tags.append(GENERATOR_CLOSED_TAG)
            exc = None
        cv.run(rec.__exit__, type(exc) if exc else None, exc, exc.__traceback__ if exc else None)

    if not is_async:

        @functools.wraps(f)
        def gen_wrapper(*args: Any, **kwargs: Any) -> Any:
            rec, gen, cv = start(args, kwargs)
            if rec is None:
                return (yield from gen)
            items: list[Any] = []
            try:
                item = cv.run(next, gen)
                while True:
                    items.append(item)
                    try:
                        sent = yield item
                    except GeneratorExit:
                        cv.run(gen.close)
                        raise
                    except BaseException as exc:  # thrown in by the consumer
                        item = cv.run(gen.throw, exc)
                    else:
                        item = cv.run(gen.send, sent)
            except StopIteration as stop:
                finish(rec, cv, items, None)
                return stop.value
            except BaseException as exc:
                finish(rec, cv, items, exc)
                raise

        return gen_wrapper

    @functools.wraps(f)
    async def agen_wrapper(*args: Any, **kwargs: Any) -> Any:
        rec, gen, cv = start(args, kwargs)
        if rec is None:
            async for item in gen:
                yield item
            return
        items: list[Any] = []
        try:
            item = await _in_context(cv, gen.__anext__())
            while True:
                items.append(item)
                try:
                    sent = yield item
                except GeneratorExit:
                    await _in_context(cv, gen.aclose())
                    raise
                except BaseException as exc:  # thrown in by the consumer
                    item = await _in_context(cv, gen.athrow(exc))
                else:
                    item = await _in_context(cv, gen.asend(sent))
        except StopAsyncIteration:
            finish(rec, cv, items, None)
        except BaseException as exc:
            finish(rec, cv, items, exc)
            raise

    return agen_wrapper


async def _in_context(cv: contextvars.Context, aw: Any) -> Any:
    """Await ``aw`` (one async generator step) with ``cv`` as its context (on 3.10 a task
    gets a copy of it, which still carries the run)."""
    import asyncio

    if sys.version_info >= (3, 11):
        task = asyncio.get_running_loop().create_task(_await(aw), context=cv)
    else:  # pragma: no cover - 3.10
        task = cv.run(asyncio.ensure_future, _await(aw))
    return await task


async def _await(aw: Any) -> Any:
    return await aw


@overload
def agent(fn: F, /) -> F: ...
@overload
def agent(
    *,
    name: str | None = None,
    tags: list[str] | None = None,
    ignore_args: tuple[str, ...] = (),
    checks: Any = None,
    env: list[str] | tuple[str, ...] | None = None,
) -> Callable[[F], F]: ...


def agent(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    tags: list[str] | None = None,
    ignore_args: tuple[str, ...] = (),
    checks: Any = None,
    env: list[str] | tuple[str, ...] | None = None,
) -> Any:
    """Mark an agent entry function: each top-level call becomes one recorded run.

    Records the entry point path (``module:qualname``), the bound input arguments, the
    return value as ``output``, status, usage totals and the git sha. Arguments listed in
    ``ignore_args`` (e.g. a client object) are left out of the recorded input.
    ``checks`` configures the runtime checks and ``env`` the recorded environment
    variables (model name, streaming flag ...), as for :class:`record`.

    A call made while another run is active is recorded as an ``invoke_agent`` step of
    the outer run instead of a separate run.
    """

    def decorate(f: Callable[..., Any]) -> Callable[..., Any]:
        agent_name = name or f.__name__
        ep = entrypoint_of(f)

        def _nested(args: tuple, kwargs: dict):
            pending = begin_step(
                "invoke_agent",
                agent_name,
                {"args": bind_arguments(f, args, kwargs, ignore_args)},
                attributes={"gen_ai.agent.name": agent_name},
            )
            if pending is not None:
                pending.replay_source()  # replay: must match the recording; body re-runs
            return pending

        if inspect.isgeneratorfunction(f) or inspect.isasyncgenfunction(f):
            wrapper: Callable[..., Any] = _generator_agent(
                f, agent_name, ep, ignore_args, dict(tags=tags, checks=checks, env=env)
            )
        elif inspect.iscoroutinefunction(f):

            @functools.wraps(f)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if bound_current() is not None:
                    pending = _nested(args, kwargs)
                    try:
                        with pending.activate():  # type: ignore[union-attr]
                            out = await f(*args, **kwargs)
                    except BaseException as exc:
                        pending.fail(exc)  # type: ignore[union-attr]
                        raise
                    pending.finish({"result": to_jsonable(out)})  # type: ignore[union-attr]
                    return out
                inp = bind_arguments(f, args, kwargs, ignore_args)
                with record(agent_name, input=inp, entrypoint=ep, tags=tags, checks=checks,
                            env=env) as run:  # fmt: skip
                    out = await f(*args, **kwargs)
                    run.output = to_jsonable(out)
                    return out

            wrapper = async_wrapper
        else:

            @functools.wraps(f)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                if bound_current() is not None:
                    pending = _nested(args, kwargs)
                    try:
                        with pending.activate():  # type: ignore[union-attr]
                            out = f(*args, **kwargs)
                    except BaseException as exc:
                        pending.fail(exc)  # type: ignore[union-attr]
                        raise
                    pending.finish({"result": to_jsonable(out)})  # type: ignore[union-attr]
                    return out
                inp = bind_arguments(f, args, kwargs, ignore_args)
                with record(agent_name, input=inp, entrypoint=ep, tags=tags, checks=checks,
                            env=env) as run:  # fmt: skip
                    out = f(*args, **kwargs)
                    run.output = to_jsonable(out)
                    return out

            wrapper = sync_wrapper
        wrapper.__crashlab_agent__ = agent_name  # type: ignore[attr-defined]
        return wrapper

    if fn is not None:
        return decorate(fn)
    return decorate
