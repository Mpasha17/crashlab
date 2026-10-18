"""Run boundaries: ``@crashlab.agent`` and ``crashlab.record(...)``."""

from __future__ import annotations

import functools
import inspect
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Any, TypeVar, overload

from .._ids import new_span_id, new_ulid
from .._version import __version__
from ..context import (
    RunContext,
    check_mode,
    current,
    reset_current,
    set_current,
    take_pending_replay,
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
    ) -> None:
        self.name = name
        self.checks = checks
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
        if current() is not None:
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
            env=_env(),
            tags=list(self.tags),
            span_id=new_span_id(),
        )
        store_dir = self.store_dir
        if replayer is not None:
            replayer.prepare_run(run)  # replay_of / forked_from, fork_step, patch, tags
            replayer.replay_run_id = run.run_id
            store_dir = store_dir or replayer.store_dir
        store = Store(store_dir)
        store.begin_run(run)
        self.ctx = RunContext(
            run=run,
            store=store,
            mode=replayer.mode if replayer else self.mode,  # type: ignore[arg-type]
            replayer=replayer,
        )
        self._setup_checks(run, store, replayer)
        self._token = set_current(self.ctx)
        self._t0 = time.perf_counter()
        return run

    def __exit__(self, exc_type, exc, tb) -> bool:
        assert self.ctx is not None and self._token is not None
        run = self.ctx.run
        try:
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
            self.ctx.store.end_run(run)
        finally:
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


@overload
def agent(fn: F, /) -> F: ...
@overload
def agent(
    *,
    name: str | None = None,
    tags: list[str] | None = None,
    ignore_args: tuple[str, ...] = (),
    checks: Any = None,
) -> Callable[[F], F]: ...


def agent(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    tags: list[str] | None = None,
    ignore_args: tuple[str, ...] = (),
    checks: Any = None,
) -> Any:
    """Mark an agent entry function: each top-level call becomes one recorded run.

    Records the entry point path (``module:qualname``), the bound input arguments, the
    return value as ``output``, status, usage totals and the git sha. Arguments listed in
    ``ignore_args`` (e.g. a client object) are left out of the recorded input.
    ``checks`` configures the runtime checks, as for :class:`record`.

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

        if inspect.iscoroutinefunction(f):

            @functools.wraps(f)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if current() is not None:
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
                with record(agent_name, input=inp, entrypoint=ep, tags=tags, checks=checks) as run:
                    out = await f(*args, **kwargs)
                    run.output = to_jsonable(out)
                    return out

            wrapper: Callable[..., Any] = async_wrapper
        else:

            @functools.wraps(f)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                if current() is not None:
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
                with record(agent_name, input=inp, entrypoint=ep, tags=tags, checks=checks) as run:
                    out = f(*args, **kwargs)
                    run.output = to_jsonable(out)
                    return out

            wrapper = sync_wrapper
        wrapper.__crashlab_agent__ = agent_name  # type: ignore[attr-defined]
        return wrapper

    if fn is not None:
        return decorate(fn)
    return decorate
