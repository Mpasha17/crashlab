"""Strict offline replay and fork-at-step-N.

``crashlab.replay(run)`` re-invokes the recorded entry point with the recorded input. Every
LLM call (through the single capture point, :func:`crashlab.capture._chat.invoke`) and
every ``@crashlab.tool`` call is served from the source run's cassette instead of running:

- a call matches when it is the next step by index, with the same operation and name,
  and its normalized request hash equals the recorded one;
- anything else raises :class:`DivergenceError` with a field-level diff;
- tools never run, so ``side_effects=True`` tools cannot fire;
- network connections are blocked and SDK clients get placeholder credentials, so a
  replay needs no API key and no network.

The replay is stored as a new run with ``replay_of`` set to the source run id.

``crashlab.fork(run, at_step=N, patch=...)`` uses the same machinery with a
:class:`Forker` policy: strict replay below N, patched live calls from N on.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import importlib.util
import inspect
import os
import socket
import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Literal

from ..context import pending_replay, run_in_thread
from ..models import Flag, Run, Step
from ..patch import PatchError, PatchSet, as_patchset, preview_patches
from ..store import Store
from ..store.redact import get_redactor
from .divergence import (
    DivergenceError,
    ForkError,
    ReplayError,
    ReplayNetworkError,
    diff_values,
)

PLACEHOLDER_KEY = "crashlab-replay-offline"
# SDK constructors that refuse to build a client without a key; nothing is ever sent.
PLACEHOLDER_ENV = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")


def _describe(step: Step) -> str:
    return f"{step.operation} {step.name!r}"


class Replayer:
    """Serves one source run's steps, in order, and records the first divergence.

    The capture layer talks to it through a small policy interface, which
    :class:`Forker` overrides: :meth:`source_for` (the recorded step a call must
    reproduce, or None to run live), :meth:`tool_decision`, :meth:`patch_chat` and
    :meth:`prepare_run`.
    """

    mode = "replay"
    edits_requests = False  # True when patch_chat() may change outgoing requests

    def __init__(self, source: Run, *, store_dir: str | os.PathLike[str] | None = None) -> None:
        self.source = source
        self.store_dir = store_dir
        self.steps: dict[int, Step] = {s.idx: s for s in source.steps}
        self.consumed: set[int] = set()
        self.divergence: DivergenceError | None = None
        self.replay_run_id: str | None = None
        self.flags: list[Flag] = []  # added to the new run when it ends
        self._lock = threading.Lock()

    # policy ----------------------------------------------------------------------------
    def prepare_run(self, run: Run) -> None:
        run.replay_of = self.source.run_id
        run.tags.append("replay")

    def source_for(self, pending: Any) -> Step | None:
        """The recorded step this call must reproduce (strict: always one, or raise)."""
        return self.match(
            pending.idx, pending.operation, pending.name, pending.request, pending.request_hash
        )

    def tool_decision(self, pending: Any, side_effects: bool) -> tuple[str, Any]:
        """``("cassette", step)`` | ``("patched", ToolResultPatch)`` | ``("live", None)``
        | ``("blocked", reason)``. Strict replay never runs a tool."""
        return "cassette", pending.replay_source()

    def patch_chat(
        self, idx: int, api: str, kwargs: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        return kwargs, []

    # matching ------------------------------------------------------------------------
    def _diverge(self, kind: str, summary: str, **kw: Any) -> DivergenceError:
        err = DivergenceError(
            kind,  # type: ignore[arg-type]
            summary,
            source_run_id=self.source.run_id,
            replay_run_id=self.replay_run_id,
            **kw,
        )
        with self._lock:
            if self.divergence is None:
                self.divergence = err
        return err

    def match(
        self, idx: int, operation: str, name: str, request: dict[str, Any], request_hash: str
    ) -> Step:
        """Return the recorded step for this call or raise :class:`DivergenceError`."""
        if self.divergence is not None:  # sticky: user code may have swallowed it
            raise self.divergence
        src = self.steps.get(idx)
        here = f"step {idx} ({operation} {name!r})"
        if src is None:
            raise self._diverge(
                "extra_step",
                f"replay made {here}, but the recording has only {len(self.steps)} steps",
                step_idx=idx,
                actual=request,
            )
        if src.operation != operation or src.name != name:
            raise self._diverge(
                "step",
                f"at step {idx} the recording has {_describe(src)}, replay made "
                f"{operation} {name!r}",
                step_idx=idx,
                expected={"operation": src.operation, "name": src.name},
                actual={"operation": operation, "name": name},
                fields=["operation"] if src.operation != operation else ["name"],
            )
        if src.request_hash != request_hash:
            actual = get_redactor().redact(request)
            fields, diff = diff_values(
                src.request, actual, fromfile=f"recorded step {idx}", tofile=f"replay step {idx}"
            )
            raise self._diverge(
                "request",
                f"{here}: request differs from the recording",
                step_idx=idx,
                expected=src.request,
                actual=actual,
                fields=fields or ["request_hash"],
                diff=diff,
            )
        with self._lock:
            self.consumed.add(idx)
        return src

    def network_attempt(self, address: Any) -> ReplayNetworkError:
        self._diverge(
            "network",
            f"replay tried to open a network connection to {address!r}; calls that are not "
            "made through a crashlab-wrapped client cannot be replayed offline",
        )
        return ReplayNetworkError(f"crashlab replay is offline: connection to {address!r} blocked")

    # end of run ----------------------------------------------------------------------
    def finalize(self, run: Run, exc: BaseException | None) -> DivergenceError | None:
        """Called when the replayed run ends: compare completeness, exception and output."""
        if self.divergence is not None:
            return self.divergence
        src = self.source
        if exc is not None and not isinstance(exc, DivergenceError):
            recorded = src.error.type.rsplit(".", 1)[-1] if src.error else None
            if src.status != "error" or recorded != type(exc).__name__:
                return self._diverge(
                    "exception",
                    f"replay raised {type(exc).__name__}: {str(exc)[:200]} but the recording "
                    + (f"raised {src.error.type}" if src.error else "finished without error"),
                )
        elif exc is None and src.status == "error":
            return self._diverge(
                "exception",
                f"the recording raised {src.error.type if src.error else 'an error'}, "
                "the replay finished without error",
            )
        missing = sorted(set(self.steps) - self.consumed)
        # invoke_agent steps are re-executed, not served; they never count as missing
        missing = [i for i in missing if self.steps[i].operation in ("chat", "execute_tool")]
        if missing:
            listed = ", ".join(f"{i} ({_describe(self.steps[i])})" for i in missing[:5])
            more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
            return self._diverge(
                "missing_steps",
                f"replay finished but {len(missing)} recorded step(s) were never reached: "
                f"{listed}{more}",
                step_idx=missing[0],
            )
        if exc is None and run.output != src.output:
            fields, diff = diff_values(src.output, run.output, fromfile="recorded output",
                                       tofile="replay output")  # fmt: skip
            return self._diverge(
                "output",
                "every step matched, but the final output differs",
                expected=src.output,
                actual=run.output,
                fields=fields,
                diff=diff,
            )
        return None


ToolsMode = Literal["cassette_if_match", "live"]


def _numbers(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: _numbers(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_numbers(v) for v in value]
    return value


def _args_key(args: Any) -> str:
    import json

    redacted = get_redactor().redact(args or {})
    return json.dumps(_numbers(redacted), sort_keys=True, ensure_ascii=False, default=str)


class Forker(Replayer):
    """Fork policy: steps ``< at_step`` are a strict replay; from ``at_step`` on, LLM calls
    go live (to whatever client the agent uses) with ``patches`` applied, and tools follow
    ``tools``:

    - ``"cassette_if_match"`` (default): the recorded result when the parent made the same
      call (same tool and arguments), otherwise run live if the tool has no side effects;
    - ``"live"``: tools without side effects always run live.

    ``side_effects=True`` tools never run live unless ``allow_live_side_effects``: they
    get the recorded result of an identical parent call (each used once) or raise
    :class:`SideEffectBlockedError`.
    """

    mode = "fork"

    def __init__(
        self,
        source: Run,
        *,
        at_step: int,
        patches: PatchSet,
        tools: ToolsMode = "cassette_if_match",
        allow_live_side_effects: bool = False,
        store_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        super().__init__(source, store_dir=store_dir)
        self.at_step = at_step
        self.patches = patches
        self.tools = tools
        self.allow_live_side_effects = allow_live_side_effects
        self.edits_requests = patches.edits_requests
        self.applied: dict[int, int] = {}  # patch index -> times applied
        self._patch_state: dict[int, dict] = {}
        self.max_idx = -1

    def _seen(self, idx: int) -> None:
        with self._lock:
            self.max_idx = max(self.max_idx, idx)

    def prepare_run(self, run: Run) -> None:
        run.forked_from = self.source.run_id
        run.fork_step = self.at_step
        run.patch = self.patches
        run.patch_description = self.patches.describe()
        run.tags.append("fork")

    def source_for(self, pending: Any) -> Step | None:
        self._seen(pending.idx)
        if pending.idx < self.at_step:
            return super().source_for(pending)
        if self.divergence is not None:
            raise self.divergence
        return None

    def _cassette_match(self, pending: Any, side_effects: bool) -> Step | None:
        """A parent call of the same tool with equal arguments (``120 == 120.0``)."""
        key = _args_key(pending.request.get("args"))
        hits = [
            s
            for s in self.source.steps
            if s.operation == "execute_tool"
            and s.name == pending.name
            and (s.request_hash == pending.request_hash or _args_key(s.args) == key)
        ]
        with self._lock:
            fresh = [s for s in hits if s.idx not in self.consumed]
            pick = fresh[0] if fresh else (None if side_effects or not hits else hits[0])
            if pick is not None:
                self.consumed.add(pick.idx)
        return pick

    def tool_decision(self, pending: Any, side_effects: bool) -> tuple[str, Any]:
        self._seen(pending.idx)
        if pending.idx < self.at_step:
            return "cassette", pending.replay_source()
        if self.divergence is not None:
            raise self.divergence
        patch = self.patches.tool_result_for(pending.idx)
        if patch is not None:
            if patch.tool and patch.tool != pending.name:
                raise PatchError(
                    f"{patch.describe()}: step {pending.idx} of the fork calls {pending.name!r}"
                )
            i = self.patches.patches.index(patch)
            with self._lock:
                self.applied[i] = self.applied.get(i, 0) + 1
            return "patched", patch
        if side_effects and self.allow_live_side_effects:
            return "live", None
        if side_effects or self.tools == "cassette_if_match":
            src = self._cassette_match(pending, side_effects)
            if src is not None:
                return "cassette", src
        if not side_effects:
            return "live", None
        return "blocked", (
            f"tool {pending.name!r} has side effects and crashlab fork never runs it live; "
            "the parent run made no identical call to serve instead (pass "
            "allow_live_side_effects=True / --allow-live-side-effects for a sandbox)"
        )

    def patch_chat(
        self, idx: int, api: str, kwargs: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        self._seen(idx)
        if idx < self.at_step:
            return kwargs, []
        new, applied = self.patches.apply_chat(
            api, idx, kwargs, fork_step=self.at_step, state=self._patch_state
        )
        with self._lock:
            for i in applied:
                self.applied[i] = self.applied.get(i, 0) + 1
        return new, [self.patches.patches[i].describe() for i in applied]

    def finalize(self, run: Run, exc: BaseException | None) -> DivergenceError | None:
        """Only the replayed prefix must match; what happens from the fork step on is the
        experiment. Unreached fork points and unused patches become run flags."""
        if self.divergence is not None:
            return self.divergence
        if self.max_idx < self.at_step:
            self.flags.append(Flag(
                detector="fork", step_idx=self.at_step, severity="warn",
                message=f"the fork never reached step {self.at_step}; nothing was patched",
            ))  # fmt: skip
        for i, p in enumerate(self.patches.patches):
            if not self.applied.get(i):
                self.flags.append(Flag(
                    detector="fork", step_idx=p.target_step(), severity="warn",
                    message=f"patch {i + 1} was never applied: {p.describe()}",
                ))  # fmt: skip
        return None


# --- guards --------------------------------------------------------------------------

_guard_lock = threading.Lock()


@contextlib.contextmanager
def block_network(replayer: Replayer) -> Iterator[None]:
    """Refuse outbound socket connections (AF_UNIX stays allowed) while replaying."""
    orig_connect = socket.socket.connect
    orig_connect_ex = socket.socket.connect_ex

    def _blocked(sock: socket.socket, address: Any) -> bool:
        return getattr(socket, "AF_UNIX", None) is None or sock.family != socket.AF_UNIX

    def connect(sock: socket.socket, address: Any) -> None:
        if _blocked(sock, address):
            raise replayer.network_attempt(address)
        return orig_connect(sock, address)

    def connect_ex(sock: socket.socket, address: Any) -> int:
        if _blocked(sock, address):
            raise replayer.network_attempt(address)
        return orig_connect_ex(sock, address)

    with _guard_lock:
        socket.socket.connect = connect  # type: ignore[method-assign]
        socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]
    try:
        yield
    finally:
        with _guard_lock:
            socket.socket.connect = orig_connect  # type: ignore[method-assign]
            socket.socket.connect_ex = orig_connect_ex  # type: ignore[method-assign]


@contextlib.contextmanager
def placeholder_credentials(names: tuple[str, ...] = PLACEHOLDER_ENV) -> Iterator[None]:
    """Set unset SDK key variables to a placeholder so clients can be constructed."""
    added = [n for n in names if not os.environ.get(n)]
    for n in added:
        os.environ[n] = PLACEHOLDER_KEY
    try:
        yield
    finally:
        for n in added:
            if os.environ.get(n) == PLACEHOLDER_KEY:
                del os.environ[n]


# --- entry points --------------------------------------------------------------------


def resolve_entrypoint(entrypoint: str | None) -> Callable[..., Any]:
    """``module:qualname`` or ``path/to/file.py:qualname`` -> the callable."""
    if not entrypoint or ":" not in entrypoint:
        raise ReplayError(
            "this run has no entry point (it was recorded with `crashlab.record(...)`); "
            "pass the function to re-run: crashlab.replay(run, fn=my_main)"
        )
    target, qualname = entrypoint.rsplit(":", 1)
    if os.getcwd() not in sys.path and "" not in sys.path:
        sys.path.insert(0, os.getcwd())
    if target.endswith(".py") or os.sep in target:
        path = Path(target).resolve()
        if not path.exists():
            raise ReplayError(f"entry point file {target!r} not found (cwd: {os.getcwd()})")
        spec = importlib.util.spec_from_file_location(f"crashlab_replay_{path.stem}", path)
        if spec is None or spec.loader is None:
            raise ReplayError(f"cannot import {target!r}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    else:
        try:
            module = importlib.import_module(target)
        except ImportError as exc:
            raise ReplayError(
                f"cannot import entry point module {target!r} ({exc}); run from the "
                "directory you recorded in, or pass fn="
            ) from exc
    obj: Any = module
    for part in qualname.split("."):
        if part == "<locals>":
            raise ReplayError(f"entry point {entrypoint!r} is a local function; pass fn=")
        obj = getattr(obj, part, None)
        if obj is None:
            raise ReplayError(f"entry point {entrypoint!r} not found")
    return obj


def _call_kwargs(fn: Callable[..., Any], data: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    """Recorded input -> call arguments (handles positional-only and *args parameters)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return [], dict(data)
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    rest = dict(data)
    for name, param in sig.parameters.items():
        if name not in rest:
            continue
        value = rest.pop(name)
        if param.kind is inspect.Parameter.POSITIONAL_ONLY:
            args.append(value)
        elif param.kind is inspect.Parameter.VAR_POSITIONAL:
            args.extend(value or [])
        else:
            kwargs[name] = value
    kwargs.update(rest)  # **kwargs parameters were flattened into the input
    return args, kwargs


def _run_callable(fn: Callable[..., Any], args: list[Any], kwargs: dict[str, Any]) -> Any:
    out = fn(*args, **kwargs)
    if not inspect.isawaitable(out):
        return out
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await(out))
    # called from async code (Jupyter, async tests): use a private loop in a worker thread
    # that carries this context (the pending replay)
    return run_in_thread(asyncio.run, _await(out))


async def _await(aw: Any) -> Any:
    return await aw


# --- public API ----------------------------------------------------------------------


def replay(
    run: Run | str = "last",
    *,
    fn: Callable[..., Any] | None = None,
    overrides: dict[str, Any] | None = None,
    store_dir: str | os.PathLike[str] | None = None,
    offline: bool = True,
) -> Run:
    """Re-execute a recorded run with every LLM and tool call served from its cassette.

    - ``run``: a :class:`Run`, a run id / unique prefix, or ``"last"``.
    - ``fn``: the function to call instead of the recorded entry point. Required for
      runs recorded with ``crashlab.record(...)``; it must open the same ``record`` block.
    - ``overrides``: extra / replacement arguments for the entry point call (e.g. a
      ``client`` that was excluded from the recorded input with ``ignore_args``).
    - ``offline``: block network connections and provide placeholder SDK credentials.

    Returns the new replay run (``replay_of`` = source id). Raises
    :class:`DivergenceError` if the run does not reproduce exactly; the diverged replay is
    still stored with ``status="error"`` so ``crashlab show`` can display it.
    """
    store = Store(store_dir)
    source = _load_source(store, run)
    target, args, kwargs = _target(source, fn, overrides)
    replayer = Replayer(source, store_dir=store.root)
    error = _execute(replayer, target, args, kwargs, offline=offline)
    if replayer.divergence is not None:
        raise replayer.divergence from (
            error if error is not None and error is not replayer.divergence else None
        )
    if error is not None and not (
        source.status == "error" and source.error
        and source.error.type.rsplit(".", 1)[-1] == type(error).__name__
    ):  # fmt: skip
        raise error
    return store.load_run(replayer.replay_run_id)  # type: ignore[arg-type]


def _load_source(store: Store, run: Run | str) -> Run:
    source = store.load_run(run.run_id if isinstance(run, Run) else run)
    if not source.steps and not store.cassette_path(source.run_id).exists():
        raise ReplayError(f"run {source.run_id} has no cassette at {store.root}")
    return source


def _target(
    source: Run, fn: Callable[..., Any] | None, overrides: dict[str, Any] | None
) -> tuple[Callable[..., Any], list[Any], dict[str, Any]]:
    target = fn if fn is not None else resolve_entrypoint(source.entrypoint)
    is_agent = getattr(target, "__crashlab_agent__", None) is not None
    data = dict(source.input) if (is_agent or fn is None) else {}
    data.update(overrides or {})
    args, kwargs = _call_kwargs(target, data)
    return target, args, kwargs


def _execute(
    replayer: Replayer,
    target: Callable[..., Any],
    args: list[Any],
    kwargs: dict[str, Any],
    *,
    offline: bool,
) -> BaseException | None:
    """Run the entry point under ``replayer``; return the exception it raised, if any."""
    guards = contextlib.ExitStack()
    if offline:
        guards.enter_context(placeholder_credentials())
        guards.enter_context(block_network(replayer))
    error: BaseException | None = None
    with guards, pending_replay(replayer):
        try:
            _run_callable(target, args, kwargs)
        except BaseException as exc:  # compared with the recording by the caller
            error = exc
    if replayer.replay_run_id is None:
        raise ReplayError(
            f"{getattr(target, '__qualname__', target)!r} did not start a crashlab run; "
            f"{replayer.mode} needs an @crashlab.agent function or a `crashlab.record(...)` "
            "block"
        ) from error
    return error


def fork(
    run: Run | str = "last",
    *,
    at_step: int,
    patch: Any = None,
    fn: Callable[..., Any] | None = None,
    overrides: dict[str, Any] | None = None,
    client: Any = None,
    tools: ToolsMode = "cassette_if_match",
    allow_live_side_effects: bool = False,
    store_dir: str | os.PathLike[str] | None = None,
    raise_errors: bool = False,
) -> Run:
    """Re-run a recorded run up to step ``at_step`` from its cassette, then continue live
    with ``patch`` applied: the "what if" experiment behind diagnosis and verification.

    - Steps ``< at_step`` are a strict replay (same requests or :class:`DivergenceError`).
    - From ``at_step`` on, LLM calls go to the agent's real client (or ``client`` /
      ``overrides={"client": ...}`` when the entry point takes one), with every patch
      applied to the outgoing request; tools follow ``tools`` (see :class:`Forker`) and
      side-effect tools never run live unless ``allow_live_side_effects=True``.
    - ``patch``: a patch, a list of patches, a :class:`~crashlab.patch.PatchSet`, a dict
      or JSON text. Patches are checked against the recorded run first
      (:class:`~crashlab.patch.PatchError` if one cannot apply).

    Returns the new run (``forked_from``, ``fork_step``, ``patch``), graded by the run's
    ``@crashlab.check`` oracles. An exception raised by the agent after the fork step is
    the experiment's result: it is stored on the run (``status="error"``) and only
    re-raised with ``raise_errors=True``.
    """
    store = Store(store_dir)
    source = _load_source(store, run)
    patches = as_patchset(patch)
    n = len(source.steps)
    if not isinstance(at_step, int) or isinstance(at_step, bool):
        raise ForkError(f"at_step must be an int, not {at_step!r}")
    if at_step < 0:
        at_step += n
    if not 0 <= at_step < max(n, 1):
        raise ForkError(f"at_step {at_step} is out of range: run {source.run_id} has {n} steps")
    early = [s for s in patches.bound_steps() if s < at_step]
    if early:
        raise ForkError(
            f"patches target step(s) {early} before the fork step {at_step}; steps before "
            "the fork are replayed from the cassette and cannot be patched"
        )
    preview_patches(patches, source, fork_step=at_step)  # PatchError early, before any call
    data = dict(overrides or {})
    if client is not None:
        data["client"] = client
    target, args, kwargs = _target(source, fn, data)
    forker = Forker(
        source,
        at_step=at_step,
        patches=patches,
        tools=tools,
        allow_live_side_effects=allow_live_side_effects,
        store_dir=store.root,
    )
    error = _execute(forker, target, args, kwargs, offline=False)
    if forker.divergence is not None:
        raise forker.divergence from (
            error if error is not None and error is not forker.divergence else None
        )
    if error is not None and (raise_errors or isinstance(error, (PatchError, KeyboardInterrupt))):
        raise error
    return store.load_run(forker.replay_run_id)  # type: ignore[arg-type]
