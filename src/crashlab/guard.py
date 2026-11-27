"""Live guardrails: turn crashlab's detectors into runtime enforcement.

``crashlab.guard(...)`` wraps a block (or decorates a function) and checks every LLM
and ``@crashlab.tool`` call made inside it *before* the call runs::

    with crashlab.guard(max_cost_usd=0.50, max_steps=40, loop_breaker=3,
                        tool_denylist=["delete_*"], validate_tool_args="repair",
                        retry_policy=True, fallback_model="gpt-4o-mini",
                        block_secrets_in_output=True):
        run_agent("Refund order A17")

Circuit breakers (``max_cost_usd``, ``max_tokens``, ``max_steps``, ``max_llm_calls``,
``max_tool_calls``, ``max_seconds``) stop the next call once a limit is reached; the
loop breaker stops a tool call that already ran N times with the same arguments, or the
next LLM call after N identical replies; tool allow / deny lists and argument
validation run before the tool. ``retry_policy`` retries transient LLM errors (429,
5xx, timeouts) with exponential backoff and jitter, ``fallback_model`` switches model
after repeated failures, and ``block_secrets_in_output`` redacts API keys / tokens
from LLM replies and tool results before your code sees them.

A violation raises :class:`GuardViolation` (``on_violation="raise"``), only warns
(``"warn"``), or raises and stops the whole recorded run (``"stop"``: every later call
raises ``CircuitBreakerError``). Each intervention is recorded on the run as a
``guard.<policy>`` finding and on the step's ``crashlab.guard`` attribute, and
violations go to optional alert sinks. ``crashlab guard-report RUN`` lists them.

Guards are stackable (``contextvars``: thread- and asyncio-safe) and cost one
``ContextVar`` read per call when none is active. Strict :func:`~crashlab.replay`
serves every call from the cassette: it never re-checks, retries or falls back, and the
recorded ``crashlab.guard`` metadata (and any recorded :class:`GuardViolation`) are
reproduced exactly.
"""

from __future__ import annotations

import asyncio
import contextvars
import fnmatch
import functools
import inspect
import json
import random
import re
import threading
import time
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any, Literal

from .context import GUARD_STACK, current
from .faults import FAULT_ATTR, take_llm_fault
from .models import Finding

GUARD_ATTR = "crashlab.guard"  # list of interventions on a step
BLOCKED_ATTR = "crashlab.guard.blocked"  # the call never ran (blocked or repaired)
ATTEMPTS_ATTR = "crashlab.guard.attempts"  # LLM attempts made under retry_policy
FALLBACK_ATTR = "crashlab.guard.fallback_model"  # the model that finally answered

OnViolation = Literal["raise", "warn", "stop"]
_ON_VIOLATION = ("raise", "warn", "stop")
POLICIES = (
    "max_cost_usd",
    "max_tokens",
    "max_steps",
    "max_llm_calls",
    "max_tool_calls",
    "max_seconds",
    "loop_breaker",
    "tool_allowlist",
    "tool_denylist",
    "validate_tool_args",
    "retry_policy",
    "fallback_model",
    "block_secrets_in_output",
)

# patched in tests; real sleeps otherwise
_sleep: Callable[[float], Any] = time.sleep


async def _async_sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


class GuardViolation(Exception):
    """A guard policy stopped a call. ``policy`` is the policy name (``max_steps``,
    ``tool_denylist`` ...), ``step_idx`` the blocked step (None outside a recorded run),
    ``detail`` a JSON-able dict with the numbers behind it."""

    def __init__(
        self,
        message: str = "",
        *,
        policy: str | None = None,
        step_idx: int | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.policy = policy
        self.step_idx = step_idx
        self.detail = detail or {}


class GuardWarning(UserWarning):
    """Emitted for a violation when ``on_violation="warn"``."""


# --- configuration -----------------------------------------------------------------------


_DEFAULT_RETRY_ON = (408, 409, 429, 500, 502, 503, 504, 529)


@dataclass(frozen=True)
class RetryPolicy:
    """Retry transient LLM errors: HTTP ``retry_on`` statuses, timeouts and connection
    errors. Delay before retry *n* is ``base_delay * 2**(n-1)`` capped at ``max_delay``;
    ``jitter`` draws it uniformly from ``[delay/2, delay]`` (``seed`` makes that
    deterministic). A numeric ``Retry-After`` header is honoured up to ``max_delay``."""

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    jitter: bool = True
    retry_on: tuple[int, ...] = _DEFAULT_RETRY_ON
    seed: int | None = None
    respect_retry_after: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.max_attempts, int) or self.max_attempts < 1:
            raise ValueError("retry_policy.max_attempts must be an int >= 1")
        if self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("retry_policy delays must be >= 0")
        object.__setattr__(self, "retry_on", tuple(int(s) for s in self.retry_on))

    @classmethod
    def coerce(cls, value: Any) -> RetryPolicy | None:
        if value is None or value is False:
            return None
        if value is True:
            return cls()
        if isinstance(value, RetryPolicy):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return cls(max_attempts=value)
        if isinstance(value, Mapping):
            known = {f.name for f in fields(cls)}
            unknown = set(value) - known
            if unknown:
                raise ValueError(f"unknown retry_policy keys: {', '.join(sorted(unknown))}")
            data = dict(value)
            if "retry_on" in data:
                data["retry_on"] = tuple(data["retry_on"])
            return cls(**data)
        raise TypeError("retry_policy must be True, an int, a dict or a RetryPolicy")

    def delay(self, failure: int, rng: random.Random, exc: BaseException | None = None) -> float:
        """Seconds to wait after the ``failure``-th failed attempt (1-based)."""
        delay = min(self.max_delay, self.base_delay * (2 ** max(0, failure - 1)))
        if self.jitter and delay > 0:
            delay = rng.uniform(delay / 2, delay)
        if self.respect_retry_after and exc is not None:
            after = _retry_after(exc)
            if after is not None:
                delay = max(delay, min(after, self.max_delay))
        return round(delay, 4)


def _retry_after(exc: BaseException) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
        return float(value) if value is not None else None
    except (TypeError, ValueError, AttributeError):
        return None


def is_transient(exc: BaseException, retry_on: Iterable[int] = _DEFAULT_RETRY_ON) -> bool:
    """Whether an LLM error is worth retrying: an HTTP status in ``retry_on``, a timeout,
    or a connection / rate-limit / overload error (by exception type name)."""
    if not isinstance(exc, Exception) or isinstance(exc, GuardViolation):
        return False
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        return status in tuple(retry_on)
    if isinstance(exc, (TimeoutError, ConnectionError, asyncio.TimeoutError)):
        return True
    name = type(exc).__name__.lower()
    return any(
        k in name
        for k in ("timeout", "ratelimit", "connectionerror", "apiconnection", "overloaded",
                  "serviceunavailable", "internalserver")
    )  # fmt: skip


def _patterns(value: Any, key: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable) and all(isinstance(v, str) for v in value):
        return tuple(value)
    raise TypeError(f"{key} must be a tool name / glob or a list of them")


def _limit(value: Any, key: str, *, integer: bool = False) -> Any:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{key} must be a non-negative number")
    if integer and not float(value).is_integer():
        raise ValueError(f"{key} must be an integer")
    return int(value) if integer else float(value)


@dataclass(frozen=True)
class GuardConfig:
    """Effective policies of one :func:`guard`. Build it with :func:`guard` or
    :meth:`from_mapping` (the ``[tool.crashlab.guard]`` keys)."""

    max_cost_usd: float | None = None
    max_tokens: int | None = None
    max_steps: int | None = None
    max_llm_calls: int | None = None
    max_tool_calls: int | None = None
    max_seconds: float | None = None
    loop_breaker: int | None = None
    tool_allowlist: tuple[str, ...] | None = None
    tool_denylist: tuple[str, ...] = ()
    validate_tool_args: bool = False
    repair_tool_args: bool = False
    retry_policy: RetryPolicy | None = None
    fallback_model: str | None = None
    fallback_after: int | None = None
    block_secrets_in_output: bool = False
    redact_pii: bool = False
    on_violation: OnViolation = "raise"
    alerts: tuple[Any, ...] = ()
    name: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> GuardConfig:
        data = dict(data)
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown guard settings: {', '.join(sorted(unknown))}")
        loop = data.get("loop_breaker")
        if loop is True:
            loop = 3
        elif loop is False:
            loop = None
        loop = _limit(loop, "loop_breaker", integer=True)
        if loop is not None and loop < 1:
            raise ValueError("loop_breaker must be >= 1")
        validate = data.get("validate_tool_args", False)
        repair = bool(data.get("repair_tool_args", False))
        if validate == "repair":
            validate, repair = True, True
        elif isinstance(validate, Mapping):
            repair = bool(validate.get("repair", repair))
            validate = bool(validate.get("enabled", True))
        elif not isinstance(validate, bool):
            raise TypeError("validate_tool_args must be a bool, 'repair' or {'repair': bool}")
        on_violation = data.get("on_violation", "raise")
        if on_violation not in _ON_VIOLATION:
            raise ValueError("on_violation must be 'raise', 'warn' or 'stop'")
        fallback = data.get("fallback_model")
        if fallback is not None and (not isinstance(fallback, str) or not fallback):
            raise ValueError("fallback_model must be a model name")
        fb_after = _limit(data.get("fallback_after"), "fallback_after", integer=True)
        if fb_after is not None and fb_after < 1:
            raise ValueError("fallback_after must be >= 1")
        alerts = data.get("alerts") or ()
        if not isinstance(alerts, (list, tuple)):
            alerts = (alerts,)
        return cls(
            max_cost_usd=_limit(data.get("max_cost_usd"), "max_cost_usd"),
            max_tokens=_limit(data.get("max_tokens"), "max_tokens", integer=True),
            max_steps=_limit(data.get("max_steps"), "max_steps", integer=True),
            max_llm_calls=_limit(data.get("max_llm_calls"), "max_llm_calls", integer=True),
            max_tool_calls=_limit(data.get("max_tool_calls"), "max_tool_calls", integer=True),
            max_seconds=_limit(data.get("max_seconds"), "max_seconds"),
            loop_breaker=loop,
            tool_allowlist=_patterns(data.get("tool_allowlist"), "tool_allowlist"),
            tool_denylist=_patterns(data.get("tool_denylist"), "tool_denylist") or (),
            validate_tool_args=bool(validate or repair),
            repair_tool_args=repair,
            retry_policy=RetryPolicy.coerce(data.get("retry_policy")),
            fallback_model=fallback,
            fallback_after=fb_after,
            block_secrets_in_output=bool(data.get("block_secrets_in_output", False)),
            redact_pii=bool(data.get("redact_pii", False)),
            on_violation=on_violation,
            alerts=tuple(alerts),
            name=data.get("name"),
        )

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "alerts":
                value = [type(a).__name__ for a in value]
            elif isinstance(value, RetryPolicy):
                value = {g.name: getattr(value, g.name) for g in fields(value)}
            elif isinstance(value, tuple):
                value = list(value)
            out[f.name] = value
        return out


def load_config(
    config: Mapping[str, Any] | GuardConfig | None = None,
    *,
    pyproject: bool = True,
    **overrides: Any,
) -> GuardConfig:
    """``[tool.crashlab.guard]`` (when ``pyproject``) < ``config`` dict < keyword
    overrides (``None`` values are ignored)."""
    if isinstance(config, GuardConfig) and not overrides:
        return config
    data: dict[str, Any] = {}
    if pyproject:
        from .checks.config import pyproject_section

        data.update(pyproject_section("guard"))
    if isinstance(config, GuardConfig):
        data.update({k: v for k, v in config.__dict__.items() if v is not None})
    elif config is not None:
        if not isinstance(config, Mapping):
            raise TypeError("guard config must be a dict or a GuardConfig")
        data.update(config)
    data.update({k: v for k, v in overrides.items() if v is not None})
    return GuardConfig.from_mapping(data)


# --- per-scope state ---------------------------------------------------------------------


@dataclass
class _Violation:
    scope: GuardScope
    policy: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


def _args_key(name: str, args: Any) -> str:
    try:
        return name + ":" + json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return name + ":" + repr(args)


def _reply_signature(response: Any) -> str | None:
    """Text and tool calls (name + arguments) of an LLM reply, without ids."""
    if not isinstance(response, dict):
        return None
    texts: list[Any] = []
    for m in response.get("messages") or []:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts += [p.get("text") for p in content
                      if isinstance(p, dict) and p.get("type") == "text"]  # fmt: skip
    sig = {
        "text": texts,
        "tool_calls": [
            {"name": c.get("name"), "arguments": c.get("arguments")}
            for c in response.get("tool_calls") or []
            if isinstance(c, dict)
        ],
    }
    try:
        return json.dumps(sig, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return None


class GuardScope:
    """One entered :func:`guard`: its policies, counters and the interventions it made.
    ``with crashlab.guard(...) as scope:`` gives you this object."""

    def __init__(self, config: GuardConfig) -> None:
        self.config = config
        self.started = time.monotonic()
        self.lock = threading.Lock()
        self.steps = 0
        self.llm_calls = 0
        self.tool_calls = 0
        self.tokens = 0
        self.cost_usd = 0.0
        self.cost_unknown = False
        self.tool_counts: dict[str, int] = {}
        self.replies: list[str] = []
        self.interventions: list[dict[str, Any]] = []

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    # accounting (RunContext.add_step) -----------------------------------------------------
    def on_step(self, step: Any) -> None:
        attrs = step.attributes or {}
        if attrs.get(BLOCKED_ATTR):
            return
        with self.lock:
            self.steps += 1
            if step.operation == "chat":
                self.llm_calls += 1
                inp = int(attrs.get("gen_ai.usage.input_tokens") or 0)
                out = int(attrs.get("gen_ai.usage.output_tokens") or 0)
                total = attrs.get("crashlab.usage.total_tokens")
                self.tokens += int(total) if total is not None else inp + out
                cost = attrs.get("crashlab.usage.cost_usd")
                if cost is None:
                    self.cost_unknown = True
                else:
                    self.cost_usd += float(cost)
                sig = _reply_signature(step.response) if step.error is None else None
                if sig is not None:
                    self.replies.append(sig)
                    del self.replies[: -max(1, self.config.loop_breaker or 1)]
                else:
                    self.replies.clear()
            elif step.operation == "embeddings":
                self.tokens += int(attrs.get("crashlab.usage.total_tokens") or 0)
            elif step.operation == "execute_tool":
                self.tool_calls += 1
                key = _args_key(step.name, (step.request or {}).get("args"))
                self.tool_counts[key] = self.tool_counts.get(key, 0) + 1

    # pre-call checks -----------------------------------------------------------------------
    def _breakers(self, kind: str) -> list[_Violation]:
        c, out = self.config, []
        if c.max_steps is not None and self.steps >= c.max_steps:
            out.append(_Violation(
                self, "max_steps", f"{self.steps} steps reached max_steps={c.max_steps}",
                {"steps": self.steps, "limit": c.max_steps},
            ))  # fmt: skip
        if kind == "chat" and c.max_llm_calls is not None and self.llm_calls >= c.max_llm_calls:
            out.append(_Violation(
                self, "max_llm_calls",
                f"{self.llm_calls} LLM calls reached max_llm_calls={c.max_llm_calls}",
                {"llm_calls": self.llm_calls, "limit": c.max_llm_calls},
            ))  # fmt: skip
        if (kind == "execute_tool" and c.max_tool_calls is not None
                and self.tool_calls >= c.max_tool_calls):  # fmt: skip
            out.append(_Violation(
                self, "max_tool_calls",
                f"{self.tool_calls} tool calls reached max_tool_calls={c.max_tool_calls}",
                {"tool_calls": self.tool_calls, "limit": c.max_tool_calls},
            ))  # fmt: skip
        if c.max_tokens is not None and self.tokens >= c.max_tokens:
            out.append(_Violation(
                self, "max_tokens", f"{self.tokens} tokens reached max_tokens={c.max_tokens}",
                {"tokens": self.tokens, "limit": c.max_tokens},
            ))  # fmt: skip
        if c.max_cost_usd is not None and self.cost_usd >= c.max_cost_usd:
            out.append(_Violation(
                self, "max_cost_usd",
                f"${self.cost_usd:.6f} spent reached max_cost_usd=${c.max_cost_usd}",
                {"cost_usd": round(self.cost_usd, 8), "limit": c.max_cost_usd,
                 "cost_unknown": self.cost_unknown},
            ))  # fmt: skip
        if c.max_seconds is not None and self.elapsed >= c.max_seconds:
            out.append(_Violation(
                self, "max_seconds",
                f"{self.elapsed:.2f}s elapsed reached max_seconds={c.max_seconds}",
                {"seconds": round(self.elapsed, 3), "limit": c.max_seconds},
            ))  # fmt: skip
        return out

    def check_llm(self) -> list[_Violation]:
        with self.lock:
            out = self._breakers("chat")
            n = self.config.loop_breaker
            if n and len(self.replies) >= n and len(set(self.replies[-n:])) == 1:
                out.append(_Violation(
                    self, "loop_breaker", f"the last {n} LLM replies were identical",
                    {"repeats": n, "kind": "llm_reply"},
                ))  # fmt: skip
        return out

    def check_tool(
        self, name: str, args: Any, schema: Callable[[], dict[str, Any] | None]
    ) -> tuple[list[_Violation], list[str]]:
        """Violations, and argument problems to repair (``repair_tool_args``)."""
        c = self.config
        out: list[_Violation] = []
        if c.tool_allowlist is not None and not any(
            fnmatch.fnmatchcase(name, p) for p in c.tool_allowlist
        ):
            out.append(_Violation(self, "tool_allowlist", f"tool {name!r} is not in the allowlist",
                                  {"tool": name, "allowlist": list(c.tool_allowlist)}))  # fmt: skip
        hit = next((p for p in c.tool_denylist if fnmatch.fnmatchcase(name, p)), None)
        if hit is not None:
            out.append(_Violation(self, "tool_denylist", f"tool {name!r} is denied ({hit!r})",
                                  {"tool": name, "pattern": hit}))  # fmt: skip
        problems: list[str] = []
        if c.validate_tool_args:
            sch = schema()
            if sch is not None:
                problems = validate_args(args, sch)
            if problems and not c.repair_tool_args:
                out.append(_Violation(
                    self, "validate_tool_args",
                    f"invalid arguments for tool {name!r}: " + "; ".join(problems[:5]),
                    {"tool": name, "problems": problems[:20]},
                ))  # fmt: skip
        with self.lock:
            n = c.loop_breaker
            if n:
                seen = self.tool_counts.get(_args_key(name, args), 0)
                if seen >= n:
                    out.append(_Violation(
                        self, "loop_breaker",
                        f"tool {name!r} already ran {seen} times with the same arguments",
                        {"tool": name, "repeats": seen, "limit": n, "kind": "tool_call"},
                    ))  # fmt: skip
            out.extend(self._breakers("execute_tool"))
        return out, problems if c.repair_tool_args else []


# --- the public API ----------------------------------------------------------------------


class Guard:
    """Context manager (sync and async) and decorator returned by :func:`guard`. Each
    ``with`` / decorated call opens a fresh :class:`GuardScope` with its own counters."""

    def __init__(self, config: GuardConfig) -> None:
        self.config = config
        self._entered: contextvars.ContextVar[tuple[Any, ...]] = contextvars.ContextVar(
            "crashlab_guard_entered", default=()
        )
        self.last_scope: GuardScope | None = None

    def _push(self) -> GuardScope:
        scope = GuardScope(self.config)
        token = GUARD_STACK.set((*GUARD_STACK.get(), scope))
        self._entered.set((*self._entered.get(), (token, scope)))
        self.last_scope = scope
        return scope

    def _pop(self) -> None:
        entered = self._entered.get()
        if not entered:
            return
        token, _ = entered[-1]
        self._entered.set(entered[:-1])
        try:
            GUARD_STACK.reset(token)
        except ValueError:  # exited in another context: drop just this scope
            GUARD_STACK.set(tuple(s for s in GUARD_STACK.get() if s is not entered[-1][1]))

    def __enter__(self) -> GuardScope:
        return self._push()

    def __exit__(self, *exc: Any) -> None:
        self._pop()

    async def __aenter__(self) -> GuardScope:
        return self._push()

    async def __aexit__(self, *exc: Any) -> None:
        self._pop()

    def __call__(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                async with self:
                    return await fn(*args, **kwargs)

            return async_wrapper

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self:
                return fn(*args, **kwargs)

        return wrapper

    @property
    def interventions(self) -> list[dict[str, Any]]:
        """Interventions of the most recently entered scope."""
        return list(self.last_scope.interventions) if self.last_scope else []


def guard(
    config: Mapping[str, Any] | GuardConfig | None = None,
    *,
    max_cost_usd: float | None = None,
    max_tokens: int | None = None,
    max_steps: int | None = None,
    max_llm_calls: int | None = None,
    max_tool_calls: int | None = None,
    max_seconds: float | None = None,
    loop_breaker: int | bool | None = None,
    tool_allowlist: Iterable[str] | str | None = None,
    tool_denylist: Iterable[str] | str | None = None,
    validate_tool_args: bool | str | Mapping[str, Any] | None = None,
    repair_tool_args: bool | None = None,
    retry_policy: RetryPolicy | Mapping[str, Any] | int | bool | None = None,
    fallback_model: str | None = None,
    fallback_after: int | None = None,
    block_secrets_in_output: bool | None = None,
    redact_pii: bool | None = None,
    on_violation: OnViolation | None = None,
    alerts: Any = None,
    name: str | None = None,
    pyproject: bool = True,
) -> Guard:
    """Enforce policies on every LLM / tool call made inside the block (or the decorated
    function). Settings come from ``[tool.crashlab.guard]`` in pyproject.toml (unless
    ``pyproject=False``), then the ``config`` dict, then these keywords.

    ``on_violation``: ``"raise"`` (default) raises :class:`GuardViolation` instead of
    making the call; ``"warn"`` records it, warns and lets the call run; ``"stop"``
    raises and also stops the recorded run (later calls raise ``CircuitBreakerError``).
    ``alerts`` is a sink (``crashlab.alerts.WebhookSink`` / ``JsonlSink`` /
    ``CallbackSink``), a callable taking an ``AlertEvent``, or a list of them.
    """
    if isinstance(tool_allowlist, str):
        tool_allowlist = [tool_allowlist]
    if isinstance(tool_denylist, str):
        tool_denylist = [tool_denylist]
    cfg = load_config(
        config,
        pyproject=pyproject,
        max_cost_usd=max_cost_usd,
        max_tokens=max_tokens,
        max_steps=max_steps,
        max_llm_calls=max_llm_calls,
        max_tool_calls=max_tool_calls,
        max_seconds=max_seconds,
        loop_breaker=loop_breaker,
        tool_allowlist=list(tool_allowlist) if tool_allowlist is not None else None,
        tool_denylist=list(tool_denylist) if tool_denylist is not None else None,
        validate_tool_args=validate_tool_args,
        repair_tool_args=repair_tool_args,
        retry_policy=retry_policy,
        fallback_model=fallback_model,
        fallback_after=fallback_after,
        block_secrets_in_output=block_secrets_in_output,
        redact_pii=redact_pii,
        on_violation=on_violation,
        alerts=alerts,
        name=name,
    )
    return Guard(cfg)


def active_scopes() -> tuple[GuardScope, ...]:
    """The guard scopes active in this context, outer first; empty when none."""
    return GUARD_STACK.get()


# --- argument validation -----------------------------------------------------------------

_JSON_TYPES: dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, (list, tuple)),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


def validate_args(
    args: Any, schema: Mapping[str, Any], *, _path: str = "", _depth: int = 0
) -> list[str]:
    """Problems with ``args`` against a JSON-Schema subset (type, required, properties,
    additionalProperties: false, enum, minimum / maximum, minLength / maxLength, pattern,
    items). An empty list means valid."""
    if _depth > 8 or not isinstance(schema, Mapping):
        return []
    where = _path or "arguments"
    problems: list[str] = []
    types = schema.get("type")
    if isinstance(types, str):
        types = [types]
    if (
        isinstance(types, list)
        and types
        and not any(_JSON_TYPES.get(t, lambda v: True)(args) for t in types)
    ):
        return [f"{where} should be {' or '.join(types)}, got {type(args).__name__}"]
    enum = schema.get("enum")
    if isinstance(enum, list) and args not in enum:
        problems.append(f"{where} must be one of {enum}")
    if isinstance(args, (int, float)) and not isinstance(args, bool):
        lo, hi = schema.get("minimum"), schema.get("maximum")
        if isinstance(lo, (int, float)) and args < lo:
            problems.append(f"{where} must be >= {lo}")
        if isinstance(hi, (int, float)) and args > hi:
            problems.append(f"{where} must be <= {hi}")
    if isinstance(args, str):
        mn, mx, pat = schema.get("minLength"), schema.get("maxLength"), schema.get("pattern")
        if isinstance(mn, int) and len(args) < mn:
            problems.append(f"{where} is shorter than {mn} characters")
        if isinstance(mx, int) and len(args) > mx:
            problems.append(f"{where} is longer than {mx} characters")
        if isinstance(pat, str):
            try:
                if re.search(pat, args) is None:
                    problems.append(f"{where} does not match {pat!r}")
            except re.error:
                pass
    if isinstance(args, (list, tuple)) and isinstance(schema.get("items"), Mapping):
        for i, item in enumerate(args[:100]):
            problems += validate_args(item, schema["items"], _path=f"{where}[{i}]",
                                      _depth=_depth + 1)  # fmt: skip
    if isinstance(args, dict):
        props = schema.get("properties") if isinstance(schema.get("properties"), Mapping) else {}
        required = schema.get("required") if isinstance(schema.get("required"), list) else []
        for key in required:
            if key not in args:
                problems.append(f"missing required argument {key!r}" if not _path
                                else f"{where}.{key} is required")  # fmt: skip
        for key, value in args.items():
            sub = props.get(key)
            if sub is None:
                if schema.get("additionalProperties") is False:
                    problems.append(f"unexpected argument {key!r}" if not _path
                                    else f"unexpected {where}.{key}")  # fmt: skip
                continue
            if value is None and key not in required:
                continue  # an optional argument left at None
            problems += validate_args(value, sub, _path=f"{_path + '.' if _path else ''}{key}",
                                      _depth=_depth + 1)  # fmt: skip
    return problems


_PY_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean",
             list: "array", tuple: "array", dict: "object"}  # fmt: skip


def schema_from_signature(fn: Callable[..., Any]) -> dict[str, Any] | None:
    """A JSON schema from a tool function's signature (simple annotations only)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    try:
        import typing

        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}
    props: dict[str, Any] = {}
    required: list[str] = []
    extra = False
    for name, p in sig.parameters.items():
        if name in ("self", "cls") or p.kind is p.VAR_POSITIONAL:
            continue
        if p.kind is p.VAR_KEYWORD:
            extra = True
            continue
        ann = hints.get(name, p.annotation)
        props[name] = {"type": _PY_TYPES[ann]} if ann in _PY_TYPES else {}
        if p.default is p.empty:
            required.append(name)
    schema: dict[str, Any] = {"type": "object", "properties": props, "required": required}
    if not extra:
        schema["additionalProperties"] = False
    return schema


def _declared_schema(ctx: Any, tool_name: str) -> dict[str, Any] | None:
    """The ``parameters`` schema the model was given for ``tool_name`` (latest chat)."""
    if ctx is None:
        return None
    for step in reversed(ctx.steps[-200:]):
        if step.operation != "chat":
            continue
        for tool in (step.request or {}).get("tools") or []:
            if isinstance(tool, dict) and tool.get("name") == tool_name:
                params = tool.get("parameters")
                return params if isinstance(params, dict) else None
    return None


# --- recording interventions -------------------------------------------------------------


def _note(pending: Any, scope: GuardScope | None, entry: dict[str, Any]) -> None:
    if scope is not None:
        with scope.lock:
            scope.interventions.append(entry)
    if pending is not None:
        pending.attributes.setdefault(GUARD_ATTR, []).append(entry)


def _finding(policy: str, severity: str, message: str, idx: int | None) -> Finding:
    return Finding(
        kind=f"guard.{policy}",
        severity=severity,  # type: ignore[arg-type]
        step=idx,
        steps=[idx] if idx is not None else [],
        message=message,
        key=f"guard.{policy}:{idx}",
    )


def _record_finding(ctx: Any, finding: Finding) -> None:
    if ctx is not None:
        ctx.add_findings([finding])


def _dispatch_alerts(scope: GuardScope, ctx: Any, finding: Finding) -> None:
    sinks = scope.config.alerts
    if not sinks:
        return
    from .alerts import AlertEvent

    run = getattr(ctx, "run", None)
    agent = scope.config.name or getattr(run, "name", None) or "agent"
    event = AlertEvent(
        kind="finding",
        signature=f"{agent}:{finding.kind}",
        run_id=getattr(run, "run_id", "") or "",
        agent_name=agent,
        message=finding.message,
        severity=finding.severity,
        status=str(getattr(run, "status", "") or ""),
        outcome=str(getattr(run, "outcome", "") or ""),
        finding_kind=finding.kind,
    )
    for sink in sinks:
        try:
            emit = getattr(sink, "emit", None)
            (emit if callable(emit) else sink)(event)
        except Exception as exc:  # noqa: BLE001 - an alert must never break the agent
            warnings.warn(f"crashlab guard: alert sink failed: {exc}", GuardWarning,
                          stacklevel=2)  # fmt: skip


def _enforce(violations: list[_Violation], pending: Any, *, is_tool: bool) -> None:
    """Record every violation; raise for the first one whose guard does not just warn."""
    if not violations:
        return
    ctx = pending.ctx if pending is not None else current()
    idx = pending.idx if pending is not None else None
    raising: _Violation | None = None
    findings: list[tuple[_Violation, Finding]] = []
    for v in violations:
        action = v.scope.config.on_violation
        if action != "warn" and raising is None:
            raising = v
        verb = "warned" if action == "warn" else ("stopped" if action == "stop" else "blocked")
        _note(pending, v.scope, {"policy": v.policy, "action": verb, "message": v.message,
                                 "detail": v.detail})  # fmt: skip
        sev = "warn" if action == "warn" else "error"
        findings.append((v, _finding(v.policy, sev, v.message, idx)))
        if action == "warn":
            warnings.warn(f"crashlab guard {v.policy}: {v.message}", GuardWarning, stacklevel=4)
    if raising is None:
        for v, f in findings:
            _record_finding(ctx, f)
            _dispatch_alerts(v.scope, ctx, f)
        return
    exc = GuardViolation(
        f"guard {raising.policy}: {raising.message}",
        policy=raising.policy,
        step_idx=idx,
        detail=raising.detail,
    )
    if pending is not None and not pending.done:
        pending.attributes[BLOCKED_ATTR] = True
        pending.fail(exc, is_tool_error=is_tool)
    stop = raising.scope.config.on_violation == "stop"
    for v, f in findings:
        if stop and v is raising:
            f.stopped_run = True
        _record_finding(ctx, f)
        _dispatch_alerts(v.scope, ctx, f)
    if stop and ctx is not None:
        ctx._trip(next(f for v, f in findings if v is raising))
    raise exc


def _info(pending: Any, scope: GuardScope | None, policy: str, action: str, message: str,
          detail: dict[str, Any]) -> None:  # fmt: skip
    _note(pending, scope, {"policy": policy, "action": action, "message": message,
                           "detail": detail})  # fmt: skip
    ctx = pending.ctx if pending is not None else None
    _record_finding(ctx, _finding(policy, "info", message, pending.idx if pending else None))


# --- redaction ---------------------------------------------------------------------------


def _redact_value(value: Any, include_pii: bool) -> tuple[Any, int]:
    from .bundle import _redact_obj

    return _redact_obj(value, include_pii=include_pii)


def _redact_llm_response(resp: Any, include_pii: bool) -> int:
    """Redact secrets in a chat completion / Anthropic message in place."""
    from .faults import _choice_message, _set_msg_content, _text_of

    count = 0
    found = _choice_message(resp)
    if found is not None:
        choices = getattr(resp, "choices", None)
        if choices is None and isinstance(resp, dict):
            choices = resp.get("choices")
        for choice in choices or []:
            msg = getattr(choice, "message", None)
            if msg is None and isinstance(choice, dict):
                msg = choice.get("message")
            text = _text_of(msg)
            if text:
                new, n = _redact_value(text, include_pii)
                if n:
                    _set_msg_content(msg, new)
                    count += n
        return count
    content = getattr(resp, "content", None)
    if content is None and isinstance(resp, dict):
        content = resp.get("content")
    if isinstance(content, list):
        for block in content:
            text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if isinstance(text, str) and text:
                new, n = _redact_value(text, include_pii)
                if n:
                    if isinstance(block, dict):
                        block["text"] = new
                    else:
                        try:
                            block.text = new
                        except Exception:  # noqa: BLE001 - frozen block: leave it
                            continue
                    count += n
    return count


def _secrets_scope(scopes: tuple[GuardScope, ...]) -> GuardScope | None:
    return next((s for s in reversed(scopes) if s.config.block_secrets_in_output), None)


# --- tool hooks (capture/tools.py) -------------------------------------------------------


def before_tool(pending: Any, tool_name: str, fn: Callable[..., Any]) -> tuple[bool, Any]:
    """Check a live tool call. Raises :class:`GuardViolation`, returns ``(True, error
    result)`` when ``repair_tool_args`` answers the model instead of running the tool,
    else ``(False, None)``."""
    scopes = GUARD_STACK.get()
    if not scopes or pending is None or pending.ctx.mode == "replay":
        return False, None
    args = (pending.request or {}).get("args")
    if (
        isinstance(args, dict)
        and set(args) == {"args", "kwargs"}
        and not args["args"]
        and isinstance(args["kwargs"], dict)
    ):  # the call did not bind to the signature (unexpected keyword): check what was sent
        args = args["kwargs"]
    cache: list[Any] = []

    def schema() -> dict[str, Any] | None:
        if not cache:
            cache.append(_declared_schema(pending.ctx, tool_name) or schema_from_signature(fn))
        return cache[0]

    violations: list[_Violation] = []
    repairs: list[tuple[GuardScope, list[str]]] = []
    for scope in scopes:
        found, problems = scope.check_tool(tool_name, args, schema)
        violations += found
        if problems:
            repairs.append((scope, problems))
    _enforce(violations, pending, is_tool=True)  # raises unless every guard only warns
    if not repairs:
        return False, None
    scope, problems = repairs[-1]
    result = {
        "error": "invalid_tool_arguments",
        "tool": tool_name,
        "problems": problems[:20],
        "message": f"The arguments for {tool_name} are invalid: "
        + "; ".join(problems[:5])
        + ". Fix them and call the tool again.",
    }
    _info(pending, scope, "validate_tool_args", "repaired",
          f"returned an argument error to the model instead of running {tool_name!r}",
          {"tool": tool_name, "problems": problems[:20]})  # fmt: skip
    pending.attributes[BLOCKED_ATTR] = True
    pending.finish({"result": result})
    return True, result


def after_tool(pending: Any, result: Any) -> Any:
    """Redact secrets from a live tool result (``block_secrets_in_output``)."""
    scopes = GUARD_STACK.get()
    scope = _secrets_scope(scopes) if scopes else None
    if scope is None or pending is None or pending.ctx.mode == "replay":
        return result
    if not isinstance(result, (str, dict, list, tuple)):
        return result
    new, n = _redact_value(result, scope.config.redact_pii)
    if n:
        _info(pending, scope, "block_secrets_in_output", "redacted",
              f"redacted {n} secret(s) from the {pending.name!r} tool result",
              {"redactions": n, "where": "tool_result"})  # fmt: skip
        return new
    return result


# --- LLM hook (capture/_chat.py invoke) ---------------------------------------------------


class _Attempts:
    """Retry / fallback bookkeeping for one LLM call."""

    def __init__(self, scopes: tuple[GuardScope, ...], call: Any, kwargs: dict[str, Any]):
        self.call = call
        self.kwargs = kwargs
        self.retry_scope = next((s for s in reversed(scopes) if s.config.retry_policy), None)
        self.fb_scope = next((s for s in reversed(scopes) if s.config.fallback_model), None)
        self.retry = self.retry_scope.config.retry_policy if self.retry_scope else None
        policy = self.retry or RetryPolicy(max_attempts=1)
        self.policy = policy
        self.rng = random.Random(policy.seed) if policy.seed is not None else random.Random()
        self.fallback = self.fb_scope.config.fallback_model if self.fb_scope else None
        if self.fallback is not None and "model" not in kwargs:
            self.fallback = None  # model passed positionally: cannot switch safely
        self.fb_after = (self.fb_scope.config.fallback_after if self.fb_scope else None) or (
            policy.max_attempts
        )
        self.primary_failures = 0
        self.fallback_failures = 0
        self.on_fallback = False
        self.errors: list[dict[str, Any]] = []
        self.n_attempts = 0
        self.delays: list[float] = []
        self.original_model = kwargs.get("model")

    @property
    def enabled(self) -> bool:
        return self.retry is not None or self.fallback is not None

    @property
    def attempts(self) -> int:
        return max(1, self.n_attempts)

    def once(self, fn: Callable[..., Any], args: tuple) -> Any:
        """One attempt: injected faults first (each retry is a new call for
        ``crashlab.faults``), then the real call."""
        call = self.call
        self.n_attempts += 1
        call._fault = None
        fault = take_llm_fault(call)
        if fault is not None:
            if call.pending is not None:
                call.pending.attributes[FAULT_ATTR] = fault.as_dict()
            call._fault = fault
            if fault.kind == "llm_error":
                raise fault.exception()
        return fn(*args, **self.kwargs)

    def on_error(self, exc: BaseException) -> float | None:
        """Seconds to wait before the next attempt, or None to give up."""
        if not self.enabled or not is_transient(exc, self.policy.retry_on):
            return None
        status = getattr(exc, "status_code", None)
        self.errors.append({
            "attempt": self.attempts, "model": self.kwargs.get("model"),
            "error": type(exc).__name__,
            "status": status if isinstance(status, int) else None,
        })  # fmt: skip
        if not self.on_fallback:
            self.primary_failures += 1
            if self.fallback is not None and self.primary_failures >= self.fb_after:
                self.on_fallback = True
                self.kwargs = {**self.kwargs, "model": self.fallback}
                self.delays.append(0.0)
                return 0.0
            if self.retry is not None and self.primary_failures < (
                self.fb_after if self.fallback is not None else self.retry.max_attempts
            ):
                return self._delay(self.primary_failures, exc)
            return None
        self.fallback_failures += 1
        if self.fallback_failures < self.policy.max_attempts:
            return self._delay(self.fallback_failures, exc)
        return None

    def _delay(self, n: int, exc: BaseException) -> float:
        d = self.policy.delay(n, self.rng, exc)
        self.delays.append(d)
        return d

    def finalize(self, ok: bool) -> None:
        pending = self.call.pending
        if not self.errors:
            return
        if pending is not None:
            pending.attributes[ATTEMPTS_ATTR] = self.attempts
        n_retry = len(self.errors) - (1 if self.on_fallback else 0)
        if n_retry > 0 or not self.on_fallback:
            _info(pending, self.retry_scope or self.fb_scope, "retry_policy",
                  "retried" if ok else "gave_up",
                  f"{len(self.errors)} transient LLM error(s); "
                  + ("succeeded" if ok else "gave up") + f" after {self.attempts} attempt(s)",
                  {"attempts": self.attempts, "errors": self.errors[:20],
                   "delays": self.delays[:20]})  # fmt: skip
        if self.on_fallback:
            if pending is not None and ok:
                pending.attributes[FALLBACK_ATTR] = self.fallback
            _info(pending, self.fb_scope, "fallback_model", "fallback",
                  f"switched from {self.original_model!r} to {self.fallback!r} after "
                  f"{self.primary_failures} failure(s)" + ("" if ok else "; fallback failed too"),
                  {"from": self.original_model, "to": self.fallback,
                   "after_failures": self.primary_failures, "ok": ok})  # fmt: skip


def _redact_reply(scopes: tuple[GuardScope, ...], call: Any, result: Any) -> Any:
    scope = _secrets_scope(scopes)
    if scope is None or call.stream:
        return result
    n = _redact_llm_response(result, scope.config.redact_pii)
    if n:
        _info(call.pending, scope, "block_secrets_in_output", "redacted",
              f"redacted {n} secret(s) from the LLM reply",
              {"redactions": n, "where": "llm_reply"})  # fmt: skip
    return result


def guarded_invoke(
    scopes: tuple[GuardScope, ...],
    call: Any,
    fn: Callable[..., Any],
    args: tuple,
    kwargs: dict[str, Any],
    *,
    is_async: bool | None,
    wrap: Callable[..., Any],
) -> Any:
    """The live LLM call under guards: pre-call checks, faults, retries / fallback,
    output redaction, then ``wrap`` (recording). Strict replay never gets here."""
    pending = call.pending
    replay = pending is not None and pending.ctx.mode == "replay"
    if not replay:
        violations: list[_Violation] = []
        for scope in scopes:
            violations += scope.check_llm()
        _enforce(violations, pending, is_tool=False)
    att = _Attempts(scopes, call, dict(kwargs))
    if replay:
        att.retry, att.fallback = None, None
    if is_async is None:
        is_async = inspect.iscoroutinefunction(fn)
    if is_async:
        return _async_attempts(scopes, att, fn, args, wrap, None)
    while True:
        try:
            result = att.once(fn, args)
        except Exception as exc:
            delay = att.on_error(exc)
            if delay is None:
                att.finalize(False)
                call.fail(exc)
                raise
            if delay:
                _sleep(delay)
            continue
        except BaseException as exc:
            att.finalize(False)
            call.fail(exc)
            raise
        if inspect.isawaitable(result):
            return _async_attempts(scopes, att, fn, args, wrap, result)
        break
    att.finalize(True)
    return wrap(call, _redact_reply(scopes, call, result), is_async=False)


async def _async_attempts(
    scopes: tuple[GuardScope, ...],
    att: _Attempts,
    fn: Callable[..., Any],
    args: tuple,
    wrap: Callable[..., Any],
    first: Any,
) -> Any:
    call = att.call
    pending_result = first
    while True:
        try:
            if pending_result is None:
                pending_result = att.once(fn, args)
            resp = await pending_result if inspect.isawaitable(pending_result) else pending_result
        except Exception as exc:
            pending_result = None
            delay = att.on_error(exc)
            if delay is None:
                att.finalize(False)
                call.fail(exc)
                raise
            if delay:
                await _async_sleep(delay)
            continue
        except BaseException as exc:
            att.finalize(False)
            call.fail(exc)
            raise
        break
    att.finalize(True)
    return wrap(call, _redact_reply(scopes, call, resp), is_async=True)


# --- reporting ---------------------------------------------------------------------------


def interventions(run: Any) -> list[dict[str, Any]]:
    """Every guard intervention recorded on a run's steps, in step order."""
    out: list[dict[str, Any]] = []
    for step in getattr(run, "steps", None) or []:
        for entry in (step.attributes or {}).get(GUARD_ATTR) or []:
            if isinstance(entry, dict):
                out.append({"step": step.idx, "operation": step.operation,
                            "call": step.name, **entry})  # fmt: skip
    return out


guard.RetryPolicy = RetryPolicy  # type: ignore[attr-defined]
guard.GuardViolation = GuardViolation  # type: ignore[attr-defined]

__all__ = [
    "ATTEMPTS_ATTR",
    "BLOCKED_ATTR",
    "FALLBACK_ATTR",
    "GUARD_ATTR",
    "POLICIES",
    "Guard",
    "GuardConfig",
    "GuardScope",
    "GuardViolation",
    "GuardWarning",
    "RetryPolicy",
    "active_scopes",
    "guard",
    "interventions",
    "is_transient",
    "load_config",
    "schema_from_signature",
    "validate_args",
]
