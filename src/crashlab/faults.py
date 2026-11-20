"""Deterministic fault injection for recorded AI agents.

Use :func:`faults` as a context manager around a live recorded run (or a
:func:`~crashlab.fork` past its fork step) to inject tool and LLM failures.
Rules are stackable and nested via ``contextvars`` (thread- and asyncio-safe).
When no policy is active, a single ``ContextVar`` read is the only cost.

Strict :func:`~crashlab.replay` never re-injects: it serves the cassette,
including any ``crashlab.fault`` metadata recorded when the fault first ran.
"""

from __future__ import annotations

import contextvars
import json
import random
import threading
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from typing import Any, Literal

from .context import AGENT_ATTR, current, current_agent

FAULT_ATTR = "crashlab.fault"

ToolResultMode = Literal["replace", "html_502", "empty", "truncated", "schema_drift"]
FaultKind = Literal[
    "tool_error",
    "tool_result",
    "tool_delay",
    "llm_error",
    "llm_empty",
    "llm_truncate",
    "llm_bad_tool_call",
]

HTML_502 = (
    "<!DOCTYPE html>\n"
    "<html><head><title>502 Bad Gateway</title></head>\n"
    "<body><center><h1>502 Bad Gateway</h1></center>"
    "<hr><center>nginx/1.25.3</center></body>\n"
    "</html>"
)

_HTTP_PHRASES = {
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    408: "Request Timeout",
    409: "Conflict",
    429: "Rate limit exceeded",
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout",
}

_MISSING = object()

# Empty tuple is the inactive default: one ContextVar get, no allocation on the
# hot path, and `if not _stack.get():` is false only when a policy is pushed.
_stack: contextvars.ContextVar[tuple[FaultPolicy, ...]] = contextvars.ContextVar(
    "crashlab_faults", default=()
)


class InjectedFaultError(Exception):
    """A tool or LLM error crashlab injected, with an optional HTTP status."""

    def __init__(
        self, message: str, *, status_code: int | None = None, kind: str = "error"
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.status = status_code
        self.kind = kind
        self.body = {"error": {"message": message, "type": "injected_fault", "code": status_code}}
        self.response = None
        self.request = None


@dataclass
class _RuleState:
    calls: int = 0
    triggers: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    rng: random.Random | None = None


def _as_on_calls(value: int | Sequence[int] | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, Sequence)):
        raise TypeError("on_calls must be a positive int or a sequence of them")
    items = (value,) if isinstance(value, int) else tuple(value)
    if not items or any(not isinstance(i, int) or isinstance(i, bool) or i < 1 for i in items):
        raise ValueError("on_calls must be one or more 1-based call numbers")
    return items


def _validate_triggers(
    *,
    on_calls: int | Sequence[int] | None,
    after_calls: int | None,
    every: int | None,
    max_triggers: int | None,
    probability: float | None,
    seed: int | None,
    agent: str | None,
) -> tuple[
    tuple[int, ...] | None, int | None, int | None, int | None, float | None, int | None, str | None
]:
    calls = _as_on_calls(on_calls)
    if after_calls is not None and (
        not isinstance(after_calls, int) or isinstance(after_calls, bool) or after_calls < 0
    ):
        raise ValueError("after_calls must be a non-negative int")
    if every is not None and (not isinstance(every, int) or isinstance(every, bool) or every < 1):
        raise ValueError("every must be an int >= 1")
    if max_triggers is not None and (
        not isinstance(max_triggers, int) or isinstance(max_triggers, bool) or max_triggers < 1
    ):
        raise ValueError("max_triggers must be an int >= 1")
    if probability is not None:
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise TypeError("probability must be a float in [0, 1]")
        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        probability = float(probability)
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise TypeError("seed must be an int")
    if agent is not None and (not isinstance(agent, str) or not agent.strip()):
        raise ValueError("agent must be a non-empty string")
    return (
        calls,
        after_calls,
        every,
        max_triggers,
        probability,
        seed,
        agent.strip() if agent else None,
    )


def _agent_matches(rule_agent: str | None, actual: str | None) -> bool:
    if rule_agent is None:
        return True
    if not actual:
        return False
    if actual == rule_agent:
        return True
    # wrap(client, agent="researcher") plus a concurrent copy labelled researcher#1
    return actual.startswith(rule_agent + "#") and "#" not in rule_agent


@dataclass
class FaultRule:
    """One injectable fault and the conditions under which it fires.

    Call numbers (``on_calls``, ``after_calls``, ``every``, ``max_triggers``) count
    only matching calls: the same tool (or any LLM call for LLM faults) and, when
    ``agent=`` is set, that agent. ``probability`` is applied after those filters;
    pass ``seed`` for a deterministic stream.
    """

    kind: FaultKind
    tool: str | None = None
    mode: ToolResultMode | None = None
    result: Any = None
    seconds: float | None = None
    status: int | None = None
    message: str | None = None
    exc: Any = None
    name: str | None = None
    arguments: str | None = None
    max_chars: int | None = None
    on_calls: tuple[int, ...] | None = None
    after_calls: int | None = None
    every: int | None = None
    max_triggers: int | None = None
    probability: float | None = None
    seed: int | None = None
    agent: str | None = None
    _state: _RuleState = field(default_factory=_RuleState, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.seed is not None:
            self._state.rng = random.Random(self.seed)

    def matches_target(self, *, tool: str | None, agent: str | None) -> bool:
        if not _agent_matches(self.agent, agent):
            return False
        if self.kind.startswith("tool"):
            return self.tool in (None, "*", tool)
        return True  # LLM faults: every chat call (agent already filtered)

    def consider(self, *, tool: str | None, agent: str | None) -> bool:
        """Count this matching call and return True if the fault should fire now."""
        if not self.matches_target(tool=tool, agent=agent):
            return False
        st = self._state
        with st.lock:
            st.calls += 1
            n = st.calls
            if self.on_calls is not None and n not in self.on_calls:
                return False
            if self.after_calls is not None and n <= self.after_calls:
                return False
            if self.every is not None and n % self.every != 0:
                return False
            if self.max_triggers is not None and st.triggers >= self.max_triggers:
                return False
            if self.probability is not None:
                rng = st.rng if st.rng is not None else random
                if rng.random() >= self.probability:
                    return False
            st.triggers += 1
            return True

    def as_dict(self) -> dict[str, Any]:
        """JSON-able metadata stored on ``step.attributes['crashlab.fault']``."""
        st = self._state
        data: dict[str, Any] = {"kind": self.kind, "call": st.calls, "triggers": st.triggers}
        if self.tool is not None:
            data["tool"] = self.tool
        if self.mode is not None:
            data["mode"] = self.mode
        if self.seconds is not None:
            data["seconds"] = self.seconds
        if self.status is not None:
            data["status"] = self.status
        if self.message is not None:
            data["message"] = self.message
        if self.name is not None:
            data["name"] = self.name
        if self.arguments is not None:
            data["arguments"] = self.arguments
        if self.max_chars is not None:
            data["max_chars"] = self.max_chars
        if self.agent is not None:
            data["agent"] = self.agent
        if self.kind == "tool_error":
            typ = (
                self.exc
                if isinstance(self.exc, type)
                else type(self.exc)
                if self.exc is not None
                else InjectedFaultError
            )
            data["exception"] = typ.__name__
        return data

    def replaces_call(self) -> bool:
        return self.kind == "tool_result" and self.mode in ("replace", "html_502", "empty")

    def exception(self) -> BaseException:
        if isinstance(self.exc, BaseException):
            return self.exc
        typ = (
            self.exc if isinstance(self.exc, type) and issubclass(self.exc, BaseException) else None
        )
        status = self.status
        message = self.message or (
            f"injected HTTP {status} ({_HTTP_PHRASES.get(status, 'Error')})"
            if status is not None
            else "injected fault"
        )
        if typ is TimeoutError or (status in (408, 504) and typ is None and self.exc is None):
            return TimeoutError(message)
        if typ is not None:
            try:
                err = typ(message)
            except TypeError:
                err = typ()
            if status is not None and not hasattr(err, "status_code"):
                with suppress(Exception):
                    err.status_code = status
            return err
        if self.kind == "llm_error":
            built = _provider_llm_error(status or 429, message)
            if built is not None:
                return built
        return InjectedFaultError(
            message, status_code=status, kind="rate_limit" if status == 429 else "error"
        )

    def materialize(self, original: Any) -> Any:
        mode = self.mode or "replace"
        if mode == "replace":
            return self.result
        if mode == "html_502":
            return HTML_502 if self.result is None else self.result
        if mode == "empty":
            if self.result is not None:
                return self.result
            if isinstance(original, dict):
                return {}
            if isinstance(original, list):
                return []
            return ""
        if mode == "truncated":
            return _truncate_payload(original if original is not None else self.result)
        if mode == "schema_drift":
            return _drift_schema(original if original is not None else self.result)
        raise ValueError(f"unknown tool_result mode {mode!r}")

    def mutate_response(self, resp: Any) -> Any:
        if self.kind == "llm_empty":
            return _llm_empty(resp)
        if self.kind == "llm_truncate":
            return _llm_truncate(resp, self.max_chars)
        if self.kind == "llm_bad_tool_call":
            return _llm_bad_tool_call(resp, self.name, self.arguments)
        return resp


@dataclass
class FaultPolicy:
    """An ordered list of :class:`FaultRule` s. The first matching rule fires."""

    rules: tuple[FaultRule, ...] = ()

    def take(self, *, kind_prefix: str, tool: str | None, agent: str | None) -> FaultRule | None:
        for rule in self.rules:
            if not rule.kind.startswith(kind_prefix):
                continue
            if rule.consider(tool=tool, agent=agent):
                return rule
        return None


def current_policies() -> tuple[FaultPolicy, ...]:
    """The stacked policies for this context, outer first; empty when inactive."""
    return _stack.get()


def take_tool_fault(pending: Any, tool_name: str) -> FaultRule | None:
    """The tool fault that should fire for this live call, or None.

    No-ops when no policy is active, in strict replay, or when ``pending`` is a
    cassette-served step (the caller only invokes this on a live path).
    """
    stack = _stack.get()
    if not stack:
        return None
    if _replay_mode(pending):
        return None
    agent = _agent_of(pending)
    for policy in reversed(stack):
        rule = policy.take(kind_prefix="tool", tool=tool_name, agent=agent)
        if rule is not None:
            return rule
    return None


def take_llm_fault(call: Any) -> FaultRule | None:
    """The LLM fault that should fire for this live call, or None."""
    stack = _stack.get()
    if not stack:
        return None
    pending = getattr(call, "pending", None)
    if _replay_mode(pending):
        return None
    agent = _agent_of(pending, getattr(call, "adapter", None))
    for policy in reversed(stack):
        rule = policy.take(kind_prefix="llm", tool=None, agent=agent)
        if rule is not None:
            return rule
    return None


def _replay_mode(pending: Any) -> bool:
    ctx = getattr(pending, "ctx", None) if pending is not None else current()
    return ctx is not None and getattr(ctx, "mode", None) == "replay"


def _agent_of(pending: Any, adapter: Any = None) -> str | None:
    if pending is not None:
        name = (pending.attributes or {}).get(AGENT_ATTR)
        if isinstance(name, str) and name:
            return name
    agent = getattr(adapter, "agent", None)
    if isinstance(agent, str) and agent:
        return agent
    return current_agent()


def _collect_rules(items: Iterable[Any]) -> list[FaultRule]:
    rules: list[FaultRule] = []
    for item in items:
        if isinstance(item, FaultRule):
            rules.append(item)
        elif isinstance(item, FaultPolicy):
            rules.extend(item.rules)
        else:
            raise TypeError(
                f"crashlab.faults() expected a FaultRule or FaultPolicy, not {type(item).__name__}"
            )
    return rules


@contextmanager
def faults(*rules: FaultRule | FaultPolicy) -> Iterator[FaultPolicy]:
    """Activate one or more fault rules for the current context (stackable).

    ::

        with crashlab.faults(crashlab.faults.tool_error(tool="get_order", on_calls=[1])):
            run_agent("Refund order A17")

    Nested ``with faults(...)`` blocks add rules; the innermost matching rule
    wins. Exiting the block restores the previous policy (if any).
    """
    policy = FaultPolicy(tuple(_collect_rules(rules)))
    token = _stack.set((*_stack.get(), policy))
    try:
        yield policy
    finally:
        _stack.reset(token)


def tool_error(
    tool: str = "*",
    exc: Any = None,
    status: int | None = None,
    message: str | None = None,
    *,
    on_calls: int | Sequence[int] | None = None,
    after_calls: int | None = None,
    every: int | None = None,
    max_triggers: int | None = None,
    seed: int | None = None,
    probability: float | None = None,
    agent: str | None = None,
) -> FaultRule:
    """Raise an exception instead of calling the tool.

    ``exc`` is an exception instance or type (default :class:`RuntimeError`, or
    :class:`TimeoutError` when ``status`` is 408 / 504). ``status`` builds a
    provider-style HTTP error (``status_code`` on the exception).
    """
    if not isinstance(tool, str) or not tool:
        raise ValueError("tool must be a non-empty string (or '*' )")
    if status is not None and (
        not isinstance(status, int) or isinstance(status, bool) or status < 100
    ):
        raise ValueError("status must be an HTTP status code")
    calls, after, ev, mx, prob, sd, ag = _validate_triggers(
        on_calls=on_calls, after_calls=after_calls, every=every,
        max_triggers=max_triggers, probability=probability, seed=seed, agent=agent,
    )  # fmt: skip
    if exc is None and status is None:
        exc = RuntimeError
    return FaultRule(
        kind="tool_error", tool=tool, exc=exc, status=status, message=message,
        on_calls=calls, after_calls=after, every=ev, max_triggers=mx,
        probability=prob, seed=sd, agent=ag,
    )  # fmt: skip


def tool_result(
    tool: str = "*",
    result: Any = _MISSING,
    mode: ToolResultMode = "replace",
    *,
    on_calls: int | Sequence[int] | None = None,
    after_calls: int | None = None,
    every: int | None = None,
    max_triggers: int | None = None,
    seed: int | None = None,
    probability: float | None = None,
    agent: str | None = None,
) -> FaultRule:
    """Replace the tool's result with a realistic corrupted payload.

    ``mode`` is ``replace`` (use ``result``), ``html_502`` (an nginx 502 HTML
    page), ``empty`` (``{}`` / ``[]`` / ``""``), ``truncated`` (cut-off JSON)
    or ``schema_drift`` (renamed / dropped keys). ``truncated`` and
    ``schema_drift`` call the tool first, then corrupt what it returned.
    """
    if not isinstance(tool, str) or not tool:
        raise ValueError("tool must be a non-empty string (or '*' )")
    if mode not in ("replace", "html_502", "empty", "truncated", "schema_drift"):
        raise ValueError(
            "mode must be 'replace', 'html_502', 'empty', 'truncated' or 'schema_drift'"
        )
    if mode == "replace" and result is _MISSING:
        raise ValueError("tool_result(mode='replace') needs a result=")
    calls, after, ev, mx, prob, sd, ag = _validate_triggers(
        on_calls=on_calls, after_calls=after_calls, every=every,
        max_triggers=max_triggers, probability=probability, seed=seed, agent=agent,
    )  # fmt: skip
    return FaultRule(
        kind="tool_result", tool=tool, mode=mode,
        result=None if result is _MISSING else result,
        on_calls=calls, after_calls=after, every=ev, max_triggers=mx,
        probability=prob, seed=sd, agent=ag,
    )  # fmt: skip


def tool_delay(
    tool: str = "*",
    seconds: float = 1.0,
    *,
    on_calls: int | Sequence[int] | None = None,
    after_calls: int | None = None,
    every: int | None = None,
    max_triggers: int | None = None,
    seed: int | None = None,
    probability: float | None = None,
    agent: str | None = None,
) -> FaultRule:
    """Sleep ``seconds`` before calling the tool (live record / fork only).

    Strict replay serves the cassette and does not sleep again. The delay is
    recorded on the step as ``crashlab.fault``.
    """
    if not isinstance(tool, str) or not tool:
        raise ValueError("tool must be a non-empty string (or '*' )")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds < 0:
        raise ValueError("seconds must be a non-negative number")
    calls, after, ev, mx, prob, sd, ag = _validate_triggers(
        on_calls=on_calls, after_calls=after_calls, every=every,
        max_triggers=max_triggers, probability=probability, seed=seed, agent=agent,
    )  # fmt: skip
    return FaultRule(
        kind="tool_delay", tool=tool, seconds=float(seconds),
        on_calls=calls, after_calls=after, every=ev, max_triggers=mx,
        probability=prob, seed=sd, agent=ag,
    )  # fmt: skip


def llm_error(
    status: int = 429,
    message: str | None = None,
    exc: Any = None,
    *,
    on_calls: int | Sequence[int] | None = None,
    after_calls: int | None = None,
    every: int | None = None,
    max_triggers: int | None = None,
    seed: int | None = None,
    probability: float | None = None,
    agent: str | None = None,
) -> FaultRule:
    """Raise a provider-style API / rate-limit / timeout error on an LLM call."""
    if not isinstance(status, int) or isinstance(status, bool) or status < 100:
        raise ValueError("status must be an HTTP status code")
    calls, after, ev, mx, prob, sd, ag = _validate_triggers(
        on_calls=on_calls, after_calls=after_calls, every=every,
        max_triggers=max_triggers, probability=probability, seed=seed, agent=agent,
    )  # fmt: skip
    return FaultRule(
        kind="llm_error", status=status, message=message, exc=exc,
        on_calls=calls, after_calls=after, every=ev, max_triggers=mx,
        probability=prob, seed=sd, agent=ag,
    )  # fmt: skip


def llm_empty(
    *,
    on_calls: int | Sequence[int] | None = None,
    after_calls: int | None = None,
    every: int | None = None,
    max_triggers: int | None = None,
    seed: int | None = None,
    probability: float | None = None,
    agent: str | None = None,
) -> FaultRule:
    """Force an empty assistant reply (no text, no tool calls) on a live LLM call."""
    calls, after, ev, mx, prob, sd, ag = _validate_triggers(
        on_calls=on_calls, after_calls=after_calls, every=every,
        max_triggers=max_triggers, probability=probability, seed=seed, agent=agent,
    )  # fmt: skip
    return FaultRule(
        kind="llm_empty",
        on_calls=calls, after_calls=after, every=ev, max_triggers=mx,
        probability=prob, seed=sd, agent=ag,
    )  # fmt: skip


def llm_truncate(
    max_chars: int | None = None,
    *,
    on_calls: int | Sequence[int] | None = None,
    after_calls: int | None = None,
    every: int | None = None,
    max_triggers: int | None = None,
    seed: int | None = None,
    probability: float | None = None,
    agent: str | None = None,
) -> FaultRule:
    """Truncate the assistant text and set ``finish_reason="length"``."""
    if max_chars is not None and (
        not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars < 0
    ):
        raise ValueError("max_chars must be a non-negative int")
    calls, after, ev, mx, prob, sd, ag = _validate_triggers(
        on_calls=on_calls, after_calls=after_calls, every=every,
        max_triggers=max_triggers, probability=probability, seed=seed, agent=agent,
    )  # fmt: skip
    return FaultRule(
        kind="llm_truncate", max_chars=max_chars,
        on_calls=calls, after_calls=after, every=ev, max_triggers=mx,
        probability=prob, seed=sd, agent=ag,
    )  # fmt: skip


def llm_bad_tool_call(
    name: str | None = None,
    arguments: str | None = None,
    *,
    on_calls: int | Sequence[int] | None = None,
    after_calls: int | None = None,
    every: int | None = None,
    max_triggers: int | None = None,
    seed: int | None = None,
    probability: float | None = None,
    agent: str | None = None,
) -> FaultRule:
    """Inject a malformed or unknown tool call into the LLM response.

    ``arguments`` is the raw arguments string (default: cut-off JSON). ``name``
    defaults to an unknown tool name so the agent has to recover.
    """
    if name is not None and not isinstance(name, str):
        raise TypeError("name must be a string")
    if arguments is not None and not isinstance(arguments, str):
        raise TypeError("arguments must be a string")
    calls, after, ev, mx, prob, sd, ag = _validate_triggers(
        on_calls=on_calls, after_calls=after_calls, every=every,
        max_triggers=max_triggers, probability=probability, seed=seed, agent=agent,
    )  # fmt: skip
    return FaultRule(
        kind="llm_bad_tool_call", name=name, arguments=arguments,
        on_calls=calls, after_calls=after, every=ev, max_triggers=mx,
        probability=prob, seed=sd, agent=ag,
    )  # fmt: skip


# ``crashlab.faults.tool_error(...)`` when `faults` is imported from the package.
faults.tool_error = tool_error  # type: ignore[attr-defined]
faults.tool_result = tool_result  # type: ignore[attr-defined]
faults.tool_delay = tool_delay  # type: ignore[attr-defined]
faults.llm_error = llm_error  # type: ignore[attr-defined]
faults.llm_empty = llm_empty  # type: ignore[attr-defined]
faults.llm_truncate = llm_truncate  # type: ignore[attr-defined]
faults.llm_bad_tool_call = llm_bad_tool_call  # type: ignore[attr-defined]
faults.HTML_502 = HTML_502  # type: ignore[attr-defined]
faults.InjectedFaultError = InjectedFaultError  # type: ignore[attr-defined]
faults.FAULT_ATTR = FAULT_ATTR  # type: ignore[attr-defined]


def _provider_llm_error(status: int, message: str) -> BaseException | None:
    """Build an openai.* error when the SDK is already installed; never required."""
    try:
        import openai
    except ImportError:
        return None
    body = {"error": {"message": message, "type": "injected_fault", "code": status}}
    names: list[str] = []
    if status == 429:
        names.append("RateLimitError")
    if status in (408, 504):
        names.append("APITimeoutError")
    names.append("APIStatusError")
    for cls_name in names:
        cls = getattr(openai, cls_name, None)
        if cls is None:
            continue
        try:
            return cls(message, response=None, body=body)
        except TypeError:
            try:
                return cls(message)
            except TypeError:
                continue
        except Exception:
            continue
    return None


def _truncate_payload(value: Any) -> str:
    if value is None:
        return "{"
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, default=str)
        except (TypeError, ValueError):
            text = str(value)
    if len(text) <= 1:
        return text or "{"
    cut = max(1, len(text) // 2)
    return text[:cut]


def _drift_schema(value: Any) -> Any:
    if not isinstance(value, dict) or not value:
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return [_drift_schema(v) for v in value]
        return {"_drifted": True} if value is None else value
    out: dict[str, Any] = {}
    items = list(value.items())
    for i, (k, v) in enumerate(items):
        if k == "total":
            out["amount_total"] = v
            continue
        if i == 0:
            out[f"{k}_v2"] = v
            continue
        if i == 1:
            continue  # dropped
        out[k] = v
    return out or {"_drifted": True}


def _choice_message(resp: Any) -> tuple[Any, Any] | None:
    choices = getattr(resp, "choices", None)
    if not choices and isinstance(resp, dict):
        choices = resp.get("choices")
    if not choices:
        return None
    choice = choices[0]
    msg = getattr(choice, "message", None)
    if msg is None and isinstance(choice, dict):
        msg = choice.get("message")
    return (choice, msg) if msg is not None else (choice, None)


def _set_finish(choice: Any, reason: str) -> None:
    if isinstance(choice, dict):
        choice["finish_reason"] = reason
        return
    for attr in ("finish_reason", "stop_reason"):
        if hasattr(choice, attr):
            try:
                setattr(choice, attr, reason)
                return
            except Exception:
                continue


def _set_msg_content(msg: Any, text: str | None) -> None:
    if isinstance(msg, dict):
        msg["content"] = text
        return
    if hasattr(msg, "content"):
        with suppress(Exception):
            msg.content = text


def _set_msg_tool_calls(msg: Any, calls: Any) -> None:
    if isinstance(msg, dict):
        msg["tool_calls"] = calls
        return
    if hasattr(msg, "tool_calls"):
        with suppress(Exception):
            msg.tool_calls = calls


def _clear_anthropic_content(resp: Any) -> bool:
    content = getattr(resp, "content", None)
    if content is None and isinstance(resp, dict):
        if "content" in resp:
            resp["content"] = []
            resp["stop_reason"] = "end_turn"
            return True
        return False
    if content is None:
        return False
    try:
        resp.content = []
        if hasattr(resp, "stop_reason"):
            resp.stop_reason = "end_turn"
        return True
    except Exception:
        return False


def _llm_empty(resp: Any) -> Any:
    found = _choice_message(resp)
    if found is not None:
        choice, msg = found
        if msg is not None:
            _set_msg_content(msg, "")
            _set_msg_tool_calls(msg, None)
        _set_finish(choice, "stop")
        return resp
    _clear_anthropic_content(resp)
    return resp


def _text_of(msg: Any) -> str:
    if msg is None:
        return ""
    if isinstance(msg, dict):
        c = msg.get("content")
        return c if isinstance(c, str) else ""
    c = getattr(msg, "content", None)
    return c if isinstance(c, str) else ""


def _llm_truncate(resp: Any, max_chars: int | None) -> Any:
    found = _choice_message(resp)
    if found is None:
        content = getattr(resp, "content", None)
        if isinstance(content, list):
            for block in content:
                text = getattr(block, "text", None)
                if isinstance(text, str) and text:
                    cut = max_chars if max_chars is not None else max(1, len(text) // 2)
                    with suppress(Exception):
                        block.text = text[:cut]
                    break
        if hasattr(resp, "stop_reason"):
            with suppress(Exception):
                resp.stop_reason = "max_tokens"
        return resp
    choice, msg = found
    text = _text_of(msg)
    if text:
        cut = max_chars if max_chars is not None else max(1, len(text) // 2)
        _set_msg_content(msg, text[:cut])
    _set_finish(choice, "length")
    return resp


class _Fn:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _Call:
    def __init__(self, name: str, arguments: str) -> None:
        self.id = "call_injected"
        self.type = "function"
        self.function = _Fn(name, arguments)


def _llm_bad_tool_call(resp: Any, name: str | None, arguments: str | None) -> Any:
    name = name or "not_a_real_tool"
    arguments = "{not json" if arguments is None else arguments
    found = _choice_message(resp)
    if found is not None:
        choice, msg = found
        existing = None
        if msg is not None:
            existing = (
                msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
            )
        if existing:
            first = existing[0]
            fn = (
                first.get("function")
                if isinstance(first, dict)
                else getattr(first, "function", first)
            )
            if isinstance(fn, dict):
                fn["name"] = name
                fn["arguments"] = arguments
            else:
                try:
                    fn.name = name
                    fn.arguments = arguments
                except Exception:
                    _set_msg_tool_calls(msg, [_Call(name, arguments)])
        elif msg is not None:
            if isinstance(msg, dict):
                msg["tool_calls"] = [
                    {"id": "call_injected", "type": "function",
                     "function": {"name": name, "arguments": arguments}}
                ]  # fmt: skip
            else:
                _set_msg_tool_calls(msg, [_Call(name, arguments)])
            _set_msg_content(msg, None)
        _set_finish(choice, "tool_calls")
        return resp
    # Anthropic-style: replace content with a tool_use block
    block = type("ToolUse", (), {"type": "tool_use", "id": "call_injected",
                                 "name": name, "input": arguments})()  # fmt: skip
    if hasattr(resp, "content"):
        try:
            resp.content = [block]
            if hasattr(resp, "stop_reason"):
                resp.stop_reason = "tool_use"
        except Exception:
            pass
    elif isinstance(resp, dict):
        resp["content"] = [
            {"type": "tool_use", "id": "call_injected", "name": name, "input": arguments}
        ]
        resp["stop_reason"] = "tool_use"
    return resp


__all__ = [
    "FAULT_ATTR",
    "HTML_502",
    "FaultPolicy",
    "FaultRule",
    "InjectedFaultError",
    "current_policies",
    "faults",
    "llm_bad_tool_call",
    "llm_empty",
    "llm_error",
    "llm_truncate",
    "take_llm_fault",
    "take_tool_fault",
    "tool_delay",
    "tool_error",
    "tool_result",
]
