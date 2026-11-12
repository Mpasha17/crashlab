"""Cost / token / latency budget oracles for ``@crashlab.check``.

Deterministic: everything is read from the recorded run (token counts the provider
reported on each chat step, prices from :mod:`crashlab.pricing`, step durations and the
run's wall time). Nothing is estimated from text.

::

    import crashlab
    from crashlab.budgets import within_budget

    @crashlab.check("refund-bot")
    def cheap_and_fast(run):
        return within_budget(run, max_cost_usd=0.02, max_tokens=20_000, max_seconds=30)

``within_budget`` returns ``None`` when every limit holds and a failure message
otherwise, the same convention as the :mod:`crashlab.rag` helpers (a check that returns
a non-empty string fails with that message).

**Missing data is never a silent pass.** A limit whose number is not in the recording
(a chat call that reported no token usage, e.g. a stream without
``stream_options={"include_usage": True}``; a model that is not in the price table; the
wall time of a replay or fork, whose cassette-served steps take no real time) fails with
a message that says what is missing, unless you pass ``unknown="pass"``.

Scope: tokens and cost come from chat (LLM) steps; embeddings calls are not counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .models import Run, Step

__all__ = ["UsageSummary", "summarize_usage", "within_budget"]

Unknown = Literal["fail", "pass"]


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class UsageSummary:
    """What a run used, read from its recorded steps. ``None`` / non-empty ``*_missing``
    lists mean the recording does not have that number."""

    steps: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = 0.0
    wall_seconds: float | None = None
    tokens_missing: list[int] = field(default_factory=list)  # chat steps without usage
    cost_missing: list[int] = field(default_factory=list)  # chat steps with unknown price
    unpriced_models: list[str] = field(default_factory=list)
    slowest_call: tuple[int, float] | None = None  # (step idx, seconds), live steps only
    largest_call_input: tuple[int, int] | None = None  # (step idx, input tokens)
    first_call_input: int | None = None  # input tokens of the first chat call with usage
    served: list[int] = field(default_factory=list)  # steps served from a cassette
    wall_note: str | None = None  # why wall_seconds is None


def _chat_tokens(step: Step) -> tuple[int, int, int] | None:
    a = step.attributes or {}
    inp = _int_or_none(a.get("gen_ai.usage.input_tokens"))
    out = _int_or_none(a.get("gen_ai.usage.output_tokens"))
    total = _int_or_none(a.get("crashlab.usage.total_tokens"))
    if inp is None and out is None and total is None:
        return None
    inp, out = inp or 0, out or 0
    return inp, out, total if total is not None else inp + out


def summarize_usage(run: Run) -> UsageSummary:
    """Sum a run's usage from its steps (``run.steps``; falls back to the stored
    ``run.usage`` totals when the run was loaded without steps)."""
    from .pricing import step_cost

    u = UsageSummary()
    steps = sorted(run.steps, key=lambda s: s.idx)
    if not steps:
        return _from_stored_totals(run, u)
    cost = 0.0
    for s in steps:
        u.steps += 1
        if s.source == "cassette":
            u.served.append(s.idx)
        # fake / patched / blocked / served steps took no live time
        if s.source == "live" and s.operation in ("chat", "execute_tool"):
            secs = (s.duration_ms or 0.0) / 1000
            if u.slowest_call is None or secs > u.slowest_call[1]:
                u.slowest_call = (s.idx, secs)
        if s.operation == "execute_tool":
            u.tool_calls += 1
            continue
        if s.operation != "chat":
            continue
        u.llm_calls += 1
        toks = _chat_tokens(s)
        if toks is None:
            if s.error is None:  # a failed request reporting nothing used nothing
                u.tokens_missing.append(s.idx)
                u.cost_missing.append(s.idx)
            continue
        inp, out, total = toks
        u.input_tokens += inp
        u.output_tokens += out
        u.total_tokens += total
        if u.first_call_input is None:
            u.first_call_input = inp
        if u.largest_call_input is None or inp > u.largest_call_input[1]:
            u.largest_call_input = (s.idx, inp)
        c = s.attributes.get("crashlab.usage.cost_usd")
        c = step_cost(s) if c is None else c
        if c is None:
            u.cost_missing.append(s.idx)
            model = str(s.attributes.get("gen_ai.response.model")
                        or s.attributes.get("gen_ai.request.model") or s.name)  # fmt: skip
            if model not in u.unpriced_models:
                u.unpriced_models.append(model)
        else:
            cost += float(c)
    u.cost_usd = None if u.cost_missing else round(cost, 8)
    _wall(run, u)
    return u


def _from_stored_totals(run: Run, u: UsageSummary) -> UsageSummary:
    us = run.usage
    u.steps, u.llm_calls, u.tool_calls = us.steps, us.llm_calls, us.tool_calls
    u.input_tokens, u.output_tokens = us.input_tokens, us.output_tokens
    u.total_tokens = us.total_tokens or us.input_tokens + us.output_tokens
    if us.llm_calls and not u.total_tokens:
        u.tokens_missing = [-1]  # unknown which steps
    u.cost_usd = us.cost_usd
    if us.llm_calls and us.cost_usd is None:
        u.cost_missing = [-1]
    _wall(run, u)
    return u


def _wall(run: Run, u: UsageSummary) -> None:
    if run.replay_of is not None or run.forked_from is not None or u.served:
        u.wall_note = "replays and forks serve recorded steps instantly: not live time"
        return
    if run.usage.wall_time_ms is not None:
        u.wall_seconds = run.usage.wall_time_ms / 1000
    elif run.ended_at is not None:
        try:
            u.wall_seconds = max((run.ended_at - run.started_at).total_seconds(), 0.0)
        except (TypeError, ValueError):  # naive vs aware datetimes in old data
            u.wall_note = "start / end times are not comparable"
    else:
        u.wall_note = "the run has not ended"


def _steps(idxs: list[int]) -> str:
    if idxs == [-1]:
        return "some chat calls"
    shown = ", ".join(str(i) for i in idxs[:8])
    return f"chat step{'s' if len(idxs) > 1 else ''} {shown}" + (
        f" (+{len(idxs) - 8} more)" if len(idxs) > 8 else ""
    )


def within_budget(
    run: Run,
    *,
    max_cost_usd: float | None = None,
    max_tokens: int | None = None,
    max_input_tokens: int | None = None,
    max_output_tokens: int | None = None,
    max_call_input_tokens: int | None = None,
    max_seconds: float | None = None,
    max_call_seconds: float | None = None,
    max_steps: int | None = None,
    max_llm_calls: int | None = None,
    max_tool_calls: int | None = None,
    unknown: Unknown = "fail",
) -> str | None:
    """``None`` when every given limit holds, else a message naming each one exceeded.

    - ``max_cost_usd``: summed estimated cost of the chat calls (``crashlab.pricing``).
    - ``max_tokens`` / ``max_input_tokens`` / ``max_output_tokens``: summed over chat
      calls (``max_tokens`` uses the provider's total when it reports one).
    - ``max_call_input_tokens``: the largest single prompt (context size of one call).
    - ``max_seconds``: the run's wall time. ``max_call_seconds``: the slowest single
      LLM or tool call (live steps only).
    - ``max_steps`` / ``max_llm_calls`` / ``max_tool_calls``: counts.

    ``unknown="fail"`` (default): a limit whose number is missing from the recording
    fails and says why. ``unknown="pass"``: such limits are skipped.
    """
    if unknown not in ("fail", "pass"):
        raise ValueError(f"unknown= must be 'fail' or 'pass', not {unknown!r}")
    limits = {
        "max_cost_usd": max_cost_usd,
        "max_tokens": max_tokens,
        "max_input_tokens": max_input_tokens,
        "max_output_tokens": max_output_tokens,
        "max_call_input_tokens": max_call_input_tokens,
        "max_seconds": max_seconds,
        "max_call_seconds": max_call_seconds,
        "max_steps": max_steps,
        "max_llm_calls": max_llm_calls,
        "max_tool_calls": max_tool_calls,
    }
    for name, value in limits.items():
        if value is not None and (isinstance(value, bool) or value < 0):
            raise ValueError(f"{name} must be a non-negative number, not {value!r}")
    u = summarize_usage(run)
    over: list[str] = []
    missing: list[str] = []

    def cmp(name: str, value: float, fmt: str, where: str = "") -> None:
        limit = limits[name]
        if limit is not None and value > limit:
            over.append(f"{fmt.format(value)}{where} > {name} {fmt.format(limit)}")

    for name, value in (("max_steps", u.steps), ("max_llm_calls", u.llm_calls),
                        ("max_tool_calls", u.tool_calls)):  # fmt: skip
        cmp(name, value, "{:,}")
    token_limits = ("max_tokens", "max_input_tokens", "max_output_tokens",
                    "max_call_input_tokens")  # fmt: skip
    if u.tokens_missing and any(limits[n] is not None for n in token_limits):
        missing.append(
            f"token usage not recorded for {_steps(u.tokens_missing)} "
            "(a stream without include_usage, or a provider that sends none)"
        )
    cmp("max_tokens", u.total_tokens, "{:,}", " tokens")
    cmp("max_input_tokens", u.input_tokens, "{:,}", " input tokens")
    cmp("max_output_tokens", u.output_tokens, "{:,}", " output tokens")
    if u.largest_call_input is not None:
        idx, toks = u.largest_call_input
        grown = (f", grew from {u.first_call_input:,}" if u.first_call_input is not None
                 and u.first_call_input < toks else "")  # fmt: skip
        cmp("max_call_input_tokens", toks, "{:,}", f" input tokens at step {idx}{grown}")
    if max_cost_usd is not None:
        if u.cost_usd is None:
            models = f": {', '.join(u.unpriced_models)}" if u.unpriced_models else ""
            missing.append(f"cost unknown for {_steps(u.cost_missing)} (no usage, or a "
                           f"model not in the price table{models})")  # fmt: skip
        else:
            cmp("max_cost_usd", u.cost_usd, "${:.4f}")
    if max_seconds is not None:
        if u.wall_seconds is None:
            missing.append(f"wall time not measurable: {u.wall_note}")
        else:
            cmp("max_seconds", u.wall_seconds, "{:.1f}s")
    if max_call_seconds is not None:
        if u.slowest_call is None:
            missing.append("no live LLM or tool call to time")
        else:
            idx, secs = u.slowest_call
            cmp("max_call_seconds", secs, "{:.2f}s", f" at step {idx}")
    if missing and unknown == "fail":
        over += [f"cannot check: {m}" for m in missing]
    if not over:
        return None
    return "budget: " + "; ".join(over)
