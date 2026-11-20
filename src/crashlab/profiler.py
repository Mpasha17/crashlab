"""Cost, token and latency profiling of a recorded run: ``crashlab profile``."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .models import Run, Step
from .pricing import price_for, provider_of, run_cost, step_cost

__all__ = [
    "DEFAULT_COMPARE_MODELS",
    "CacheOpportunity",
    "ModelSwap",
    "RunProfile",
    "StepProfile",
    "ToolProfile",
    "profile_run",
]

# Cheap / mid / flagship defaults when the caller does not pass compare_models.
DEFAULT_COMPARE_MODELS: tuple[str, ...] = (
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-5",
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
)

_TOP_N = 5
Category = Literal["llm", "tool", "other"]


@dataclass
class StepProfile:
    """One step's share of a run's time, tokens and estimated cost."""

    idx: int
    operation: str
    name: str
    category: Category
    duration_ms: float
    duration_pct: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolProfile:
    """Aggregated latency and call count for one tool name."""

    name: str
    calls: int
    total_duration_ms: float
    mean_duration_ms: float
    max_duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CachePair:
    """Shared prompt prefix between two consecutive LLM steps."""

    from_idx: int
    to_idx: int
    prefix_chars: int
    prompt_chars: int
    estimated_redundant_tokens: int


@dataclass
class CacheOpportunity:
    """Repeated prompt-prefix estimate across consecutive LLM calls in one run."""

    consecutive_pairs: int = 0
    estimated_redundant_tokens: int = 0
    total_input_tokens: int = 0
    pairs: list[CachePair] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelSwap:
    """What-if cost of the same token counts billed at another model's list price."""

    model: str
    estimated_cost_usd: float | None
    priced: bool = True
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunProfile:
    """Step-by-step time, token, cost, cache-opportunity and model-swap profile."""

    run_id: str
    name: str
    total_duration_ms: float
    llm_duration_ms: float
    tool_duration_ms: float
    other_duration_ms: float
    llm_pct: float
    tool_pct: float
    other_pct: float
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    steps: list[StepProfile] = field(default_factory=list)
    slowest_steps: list[StepProfile] = field(default_factory=list)
    costliest_steps: list[StepProfile] = field(default_factory=list)
    tools: list[ToolProfile] = field(default_factory=list)
    cache: CacheOpportunity = field(default_factory=CacheOpportunity)
    what_if: list[ModelSwap] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_run(run: Run, *, compare_models: Iterable[str] | None = None) -> RunProfile:
    """Break down ``run`` by LLM vs tool time, tokens, cost, cache opportunity and swaps.

    ``compare_models`` is the list of model names for what-if cost estimates; when
    omitted, :data:`DEFAULT_COMPARE_MODELS` is used (cheap / mid / flagship).
    """
    models = list(compare_models) if compare_models is not None else list(DEFAULT_COMPARE_MODELS)
    models = [m.strip() for m in models if str(m).strip()]
    total_ms = _total_duration_ms(run)
    llm_ms = _sum_ms(run.steps, "chat")
    tool_ms = _sum_ms(run.steps, "execute_tool")
    other_step_ms = sum(
        float(s.duration_ms or 0) for s in run.steps if s.operation not in ("chat", "execute_tool")
    )
    # leftover wall-clock time between steps is overhead, not LLM or tool time
    leftover = max(0.0, total_ms - llm_ms - tool_ms - other_step_ms)
    other_ms = other_step_ms + leftover
    if total_ms <= 0:
        total_ms = llm_ms + tool_ms + other_ms
    llm_pct = _pct(llm_ms, total_ms)
    tool_pct = _pct(tool_ms, total_ms)
    # leftover percentage so the three shares always sum to 100
    other_pct = round(100.0 - llm_pct - tool_pct, 2) if total_ms > 0 else 0.0
    steps = [_step_profile(s, total_ms) for s in sorted(run.steps, key=lambda s: s.idx)]
    inp = sum(s.input_tokens for s in steps)
    out = sum(s.output_tokens for s in steps)
    if run.usage.input_tokens:
        inp = int(run.usage.input_tokens)
    if run.usage.output_tokens:
        out = int(run.usage.output_tokens)
    slowest = sorted(steps, key=lambda s: s.duration_ms, reverse=True)[:_TOP_N]
    with_cost = [s for s in steps if s.cost_usd is not None]
    costliest = sorted(with_cost, key=lambda s: s.cost_usd or 0.0, reverse=True)[:_TOP_N]
    return RunProfile(
        run_id=run.run_id,
        name=run.name,
        total_duration_ms=_round(total_ms),
        llm_duration_ms=_round(llm_ms),
        tool_duration_ms=_round(tool_ms),
        other_duration_ms=_round(other_ms),
        llm_pct=llm_pct,
        tool_pct=tool_pct,
        other_pct=other_pct,
        input_tokens=inp,
        output_tokens=out,
        cost_usd=run_cost(run),
        steps=steps,
        slowest_steps=slowest,
        costliest_steps=costliest,
        tools=_tool_leaderboard(run),
        cache=_cache_opportunity(run),
        what_if=_what_if(run, models),
    )


def _step_profile(step: Step, total_ms: float) -> StepProfile:
    duration = float(step.duration_ms or 0)
    inp, out = _step_tokens(step)
    cost = step.attributes.get("crashlab.usage.cost_usd") if step.attributes else None
    if cost is None:
        cost = step_cost(step)
    else:
        try:
            cost = float(cost)
        except (TypeError, ValueError):
            cost = step_cost(step)
    if step.operation == "chat":
        category: Category = "llm"
    elif step.operation == "execute_tool":
        category = "tool"
    else:
        category = "other"
    return StepProfile(
        idx=step.idx,
        operation=step.operation,
        name=step.name,
        category=category,
        duration_ms=_round(duration),
        duration_pct=_pct(duration, total_ms),
        input_tokens=inp,
        output_tokens=out,
        cost_usd=None if cost is None else round(float(cost), 8),
    )


def _step_tokens(step: Step) -> tuple[int, int]:
    attrs = step.attributes or {}
    return _as_int(attrs.get("gen_ai.usage.input_tokens")), _as_int(
        attrs.get("gen_ai.usage.output_tokens")
    )


def _tool_leaderboard(run: Run) -> list[ToolProfile]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for step in run.steps:
        if step.operation != "execute_tool":
            continue
        buckets[step.name].append(float(step.duration_ms or 0))
    out: list[ToolProfile] = []
    for name, durs in buckets.items():
        total = sum(durs)
        out.append(
            ToolProfile(
                name=name,
                calls=len(durs),
                total_duration_ms=_round(total),
                mean_duration_ms=_round(total / len(durs)),
                max_duration_ms=_round(max(durs)),
            )
        )
    out.sort(key=lambda t: (-t.total_duration_ms, -t.calls, t.name))
    return out


def _cache_opportunity(run: Run) -> CacheOpportunity:
    chats = [s for s in sorted(run.steps, key=lambda s: s.idx) if s.operation == "chat"]
    pairs: list[CachePair] = []
    redundant = 0
    total_input = 0
    for step in chats:
        total_input += _step_tokens(step)[0]
    for prev, cur in zip(chats, chats[1:], strict=False):
        prev_text = _prompt_text(prev)
        cur_text = _prompt_text(cur)
        prefix = _common_prefix_len(prev_text, cur_text)
        inp, _ = _step_tokens(cur)
        est = _prefix_tokens(prefix, len(cur_text), inp)
        pairs.append(
            CachePair(
                from_idx=prev.idx,
                to_idx=cur.idx,
                prefix_chars=prefix,
                prompt_chars=len(cur_text),
                estimated_redundant_tokens=est,
            )
        )
        redundant += est
    return CacheOpportunity(
        consecutive_pairs=len(pairs),
        estimated_redundant_tokens=redundant,
        total_input_tokens=total_input,
        pairs=pairs,
    )


def _prompt_text(step: Step) -> str:
    """A stable serialization of the prompt a chat step sent (system + messages + tools)."""
    req = step.request or {}
    parts: list[str] = []
    system = req.get("system")
    if system:
        parts.append(_as_text(system))
    for msg in req.get("messages") or []:
        parts.append(_as_text(msg))
    tools = req.get("tools")
    if tools:
        parts.append(_as_text(tools))
    return "\n".join(parts)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _prefix_tokens(prefix_chars: int, prompt_chars: int, reported_input: int) -> int:
    if prefix_chars <= 0:
        return 0
    if reported_input > 0 and prompt_chars > 0:
        return int(round(reported_input * (prefix_chars / prompt_chars)))
    # ~4 characters per token when the step has no usage
    return max(0, (prefix_chars + 3) // 4)


def _what_if(run: Run, models: list[str]) -> list[ModelSwap]:
    usage: list[tuple[int, int, Any]] = []
    for step in run.steps:
        if step.operation != "chat":
            continue
        inp, out = _step_tokens(step)
        if not (inp or out):
            continue
        usage.append((inp, out, step.started_at))
    out: list[ModelSwap] = []
    for model in models:
        total = 0.0
        priced = True
        for inp, output, when in usage:
            provider = _provider_guess(model)
            price = price_for(model, provider, when)
            if price is None:
                priced = False
                break
            total += (inp * price.input + output * price.output) / 1e6
        if not usage:
            out.append(
                ModelSwap(
                    model=model,
                    estimated_cost_usd=0.0 if price_for(model, _provider_guess(model)) else None,
                    priced=price_for(model, _provider_guess(model)) is not None,
                    note=None if price_for(model, _provider_guess(model)) else "no list price",
                )
            )
            continue
        if not priced:
            out.append(
                ModelSwap(model=model, estimated_cost_usd=None, priced=False, note="no list price")
            )
            continue
        out.append(ModelSwap(model=model, estimated_cost_usd=round(total, 8), priced=True))
    return out


def _provider_guess(model: str) -> str | None:
    name = model.lower()
    if name.startswith("claude") or name.startswith("anthropic/"):
        return "anthropic"
    if name.startswith("gemini") or name.startswith("gemini/"):
        return "gemini"
    if name.startswith("groq/") or "gpt-oss" in name:
        return "groq"
    if name.startswith(("gpt-", "o1", "o3", "o4", "openai/")):
        return "openai"
    return provider_of({}, model)


def _total_duration_ms(run: Run) -> float:
    if run.usage.wall_time_ms is not None:
        try:
            return max(0.0, float(run.usage.wall_time_ms))
        except (TypeError, ValueError):
            pass
    if run.ended_at is not None and run.started_at is not None:
        return max(0.0, (run.ended_at - run.started_at).total_seconds() * 1000.0)
    return sum(float(s.duration_ms or 0) for s in run.steps)


def _sum_ms(steps: list[Step], operation: str) -> float:
    return sum(float(s.duration_ms or 0) for s in steps if s.operation == operation)


def _as_int(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _pct(part: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return round(100.0 * part / total, 2)


def _round(value: float) -> float:
    return round(float(value), 3)
