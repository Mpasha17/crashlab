"""Terminal ASCII waterfall of a recorded run: ``crashlab timeline``."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from .models import Handoff, Run, Step
from .otel import linked_runs
from .pricing import run_cost, step_cost
from .store import Store

__all__ = ["TimelineEvent", "render_timeline", "timeline_events"]

_BAR = "█"
_PAD = "░"
_HANDOFF = "↳"


def _utc(ts: datetime | None) -> datetime | None:
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _step_tokens(step: Step) -> tuple[int | None, int | None]:
    a = step.attributes or {}

    def _i(value: Any) -> int | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    return _i(a.get("gen_ai.usage.input_tokens")), _i(a.get("gen_ai.usage.output_tokens"))


def _step_usd(step: Step) -> float | None:
    raw = (step.attributes or {}).get("crashlab.usage.cost_usd")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    if step.operation != "chat":
        return None
    try:
        return step_cost(step)
    except Exception:
        return None


def _fmt_dur(ms: float) -> str:
    if ms < 1:
        return f"{ms:.2f}ms"
    if ms < 1000:
        return f"{ms:.0f}ms" if ms >= 10 else f"{ms:.1f}ms"
    return f"{ms / 1000:.2f}s"


def _fmt_cost(value: float | None) -> str:
    if value is None:
        return ""
    if value == 0:
        return "$0"
    return f"${value:.4f}" if value >= 0.0001 else f"${value:.6f}"


def _fmt_tokens(inp: int | None, out: int | None) -> str:
    if inp is None and out is None:
        return ""
    return f"{inp or 0}/{out or 0}"


def _window(run: Run) -> tuple[datetime, float]:
    """Start of the first step (or the run) and the span in ms covering every step."""
    starts = [_utc(s.started_at) for s in run.steps if s.started_at is not None]
    origin = min(starts) if starts else (_utc(run.started_at) or datetime.now(timezone.utc))
    span = 0.0
    for s in run.steps:
        st = _utc(s.started_at) or origin
        off = (st - origin).total_seconds() * 1000.0
        span = max(span, off + max(s.duration_ms, 0.0))
    if span <= 0 and run.usage.wall_time_ms:
        span = max(run.usage.wall_time_ms, 1.0)
    if span <= 0:
        span = 1.0
    return origin, span


def _markers(run: Run, step: Step, handoffs: list[Handoff]) -> str:
    marks: list[str] = []
    if step.error is not None:
        marks.append("ERR")
    kinds = []
    for f in run.findings:
        if f.step == step.idx or step.idx in (f.steps or []):
            if f.severity in ("error", "critical"):
                kinds.append("F!")
            elif f.severity == "warn":
                kinds.append("F")
            else:
                kinds.append("i")
    # unique, keep order
    for k in dict.fromkeys(kinds):
        marks.append(k)
    for h in handoffs:
        if h.via_step == step.idx:
            marks.append(f"{_HANDOFF}{h.to_agent}")
        elif h.step == step.idx:
            marks.append(f"from {h.from_agent}")
    return " ".join(marks)


def _label(step: Step) -> str:
    if step.operation == "execute_tool":
        return f"tool {step.name}"
    if step.operation == "chat":
        return f"chat {step.name}"
    return f"{step.operation} {step.name}".strip()


@dataclass
class TimelineEvent:
    """One row of the waterfall (a run header, a step, a handoff, or a child tree)."""

    kind: str  # run, step, child
    run_id: str
    name: str
    idx: int | None = None
    agent: str | None = None
    operation: str | None = None
    offset_ms: float = 0.0
    duration_ms: float = 0.0
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    error: str | None = None
    markers: str = ""
    depth: int = 0
    outcome: str | None = None
    status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run_events(run: Run, *, depth: int = 0) -> list[TimelineEvent]:
    origin, _span = _window(run)
    cost = None
    try:
        cost = run_cost(run)
    except Exception:
        cost = None
    start = _utc(run.started_at) or origin
    if run.ended_at is not None:
        dur = max((_utc(run.ended_at) - start).total_seconds() * 1000.0, 0.0)  # type: ignore[operator]
    elif run.usage.wall_time_ms is not None:
        dur = max(run.usage.wall_time_ms, 0.0)
    else:
        dur = max(_window(run)[1], 0.0)
    err = run.error.message if run.error is not None else None
    events = [
        TimelineEvent(
            kind="run",
            run_id=run.run_id,
            name=run.name,
            agent=run.name,
            offset_ms=0.0,
            duration_ms=dur,
            tokens_in=run.usage.input_tokens or None,
            tokens_out=run.usage.output_tokens or None,
            cost_usd=cost,
            error=err,
            depth=depth,
            outcome=run.outcome,
            status=run.status,
        )
    ]
    handoffs = run.handoffs() if run.agents else []
    instances = run.agent_instances() if run.agents else {}
    for step in sorted(run.steps, key=lambda s: s.idx):
        st = _utc(step.started_at) or origin
        off = max((st - origin).total_seconds() * 1000.0, 0.0)
        inp, out = _step_tokens(step)
        events.append(
            TimelineEvent(
                kind="step",
                run_id=run.run_id,
                name=_label(step),
                idx=step.idx,
                agent=instances.get(step.idx) or step.agent or run.name,
                operation=step.operation,
                offset_ms=off,
                duration_ms=max(step.duration_ms, 0.0),
                tokens_in=inp,
                tokens_out=out,
                cost_usd=_step_usd(step) if step.operation == "chat" else None,
                error=step.error.message if step.error is not None else None,
                markers=_markers(run, step, handoffs),
                depth=depth,
                outcome=run.outcome,
                status=run.status,
            )
        )
    return events


def timeline_events(
    run: Run, *, store: Store | None = None, include_children: bool = True
) -> list[TimelineEvent]:
    """Structured waterfall events for ``run`` (and linked children when ``store`` is given)."""
    tree = linked_runs(run, store=store, include_children=include_children)
    depth_of = {run.run_id: 0}
    events: list[TimelineEvent] = []
    for item in tree:
        parent = item.parent_run
        depth = (depth_of.get(parent, 0) + 1) if parent and parent in depth_of else 0
        if item.run_id == run.run_id:
            depth = 0
        depth_of[item.run_id] = depth
        events.extend(_run_events(item, depth=depth))
    return events


def _bar(offset_ms: float, duration_ms: float, span_ms: float, width: int) -> str:
    if width <= 0 or span_ms <= 0:
        return ""
    left = int(round(offset_ms / span_ms * width))
    raw = int(round(duration_ms / span_ms * width))
    w = max(raw, 1 if duration_ms > 0 else 0)
    if left + w > width:
        w = max(width - left, 0)
        if w == 0 and width:
            left = max(width - 1, 0)
            w = 1
    left = max(min(left, width - w), 0) if width else 0
    return " " * left + _BAR * w + _PAD * max(width - left - w, 0)


def render_timeline(
    run: Run,
    *,
    store: Store | None = None,
    include_children: bool = True,
    width: int = 100,
) -> str:
    """ASCII waterfall: step index, agent, operation, a timing bar, duration, tokens,
    cost, and error / finding / handoff markers. Linked child runs are indented under
    the parent when ``store`` is given.
    """
    events = timeline_events(run, store=store, include_children=include_children)
    width = max(int(width), 40)
    # layout: indent + idx(4) + agent(14) + name(22) + bar + dur + toks + cost + marks
    # the bar shrinks first when the line budget is tight
    lines: list[str] = []
    # group by run for per-run bar scaling
    groups: list[list[TimelineEvent]] = []
    current: list[TimelineEvent] = []
    for ev in events:
        if ev.kind == "run" and current:
            groups.append(current)
            current = [ev]
        else:
            current.append(ev)
    if current:
        groups.append(current)

    for group in groups:
        header = group[0]
        indent = "  " * header.depth
        bits = [header.name, header.run_id]
        if header.outcome:
            bits.append(header.outcome)
        if header.status and header.status != "ok":
            bits.append(header.status)
        bits.append(_fmt_dur(header.duration_ms))
        if header.tokens_in is not None or header.tokens_out is not None:
            bits.append(_fmt_tokens(header.tokens_in, header.tokens_out) + " tok")
        elif run.usage.total_tokens and header.depth == 0 and header.run_id == run.run_id:
            bits.append(f"{run.usage.total_tokens} tok")
        elif header.kind == "run":
            # prefer this run's own token total (looks better on child headers)
            pass
        cost_s = _fmt_cost(header.cost_usd)
        if cost_s:
            bits.append(cost_s)
        if header.error:
            bits.append(f"ERR {header.error}")
        prefix = "child " if header.depth else ""
        lines.append(f"{indent}{prefix}{'  '.join(bits)}")
        steps = [e for e in group if e.kind == "step"]
        if not steps:
            continue
        span = max((e.offset_ms + e.duration_ms) for e in steps) or 1.0
        # remaining columns after indent, idx, agent, name
        idx_w, agent_w, name_w = 4, 14, 22
        gutter = len(indent) + idx_w + 1 + agent_w + 1 + name_w + 1
        # reserve ~28 cols for dur / tokens / cost / markers
        bar_w = max(width - gutter - 28, 8)
        for ev in steps:
            idx = f"{ev.idx:<{idx_w}}" if ev.idx is not None else f"{'?':<{idx_w}}"
            agent = (ev.agent or "-")[:agent_w].ljust(agent_w)
            name = (ev.name or "")[:name_w].ljust(name_w)
            bar = _bar(ev.offset_ms, ev.duration_ms, span, bar_w)
            meta = [_fmt_dur(ev.duration_ms)]
            toks = _fmt_tokens(ev.tokens_in, ev.tokens_out)
            if toks:
                meta.append(toks)
            cost_s = _fmt_cost(ev.cost_usd)
            if cost_s:
                meta.append(cost_s)
            if ev.markers:
                meta.append(ev.markers)
            lines.append(f"{indent}{idx} {agent} {name} {bar}  {'  '.join(meta)}")
    return "\n".join(lines) + ("\n" if lines else "")
