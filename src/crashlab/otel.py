"""Export a recorded run as OpenTelemetry OTLP/JSON or a Chrome trace.

Step attributes already follow the OpenTelemetry GenAI semantic conventions, so the
export is a straight mapping plus a root run span, parent links, and error / finding
events. The conventions are still in Development status.

Load a Chrome trace in https://ui.perfetto.dev (or chrome://tracing).
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any

from ._version import __version__
from .models import Finding, Run, Step
from .pricing import run_cost, step_cost
from .store import Store

__all__ = ["linked_runs", "to_chrome_trace", "to_otlp"]

_HEX16 = re.compile(r"^[0-9a-fA-F]{16}$")
_HEX32 = re.compile(r"^[0-9a-fA-F]{32}$")

# OTLP status / span kind (proto enum numbers)
_STATUS_UNSET, _STATUS_OK, _STATUS_ERROR = 0, 1, 2
_KIND_INTERNAL, _KIND_CLIENT = 1, 3


def _hex_id(value: str, n: int) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:n]


def _span_hex(value: str | None, fallback: str) -> str:
    if value and _HEX16.fullmatch(value):
        return value.lower()
    return _hex_id(fallback, 16)


def _trace_hex(value: str) -> str:
    if _HEX32.fullmatch(value):
        return value.lower()
    return _hex_id(f"crashlab.trace:{value}", 32)


def _as_utc(ts: datetime | None) -> datetime | None:
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _nano(ts: datetime) -> str:
    ts = _as_utc(ts) or ts
    return str(int(ts.timestamp() * 1_000_000_000))


def _add_ms(ts: datetime, duration_ms: float) -> datetime:
    from datetime import timedelta

    return ts + timedelta(milliseconds=max(duration_ms, 0.0))


def linked_runs(
    run: Run, *, store: Store | None = None, include_children: bool = True
) -> list[Run]:
    """``run`` then its linked children (and theirs), depth-first, oldest sibling first.

    Cycles and unreadable rows are skipped. Without a store, or with
    ``include_children=False``, this is just ``[run]``.
    """
    out = [run]
    if not include_children or store is None:
        return out
    seen = {run.run_id}

    def walk(parent: Run) -> None:
        try:
            rows = store.child_runs(parent.run_id)
        except Exception:  # a broken index must not abort an export
            return
        for row in rows:
            rid = row.get("run_id")
            if not isinstance(rid, str) or rid in seen:
                continue
            seen.add(rid)
            try:
                child = store.load_run(rid)
            except Exception:
                continue
            out.append(child)
            walk(child)

    walk(run)
    return out


def _run_span_id(run: Run) -> str:
    return _span_hex(run.span_id, f"crashlab.run:{run.run_id}")


def _step_span_id(run: Run, step: Step) -> str:
    return _span_hex(step.span_id, f"crashlab.step:{run.run_id}:{step.idx}")


def _parent_link(run: Run) -> dict[str, Any]:
    link = (run.env or {}).get("link")
    return link if isinstance(link, dict) else {}


def _parent_span_for_child(child: Run, parent: Run | None) -> str | None:
    """Span id a child run's root should point at: a named parent step / span, else the
    parent run's root. ``None`` when the parent is not available."""
    if parent is None:
        return None
    link = _parent_link(child)
    raw_span = link.get("parent_span") or link.get("parent_span_id")
    if isinstance(raw_span, str) and raw_span.strip():
        return _span_hex(raw_span, f"crashlab.parent_span:{parent.run_id}:{raw_span}")
    raw_step = link.get("parent_step")
    if raw_step is not None and not isinstance(raw_step, bool):
        try:
            idx = int(raw_step)
        except (TypeError, ValueError):
            idx = None
        if idx is not None:
            step = next((s for s in parent.steps if s.idx == idx), None)
            if step is not None:
                return _step_span_id(parent, step)
    return _run_span_id(parent)


def _step_parent_id(run: Run, step: Step, root_id: str) -> str:
    """Tool steps hang off the chat call that requested them when we can see it; otherwise
    a recorded parent span (if it is in this run) or the run root."""
    if step.operation == "execute_tool":
        caller = run.caller_chat(step.idx)
        if caller is not None:
            chat = next((s for s in run.steps if s.idx == caller), None)
            if chat is not None:
                return _step_span_id(run, chat)
    if step.parent_span_id:
        owner = next((s for s in run.steps if s.span_id == step.parent_span_id), None)
        if owner is not None and owner.idx != step.idx:
            return _step_span_id(run, owner)
        if _HEX16.fullmatch(step.parent_span_id) and step.parent_span_id.lower() != _step_span_id(
            run, step
        ):
            return step.parent_span_id.lower()
    return root_id


def _attr(key: str, value: Any) -> dict[str, Any] | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    if isinstance(value, (list, tuple)):
        return {
            "key": key,
            "value": {"arrayValue": {"values": [{"stringValue": str(v)} for v in value]}},
        }
    return {"key": key, "value": {"stringValue": str(value)}}


def _attrs(*pairs: tuple[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, value in pairs:
        item = _attr(key, value)
        if item is not None:
            out.append(item)
    return out


def _step_tokens(step: Step) -> tuple[int | None, int | None]:
    a = step.attributes or {}
    inp, out = a.get("gen_ai.usage.input_tokens"), a.get("gen_ai.usage.output_tokens")
    try:
        inp_i = int(inp) if inp is not None and not isinstance(inp, bool) else None
    except (TypeError, ValueError):
        inp_i = None
    try:
        out_i = int(out) if out is not None and not isinstance(out, bool) else None
    except (TypeError, ValueError):
        out_i = None
    return inp_i, out_i


def _step_usd(step: Step) -> float | None:
    raw = (step.attributes or {}).get("crashlab.usage.cost_usd")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    try:
        return step_cost(step)
    except Exception:
        return None


def _provider(step: Step) -> str | None:
    a = step.attributes or {}
    for key in ("gen_ai.system", "gen_ai.provider.name"):
        value = a.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _model(step: Step, which: str) -> str | None:
    a = step.attributes or {}
    value = a.get(f"gen_ai.{which}.model")
    if isinstance(value, str) and value.strip():
        return value
    if which == "request" and step.operation == "chat" and step.name:
        return step.name
    if which == "response":
        model = (step.response or {}).get("model")
        return model if isinstance(model, str) and model else None
    return None


def _run_window(run: Run) -> tuple[datetime, datetime]:
    start = _as_utc(run.started_at) or datetime.now(timezone.utc)
    if run.ended_at is not None:
        end = _as_utc(run.ended_at) or start
        if end < start:
            end = start
        return start, end
    last = start
    for step in run.steps:
        st = _as_utc(step.started_at) or start
        en = _add_ms(st, step.duration_ms)
        if en > last:
            last = en
    if run.usage.wall_time_ms is not None:
        wall_end = _add_ms(start, run.usage.wall_time_ms)
        if wall_end > last:
            last = wall_end
    return start, last if last >= start else start


def _findings_for_step(run: Run, idx: int) -> list[Finding]:
    out = []
    for f in run.findings:
        if f.step == idx or idx in (f.steps or []):
            out.append(f)
    return out


def _error_event(ts: datetime, typ: str, message: str, **extra: Any) -> dict[str, Any]:
    attrs = _attrs(("exception.type", typ), ("exception.message", message))
    for key, value in extra.items():
        item = _attr(key, value)
        if item is not None:
            attrs.append(item)
    return {"timeUnixNano": _nano(ts), "name": "exception", "attributes": attrs}


def _finding_event(ts: datetime, finding: Finding) -> dict[str, Any]:
    return {
        "timeUnixNano": _nano(ts),
        "name": "crashlab.finding",
        "attributes": _attrs(
            ("crashlab.finding.kind", finding.kind),
            ("crashlab.finding.severity", finding.severity),
            ("crashlab.finding.message", finding.message),
            ("crashlab.finding.stopped_run", finding.stopped_run),
        ),
    }


def _run_span(run: Run, trace_id: str, parent_span_id: str | None) -> dict[str, Any]:
    start, end = _run_window(run)
    events: list[dict[str, Any]] = []
    status_code = _STATUS_OK if run.status == "ok" else _STATUS_UNSET
    status_msg = None
    if run.status == "error" or run.error is not None:
        status_code = _STATUS_ERROR
        err = run.error
        status_msg = err.message if err is not None else "run error"
        events.append(
            _error_event(
                end,
                err.type if err is not None else "error",
                status_msg,
                **(
                    {"exception.stacktrace": err.traceback}
                    if err is not None and err.traceback
                    else {}
                ),
            )
        )
    elif run.outcome == "fail":
        status_code = _STATUS_ERROR
        status_msg = run.outcome_reason or "outcome=fail"
    for f in run.findings:
        if f.step is not None:
            continue
        events.append(_finding_event(end, f))
    inp = run.usage.input_tokens or None
    out = run.usage.output_tokens or None
    cost = run_cost(run)
    status: dict[str, Any] = {"code": status_code}
    if status_msg:
        status["message"] = status_msg
    span: dict[str, Any] = {
        "traceId": trace_id,
        "spanId": _run_span_id(run),
        "name": run.name or "run",
        "kind": _KIND_INTERNAL,
        "startTimeUnixNano": _nano(start),
        "endTimeUnixNano": _nano(end),
        "attributes": _attrs(
            ("gen_ai.operation.name", "invoke_agent"),
            ("gen_ai.agent.name", run.name),
            ("crashlab.run_id", run.run_id),
            ("crashlab.outcome", run.outcome),
            ("crashlab.status", run.status),
            ("crashlab.cost_usd", cost),
            ("gen_ai.usage.input_tokens", inp if inp else None),
            ("gen_ai.usage.output_tokens", out if out else None),
        ),
        "status": status,
    }
    if parent_span_id:
        span["parentSpanId"] = parent_span_id
    if events:
        span["events"] = events
    return span


def _step_span(run: Run, step: Step, trace_id: str, parent_span_id: str) -> dict[str, Any]:
    start = _as_utc(step.started_at) or _as_utc(run.started_at) or datetime.now(timezone.utc)
    end = _add_ms(start, step.duration_ms)
    inp, out_tok = _step_tokens(step)
    cost = _step_usd(step) if step.operation == "chat" else None
    system = _provider(step)
    tool_id = (step.attributes or {}).get("gen_ai.tool.call.id")
    name = step.name or step.operation
    if step.operation == "execute_tool":
        span_name = f"tool {name}"
        kind = _KIND_INTERNAL
        op = (step.attributes or {}).get("gen_ai.operation.name") or "execute_tool"
    elif step.operation in ("chat", "embeddings"):
        span_name = f"{step.operation} {name}"
        kind = _KIND_CLIENT
        op = (step.attributes or {}).get("gen_ai.operation.name") or step.operation
    else:
        span_name = f"{step.operation} {name}".strip()
        kind = _KIND_INTERNAL
        op = (step.attributes or {}).get("gen_ai.operation.name") or step.operation
    events: list[dict[str, Any]] = []
    status_code, status_msg = _STATUS_OK, None
    if step.error is not None:
        status_code = _STATUS_ERROR
        status_msg = step.error.message
        extra = {}
        if step.error.traceback:
            extra["exception.stacktrace"] = step.error.traceback
        if step.error.is_tool_error:
            extra["crashlab.is_tool_error"] = True
        events.append(_error_event(end, step.error.type, step.error.message, **extra))
    for f in _findings_for_step(run, step.idx):
        events.append(_finding_event(end, f))
        if f.severity in ("error", "critical") and status_code != _STATUS_ERROR:
            status_code = _STATUS_ERROR
            status_msg = status_msg or f.message
    status: dict[str, Any] = {"code": status_code}
    if status_msg:
        status["message"] = status_msg
    span: dict[str, Any] = {
        "traceId": trace_id,
        "spanId": _step_span_id(run, step),
        "parentSpanId": parent_span_id,
        "name": span_name,
        "kind": kind,
        "startTimeUnixNano": _nano(start),
        "endTimeUnixNano": _nano(end),
        "attributes": _attrs(
            ("gen_ai.system", system),
            ("gen_ai.operation.name", op),
            ("gen_ai.request.model", _model(step, "request")),
            ("gen_ai.response.model", _model(step, "response")),
            ("gen_ai.usage.input_tokens", inp),
            ("gen_ai.usage.output_tokens", out_tok),
            ("gen_ai.agent.name", step.agent),
            ("gen_ai.tool.name", step.name if step.operation == "execute_tool" else None),
            ("gen_ai.tool.call.id", tool_id if isinstance(tool_id, str) else None),
            ("crashlab.cost_usd", cost),
            ("crashlab.run_id", run.run_id),
            ("crashlab.outcome", run.outcome),
            ("crashlab.step", step.idx),
            ("crashlab.source", step.source),
        ),
        "status": status,
    }
    if events:
        span["events"] = events
    return span


def _parent_map(runs: list[Run], store: Store | None) -> dict[str, Run]:
    by_id = {r.run_id: r for r in runs}
    if store is None:
        return by_id
    for run in runs:
        pid = run.parent_run
        if pid and pid not in by_id:
            try:
                by_id[pid] = store.load_run(pid)
            except Exception:
                continue
    return by_id


def to_otlp(
    run: Run, *, store: Store | None = None, include_children: bool = True
) -> dict[str, Any]:
    """Export ``run`` (and linked child runs when ``store`` is given) as OTLP/JSON.

    The payload has the usual ``resourceSpans`` → ``scopeSpans`` → ``spans`` shape, with
    GenAI semantic conventions on each span, a root span per run, tool steps parented to
    the chat call that requested them, and child-run roots parented to the parent run (or
    a ``parent_step`` / ``parent_span`` in ``run.env["link"]`` when set).
    """
    runs = linked_runs(run, store=store, include_children=include_children)
    parents = _parent_map(runs, store)
    # one trace for the tree we were asked to export
    trace_id = _trace_hex(run.trace_id)
    spans: list[dict[str, Any]] = []
    for item in runs:
        parent = parents.get(item.parent_run) if item.parent_run else None
        # a linked child keeps a parentSpanId even when the parent is not in this export
        parent_span = _parent_span_for_child(item, parent) if parent is not None else None
        root_id = _run_span_id(item)
        spans.append(_run_span(item, trace_id, parent_span))
        for step in sorted(item.steps, key=lambda s: s.idx):
            spans.append(_step_span(item, step, trace_id, _step_parent_id(item, step, root_id)))
    resource_attrs = _attrs(
        ("service.name", "crashlab"),
        ("service.version", __version__),
        ("telemetry.sdk.language", "python"),
        ("telemetry.sdk.name", "crashlab"),
        ("crashlab.run_id", run.run_id),
        ("crashlab.outcome", run.outcome),
        ("crashlab.trace_id", run.trace_id),
    )
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": resource_attrs},
                "scopeSpans": [
                    {
                        "scope": {"name": "crashlab", "version": __version__},
                        "spans": spans,
                    }
                ],
            }
        ]
    }


def _chrome_pid(run: Run) -> str:
    return (
        f"{run.name} ({run.run_id[-8:]})" if len(run.run_id) >= 8 else f"{run.name} ({run.run_id})"
    )


def _chrome_tid(run: Run, step: Step | None) -> str:
    if step is not None and step.agent:
        return step.agent
    if run.agents:
        return run.agents[0]
    return run.name or "run"


def _us(ts: datetime) -> int:
    ts = _as_utc(ts) or ts
    return int(ts.timestamp() * 1_000_000)


def _chrome_event(
    *,
    name: str,
    cat: str,
    start: datetime,
    duration_ms: float,
    pid: str,
    tid: str,
    args: dict[str, Any],
) -> dict[str, Any]:
    dur = max(int(round(max(duration_ms, 0.0) * 1000)), 0)
    ev: dict[str, Any] = {
        "name": name,
        "cat": cat,
        "ph": "X",
        "ts": _us(start),
        "dur": max(dur, 0),
        "pid": pid,
        "tid": tid,
        "args": args,
    }
    return ev


def to_chrome_trace(
    run: Run, *, store: Store | None = None, include_children: bool = True
) -> dict[str, Any]:
    """Export ``run`` (and linked child runs when ``store`` is given) as a Chrome
    ``traceEvents`` log, loadable in https://ui.perfetto.dev. Timestamps are microseconds
    (Chrome / Perfetto convention); ``displayTimeUnit`` is ``ms``.
    """
    runs = linked_runs(run, store=store, include_children=include_children)
    events: list[dict[str, Any]] = []
    # metadata events help Perfetto label process / thread swimlanes
    seen_pid: set[str] = set()
    seen_tid: set[tuple[str, str]] = set()
    for item in runs:
        pid = _chrome_pid(item)
        start, end = _run_window(item)
        dur_ms = max((end - start).total_seconds() * 1000.0, 0.0)
        if pid not in seen_pid:
            events.append({"name": "process_name", "ph": "M", "pid": pid, "args": {"name": pid}})
            seen_pid.add(pid)
        tid0 = _chrome_tid(item, None)
        if (pid, tid0) not in seen_tid:
            events.append(
                {"name": "thread_name", "ph": "M", "pid": pid, "tid": tid0, "args": {"name": tid0}}
            )
            seen_tid.add((pid, tid0))
        cost = run_cost(item)
        events.append(
            _chrome_event(
                name=item.name or "run",
                cat="run",
                start=start,
                duration_ms=dur_ms,
                pid=pid,
                tid=tid0,
                args={
                    "crashlab.run_id": item.run_id,
                    "crashlab.outcome": item.outcome,
                    "status": item.status,
                    "tokens": item.usage.total_tokens,
                    "cost": cost,
                    "error": item.error.message if item.error is not None else None,
                },
            )
        )
        for step in sorted(item.steps, key=lambda s: s.idx):
            tid = _chrome_tid(item, step)
            if (pid, tid) not in seen_tid:
                events.append(
                    {
                        "name": "thread_name",
                        "ph": "M",
                        "pid": pid,
                        "tid": tid,
                        "args": {"name": tid},
                    }
                )
                seen_tid.add((pid, tid))
            st = _as_utc(step.started_at) or start
            inp, out_tok = _step_tokens(step)
            usd = _step_usd(step) if step.operation == "chat" else None
            err = step.error.message if step.error is not None else None
            cat = "tool" if step.operation == "execute_tool" else step.operation
            if step.error is not None:
                cat = f"{cat},error"
            name = (
                f"tool {step.name}"
                if step.operation == "execute_tool"
                else f"{step.operation} {step.name}"
            ).strip()
            events.append(
                _chrome_event(
                    name=name,
                    cat=cat,
                    start=st,
                    duration_ms=step.duration_ms,
                    pid=pid,
                    tid=tid,
                    args={
                        "step": step.idx,
                        "operation": step.operation,
                        "agent": step.agent,
                        "tokens_in": inp,
                        "tokens_out": out_tok,
                        "cost": usd,
                        "error": err,
                        "crashlab.run_id": item.run_id,
                    },
                )
            )
    return {"displayTimeUnit": "ms", "traceEvents": events}
