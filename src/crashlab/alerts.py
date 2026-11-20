"""Live production alerts for recorded runs: webhooks, a JSONL file, or a callback.

A run can raise one or more :class:`AlertEvent` s — a failed outcome, a runtime
finding at or above a severity floor, a cost cap, a latency cap. :class:`AlertDispatcher`
sends them to one or more sinks and applies a cooldown so a tight loop of the same
failure does not spam a webhook. The key is ``(agent_name, alert_kind, signature)``.

No third-party HTTP stack: sinks use :mod:`urllib.request` and :mod:`pathlib`.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from .models import SEVERITY_RANK, Run

__all__ = [
    "AlertDispatcher",
    "AlertEvent",
    "AlertRule",
    "CallbackSink",
    "JsonlSink",
    "WebhookSink",
    "evaluate_run_alerts",
]

log = logging.getLogger("crashlab.alerts")
log.addHandler(logging.NullHandler())

AlertKind = Literal["failed", "finding", "cost", "latency"]
WebhookFormat = Literal["json", "slack", "discord"]
_WEBHOOK_FORMATS = ("json", "slack", "discord")
_WEBHOOK_PATH_MARKERS = ("/services/", "/webhooks/", "/hooks/")


def _safe_url(url: str) -> str:
    """A URL safe to put in a log line: no query string, no webhook path token."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<invalid-url>"
    path = parts.path or ""
    lower = path.lower()
    redacted = path
    for marker in _WEBHOOK_PATH_MARKERS:
        i = lower.find(marker)
        if i >= 0:
            redacted = path[: i + len(marker)] + "***"
            break
    return urlunsplit((parts.scheme, parts.netloc, redacted, "", ""))


def _latency_ms(run: Run) -> float | None:
    if run.usage.wall_time_ms is not None:
        return max(float(run.usage.wall_time_ms), 0.0)
    if run.ended_at is not None and run.started_at is not None:
        return max((run.ended_at - run.started_at).total_seconds() * 1000.0, 0.0)
    return None


def _cost_usd(run: Run) -> float | None:
    cost = run.usage.cost_usd
    return None if cost is None else float(cost)


def _iso(now: datetime | None = None) -> str:
    stamp = now or datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.isoformat()


def _failed_signature(run: Run) -> str:
    if run.error is not None:
        err = run.error.type.rsplit(".", 1)[-1]
    elif run.status == "error":
        err = "error"
    else:
        err = "fail"
    return f"failed:{err}"


def _failed_message(run: Run) -> str:
    bits = [f"status={run.status}", f"outcome={run.outcome}"]
    if run.error is not None:
        bits.append(f"raised {run.error.type.rsplit('.', 1)[-1]}")
    if run.outcome_reason:
        bits.append(run.outcome_reason)
    return "run failed: " + " ".join(bits)


@dataclass
class AlertRule:
    """What about a finished run should raise an alert.

    Failed outcomes (``outcome="fail"`` or ``status="error"``) always raise when this
    rule is used. Findings use ``min_severity`` as a floor. Cost and latency caps are
    off until a threshold is set. ``cooldown_seconds`` is applied by the dispatcher,
    not by :func:`evaluate_run_alerts`.
    """

    min_severity: str = "warn"
    max_cost_usd: float | None = None
    max_latency_ms: float | None = None
    cooldown_seconds: float = 300.0

    def __post_init__(self) -> None:
        if self.min_severity not in SEVERITY_RANK:
            raise ValueError(f"min_severity must be one of {', '.join(SEVERITY_RANK)}")
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be >= 0")
        if self.max_cost_usd is not None and self.max_cost_usd < 0:
            raise ValueError("max_cost_usd must be >= 0")
        if self.max_latency_ms is not None and self.max_latency_ms < 0:
            raise ValueError("max_latency_ms must be >= 0")


@dataclass
class AlertEvent:
    """One alert raised from a run. ``signature`` groups repeats for the cooldown."""

    kind: AlertKind
    signature: str
    run_id: str
    agent_name: str
    message: str
    severity: str = "error"
    status: str = ""
    outcome: str = ""
    cost_usd: float | None = None
    latency_ms: float | None = None
    finding_kind: str | None = None
    created_at: str = field(default_factory=_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def text(self) -> str:
        """One-line Slack / Discord body."""
        return f"[crashlab] {self.agent_name}: {self.message}"


def evaluate_run_alerts(run: Run, rule: AlertRule) -> list[AlertEvent]:
    """Return the alerts ``rule`` raises for ``run``, with no cooldown or delivery."""
    events: list[AlertEvent] = []
    cost = _cost_usd(run)
    latency = _latency_ms(run)
    base = dict(
        run_id=run.run_id,
        agent_name=run.name,
        status=run.status,
        outcome=run.outcome,
        cost_usd=cost,
        latency_ms=latency,
    )
    if run.outcome == "fail" or run.status == "error" or run.error is not None:
        events.append(
            AlertEvent(
                kind="failed",
                signature=_failed_signature(run),
                message=_failed_message(run),
                severity="error",
                **base,
            )
        )
    floor = SEVERITY_RANK[rule.min_severity]
    for finding in run.findings:
        if SEVERITY_RANK.get(finding.severity, 0) < floor:
            continue
        step = f" step {finding.step}" if finding.step is not None else ""
        events.append(
            AlertEvent(
                kind="finding",
                signature=f"finding:{finding.kind}",
                message=f"finding {finding.kind} [{finding.severity}]{step}: {finding.message}",
                severity=finding.severity,
                finding_kind=finding.kind,
                **base,
            )
        )
    if rule.max_cost_usd is not None and cost is not None and cost > rule.max_cost_usd:
        events.append(
            AlertEvent(
                kind="cost",
                signature="cost",
                message=f"cost {cost:.4f} USD exceeds max {rule.max_cost_usd:.4f} USD",
                severity="error",
                **base,
            )
        )
    if rule.max_latency_ms is not None and latency is not None and latency > rule.max_latency_ms:
        events.append(
            AlertEvent(
                kind="latency",
                signature="latency",
                message=(f"latency {latency:.1f} ms exceeds max {rule.max_latency_ms:.1f} ms"),
                severity="error",
                **base,
            )
        )
    return events


class WebhookSink:
    """POST one JSON payload per event to ``url``.

    ``format="json"`` is the event as an object; ``slack`` is ``{"text": ...}`` for
    an Incoming Webhook; ``discord`` is ``{"content": ...}``. ``headers`` are merged
    onto the request (``Content-Type`` is always JSON). Delivery errors are logged
    with the URL redacted; they are not raised.
    """

    def __init__(
        self,
        url: str,
        format: str = "json",
        headers: dict[str, str] | None = None,
        timeout: float = 5.0,
    ) -> None:
        if format not in _WEBHOOK_FORMATS:
            raise ValueError(f"format must be one of {', '.join(_WEBHOOK_FORMATS)}")
        if timeout <= 0:
            raise ValueError("timeout must be > 0")
        self.url = url
        self.format: WebhookFormat = format  # type: ignore[assignment]
        self.headers = dict(headers or {})
        self.timeout = timeout

    def payload(self, event: AlertEvent) -> dict[str, Any]:
        if self.format == "slack":
            return {"text": event.text()}
        if self.format == "discord":
            return {"content": event.text()}
        return event.to_dict()

    def emit(self, event: AlertEvent) -> None:
        body = json.dumps(self.payload(event), ensure_ascii=False, default=str).encode("utf-8")
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "crashlab-alerts",
            **self.headers,
        }
        req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(req, timeout=self.timeout) as resp:
                resp.read()
        except Exception as exc:  # delivery must not break watch
            _log_webhook_error(self.url, exc)


def _log_webhook_error(url: str, exc: BaseException) -> None:
    status = ""
    if isinstance(exc, urllib.error.HTTPError):
        status = f" HTTP {exc.code}"
    # do not log str(exc): urlopen sometimes repeats the request URL
    log.warning("webhook%s delivery failed for %s: %s", status, _safe_url(url), type(exc).__name__)


class JsonlSink:
    """Append one JSON object per event to ``path`` (created if needed)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def emit(self, event: AlertEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event.to_dict(), ensure_ascii=False, default=str) + "\n"
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line)


class CallbackSink:
    """Call ``fn(event)`` for each event (tests and in-process hooks)."""

    def __init__(self, fn: Callable[[AlertEvent], Any]) -> None:
        self.fn = fn

    def emit(self, event: AlertEvent) -> None:
        self.fn(event)


class AlertDispatcher:
    """Evaluate a run against a rule and send new events to every sink.

    Cooldown is keyed by ``(agent_name, alert_kind, signature)``. A matching event
    inside the window is dropped, including when a previous delivery failed, so a
    tight loop cannot hammer a down webhook. ``clock`` is a monotonic-style
    ``() -> seconds`` hook for tests (default :func:`time.monotonic`).
    """

    def __init__(
        self,
        rule: AlertRule,
        sinks: Sequence[Any] | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.rule = rule
        self.sinks = list(sinks or [])
        self.clock = clock
        self._last: dict[tuple[str, str, str], float] = {}

    def dispatch(self, run: Run) -> list[AlertEvent]:
        sent: list[AlertEvent] = []
        now = self.clock()
        for ev in evaluate_run_alerts(run, self.rule):
            key = (ev.agent_name, ev.kind, ev.signature)
            last = self._last.get(key)
            if last is not None and (now - last) < self.rule.cooldown_seconds:
                continue
            self._last[key] = now
            for sink in self.sinks:
                try:
                    sink.emit(ev)
                except Exception:  # a bad callback must not skip the other sinks
                    log.exception("alert sink %s failed", type(sink).__name__)
            sent.append(ev)
        return sent
