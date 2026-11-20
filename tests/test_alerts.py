"""Live production alerts: evaluation, cooldown, sinks, and `crashlab watch` flags."""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from typer.testing import CliRunner

from crashlab.alerts import (
    AlertDispatcher,
    AlertEvent,
    AlertRule,
    CallbackSink,
    JsonlSink,
    WebhookSink,
    _safe_url,
    evaluate_run_alerts,
)
from crashlab.cli import app
from crashlab.models import ErrorInfo, Finding, Run, Usage
from crashlab.store import Store
from tests.debug_fixtures import make_run, run_id

runner = CliRunner()
ENV = {"COLUMNS": "200", "TERM": "xterm-256color", "NO_COLOR": "1"}
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _run(**kwargs) -> Run:
    vals = dict(run_id="r1", name="refund-bot", status="ok", outcome="pass")
    vals.update(kwargs)
    return Run(**vals)


def test_evaluate_failed_outcome_and_error_status():
    fail = _run(outcome="fail")
    err = _run(
        status="error", outcome="unknown", error=ErrorInfo(type="TimeoutError", message="slow")
    )
    ok = _run()
    rule = AlertRule()

    def kinds(r):
        return [e.kind for e in evaluate_run_alerts(r, rule)]

    assert kinds(fail) == ["failed"]
    assert kinds(err) == ["failed"]
    assert kinds(ok) == []
    ev = evaluate_run_alerts(err, rule)[0]
    assert ev.signature == "failed:TimeoutError" and ev.agent_name == "refund-bot"
    assert "raised TimeoutError" in ev.message


def test_evaluate_findings_respects_min_severity():
    run = _run(
        findings=[
            Finding(kind="loop", severity="info", message="maybe"),
            Finding(kind="step_repetition", severity="warn", step=2, message="looped"),
            Finding(kind="budget", severity="critical", message="blown"),
        ]
    )

    def kinds(sev):
        return [
            (e.finding_kind, e.severity)
            for e in evaluate_run_alerts(run, AlertRule(min_severity=sev))
        ]

    assert kinds("info") == [
        ("loop", "info"),
        ("step_repetition", "warn"),
        ("budget", "critical"),
    ]
    assert kinds("warn") == [("step_repetition", "warn"), ("budget", "critical")]
    assert kinds("error") == [("budget", "critical")]
    assert kinds("critical") == [("budget", "critical")]
    ev = evaluate_run_alerts(run, AlertRule(min_severity="warn"))[0]
    assert ev.kind == "finding" and ev.signature == "finding:step_repetition"
    assert "step 2" in ev.message


def test_evaluate_cost_and_latency_caps():
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    run = _run(
        usage=Usage(cost_usd=1.25, wall_time_ms=5000),
        started_at=started,
        ended_at=started + timedelta(seconds=9),
    )
    evs = evaluate_run_alerts(run, AlertRule(max_cost_usd=1.0, max_latency_ms=1000))
    assert [e.kind for e in evs] == ["cost", "latency"]
    assert evs[0].signature == "cost" and "1.2500" in evs[0].message
    assert evs[1].signature == "latency" and evs[1].latency_ms == 5000
    # equal to the cap does not fire; missing figures are skipped, not zero
    assert evaluate_run_alerts(run, AlertRule(max_cost_usd=1.25, max_latency_ms=5000)) == []
    bare = _run()
    assert evaluate_run_alerts(bare, AlertRule(max_cost_usd=0.0, max_latency_ms=0.0)) == []
    # wall_time_ms missing: fall back to ended_at - started_at
    timed = _run(started_at=started, ended_at=started + timedelta(milliseconds=1500))
    (lat,) = evaluate_run_alerts(timed, AlertRule(max_latency_ms=1000))
    assert lat.kind == "latency" and lat.latency_ms == 1500


def test_alert_rule_rejects_bad_values():
    with pytest.raises(ValueError, match="min_severity"):
        AlertRule(min_severity="loud")
    with pytest.raises(ValueError, match="cooldown"):
        AlertRule(cooldown_seconds=-1)
    with pytest.raises(ValueError, match="max_cost"):
        AlertRule(max_cost_usd=-0.1)
    with pytest.raises(ValueError, match="max_latency"):
        AlertRule(max_latency_ms=-1)


def test_cooldown_dedup_by_agent_kind_and_signature():
    seen: list[AlertEvent] = []
    t = [0.0]
    disp = AlertDispatcher(
        AlertRule(cooldown_seconds=60),
        sinks=[CallbackSink(seen.append)],
        clock=lambda: t[0],
    )
    fail = _run(outcome="fail", error=ErrorInfo(type="TimeoutError", message="slow"))
    other = _run(
        run_id="r2",
        name="other-bot",
        outcome="fail",
        error=ErrorInfo(type="TimeoutError", message="slow"),
    )
    different = _run(run_id="r3", outcome="fail", error=ErrorInfo(type="ValueError", message="bad"))
    assert [e.kind for e in disp.dispatch(fail)] == ["failed"]
    assert disp.dispatch(fail) == []  # same agent, kind, signature: cooled down
    t[0] = 59.9
    assert disp.dispatch(fail) == []
    # a different agent or signature is not cooled down
    assert disp.dispatch(other)[0].agent_name == "other-bot"
    assert disp.dispatch(different)[0].signature == "failed:ValueError"
    t[0] = 60.0
    again = disp.dispatch(fail)
    assert again and again[0].signature == "failed:TimeoutError"
    assert [e.run_id for e in seen] == ["r1", "r2", "r3", "r1"]


def test_jsonl_sink_appends(tmp_path):
    path = tmp_path / "nested" / "alerts.jsonl"
    sink = JsonlSink(path)
    rule = AlertRule()
    evs = evaluate_run_alerts(_run(outcome="fail"), rule)
    sink.emit(evs[0])
    sink.emit(evs[0])
    lines = path.read_text("utf-8").splitlines()
    assert len(lines) == 2
    row = json.loads(lines[0])
    assert row["kind"] == "failed" and row["run_id"] == "r1" and row["agent_name"] == "refund-bot"


def test_webhook_slack_and_discord_payload_formats():
    ev = AlertEvent(
        kind="failed",
        signature="failed:TimeoutError",
        run_id="r1",
        agent_name="refund-bot",
        message="run failed: status=error outcome=fail raised TimeoutError",
        status="error",
        outcome="fail",
    )
    json_payload = WebhookSink("http://127.0.0.1/hook", format="json").payload(ev)
    assert json_payload["kind"] == "failed" and json_payload["run_id"] == "r1"
    slack = WebhookSink("http://127.0.0.1/hook", format="slack").payload(ev)
    assert list(slack) == ["text"]
    assert slack["text"].startswith("[crashlab] refund-bot:")
    assert "TimeoutError" in slack["text"]
    discord = WebhookSink("http://127.0.0.1/hook", format="discord").payload(ev)
    assert list(discord) == ["content"]
    assert discord["content"] == slack["text"]
    with pytest.raises(ValueError, match="format"):
        WebhookSink("http://127.0.0.1/hook", format="teams")


def test_webhook_delivery_to_localhost_httpserver():
    received: list[tuple[str, dict[str, str], bytes]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length)
            received.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, format, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/alerts"
        ev = evaluate_run_alerts(_run(outcome="fail"), AlertRule())[0]
        WebhookSink(url, format="json", timeout=2.0, headers={"X-Token": "t"}).emit(ev)
        WebhookSink(url, format="slack", timeout=2.0).emit(ev)
    finally:
        server.shutdown()
        thread.join(timeout=2)
    assert len(received) == 2
    path, headers, body = received[0]
    assert path == "/alerts"
    assert "application/json" in headers["content-type"]
    assert headers.get("x-token") == "t"
    assert json.loads(body)["kind"] == "failed"
    assert json.loads(received[1][2]) == {"text": ev.text()}


def test_webhook_error_is_swallowed_and_url_is_redacted(caplog):
    ev = evaluate_run_alerts(_run(outcome="fail"), AlertRule())[0]
    url = "https://hooks.slack.com/services/TSECRET/BSECRET/token123?x=1"
    # nothing listens: delivery fails, watch must not raise
    WebhookSink("http://127.0.0.1:1/hook", format="json", timeout=0.2).emit(ev)
    safe = _safe_url(url)
    assert "TSECRET" not in safe and "token123" not in safe and "x=1" not in safe
    assert safe.startswith("https://hooks.slack.com/services/")
    assert _safe_url("https://discord.com/api/webhooks/123/secret") == (
        "https://discord.com/api/webhooks/***"
    )
    assert _safe_url("https://example.com/alerts?token=secret") == "https://example.com/alerts"
    # the secret URL must never appear in a log line
    with caplog.at_level("WARNING", logger="crashlab.alerts"):
        # trigger the logger by posting to a closed port and checking format via _safe_url
        WebhookSink("http://127.0.0.1:1/hooks/secret-token", timeout=0.2).emit(ev)
    text = caplog.text
    assert "delivery failed" in text
    assert "secret-token" not in text
    assert _safe_url("http://127.0.0.1:1/hooks/secret-token") in text


def test_watch_existing_alert_file(crashlab_dir, tmp_path):
    s = Store()
    make_run(s, 1)
    bad = make_run(
        s,
        2,
        status="error",
        outcome="fail",
        error=ErrorInfo(type="TimeoutError", message="slow"),
        findings=[Finding(kind="step_repetition", severity="error", message="looped")],
    )
    bad.usage.cost_usd = 2.0
    bad.usage.wall_time_ms = 9000
    s.end_run(bad)
    path = tmp_path / "out.jsonl"
    res = runner.invoke(
        app,
        [
            "watch",
            "--existing",
            "--timeout",
            "0.01",
            "--interval",
            "0.01",
            "--alert-file",
            str(path),
            "--max-cost-usd",
            "1",
            "--max-latency-ms",
            "3000",
            "--cooldown",
            "0",
        ],
        env=ENV,
    )
    assert res.exit_code == 0, res.output
    rows = [json.loads(ln) for ln in path.read_text("utf-8").splitlines()]
    # make_run sets ended_at = started + 2s (2000 ms); 3000 ms leaves the passing run quiet
    assert [r["kind"] for r in rows] == ["failed", "finding", "cost", "latency"]
    assert {r["run_id"] for r in rows} == {run_id(2)}


def test_watch_existing_webhook(crashlab_dir):
    s = Store()
    make_run(
        s, 1, status="error", outcome="fail", error=ErrorInfo(type="RuntimeError", message="x")
    )
    received: list[bytes] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            received.append(self.rfile.read(length))
            self.send_response(204)
            self.end_headers()

        def log_message(self, format, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/h"
        res = runner.invoke(
            app,
            [
                "watch",
                "--existing",
                "--timeout",
                "0.01",
                "--interval",
                "0.01",
                "--webhook",
                url,
                "--webhook-format",
                "json",
                "--cooldown",
                "0",
            ],
            env=ENV,
        )
    finally:
        server.shutdown()
        thread.join(timeout=2)
    assert res.exit_code == 0, res.output
    assert len(received) == 1
    payload = json.loads(received[0])
    assert payload["kind"] == "failed" and payload["run_id"] == run_id(1)


def test_watch_help_lists_alert_flags():
    res = runner.invoke(app, ["watch", "--help"], env=ENV)
    out = _ANSI.sub("", res.output)
    assert res.exit_code == 0, out
    for flag in (
        "--webhook",
        "--webhook-format",
        "--slack-webhook",
        "--alert-file",
        "--min-severity",
        "--max-cost-usd",
        "--max-latency-ms",
        "--cooldown",
        "--existing",
        "--timeout",
        "--json",
    ):
        assert flag in out


@pytest.mark.parametrize(
    "args",
    [
        ["--min-severity", "loud"],
        ["--webhook-format", "teams"],
        ["--cooldown", "-1"],
        ["--max-cost-usd", "-1"],
        ["--max-latency-ms", "-5"],
        ["--interval", "0"],
    ],
)
def test_watch_rejects_bad_alert_options(args):
    res = runner.invoke(app, ["watch", *args], env=ENV)
    assert res.exit_code == 2, res.output


def test_watch_without_sinks_is_unchanged(crashlab_dir):
    s = Store()
    make_run(s, 1)
    res = runner.invoke(
        app, ["watch", "--existing", "--timeout", "0.01", "--interval", "0.01", "--json"], env=ENV
    )
    assert res.exit_code == 0, res.output
    lines = [json.loads(ln) for ln in res.output.splitlines()]
    assert [(d["kind"], d["run_id"]) for d in lines] == [("finished", run_id(1))]
