"""Local trace explorer HTTP API and security. Offline; no provider SDKs."""

from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager

import pytest
from typer.testing import CliRunner

from crashlab.cli import app
from crashlab.models import CheckResult, Finding
from crashlab.store import Store
from crashlab.ui import create_server, make_handler
from tests.debug_fixtures import make_run
from tests.test_otel import _store_linked
from tests.test_stats import _fleet, _tune

runner = CliRunner()


@contextmanager
def _server(store: Store):
    httpd = create_server(store, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        port = httpd.server_address[1]
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _get(url: str, timeout: float = 5.0):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return resp.status, resp.headers, body
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers, exc.read()


def test_create_server_rejects_non_localhost(crashlab_dir):
    with pytest.raises(ValueError, match="localhost"):
        create_server(Store(), host="0.0.0.0", port=0)
    handler = make_handler(Store())
    assert issubclass(handler, object)


def test_ui_endpoints_and_404(crashlab_dir):
    store = Store()
    parent, child = _store_linked(store)
    _fleet(store)
    # XSS payload in a run the HTML page must not execute (API is JSON; / is static)
    xss = make_run(
        store, 20, name="<script>alert(1)</script>", outcome="fail", user="<img onerror=1>"
    )
    xss = _tune(
        store,
        xss,
        outcome="fail",
        findings=[Finding(kind="x", severity="error", step=0, message="<script>steal()</script>")],
        checks=[CheckResult(name="safe", status="fail", message="<b>raw</b>")],
    )

    with _server(store) as base:
        assert create_server(store).server_address[0] in {"127.0.0.1", "localhost"}
        status, headers, body = _get(base + "/")
        assert status == 200
        ctype = headers.get("Content-Type", "")
        assert "text/html" in ctype
        html = body.decode("utf-8")
        assert "crashlab" in html
        # the page is a static shell: untrusted run text is not inlined
        assert "<script>alert(1)</script>" not in html
        assert "<script>steal()</script>" not in html
        assert html.count("<script>") == 1  # the explorer's own script

        status, headers, body = _get(base + "/api/runs")
        assert status == 200 and "application/json" in headers.get("Content-Type", "")
        listing = json.loads(body)
        ids = {r["run_id"] for r in listing["runs"]}
        assert parent.run_id in ids and xss.run_id in ids

        status, _, body = _get(base + "/api/runs/" + xss.run_id)
        assert status == 200
        detail = json.loads(body)
        assert detail["name"] == "<script>alert(1)</script>"
        assert detail["findings"][0]["message"] == "<script>steal()</script>"
        assert detail["input"]["user_msg"] == "<img onerror=1>"
        assert any(s["operation"] == "execute_tool" for s in detail["steps"])
        assert "children" in detail

        status, _, body = _get(base + "/api/runs/" + parent.run_id)
        assert status == 200
        detail = json.loads(body)
        assert any(c["run_id"] == child.run_id for c in detail["children"])
        assert detail["findings"] and detail["checks"]

        status, _, body = _get(base + "/api/runs/" + parent.run_id + "/otel")
        assert status == 200
        otlp = json.loads(body)
        assert "resourceSpans" in otlp
        names = [s["name"] for s in otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]]
        assert "refund-bot" in names and "worker" in names

        status, _, body = _get(base + f"/api/diff?a={parent.run_id}&b={child.run_id}")
        assert status == 200
        diff = json.loads(body)
        assert "summary" in diff and "steps" in diff
        assert diff["a"]["run_id"] == parent.run_id and diff["b"]["run_id"] == child.run_id

        status, _, body = _get(base + "/api/diff")
        assert status == 400

        status, _, body = _get(base + "/api/stats")
        assert status == 200
        stats = json.loads(body)
        assert stats["n"] >= 4

        status, _, body = _get(base + "/api/runs/DOESNOTEXIST")
        assert status == 404
        assert "error" in json.loads(body)

        status, _, body = _get(base + "/api/runs/../etc/passwd")
        assert status == 404

        status, _, body = _get(base + "/nope")
        assert status == 404


def test_cli_ui_help_and_bad_port(crashlab_dir):
    res = runner.invoke(app, ["ui", "--help"])
    assert res.exit_code == 0
    clean = re.sub(r"\x1b\[[0-9;]*m", "", res.output)
    assert "--port" in clean and "--no-open" in clean
    res = runner.invoke(app, ["ui", "--port", "-3"])
    assert res.exit_code == 2
