"""Localhost recording / replay HTTP proxy. Offline; no provider SDKs."""

from __future__ import annotations

import json
import re
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from typer.testing import CliRunner

from crashlab.cli import app
from crashlab.proxy import ProxyConfig, create_proxy_server
from crashlab.store import Store

runner = CliRunner()
ENV = {"COLUMNS": "220", "TERM": "xterm-256color", "NO_COLOR": "1"}
SECRET = "sk-test-SECRET-DO-NOT-STORE-xyz"
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return ANSI.sub("", text)


def _chat_completion(text: str = "hello from upstream", model: str = "gpt-test") -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
    }


def _chat_sse(text: str = "hello from upstream", model: str = "gpt-test") -> bytes:
    base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": model}
    chunks = [
        {
            **base,
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
            ],
        },
        {**base, "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]},
        {
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
        },
    ]
    parts = [f"data: {json.dumps(c)}\n\n" for c in chunks]
    parts.append("data: [DONE]\n\n")
    return "".join(parts).encode("utf-8")


def _anthropic_message(text: str = "hello from claude", model: str = "claude-test") -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 6, "output_tokens": 3},
    }


def _anthropic_sse(text: str = "hello from claude", model: str = "claude-test") -> bytes:
    start = {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "usage": {"input_tokens": 6, "output_tokens": 0},
        },
    }
    events = [
        ("message_start", start),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 3},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    out = []
    for event, data in events:
        out.append(f"event: {event}\ndata: {json.dumps(data)}\n\n")
    return "".join(out).encode("utf-8")


class _Upstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # noqa: ARG002
        return

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            body = {}
        self.server.seen.append({"path": self.path, "headers": dict(self.headers), "body": body})
        stream = bool(isinstance(body, dict) and body.get("stream") is True)
        if self.path.endswith("/messages"):
            payload = _anthropic_sse() if stream else json.dumps(_anthropic_message()).encode()
            ctype = "text/event-stream" if stream else "application/json"
        elif self.path.endswith("/responses"):
            resp = {
                "id": "resp_1",
                "object": "response",
                "status": "completed",
                "model": "gpt-test",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "hello from responses"}],
                    }
                ],
                "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
            }
            if stream:
                created = {"type": "response.created", "response": {**resp, "output": []}}
                delta = {
                    "type": "response.output_text.delta",
                    "delta": "hello from responses",
                    "output_index": 0,
                    "content_index": 0,
                }
                done = {"type": "response.completed", "response": resp}
                payload = "".join(
                    f"data: {json.dumps(ev)}\n\n" for ev in (created, delta, done)
                ).encode()
                ctype = "text/event-stream"
            else:
                payload = json.dumps(resp).encode()
                ctype = "application/json"
        else:
            payload = _chat_sse() if stream else json.dumps(_chat_completion()).encode()
            ctype = "text/event-stream" if stream else "application/json"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()


@contextmanager
def _serve(handler, *, host="127.0.0.1"):
    httpd = ThreadingHTTPServer((host, 0), handler)
    httpd.daemon_threads = True
    httpd.seen = []
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd, f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@contextmanager
def _proxy(config: ProxyConfig):
    httpd = create_proxy_server(config)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        port = httpd.server_address[1]
        yield httpd, f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _post(url: str, payload: dict, headers: dict | None = None, timeout: float = 5.0):
    import urllib.error
    import urllib.request

    data = json.dumps(payload).encode("utf-8")
    hdrs = {"Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def test_create_proxy_server_rejects_non_localhost(crashlab_dir):
    with pytest.raises(ValueError, match="localhost"):
        create_proxy_server(ProxyConfig(host="0.0.0.0", port=0, upstream_url="http://127.0.0.1:9"))
    with pytest.raises(ValueError, match="upstream_url"):
        create_proxy_server(ProxyConfig(host="127.0.0.1", port=0))


def test_json_record_scrubs_auth_and_groups_sessions(crashlab_dir):
    with _serve(_Upstream) as (up, upstream):
        config = ProxyConfig(host="127.0.0.1", port=0, upstream_url=upstream, agent="proxy-bot")
        with _proxy(config) as (httpd, base):
            assert httpd.server_address[0] in {"127.0.0.1", "localhost"}
            status, headers, body = _post(
                base + "/v1/chat/completions",
                {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
                headers={"Authorization": f"Bearer {SECRET}", "x-api-key": SECRET},
            )
            assert status == 200
            data = json.loads(body)
            assert data["choices"][0]["message"]["content"] == "hello from upstream"
            status2, _, body2 = _post(
                base + "/v1/chat/completions",
                {"model": "gpt-test", "messages": [{"role": "user", "content": "again"}]},
                headers={"Authorization": f"Bearer {SECRET}"},
            )
            assert status2 == 200
            json.loads(body2)
            run_id = headers.get("X-Crashlab-Run-Id") or headers.get("x-crashlab-run-id")

    # upstream saw the auth header; the store must not
    assert up.seen and any(
        v == f"Bearer {SECRET}" or v == SECRET for rec in up.seen for v in rec["headers"].values()
    )
    store = Store()
    run = store.load_run(run_id or "last")
    assert run.name.startswith("proxy-bot")
    assert len(run.steps) == 2
    assert run.status == "ok"
    assert run.findings is not None
    assert run.checks is not None or run.check_config is not None
    step = run.steps[0]
    assert step.operation == "chat"
    assert step.duration_ms >= 0
    usage = (step.response or {}).get("usage") or {}
    assert (
        usage.get("total_tokens") == 12 or step.attributes.get("crashlab.usage.total_tokens") == 12
    )
    blob = json.dumps(run.model_dump(mode="json"), default=str)
    cassette = store.cassette_path(run.run_id).read_text("utf-8")
    run_json = (store.run_dir(run.run_id) / "run.json").read_text("utf-8")
    for text in (blob, cassette, run_json):
        assert SECRET not in text
        assert "Authorization" not in text
        assert "x-api-key" not in text.lower() or SECRET not in text


def test_sse_stream_is_recorded(crashlab_dir):
    with _serve(_Upstream) as (_, upstream):
        config = ProxyConfig(host="127.0.0.1", port=0, upstream_url=upstream)
        with _proxy(config) as (_, base):
            status, _, body = _post(
                base + "/v1/chat/completions",
                {
                    "model": "gpt-test",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"Authorization": f"Bearer {SECRET}", "Accept": "text/event-stream"},
            )
            assert status == 200
            text = body.decode("utf-8")
            assert "data:" in text
            assert "hello from upstream" in text
            assert "[DONE]" in text

    run = Store().load_run("last")
    assert len(run.steps) == 1
    step = run.steps[0]
    assert step.attributes.get("crashlab.stream") is True
    content = json.dumps(step.response or step.raw_response or {}, default=str)
    assert "hello from upstream" in content
    assert SECRET not in json.dumps(step.model_dump(mode="json"), default=str)


def test_anthropic_json_and_sse_record(crashlab_dir):
    with _serve(_Upstream) as (_, upstream):
        config = ProxyConfig(host="127.0.0.1", port=0, upstream_url=upstream, agent="claude-bot")
        with _proxy(config) as (_, base):
            status, _, body = _post(
                base + "/v1/messages",
                {
                    "model": "claude-test",
                    "max_tokens": 32,
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"x-api-key": SECRET, "anthropic-version": "2023-06-01"},
            )
            assert status == 200
            assert json.loads(body)["content"][0]["text"] == "hello from claude"
            status, _, body = _post(
                base + "/v1/messages",
                {
                    "model": "claude-test",
                    "max_tokens": 32,
                    "stream": True,
                    "messages": [{"role": "user", "content": "stream"}],
                },
                headers={"x-api-key": SECRET, "X-Crashlab-Run": "session-b"},
            )
            assert status == 200
            assert "hello from claude" in body.decode("utf-8")

    store = Store()
    runs = store.list_runs(limit=None)
    assert len(runs) == 2
    # default session vs X-Crashlab-Run
    names = {r["name"] if isinstance(r, dict) else r.name for r in runs}
    assert any("session-b" in n for n in names)


def test_offline_replay_with_upstream_down(crashlab_dir):
    with _serve(_Upstream) as (up, upstream):
        config = ProxyConfig(host="127.0.0.1", port=0, upstream_url=upstream)
        with _proxy(config) as (_, base):
            req = {
                "model": "gpt-test",
                "messages": [{"role": "user", "content": "hi"}],
            }
            status, _, body = _post(
                base + "/v1/chat/completions", req, headers={"Authorization": f"Bearer {SECRET}"}
            )
            assert status == 200
            recorded = json.loads(body)
        up_port = up.server_address[1]
    # upstream is shut down
    run = Store().load_run("last")
    replay = ProxyConfig(host="127.0.0.1", port=0, replay_run_id=run.run_id)
    with _proxy(replay) as (_, base):
        status, _, body = _post(
            base + "/v1/chat/completions",
            {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {SECRET}"},
        )
        assert status == 200
        assert (
            json.loads(body)["choices"][0]["message"]["content"]
            == recorded["choices"][0]["message"]["content"]
        )
        # exhausted cassette
        status, _, body = _post(
            base + "/v1/chat/completions",
            {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert status == 404
        err = json.loads(body)
        assert err["error"]["code"] == "cassette_exhausted"
        # a matching second recording is not there; divergence on a different request
        # after exhaustion is still 404. Record two steps and test 409 separately.

    # confirm nothing is listening on the old upstream port
    import socket

    sock = socket.socket()
    sock.settimeout(0.2)
    with pytest.raises((OSError, ConnectionRefusedError, TimeoutError)):
        # should fail: upstream is down, and the replay path must not have needed it
        sock.connect(("127.0.0.1", up_port))
    sock.close()


def test_replay_divergence_is_conflict(crashlab_dir):
    with _serve(_Upstream) as (_, upstream):
        config = ProxyConfig(host="127.0.0.1", port=0, upstream_url=upstream)
        with _proxy(config) as (_, base):
            _post(
                base + "/v1/chat/completions",
                {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
            )
    run = Store().load_run("last")
    replay = ProxyConfig(host="127.0.0.1", port=0, replay_run_id=run.run_id)
    with _proxy(replay) as (_, base):
        status, _, body = _post(
            base + "/v1/chat/completions",
            {
                "model": "gpt-test",
                "messages": [{"role": "user", "content": "something else entirely"}],
            },
        )
        assert status == 409
        err = json.loads(body)
        assert err["error"]["code"] == "cassette_divergence"


def test_replay_sse_from_recorded_stream(crashlab_dir):
    with _serve(_Upstream) as (_, upstream):
        config = ProxyConfig(host="127.0.0.1", port=0, upstream_url=upstream)
        with _proxy(config) as (_, base):
            _post(
                base + "/v1/chat/completions",
                {
                    "model": "gpt-test",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
    run = Store().load_run("last")
    replay = ProxyConfig(host="127.0.0.1", port=0, replay_run_id=run.run_id)
    with _proxy(replay) as (_, base):
        status, _, body = _post(
            base + "/v1/chat/completions",
            {
                "model": "gpt-test",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert status == 200
        text = body.decode("utf-8")
        assert "data:" in text and "hello from upstream" in text


def test_cli_proxy_help_and_validation(crashlab_dir):
    res = runner.invoke(app, ["proxy", "--help"], env=ENV)
    assert res.exit_code == 0
    clean = _plain(res.output)
    assert "--upstream" in clean
    assert "--port" in clean
    assert "--agent" in clean
    assert "--replay" in clean
    assert "--once" in clean
    res = runner.invoke(app, ["proxy"], env=ENV)
    assert res.exit_code == 2
    res = runner.invoke(app, ["proxy", "--port", "-3", "--upstream", "http://127.0.0.1:9"], env=ENV)
    assert res.exit_code == 2
    res = runner.invoke(app, ["proxy", "--replay", "NOPE"], env=ENV)
    assert res.exit_code == 2
