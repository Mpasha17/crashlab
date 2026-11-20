"""Zero-dependency localhost HTTP recording and offline-replay proxy.

Agents in any language can point ``OPENAI_BASE_URL`` / ``ANTHROPIC_BASE_URL`` at
``http://127.0.0.1:8787`` (plus ``/v1`` for OpenAI-compatible clients). In record mode
the proxy forwards each request to ``upstream_url``, streams JSON or SSE back to the
client in real time, and writes a ``Run`` / ``Step`` into the local store. In replay
mode (``replay_run_id``) it serves the recorded cassette with no upstream network
calls.

Auth headers (``Authorization``, ``x-api-key``) are forwarded to the upstream when
recording and are never persisted in the store.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import socket
import ssl
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from ._ids import new_span_id, new_ulid
from .capture import normalize as N
from .capture._chat import _request_attributes, _response_attributes
from .capture.agent import git_version
from .capture.normalize import request_hash
from .capture.streams import AnthropicAssembler, OpenAIChatAssembler, ResponsesAssembler
from .checks.config import resolve_config
from .checks.engine import CheckEngine
from .context import RunContext
from .models import ErrorInfo, Run, Step, utcnow
from .store import Store
from .store.redact import get_redactor

__all__ = ["ProxyConfig", "create_proxy_server"]

_LOOPBACK = {"127.0.0.1", "localhost", "::1"}
_SECRET_HEADERS = frozenset({"authorization", "x-api-key", "api-key"})
_SECRET_BODY_KEYS = frozenset({"api_key", "authorization", "x-api-key"})
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)
_SESSION_HEADER = "X-Crashlab-Run"
_DEFAULT_SESSION = "_default"
_UPSTREAM_TIMEOUT = 300.0
_MAX_BODY = 32 * 1024 * 1024

# path suffixes -> (provider, api, normalize_request, normalize_response, assembler)
_ROUTES: tuple[tuple[tuple[str, ...], str, str, Any, Any, Any], ...] = (
    (
        ("/v1/chat/completions", "/chat/completions"),
        "openai",
        "openai.chat.completions",
        N.normalize_openai_chat_request,
        N.normalize_openai_chat_response,
        OpenAIChatAssembler,
    ),
    (
        ("/v1/responses", "/responses"),
        "openai",
        "openai.responses",
        N.normalize_openai_responses_request,
        N.normalize_openai_responses_response,
        ResponsesAssembler,
    ),
    (
        ("/v1/messages", "/messages"),
        "anthropic",
        "anthropic.messages",
        N.normalize_anthropic_request,
        N.normalize_anthropic_response,
        AnthropicAssembler,
    ),
)


def _route_for(path: str) -> dict[str, Any] | None:
    clean = path.split("?", 1)[0].rstrip("/") or "/"
    for suffixes, provider, api, nreq, nresp, assembler in _ROUTES:
        if clean in suffixes or any(clean.endswith(s) for s in suffixes):
            return {
                "provider": provider,
                "api": api,
                "normalize_request": nreq,
                "normalize_response": nresp,
                "assembler": assembler,
            }
    return None


@dataclass
class ProxyConfig:
    """Listen settings for :func:`create_proxy_server`.

    ``replay_run_id`` selects offline replay (no upstream calls). Otherwise the
    proxy is in record mode and ``upstream_url`` is required. Binding is
    localhost-only unless ``allow_remote`` is true.
    """

    host: str = "127.0.0.1"
    port: int = 8787
    upstream_url: str | None = None
    agent: str = "proxy"
    store_dir: str | None = None
    replay_run_id: str | None = None
    fault: Any = None
    allow_remote: bool = False
    checks: Any = None
    once: bool = False
    timeout: float = _UPSTREAM_TIMEOUT


@dataclass
class _Session:
    key: str
    ctx: RunContext
    t0: float


@dataclass
class _ProxyState:
    config: ProxyConfig
    store: Store
    lock: threading.Lock = field(default_factory=threading.Lock)
    sessions: dict[str, _Session] = field(default_factory=dict)
    source: Run | None = None
    used: set[int] = field(default_factory=set)
    closed: bool = False
    handled: int = 0

    def session(self, key: str) -> _Session:
        with self.lock:
            existing = self.sessions.get(key)
            if existing is not None:
                return existing
            sess = self._new_session(key)
            self.sessions[key] = sess
            return sess

    def _new_session(self, key: str) -> _Session:
        name = self.config.agent or "proxy"
        if key != _DEFAULT_SESSION:
            name = f"{name}:{key}"
        run = Run(
            run_id=new_ulid(),
            name=name,
            entrypoint="crashlab.proxy",
            input={"session": None if key == _DEFAULT_SESSION else key},
            status="running",
            started_at=utcnow(),
            code_version=git_version(),
            env={
                "proxy": {
                    "mode": "replay" if self.config.replay_run_id else "record",
                    "upstream": self.config.upstream_url,
                    "agent": self.config.agent,
                }
            },
            tags=["proxy"],
            span_id=new_span_id(),
        )
        cfg = resolve_config(self.config.checks)
        run.check_config = cfg.model_dump(mode="json")
        self.store.begin_run(run)
        ctx = RunContext(run=run, store=self.store, mode="record")
        if cfg.enabled:
            ctx.checks = CheckEngine(cfg, run)
            ctx.circuit_breaker = cfg.circuit_breaker
        return _Session(key=key, ctx=ctx, t0=time.perf_counter())

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
            sessions = list(self.sessions.values())
            self.sessions.clear()
        for sess in sessions:
            _finalize_session(sess)

    def next_replay_step(self, path: str, body: dict[str, Any]) -> tuple[Step | None, int, str]:
        """Return ``(step, 0, '')`` or ``(None, status, message)``."""
        if self.source is None:
            return None, 404, "no cassette loaded"
        route = _route_for(path)
        remaining = [
            s for s in self.source.steps if s.operation == "chat" and s.idx not in self.used
        ]
        if not remaining:
            return None, 404, "cassette exhausted"
        hashed = None
        model = None
        want_msgs = None
        want_sys = None
        if route is not None and isinstance(body, dict):
            try:
                norm = route["normalize_request"](body)
            except Exception:
                norm = None
            if isinstance(norm, dict):
                hashed = request_hash(get_redactor().redact(norm))
                model = (norm.get("params") or {}).get("model")
                want_msgs = N.canonical_json(norm.get("messages"))
                want_sys = norm.get("system")
        if hashed:
            for step in remaining:
                if step.request_hash == hashed:
                    self.used.add(step.idx)
                    return step, 0, ""
        if want_msgs is not None:
            for step in remaining:
                req = step.request or {}
                params = req.get("params") or {}
                if (
                    params.get("model") == model
                    and req.get("system") == want_sys
                    and N.canonical_json(req.get("messages")) == want_msgs
                ):
                    self.used.add(step.idx)
                    return step, 0, ""
        next_step = remaining[0]
        detail = f"request does not match recorded step {next_step.idx} ({next_step.name})"
        return None, 409, detail


def _finalize_session(sess: _Session) -> None:
    ctx = sess.ctx
    run = ctx.run
    if run.status == "running":
        run.status = "ok"
    run.ended_at = utcnow()
    run.usage.wall_time_ms = round((time.perf_counter() - sess.t0) * 1000, 3)
    if ctx.checks is not None:
        ctx.add_findings(ctx.checks.on_end(run))
    if run.outcome == "unknown":
        if run.findings and any(f.severity in {"error", "critical"} for f in run.findings):
            run.outcome = "fail"
            run.outcome_reason = "runtime findings"
        else:
            run.outcome = "pass"
    ctx.write(ctx.store.end_run, run)


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")


def _error_payload(message: str, typ: str, code: str | None = None) -> dict[str, Any]:
    return {"error": {"message": message, "type": typ, "code": code or typ}}


def _scrub_body(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    return {k: v for k, v in value.items() if k.lower() not in _SECRET_BODY_KEYS}


def _header_map(headers: Any) -> dict[str, str]:
    if hasattr(headers, "items"):
        return {str(k): str(v) for k, v in headers.items()}
    return {}


def _target(upstream: str, path: str) -> tuple[str, str, str, int]:
    """Return ``(scheme, host, full_path, port)`` for the upstream request."""
    parsed_path = urlparse(path)
    req_path = parsed_path.path or "/"
    query = parsed_path.query
    u = urlparse(upstream)
    if not u.scheme or not u.netloc:
        raise ValueError(f"upstream_url must include a scheme and host: {upstream!r}")
    base_path = (u.path or "").rstrip("/")
    if base_path and (req_path == base_path or req_path.startswith(base_path + "/")):
        full = req_path
    elif base_path:
        full = base_path + (req_path if req_path.startswith("/") else "/" + req_path)
    else:
        full = req_path
    if query:
        full = full + "?" + query
    port = u.port or (443 if u.scheme == "https" else 80)
    host = u.hostname or u.netloc.split("@")[-1].split(":")[0]
    return u.scheme, host, full, port


def _parse_sse(payload: bytes) -> list[dict[str, Any]]:
    text = payload.decode("utf-8", "replace")
    events: list[dict[str, Any]] = []
    for block in text.split("\n\n"):
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if not data_lines:
            continue
        raw = "\n".join(data_lines).strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


def _assemble(route: dict[str, Any] | None, body: bytes, streamed: bool) -> dict[str, Any] | None:
    if not body:
        return None
    if streamed:
        events = _parse_sse(body)
        if route is None:
            return events[-1] if events else None
        asm = route["assembler"]()
        for ev in events:
            asm.add(ev)
        try:
            result = asm.result()
        except Exception:
            return events[-1] if events else None
        return result if isinstance(result, dict) else None
    try:
        obj = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


def _openai_chat_sse(raw: dict[str, Any]) -> bytes:
    model = raw.get("model") or "unknown"
    ident = raw.get("id") or "chatcmpl-replay"
    created = raw.get("created") or 0
    choice = (raw.get("choices") or [{}])[0] if isinstance(raw.get("choices"), list) else {}
    msg = choice.get("message") or {}
    finish = choice.get("finish_reason") or "stop"
    base = {"id": ident, "object": "chat.completion.chunk", "created": created, "model": model}
    chunks: list[dict[str, Any]] = []
    chunks.append(
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
    )
    content = msg.get("content")
    if isinstance(content, str) and content:
        chunks.append(
            {
                **base,
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
            }
        )
    if msg.get("tool_calls"):
        for i, call in enumerate(msg["tool_calls"]):
            fn = call.get("function") or {}
            chunks.append(
                {
                    **base,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": i,
                                        "id": call.get("id"),
                                        "type": call.get("type") or "function",
                                        "function": {
                                            "name": fn.get("name"),
                                            "arguments": fn.get("arguments") or "",
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            )
    usage = raw.get("usage")
    last: dict[str, Any] = {
        **base,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
    }
    if usage:
        last["usage"] = usage
    chunks.append(last)
    parts = [f"data: {json.dumps(c, ensure_ascii=False)}\n\n" for c in chunks]
    parts.append("data: [DONE]\n\n")
    return "".join(parts).encode("utf-8")


def _anthropic_sse(raw: dict[str, Any]) -> bytes:
    message = {
        k: raw.get(k) for k in ("id", "type", "role", "model", "usage") if k in raw or k == "role"
    }
    message.setdefault("type", "message")
    message.setdefault("role", "assistant")
    message["content"] = []
    blocks: list[bytes] = []

    def ev(event: str, data: dict[str, Any]) -> None:
        blocks.append(f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode())

    ev("message_start", {"type": "message_start", "message": message})
    for i, block in enumerate(raw.get("content") or []):
        if not isinstance(block, dict):
            continue
        start = {k: v for k, v in block.items() if k != "text" or not block.get("text")}
        if block.get("type") == "text":
            start = {"type": "text", "text": ""}
        ev(
            "content_block_start",
            {"type": "content_block_start", "index": i, "content_block": start},
        )
        if block.get("type") == "text" and isinstance(block.get("text"), str) and block["text"]:
            ev(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": i,
                    "delta": {"type": "text_delta", "text": block["text"]},
                },
            )
        ev("content_block_stop", {"type": "content_block_stop", "index": i})
    ev(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": raw.get("stop_reason") or "end_turn",
                "stop_sequence": raw.get("stop_sequence"),
            },
            "usage": raw.get("usage") or {},
        },
    )
    ev("message_stop", {"type": "message_stop"})
    return b"".join(blocks)


def _responses_sse(raw: dict[str, Any]) -> bytes:
    text_parts: list[str] = []
    for item in raw.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            for part in item.get("content") or []:
                if (
                    isinstance(part, dict)
                    and part.get("type") in {"output_text", "text"}
                    and isinstance(part.get("text"), str)
                ):
                    text_parts.append(part["text"])
    text = "".join(text_parts)
    events = [
        {"type": "response.created", "response": {**raw, "status": "in_progress", "output": []}},
        {
            "type": "response.output_text.delta",
            "delta": text,
            "output_index": 0,
            "content_index": 0,
        },
        {"type": "response.completed", "response": raw},
    ]
    return "".join(f"data: {json.dumps(e, ensure_ascii=False)}\n\n" for e in events).encode("utf-8")


def _sse_from_raw(route: dict[str, Any] | None, raw: dict[str, Any]) -> bytes:
    api = (route or {}).get("api") or ""
    if api == "anthropic.messages":
        return _anthropic_sse(raw)
    if api == "openai.responses":
        return _responses_sse(raw)
    return _openai_chat_sse(raw)


def _wants_stream(body: dict[str, Any] | None, headers: dict[str, str]) -> bool:
    if isinstance(body, dict) and body.get("stream") is True:
        return True
    accept = headers.get("Accept") or headers.get("accept") or ""
    return "text/event-stream" in accept.lower()


class _Adapter:
    def __init__(self, route: dict[str, Any], agent: str) -> None:
        self.provider = route["provider"]
        self.api = route["api"]
        self.agent = agent


def _record_step(
    sess: _Session,
    *,
    route: dict[str, Any] | None,
    path: str,
    body: dict[str, Any] | None,
    raw_body: dict[str, Any] | None,
    assembled: dict[str, Any] | None,
    sse_text: str | None,
    streamed: bool,
    status: int,
    duration_ms: float,
    started_at: Any,
    source: str,
    error: ErrorInfo | None,
    server: str | None,
    config: ProxyConfig,
) -> Step:
    ctx = sess.ctx
    req_body = body if isinstance(body, dict) else {}
    if route is not None and req_body:
        try:
            normalized = route["normalize_request"](req_body)
        except Exception:
            normalized = {"params": {}, "messages": [], "system": None, "tools": []}
    else:
        normalized = {"params": {}, "messages": [], "system": None, "tools": [], "path": path}
    model = (normalized.get("params") or {}).get("model") if isinstance(normalized, dict) else None
    attrs: dict[str, Any] = {
        "crashlab.proxy.path": path,
        "crashlab.proxy.status": status,
    }
    if route is not None:
        adapter = _Adapter(route, config.agent)
        attrs.update(_request_attributes(adapter, normalized, server, streamed))  # type: ignore[arg-type]
        attrs["gen_ai.agent.name"] = config.agent
    if streamed:
        attrs["crashlab.stream"] = True
    if sse_text is not None:
        attrs["crashlab.proxy.sse"] = sse_text
    norm_resp: dict[str, Any] | None = None
    raw_resp = assembled
    if route is not None and assembled is not None:
        try:
            norm_resp, raw_resp = route["normalize_response"](assembled)
        except Exception:
            norm_resp, raw_resp = assembled, assembled
        if isinstance(norm_resp, dict):
            attrs.update(_response_attributes(norm_resp))
    hashed = request_hash(get_redactor().redact(normalized))
    step = Step(
        run_id=ctx.run.run_id,
        idx=ctx.next_idx(),
        span_id=new_span_id(),
        parent_span_id=ctx.run.span_id,
        operation="chat",
        name=str(model) if model else "unknown-model",
        attributes=attrs,
        request=normalized if isinstance(normalized, dict) else {},
        response=norm_resp,
        raw_request=_scrub_body(raw_body if raw_body is not None else req_body),
        raw_response=raw_resp if isinstance(raw_resp, dict) else assembled,
        error=error,
        request_hash=hashed,
        started_at=started_at,
        duration_ms=round(duration_ms, 3),
        source=source,  # type: ignore[arg-type]
    )
    ctx.add_step(step)
    ctx.write(ctx.store.update_run, ctx.run)
    return step


def _apply_fault(fault: Any) -> tuple[int, bytes, str, bool] | float | None:
    """Return a canned ``(status, body, content_type, streamed)``, a delay in seconds, or None."""
    if fault is None:
        return None
    if isinstance(fault, (int, float)) and not isinstance(fault, bool) and fault < 100:
        return float(fault)
    if isinstance(fault, int):
        payload = _error_payload(f"injected fault {fault}", "injected_fault", str(fault))
        return fault, _json_bytes(payload), "application/json", False
    if isinstance(fault, str):
        key = fault.strip().lower()
        if key in {"timeout", "delay"}:
            return 1.0
        if key in {"empty", "llm_empty"}:
            return 200, b"{}", "application/json", False
        if key.isdigit():
            return _apply_fault(int(key))
        payload = _error_payload(fault, "injected_fault", key)
        return 500, _json_bytes(payload), "application/json", False
    if isinstance(fault, dict):
        delay = fault.get("delay_ms") or fault.get("delay")
        if delay and not fault.get("status"):
            return float(delay) / (
                1000.0 if fault.get("delay_ms") is not None or delay > 10 else 1.0
            )
        if delay:
            time.sleep(
                float(delay) / (1000.0 if fault.get("delay_ms") is not None or delay > 10 else 1.0)
            )
        status = int(fault.get("status") or 500)
        body = fault.get("body")
        raw = _json_bytes(body) if not isinstance(body, (bytes, bytearray)) else bytes(body)
        ctype = str(fault.get("content_type") or "application/json")
        return status, raw, ctype, bool(fault.get("stream"))
    return None


class _StreamingForward:
    """Forward an upstream response while writing bytes to the client as they arrive."""

    def __init__(
        self,
        config: ProxyConfig,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
        write: Any,
        start: Any,
    ) -> None:
        self.status = 502
        self.headers: dict[str, str] = {}
        self.body = b""
        self._error: Exception | None = None
        if not config.upstream_url:
            self._error = RuntimeError("upstream_url is required in record mode")
            return
        try:
            scheme, host, full, port = _target(config.upstream_url, path)
        except ValueError as exc:
            self._error = exc
            return
        fwd = {
            k: v
            for k, v in headers.items()
            if k.lower() not in _HOP_BY_HOP and k.lower() != "x-crashlab-run"
        }
        fwd["Host"] = (
            host
            if (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
            else f"{host}:{port}"
        )
        if scheme == "https":
            conn: http.client.HTTPConnection = http.client.HTTPSConnection(
                host, port, timeout=config.timeout, context=ssl.create_default_context()
            )
        else:
            conn = http.client.HTTPConnection(host, port, timeout=config.timeout)
        chunks: list[bytes] = []
        started = False
        try:
            conn.request(method, full, body=body or None, headers=fwd)
            resp = conn.getresponse()
            self.status = resp.status
            self.headers = {k: v for k, v in resp.getheaders()}
            start(self.status, self.headers)
            started = True
            while True:
                piece = resp.read(4096)
                if not piece:
                    break
                chunks.append(piece)
                write(piece)
            self.body = b"".join(chunks)
        except Exception as exc:
            self._error = exc
            if not started:
                self.status = 502
                self.headers = {"Content-Type": "application/json"}
                self.body = _json_bytes(_error_payload(str(exc), "bad_gateway", "upstream_error"))
        finally:
            conn.close()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 300

    @property
    def state(self) -> _ProxyState:
        return self.server.proxy_state  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _read_body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length < 0 or length > _MAX_BODY:
            return b""
        return self.rfile.read(length) if length else b""

    def _send(
        self, status: int, body: bytes, content_type: str, extra: dict[str, str] | None = None
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        if extra:
            for k, v in extra.items():
                self.send_header(k, v)
        self.end_headers()
        if body and self.command != "HEAD":
            self.wfile.write(body)
            self.wfile.flush()

    def _send_stream(
        self, status: int, body: bytes, content_type: str, extra: dict[str, str] | None = None
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        if extra:
            for k, v in extra.items():
                self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)
            self.wfile.flush()

    def _maybe_stop(self, is_llm: bool) -> None:
        if is_llm:
            with self.state.lock:
                self.state.handled += 1
                count = self.state.handled
            if self.state.config.once and count >= 1:
                threading.Thread(target=self.server.shutdown, daemon=True).start()

    def do_POST(self) -> None:  # noqa: N802
        self._handle()

    def do_GET(self) -> None:  # noqa: N802
        self._handle()

    def do_PUT(self) -> None:  # noqa: N802
        self._handle()

    def do_DELETE(self) -> None:  # noqa: N802
        self._handle()

    def do_HEAD(self) -> None:  # noqa: N802
        self._handle()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._handle()

    def _handle(self) -> None:
        raw = self._read_body()
        headers = _header_map(self.headers)
        path = self.path
        route = _route_for(path)
        is_llm = route is not None and self.command == "POST"
        body_obj: dict[str, Any] | None
        try:
            parsed = json.loads(raw.decode("utf-8")) if raw else None
            body_obj = parsed if isinstance(parsed, dict) else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            body_obj = None
        try:
            if self.state.config.replay_run_id:
                self._handle_replay(path, route, headers, body_obj, is_llm)
            else:
                self._handle_record(path, route, headers, raw, body_obj, is_llm)
        except Exception as exc:
            payload = _json_bytes(_error_payload(str(exc), "proxy_error", "internal"))
            with contextlib.suppress(OSError):
                self._send(500, payload, "application/json")
        finally:
            self._maybe_stop(is_llm)

    def _run_header(self, headers: dict[str, str]) -> str:
        for key, value in headers.items():
            if key.lower() == _SESSION_HEADER.lower() and value.strip():
                return value.strip()[:128]
        return _DEFAULT_SESSION

    def _handle_replay(
        self,
        path: str,
        route: dict[str, Any] | None,
        headers: dict[str, str],
        body_obj: dict[str, Any] | None,
        is_llm: bool,
    ) -> None:
        if not is_llm:
            self._send(
                404,
                _json_bytes(_error_payload("not found", "not_found", "not_found")),
                "application/json",
            )
            return
        faulted = _maybe_sleep_or_fault(self.state.config.fault)
        if isinstance(faulted, tuple):
            status, body, ctype, streamed = faulted
            (self._send_stream if streamed else self._send)(status, body, ctype)
            return
        with self.state.lock:
            step, status, message = self.state.next_replay_step(path, body_obj or {})
        if step is None:
            typ = "cassette_exhausted" if status == 404 else "cassette_divergence"
            self._send(status, _json_bytes(_error_payload(message, typ, typ)), "application/json")
            return
        raw = step.raw_response if isinstance(step.raw_response, dict) else (step.response or {})
        stream = _wants_stream(body_obj, headers)
        extra = {"X-Crashlab-Run-Id": step.run_id}
        if stream:
            sse = (step.attributes or {}).get("crashlab.proxy.sse")
            payload = (
                sse.encode("utf-8")
                if isinstance(sse, str) and sse
                else _sse_from_raw(route, raw or {})
            )
            self._send_stream(200, payload, "text/event-stream; charset=utf-8", extra)
        else:
            self._send(200, _json_bytes(raw or {}), "application/json", extra)

    def _handle_record(
        self,
        path: str,
        route: dict[str, Any] | None,
        headers: dict[str, str],
        raw: bytes,
        body_obj: dict[str, Any] | None,
        is_llm: bool,
    ) -> None:
        started_at = utcnow()
        t0 = time.perf_counter()
        sess = self.state.session(self._run_header(headers)) if is_llm else None
        extra = {"X-Crashlab-Run-Id": sess.ctx.run.run_id} if sess is not None else None
        faulted = _maybe_sleep_or_fault(self.state.config.fault)
        streamed_req = _wants_stream(body_obj, headers)
        status = 502
        resp_headers: dict[str, str] = {}
        resp_body = b""
        error: ErrorInfo | None = None
        if isinstance(faulted, tuple):
            status, resp_body, ctype, streamed_req = faulted
            if streamed_req:
                self._send_stream(status, resp_body, ctype, extra)
            else:
                self._send(status, resp_body, ctype, extra)
        elif not self.state.config.upstream_url:
            payload = _json_bytes(
                _error_payload("upstream_url is required in record mode", "config", "no_upstream")
            )
            self._send(500, payload, "application/json", extra)
            return
        else:
            started = {"done": False}

            def start(st: int, hdrs: dict[str, str]) -> None:
                started["done"] = True
                self.send_response(st)
                ctype = (
                    hdrs.get("Content-Type")
                    or hdrs.get("content-type")
                    or ("text/event-stream; charset=utf-8" if streamed_req else "application/json")
                )
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", hdrs.get("Cache-Control") or "no-cache")
                self.send_header("Connection", "close")
                if not streamed_req:
                    clen = hdrs.get("Content-Length") or hdrs.get("content-length")
                    if clen:
                        self.send_header("Content-Length", clen)
                if extra:
                    for k, v in extra.items():
                        self.send_header(k, v)
                skip = (
                    _HOP_BY_HOP
                    | _SECRET_HEADERS
                    | {
                        "content-type",
                        "cache-control",
                        "connection",
                        "x-crashlab-run-id",
                    }
                )
                for k, v in hdrs.items():
                    if k.lower() not in skip:
                        self.send_header(k, v)
                self.end_headers()

            def write(piece: bytes) -> None:
                self.wfile.write(piece)
                self.wfile.flush()

            fwd = _StreamingForward(
                self.state.config, self.command, path, headers, raw, write, start
            )
            status, resp_headers, resp_body = fwd.status, fwd.headers, fwd.body
            if fwd._error is not None and not started["done"]:
                error = ErrorInfo(
                    type=type(fwd._error).__qualname__, message=str(fwd._error)[:4000]
                )
                if not resp_body:
                    resp_body = _json_bytes(
                        _error_payload(str(fwd._error), "bad_gateway", "upstream_error")
                    )
                self._send(status or 502, resp_body, "application/json", extra)
            elif fwd._error is not None:
                error = ErrorInfo(
                    type=type(fwd._error).__qualname__, message=str(fwd._error)[:4000]
                )
        duration_ms = (time.perf_counter() - t0) * 1000
        if not is_llm or sess is None:
            return
        ctype = resp_headers.get("Content-Type") or resp_headers.get("content-type") or ""
        streamed = streamed_req or "text/event-stream" in ctype
        assembled = _assemble(route, resp_body, streamed)
        sse_text = resp_body.decode("utf-8", "replace") if streamed and resp_body else None
        if error is None and status >= 400:
            message = ""
            if isinstance(assembled, dict):
                err = assembled.get("error")
                if isinstance(err, dict):
                    message = str(err.get("message") or "")
                elif isinstance(err, str):
                    message = err
            error = ErrorInfo(
                type="UpstreamError",
                message=(message or f"upstream status {status}")[:4000],
                status_code=status,
                body=assembled,
            )
        try:
            host = None
            if self.state.config.upstream_url:
                host = urlparse(self.state.config.upstream_url).hostname
            _record_step(
                sess,
                route=route,
                path=path,
                body=body_obj,
                raw_body=_scrub_body(body_obj),
                assembled=assembled,
                sse_text=sse_text,
                streamed=streamed,
                status=status,
                duration_ms=duration_ms,
                started_at=started_at,
                source="live",
                error=error,
                server=host,
                config=self.state.config,
            )
        except Exception:
            # recording must never hide a response already sent to the client
            pass


def _maybe_sleep_or_fault(fault: Any) -> tuple[int, bytes, str, bool] | None:
    applied = _apply_fault(fault)
    if isinstance(applied, float):
        time.sleep(applied)
        return None
    if isinstance(applied, tuple):
        return applied
    return None


def _validate_config(config: ProxyConfig) -> None:
    if config.host not in _LOOPBACK and not config.allow_remote:
        raise ValueError(
            "crashlab proxy binds only to localhost (127.0.0.1, ::1, or localhost); "
            "set allow_remote=True to override"
        )
    if not config.replay_run_id and not config.upstream_url:
        raise ValueError("upstream_url is required unless replay_run_id is set")
    if config.port < 0:
        raise ValueError("port must be 0 or a positive port")


class ProxyHTTPServer(ThreadingHTTPServer):
    """``ThreadingHTTPServer`` that finalizes open proxy sessions on close."""

    def __init__(
        self, addr: tuple[str, int], handler: type[BaseHTTPRequestHandler], state: _ProxyState
    ) -> None:
        if addr[0] == "::1":
            self.address_family = socket.AF_INET6
        self.proxy_state = state
        super().__init__(addr, handler)
        self.daemon_threads = True

    def server_close(self) -> None:
        try:
            self.proxy_state.close()
        finally:
            super().server_close()


def create_proxy_server(config: ProxyConfig) -> ThreadingHTTPServer:
    """A localhost HTTP recording / replay proxy. ``port=0`` picks an ephemeral port
    (read it back from ``server.server_address[1]``). Call ``serve_forever`` to start
    and ``shutdown`` / ``server_close`` for a clean stop.
    """
    _validate_config(config)
    store = Store(config.store_dir)
    state = _ProxyState(config=config, store=store)
    if config.replay_run_id:
        state.source = store.load_run(config.replay_run_id)
    handler = type("ProxyHandler", (_Handler,), {})
    handler.protocol_version = "HTTP/1.1"
    return ProxyHTTPServer((config.host, config.port), handler, state)
