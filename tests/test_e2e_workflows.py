"""Cross-feature end-to-end workflows. Offline; no optional provider SDK imports."""

from __future__ import annotations

import json
import sys
import threading
import types
import zipfile
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

import crashlab
from crashlab.alerts import AlertDispatcher, AlertRule, JsonlSink
from crashlab.assertions import evaluate_rules, parse_rules
from crashlab.baseline import compare_baseline, save_baseline
from crashlab.bundle import export_bundle, import_bundle
from crashlab.context_audit import analyze_context, audit_tool_schemas
from crashlab.diff_report import has_regression, render_diff_html, render_diff_markdown
from crashlab.faults import tool_error
from crashlab.integrations import OpenAIAgentsTracingProcessor, crashlab_tracing_processor
from crashlab.integrations.instrument import _reset
from crashlab.proxy import ProxyConfig, create_proxy_server
from crashlab.rubrics import DEFAULT_RUBRIC, score_run
from crashlab.simulate import Scenario, ScenarioTurn, simulate
from crashlab.store import Store
from tests.fakes import FakeOpenAI, text_response, tool_call_response

# ---------------------------------------------------------------------------
# Synthetic secret + tool schemas (assembled at runtime so scanners stay quiet)
# ---------------------------------------------------------------------------

TOOL_NAME = "e2e_lookup"


def _synthetic_secret() -> str:
    """OpenAI-shaped token built from pieces so the source never holds a whole value."""
    return "".join(("sk-", "test", "E2E", "A" * 24))


def _good_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": TOOL_NAME,
                "description": "Look up a record by key and return its stored value.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "The lookup key to resolve.",
                        }
                    },
                    "required": ["key"],
                },
            },
        }
    ]


def _bad_tools() -> list[dict[str, Any]]:
    # Groq-style bug: required keys, but no properties object.
    broken = {
        "type": "function",
        "function": {
            "name": "transfer_to_refunds",
            "description": "Hand the customer to the refunds agent.",
            "parameters": {"type": "object", "required": ["reason"]},
        },
    }
    return [*_good_tools(), broken]


def _disable_store_redactor(monkeypatch) -> None:
    """Let this test plant a synthetic secret (the live redactor would strip it)."""
    monkeypatch.setattr("crashlab.store.redact.Redactor.redact", lambda self, obj: obj)
    monkeypatch.setattr("crashlab.store.redact.Redactor.redact_str", lambda self, value: value)


# ---------------------------------------------------------------------------
# Recorded agent used by Workflow A
# ---------------------------------------------------------------------------


@crashlab.tool()
def e2e_lookup(key: str) -> dict:
    return {"key": key, "value": 42}


@crashlab.agent(name="e2e-workflow-bot", ignore_args=("client",))
def run_workflow(user: str, client, tools: list | None = None) -> str:
    r = client.chat.completions.create(
        model="fake-model",
        messages=[{"role": "user", "content": user}],
        tools=tools or _good_tools(),
    )
    msg = r.choices[0].message
    if not msg.tool_calls:
        return msg.content or ""
    tc = msg.tool_calls[0]
    args = json.loads(tc.function.arguments)
    try:
        result = e2e_lookup(**args)
    except Exception as exc:  # noqa: BLE001 - the agent under test recovers
        result = {"error": f"{type(exc).__name__}: {exc}"}
    r2 = client.chat.completions.create(
        model="fake-model",
        messages=[
            {"role": "user", "content": user},
            {"role": "assistant", "tool_calls": [tc], "content": None},
            {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)},
        ],
        tools=tools or _good_tools(),
    )
    return r2.choices[0].message.content or ""


@crashlab.check("e2e-workflow-bot")
def e2e_no_tool_errors(run):
    for step in run.tool_calls():
        if step.error is not None:
            raise AssertionError("a tool step recorded an error")


@pytest.fixture(autouse=True)
def _clean_oracles():
    from crashlab.verify.oracles import _registry

    before = dict(_registry)
    yield
    _registry.clear()
    _registry.update(before)


def _script(*, text: str) -> list[Any]:
    return [
        tool_call_response(1, TOOL_NAME, {"key": "abc"}),
        text_response(2, text),
    ]


def _client(text: str):
    return crashlab.wrap(FakeOpenAI(_script(text=text)))


# ---------------------------------------------------------------------------
# Workflow A
# ---------------------------------------------------------------------------


def test_workflow_a_record_faults_audit_rubric_baseline_diff_bundle(
    crashlab_dir, tmp_path, monkeypatch
):
    _disable_store_redactor(monkeypatch)
    secret = _synthetic_secret()
    store = Store(crashlab_dir)

    # --- good run: tool call + final LLM reply, no faults ------------------------
    good_text = "Lookup ok: key=abc value=42"
    good_out = run_workflow("please look up abc", _client(good_text), tools=_good_tools())
    assert good_out == good_text
    good = crashlab.load("last")
    assert good.name == "e2e-workflow-bot"
    assert good.status == "ok" and good.outcome == "pass"
    assert any(s.operation == "execute_tool" and s.name == TOOL_NAME for s in good.steps)
    assert any(s.operation == "chat" for s in good.steps)
    assert secret not in json.dumps(good.model_dump(mode="json"), default=str)

    baseline_path = tmp_path / "baseline.json"
    payload = save_baseline(crashlab_dir, baseline_path)
    assert baseline_path.is_file()
    assert payload["n"] == 1
    assert payload["pass_rate"] == 1.0
    assert compare_baseline(baseline_path, crashlab_dir).ok

    # --- bad run: faults + broken tool schema + synthetic secret in the last reply
    bad_text = f"Lookup failed; debug token {secret}"
    with crashlab.faults(tool_error(TOOL_NAME, message="upstream 502", status=502)):
        bad_out = run_workflow("please look up abc", _client(bad_text), tools=_bad_tools())
    assert secret in bad_out
    bad = crashlab.load("last")
    assert bad.run_id != good.run_id
    assert bad.outcome == "fail"
    assert any(s.operation == "execute_tool" and s.error is not None for s in bad.steps)
    assert secret in json.dumps(bad.model_dump(mode="json"), default=str)

    # --- context + tool-schema audit --------------------------------------------
    good_ctx = analyze_context(good)
    bad_ctx = analyze_context(bad)
    assert good_ctx.steps and bad_ctx.steps
    assert good_ctx.run_id == good.run_id
    assert any(s.sections for s in good_ctx.steps)

    good_audit = audit_tool_schemas(good)
    bad_audit = audit_tool_schemas(bad)
    assert TOOL_NAME in good_audit.called
    assert good_audit.ok
    assert not bad_audit.ok
    kinds = {i.kind: i for i in bad_audit.issues}
    assert "required_missing_properties" in kinds
    assert kinds["required_missing_properties"].tool == "transfer_to_refunds"
    assert "transfer_to_refunds" in bad_audit.unused

    # --- assertions + rubric scoring --------------------------------------------
    rules = parse_rules(
        [
            {"type": "must_call", "tool": TOOL_NAME},
            {"type": "reply_contains", "text": "Lookup ok"},
            {"type": "no_findings_above", "severity": "error"},
        ]
    )
    good_rules = evaluate_rules(good, rules)
    bad_rules = evaluate_rules(bad, rules)
    assert all(r.status == "pass" for r in good_rules), [r.message for r in good_rules]
    assert any(r.status == "fail" for r in bad_rules)

    good_score = score_run(good, DEFAULT_RUBRIC)
    bad_score = score_run(bad, DEFAULT_RUBRIC)
    assert good_score.passed and good_score.score >= DEFAULT_RUBRIC.min_score
    assert good_score.score > bad_score.score
    assert bad_score.passed is False

    # --- baseline check detects the failed run as a fleet regression ------------
    check = compare_baseline(baseline_path, crashlab_dir)
    assert not check.ok
    assert any(r.metric == "pass_rate" for r in check.regressions)

    # --- HTML / Markdown diff + has_regression ----------------------------------
    delta = crashlab.diff(good, bad)
    assert has_regression(delta)
    html = render_diff_html(delta, good, bad)
    markdown = render_diff_markdown(delta, good, bad)
    assert html.startswith("<!doctype html>")
    assert good.run_id in html and bad.run_id in html
    assert "pass" in html and "fail" in html
    assert good.run_id in markdown and bad.run_id in markdown
    assert "pass" in markdown.lower() and "fail" in markdown.lower()

    # --- bundle (redacted) → fresh store → diagnose + replay --------------------
    dest = tmp_path / "debug.crashlab.zip"
    manifest = export_bundle(store, bad.run_id, dest, redact=True)
    assert dest.is_file()
    assert manifest.redacted is True
    assert bad.run_id in manifest.run_ids

    archive = dest.read_bytes().decode("latin-1")
    with zipfile.ZipFile(dest) as zf:
        raw_members = b"".join(zf.read(name) for name in zf.namelist()).decode("latin-1")
    assert secret not in archive
    assert secret not in raw_members

    fresh = Store(tmp_path / "fresh-store")
    imported_ids = import_bundle(fresh, dest)
    assert bad.run_id in imported_ids
    imported = fresh.load_run(bad.run_id)
    imported_blob = json.dumps(imported.model_dump(mode="json"), default=str)
    assert secret not in imported_blob
    assert "[REDACTED:openai_api_key]" in imported_blob
    assert any(s.operation == "execute_tool" for s in imported.steps)

    diagnosis = crashlab.diagnose(imported, heuristic=True, store=fresh, save=False)
    assert diagnosis.model == "heuristic"
    assert diagnosis.label is not None

    replayed = crashlab.replay(
        imported,
        fn=run_workflow,
        overrides={"client": crashlab.wrap(FakeOpenAI([]))},
        store_dir=fresh.root,
        offline=True,
    )
    assert replayed.replay_of == imported.run_id
    replay_blob = json.dumps(replayed.model_dump(mode="json"), default=str)
    assert secret not in replay_blob


# ---------------------------------------------------------------------------
# Workflow B helpers: fake OpenAI module, proxy upstream, framework spans
# ---------------------------------------------------------------------------

_WATCHED = ("openai", "anthropic", "litellm", "google.genai", "google")


def _install_openai() -> types.ModuleType:
    """Duck-typed ``openai`` module so instrument() works without the SDK."""
    mod = types.ModuleType("openai")

    class _Completions:
        def __init__(self, owner: Any) -> None:
            self.owner = owner

        def create(self, **kwargs: Any) -> Any:
            self.owner.calls.append(kwargs)
            return text_response(1, "hello from openai", model=kwargs.get("model", "fake-model"))

    class _Chat:
        def __init__(self, owner: Any) -> None:
            self.completions = _Completions(owner)

    class OpenAI:
        base_url = "http://fake.local/v1"

        def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
            self.api_key = api_key
            self.calls: list[dict[str, Any]] = []
            self.chat = _Chat(self)

    mod.OpenAI = OpenAI
    sys.modules["openai"] = mod
    return mod


class _Upstream(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        payload = json.dumps(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello from upstream"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()


@contextmanager
def _serve(handler, *, host="127.0.0.1"):
    httpd = ThreadingHTTPServer((host, 0), handler)
    httpd.daemon_threads = True
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
        yield httpd, f"http://127.0.0.1:{httpd.server_address[1]}"
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


class _FakeSpanData:
    def __init__(self, typ: str, **fields: Any) -> None:
        self.type = typ
        for key, value in fields.items():
            setattr(self, key, value)


class _FakeSpan:
    def __init__(self, typ: str, *, span_id: str = "span-1", **fields: Any) -> None:
        self.span_id = span_id
        self.trace_id = "trace-e2e"
        self.parent_id = None
        self.error = None
        self.started_at = "2026-11-20T00:00:00+00:00"
        self.ended_at = "2026-11-20T00:00:01+00:00"
        self.span_data = _FakeSpanData(typ, **fields)


class _FakeTrace:
    def __init__(self, name: str = "e2e-workflow", output: Any = None) -> None:
        self.name = name
        self.trace_id = "trace-e2e"
        self.output = output


# ---------------------------------------------------------------------------
# Workflow B
# ---------------------------------------------------------------------------


def test_workflow_b_instrument_simulate_alerts_proxy_and_framework(crashlab_dir, tmp_path):
    saved = {name: sys.modules[name] for name in _WATCHED if name in sys.modules}
    _reset()
    try:
        openai = _install_openai()
        with crashlab.instrument(providers=["openai"], agent="e2e-sim-bot"):
            assert crashlab.instrumented()["openai"] is True
            client = openai.OpenAI(api_key="x")

            def agent(user_msg: str) -> str:
                resp = client.chat.completions.create(
                    model="fake-model",
                    messages=[{"role": "user", "content": user_msg}],
                )
                return resp.choices[0].message.content or ""

            # Auto-instrumented client driven by the conversation simulator.
            passing = simulate(
                agent,
                Scenario(
                    name="e2e-sim",
                    tags=["e2e"],
                    turns=[
                        ScenarioTurn(user="hi", expect_reply_matches="(?i)hello"),
                        ScenarioTurn(user="again", expect_reply_matches="(?i)hello"),
                    ],
                ),
            )
            assert passing.passed, passing.summary()
            assert passing.turns_executed == 2
            sim_run = crashlab.load(passing.run_id)
            assert "e2e" in sim_run.tags
            assert sum(s.operation == "chat" for s in sim_run.steps) == 2

            # A failing scenario + declarative rule produces a failed run for alerts.
            failing = simulate(
                agent,
                Scenario(
                    name="e2e-sim-fail",
                    turns=[
                        ScenarioTurn(
                            user="hi",
                            expect_reply_matches="this-will-not-match",
                        )
                    ],
                ),
                rules=[{"type": "reply_contains", "text": "this-will-not-appear"}],
            )
            assert not failing.passed
            fail_run = crashlab.load(failing.run_id)
            assert fail_run.outcome == "fail"

            alert_path = tmp_path / "alerts.jsonl"
            dispatcher = AlertDispatcher(AlertRule(), sinks=[JsonlSink(alert_path)])
            events = dispatcher.dispatch(fail_run)
            assert events and events[0].kind == "failed"
            lines = alert_path.read_text("utf-8").splitlines()
            assert len(lines) == 1
            row = json.loads(lines[0])
            assert row["kind"] == "failed" and row["run_id"] == fail_run.run_id

            # Framework adapter: OpenAI Agents SDK tracing processor span capture.
            proc = crashlab_tracing_processor(name="e2e-shop")
            assert isinstance(proc, OpenAIAgentsTracingProcessor)
            proc.on_trace_start(_FakeTrace("shop-workflow"))
            try:
                proc.on_span_start(_FakeSpan("agent", span_id="a1", name="shopper"))
                gen = dict(
                    model="gpt-x",
                    input=[
                        {"role": "system", "content": "You sell tea."},
                        {"role": "user", "content": "price of tea?"},
                    ],
                    output=[{"role": "assistant", "content": "let me look it up"}],
                    usage={"input_tokens": 4, "output_tokens": 6, "total_tokens": 10},
                )
                proc.on_span_start(_FakeSpan("generation", span_id="g1", **gen))
                proc.on_span_end(_FakeSpan("generation", span_id="g1", **gen))
                tool_fields = dict(
                    name="get_price",
                    input='{"item": "tea"}',
                    output={"item": "tea", "price": 3},
                )
                proc.on_span_start(_FakeSpan("function", span_id="f1", **tool_fields))
                proc.on_span_end(_FakeSpan("function", span_id="f1", **tool_fields))
                proc.on_span_end(_FakeSpan("agent", span_id="a1", name="shopper"))
            finally:
                proc.on_trace_end(_FakeTrace("shop-workflow", output="tea is $3"))
            traced = crashlab.load("last")
            assert traced.name == "e2e-shop"
            ops = {s.operation for s in traced.steps}
            assert "chat" in ops and "execute_tool" in ops
            assert traced.output == "tea is $3"
            proc.shutdown()
    finally:
        _reset()
        for name in _WATCHED:
            if name in saved:
                sys.modules[name] = saved[name]
            else:
                sys.modules.pop(name, None)

    # Proxy record + offline replay (upstream is down during replay).
    with _serve(_Upstream) as (up, upstream):
        config = ProxyConfig(host="127.0.0.1", port=0, upstream_url=upstream, agent="e2e-proxy")
        with _proxy(config) as (_, base):
            req = {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]}
            status, headers, body = _post(base + "/v1/chat/completions", req)
            assert status == 200
            recorded = json.loads(body)
            run_id = headers.get("X-Crashlab-Run-Id") or headers.get("x-crashlab-run-id")
        up_port = up.server_address[1]

    proxy_run = Store().load_run(run_id or "last")
    assert proxy_run.name.startswith("e2e-proxy")
    assert len(proxy_run.steps) == 1
    replay_cfg = ProxyConfig(host="127.0.0.1", port=0, replay_run_id=proxy_run.run_id)
    with _proxy(replay_cfg) as (_, base):
        status, _, body = _post(
            base + "/v1/chat/completions",
            {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert status == 200
        assert (
            json.loads(body)["choices"][0]["message"]["content"]
            == recorded["choices"][0]["message"]["content"]
        )

    import socket

    sock = socket.socket()
    sock.settimeout(0.2)
    with pytest.raises((OSError, ConnectionRefusedError, TimeoutError)):
        sock.connect(("127.0.0.1", up_port))
    sock.close()
