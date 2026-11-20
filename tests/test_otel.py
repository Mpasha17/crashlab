"""OTLP/JSON and Chrome trace export. Offline; no provider SDKs."""

from __future__ import annotations

import json
from datetime import timedelta

from typer.testing import CliRunner

from crashlab.cli import app
from crashlab.models import CheckResult, ErrorInfo, Finding, Run, Usage
from crashlab.otel import to_chrome_trace, to_otlp
from crashlab.store import Store
from tests.debug_fixtures import T0, chat, make_run, save, tool

runner = CliRunner()


def _attrs(*ids):
    spans = ids if len(ids) != 1 or not isinstance(ids[0], list) else ids[0]
    out = {}
    for s in spans:
        for a in s.get("attributes") or []:
            val = a["value"]
            out.setdefault(s["spanId"], {})[a["key"]] = next(iter(val.values()))
    return out


def _spans(payload):
    return payload["resourceSpans"][0]["scopeSpans"][0]["spans"]


def _by_name(spans, name):
    return [s for s in spans if s["name"] == name]


def _chat_with_tool(rid, idx, user, tool_name, args, *, model="gpt-4o-mini", at=None, agent=None,
                    tokens=(100, 10), cost=0.0002, duration_ms=120.0, call_id=None):  # fmt: skip
    call_id = call_id or f"call_{idx}"
    step = chat(
        rid,
        idx,
        user,
        tool_call=(tool_name, args),
        model=model,
        tokens=tokens,
        cost=cost,
        duration_ms=duration_ms,
        at=at or T0,
    )
    parts = (step.response or {}).get("messages", [{}])[0].get("parts") or []
    step.response = {
        **(step.response or {}),
        "model": model,
        "tool_calls": [{"id": call_id, "name": tool_name, "arguments": args}],
        "messages": [{"role": "assistant", "parts": parts}],
    }
    if agent:
        step.attributes["gen_ai.agent.name"] = agent
    return step, call_id


def _tool(rid, idx, name, args, result, *, call_id, at=None, agent=None, error=None,
          duration_ms=8.0):  # fmt: skip
    step = tool(rid, idx, name, args, result, error=error, duration_ms=duration_ms, at=at or T0)
    step.attributes["gen_ai.tool.call.id"] = call_id
    step.attributes["gen_ai.tool.name"] = name
    step.attributes["gen_ai.operation.name"] = "execute_tool"
    if agent:
        step.attributes["gen_ai.agent.name"] = agent
    return step


def _store_linked(store: Store):
    """A parent run with a chat→tool→chat, plus a child linked at tool step 1."""
    rid = "01OTELPARENT00000000000001"
    started = T0
    chat0, call_id = _chat_with_tool(
        rid,
        0,
        "lookup A17",
        "get_order",
        {"order_id": "A17"},
        at=started,
        agent="refund-bot",
    )
    t1 = _tool(
        rid,
        1,
        "get_order",
        {"order_id": "A17"},
        {"status": "delivered"},
        call_id=call_id,
        at=started + timedelta(milliseconds=130),
        agent="refund-bot",
    )
    chat2 = chat(
        rid,
        2,
        None,
        "Refund issued.",
        history=[{"role": "user", "parts": [{"type": "text", "content": "lookup A17"}]}],
        at=started + timedelta(milliseconds=150),
        duration_ms=80,
        tokens=(120, 20),
        cost=0.0003,
    )
    chat2.attributes["gen_ai.agent.name"] = "refund-bot"
    parent = Run(
        run_id=rid,
        name="refund-bot",
        status="ok",
        outcome="fail",
        outcome_reason="refund_within_policy failed",
        span_id="aa" * 8,
        started_at=started,
        ended_at=started + timedelta(seconds=2),
        findings=[
            Finding(
                kind="tool_error_streak",
                severity="error",
                step=1,
                steps=[1],
                message="tool errored",
            ),  # fmt: skip
        ],
        checks=[CheckResult(name="refund_within_policy", status="fail", message="over cap")],
        usage=Usage(
            steps=3,
            llm_calls=2,
            tool_calls=1,
            input_tokens=220,
            output_tokens=30,
            total_tokens=250,
            cost_usd=0.0005,
            wall_time_ms=2000,
        ),
        input={"user_msg": "lookup A17"},
    )
    save(store, parent, [chat0, t1, chat2])
    # a later update keeps the usage/span we set (save() overwrites usage)
    parent = store.load_run(rid)
    parent.span_id = "aa" * 8
    parent.outcome = "fail"
    parent.outcome_reason = "refund_within_policy failed"
    parent.findings = [
        Finding(
            kind="tool_error_streak", severity="error", step=1, steps=[1], message="tool errored"
        )
    ]
    parent.checks = [CheckResult(name="refund_within_policy", status="fail", message="over cap")]
    parent.usage.cost_usd = 0.0005
    parent.usage.input_tokens = 220
    parent.usage.output_tokens = 30
    parent.usage.total_tokens = 250
    parent.usage.wall_time_ms = 2000
    store.update_run(parent)
    parent = store.load_run(rid)

    cid = "01OTELCHILD000000000000001"
    child_chat = chat(cid, 0, "work", "done", at=started + timedelta(seconds=1), tokens=(10, 4),
                      cost=0.00005, duration_ms=40)  # fmt: skip
    child = Run(
        run_id=cid,
        name="worker",
        status="ok",
        outcome="pass",
        span_id="bb" * 8,
        started_at=started + timedelta(seconds=1),
        ended_at=started + timedelta(seconds=2),
        env={"link": {"parent_run": rid, "trace": rid, "parent_step": 1}},
        usage=Usage(steps=1, llm_calls=1, total_tokens=14, cost_usd=0.00005, wall_time_ms=400),
    )
    save(store, child, [child_chat])
    child = store.load_run(cid)
    child.span_id = "bb" * 8
    child.env = {"link": {"parent_run": rid, "trace": rid, "parent_step": 1}}
    child.usage.cost_usd = 0.00005
    child.usage.total_tokens = 14
    child.usage.wall_time_ms = 400
    child.outcome = "pass"
    store.update_run(child)
    return store.load_run(rid), store.load_run(cid)


def test_otlp_shape_and_gen_ai_conventions(crashlab_dir):
    store = Store()
    parent, child = _store_linked(store)
    payload = to_otlp(parent, store=store)
    assert list(payload) == ["resourceSpans"]
    resource = payload["resourceSpans"][0]["resource"]["attributes"]
    keys = {a["key"] for a in resource}
    assert "service.name" in keys and "crashlab.run_id" in keys
    spans = _spans(payload)
    # root + 3 parent steps + child root + 1 child step
    assert len(spans) >= 6
    for s in spans:
        assert len(s["traceId"]) == 32 and int(s["traceId"], 16) >= 0
        assert len(s["spanId"]) == 16 and int(s["spanId"], 16) >= 0
        assert s["startTimeUnixNano"].isdigit() and s["endTimeUnixNano"].isdigit()
        assert int(s["endTimeUnixNano"]) >= int(s["startTimeUnixNano"])
        assert s["traceId"] == spans[0]["traceId"]  # one tree, one trace

    roots = [s for s in spans if s["name"] == "refund-bot"]
    assert len(roots) == 1
    root = roots[0]
    assert "parentSpanId" not in root
    rattrs = {a["key"]: next(iter(a["value"].values())) for a in root["attributes"]}
    assert rattrs["gen_ai.operation.name"] == "invoke_agent"
    assert rattrs["gen_ai.agent.name"] == "refund-bot"
    assert rattrs["crashlab.run_id"] == parent.run_id
    assert rattrs["crashlab.outcome"] == "fail"
    assert "crashlab.cost_usd" in rattrs

    tool_spans = [s for s in spans if s["name"].startswith("tool ")]
    chat_spans = [s for s in spans if s["name"].startswith("chat ")]
    assert tool_spans and chat_spans
    tool_span = next(
        s
        for s in tool_spans
        if s["attributes"]
        and any(
            a["key"] == "crashlab.step" and a["value"].get("intValue") == "1"
            for a in s["attributes"]
        )
    )
    # tool step is parented to the chat that requested it (step 0), not the run root
    caller = next(
        s
        for s in chat_spans
        if any(
            a["key"] == "crashlab.step" and a["value"].get("intValue") == "0"
            for a in s["attributes"]
        )
    )
    assert tool_span["parentSpanId"] == caller["spanId"]
    assert caller["parentSpanId"] == root["spanId"]

    tattrs = {a["key"]: next(iter(a["value"].values())) for a in tool_span["attributes"]}
    assert tattrs["gen_ai.operation.name"] == "execute_tool"
    assert tattrs["gen_ai.tool.name"] == "get_order"
    assert tattrs["gen_ai.tool.call.id"] == "call_0"
    assert tattrs["gen_ai.agent.name"] == "refund-bot"
    cattrs = {a["key"]: next(iter(a["value"].values())) for a in caller["attributes"]}
    assert cattrs["gen_ai.system"] == "openai"
    assert cattrs["gen_ai.request.model"] == "gpt-4o-mini"
    assert cattrs["gen_ai.response.model"] == "gpt-4o-mini"
    assert cattrs["gen_ai.usage.input_tokens"] == "100"
    assert cattrs["gen_ai.usage.output_tokens"] == "10"
    assert "crashlab.cost_usd" in cattrs

    # findings on step 1 become events; run outcome=fail marks the root
    assert tool_span["status"]["code"] == 2
    ev_names = [e["name"] for e in tool_span.get("events") or []]
    assert "crashlab.finding" in ev_names
    assert root["status"]["code"] == 2

    child_root = next(s for s in spans if s["name"] == "worker")
    # child run is parented to the parent's tool step (parent_step=1)
    assert child_root["parentSpanId"] == tool_span["spanId"]
    cr = {a["key"]: next(iter(a["value"].values())) for a in child_root["attributes"]}
    assert cr["crashlab.run_id"] == child.run_id
    assert cr["crashlab.outcome"] == "pass"


def test_otlp_error_events_on_failing_steps(crashlab_dir):
    store = Store()
    rid = "01OTELERR00000000000000001"
    err = ErrorInfo(type="RuntimeError", message="boom <script>", traceback="Traceback: boom")
    step = chat(rid, 0, "hi", error=err, duration_ms=15, cost=None, tokens=(8, 0))
    run = Run(
        run_id=rid,
        name="broken",
        status="error",
        outcome="fail",
        error=err,
        started_at=T0,
        ended_at=T0 + timedelta(seconds=1),
    )
    save(store, run, [step])
    run = store.load_run(rid)
    spans = _spans(to_otlp(run))
    root = next(s for s in spans if s["name"] == "broken")
    assert root["status"]["code"] == 2
    assert any(e["name"] == "exception" for e in root["events"])
    chat_span = next(s for s in spans if s["name"].startswith("chat "))
    assert chat_span["status"]["code"] == 2
    ev = next(e for e in chat_span["events"] if e["name"] == "exception")
    keys = {a["key"]: next(iter(a["value"].values())) for a in ev["attributes"]}
    assert keys["exception.type"] == "RuntimeError"
    assert "boom <script>" in keys["exception.message"]


def test_include_children_false_and_without_store(crashlab_dir):
    store = Store()
    parent, child = _store_linked(store)
    only = _spans(to_otlp(parent, store=store, include_children=False))
    ids = {
        next(iter(a["value"].values()))
        for s in only
        for a in s["attributes"]
        if a["key"] == "crashlab.run_id"
    }
    assert parent.run_id in ids and child.run_id not in ids
    lone = _spans(to_otlp(parent))  # no store: cannot walk children
    ids = {
        next(iter(a["value"].values()))
        for s in lone
        for a in s["attributes"]
        if a["key"] == "crashlab.run_id"
    }
    assert ids == {parent.run_id}


def test_child_defaults_to_parent_run_span_when_no_parent_step(crashlab_dir):
    store = Store()
    parent = make_run(store, 1, name="orch")
    parent = store.load_run(parent.run_id)
    cid = "01OTELCHILD000000000000099"
    step = chat(cid, 0, "go", "ok", duration_ms=10)
    child = Run(
        run_id=cid,
        name="worker",
        status="ok",
        outcome="pass",
        started_at=T0,
        env={"link": {"parent_run": parent.run_id, "trace": parent.run_id}},
    )
    save(store, child, [step])
    child = store.load_run(cid)
    payload = to_otlp(parent, store=store)
    spans = _spans(payload)
    root = next(s for s in spans if s["name"] == "orch")
    child_root = next(s for s in spans if s["name"] == "worker")
    assert child_root["parentSpanId"] == root["spanId"]


def test_chrome_trace_shape(crashlab_dir):
    store = Store()
    parent, child = _store_linked(store)
    doc = to_chrome_trace(parent, store=store)
    assert doc["displayTimeUnit"] == "ms"
    events = doc["traceEvents"]
    complete = [e for e in events if e.get("ph") == "X"]
    assert complete
    pids = {e["pid"] for e in complete}
    tids = {e["tid"] for e in complete}
    assert any("refund-bot" in p for p in pids)
    assert any("worker" in p for p in pids)
    assert "refund-bot" in tids
    for e in complete:
        assert "ts" in e and "dur" in e and e["dur"] >= 0
        assert "args" in e
    tool_ev = next(e for e in complete if str(e["name"]).startswith("tool "))
    assert "tokens_in" in tool_ev["args"] or "cost" in tool_ev["args"] or "error" in tool_ev["args"]
    # loadable as JSON (Perfetto)
    json.loads(json.dumps(doc))
    none = to_chrome_trace(parent, include_children=False)
    names = {e["name"] for e in none["traceEvents"] if e.get("ph") == "X"}
    assert any("chat" in n or n == "refund-bot" for n in names)
    assert "worker" not in names


def test_cli_export_formats(crashlab_dir, tmp_path):
    store = Store()
    parent, _ = _store_linked(store)
    otel = tmp_path / "r.otel.json"
    res = runner.invoke(app, ["export", parent.run_id, "--format", "otel", "-o", str(otel)])
    assert res.exit_code == 0, res.output
    data = json.loads(otel.read_text("utf-8"))
    assert "resourceSpans" in data
    trace = tmp_path / "r.trace.json"
    res = runner.invoke(
        app, ["export", parent.run_id, "--format", "chrome-trace", "-o", str(trace)]
    )
    assert res.exit_code == 0, res.output
    doc = json.loads(trace.read_text("utf-8"))
    assert doc["displayTimeUnit"] == "ms" and "traceEvents" in doc
    # default format still writes jsonl
    res = runner.invoke(app, ["export", parent.run_id, "-o", str(tmp_path / "r.jsonl")])
    assert res.exit_code == 0 and "exported" in res.output
    assert (tmp_path / "r.jsonl").read_text("utf-8").lstrip().startswith("{")
    res = runner.invoke(app, ["export", parent.run_id, "--format", "nope"])
    assert res.exit_code == 2
