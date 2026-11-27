"""Importing traces from other tools: OTLP/JSON GenAI, Langfuse, OpenInference / Phoenix,
LangSmith. The imported runs feed the same commands as recorded ones."""

from __future__ import annotations

import contextlib
import json
import re
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

import crashlab
from crashlab import rag
from crashlab.cli import app
from crashlab.ingest import (
    IngestError,
    detect_format,
    from_langfuse,
    from_langsmith,
    from_openinference,
    from_otlp,
    load_path,
    save_runs,
)
from crashlab.models import Run
from crashlab.otel import to_otlp
from crashlab.store import Store
from tests.fakes import FakeOpenAI, refund_script, text_response, tool_call_response

FIX = Path(__file__).parent / "fixtures" / "ingest"
runner = CliRunner()


def _load(name: str):
    return json.loads((FIX / name).read_text("utf-8"))


def _out(result) -> str:
    return " ".join(re.sub(r"\x1b\[[0-9;]*m", "", result.output).split())


def _shape(run: Run):
    return [
        (
            s.idx,
            s.operation,
            s.name,
            s.agent,
            s.attributes.get("gen_ai.usage.input_tokens"),
            s.attributes.get("gen_ai.usage.output_tokens"),
            s.attributes.get("gen_ai.tool.call.id"),
            (s.error.type, s.error.message, s.error.is_tool_error) if s.error else None,
        )
        for s in run.steps
    ]


# --- OTLP ---


def test_otlp_genai_new_conventions():
    (run,) = from_otlp(_load("otel_genai.json"))  # the /healthz span is not GenAI
    assert run.name == "support-bot" and run.status == "ok"
    assert run.env["metadata"]["source"] == "otel" and "imported" in run.tags
    assert run.input == {"task": "Where is order A17?"}
    assert run.output == "Order A17 shipped and arrives on 2026-12-02."
    assert [(s.operation, s.name) for s in run.steps] == [
        ("chat", "gpt-4o-mini"),
        ("execute_tool", "get_order"),
        ("chat", "gpt-4o-mini"),
    ]
    chat, tool, final = run.steps
    assert chat.request["system"] == "You are a support agent. Use tools."
    assert chat.request["messages"][0]["parts"] == [
        {"type": "text", "content": "Where is order A17?"}
    ]
    assert chat.request["tools"][0]["name"] == "get_order"
    assert chat.request["params"] == {"model": "gpt-4o-mini", "temperature": 0.2, "max_tokens": 512}
    assert chat.response["tool_calls"] == [
        {"id": "call_1", "type": "function", "name": "get_order", "arguments": {"order_id": "A17"}}
    ]
    assert chat.response["finish_reason"] == "tool_calls"
    assert chat.attributes["gen_ai.usage.input_tokens"] == 120
    assert chat.attributes["gen_ai.response.model"] == "gpt-4o-mini-2024-07-18"
    assert tool.args == {"order_id": "A17"} and tool.result["status"] == "shipped"
    assert tool.attributes["gen_ai.tool.call.id"] == "call_1"
    assert run.caller_chat(tool.idx) == chat.idx
    assert final.response["messages"][0]["parts"][0]["content"].startswith("Order A17 shipped")
    assert chat.duration_ms == pytest.approx(390.0) and tool.duration_ms == pytest.approx(110.0)
    assert run.usage.input_tokens == 291 and run.usage.output_tokens == 34
    assert run.usage.llm_calls == 2 and run.usage.tool_calls == 1
    assert all(s.parent_span_id == run.span_id for s in run.steps)
    assert run.started_at < run.ended_at


def test_otlp_legacy_attributes_tool_errors_and_inferred_call_ids():
    (run,) = from_otlp(_load("otel_legacy_tool_error.json"))
    assert run.name == "billing-agent.run"
    assert run.input == {"task": "Charge order B9 for 40 dollars"}
    chat0 = run.steps[0]
    assert [m["role"] for m in chat0.request["messages"]] == ["user"]
    assert chat0.request["system"] == "You bill customers."
    assert chat0.response["tool_calls"][0]["arguments"] == {"order_id": "B9", "amount": "forty"}
    assert chat0.attributes["gen_ai.usage.input_tokens"] == 90
    err = run.steps[1].error
    assert err is not None and err.type == "HTTPError" and err.is_tool_error
    assert "502" in err.message
    # the second tool span had no call id: it is matched to the chat call that asked for it
    assert [s.attributes.get("gen_ai.tool.call.id") for s in run.tool_steps()] == ["c1", "c2", "c3"]
    assert run.steps[3].response is None
    assert run.usage.errors == 3 and len(run.steps) == 7


def test_otlp_message_events_and_status_error():
    attrs = lambda d: [{"key": k, "value": {"stringValue": str(v)}} for k, v in d.items()]  # noqa: E731
    span = {
        "traceId": "t1",
        "spanId": "s1",
        "name": "chat gpt-4o",
        "startTimeUnixNano": "1766000000000000000",
        "endTimeUnixNano": "1766000001000000000",
        "attributes": attrs({"gen_ai.operation.name": "chat", "gen_ai.request.model": "gpt-4o"}),
        "status": {"code": 2, "message": "rate limited"},
        "events": [
            {"name": "gen_ai.user.message", "attributes": attrs({"content": "hi"})},
            {
                "name": "gen_ai.choice",
                "attributes": attrs(
                    {"finish_reason": "stop", "message": json.dumps({"content": "hello"})}
                ),
            },
        ],
    }
    doc = {"resourceSpans": [{"scopeSpans": [{"spans": [span]}]}]}
    (run,) = from_otlp(doc)
    (step,) = run.steps
    assert step.request["messages"][0]["parts"][0]["content"] == "hi"
    assert step.response["messages"][0]["parts"][0]["content"] == "hello"
    assert step.error is not None and step.error.message == "rate limited"
    assert run.status == "error" and run.error.message == "rate limited"


def test_otlp_round_trip_of_a_recorded_run():
    from examples.refund_bot import agent as refund

    refund.run_agent(
        "Refund my order A17",
        client=crashlab.wrap(FakeOpenAI(refund_script())),
        model="fake-model",
        faults=[],
    )
    src = crashlab.load("last")
    (back,) = from_otlp(to_otlp(src))
    assert back.run_id != src.run_id and back.name == src.name
    assert _shape(back) == _shape(src)
    assert [
        s.request_hash == o.request_hash for s, o in zip(back.steps, src.steps, strict=True)
    ] != []
    assert back.usage.input_tokens == src.usage.input_tokens
    assert back.usage.output_tokens == src.usage.output_tokens
    assert (back.usage.llm_calls, back.usage.tool_calls) == (3, 2)
    for a, b in zip(back.steps, src.steps, strict=True):
        assert a.duration_ms == pytest.approx(b.duration_ms, abs=0.01)
        assert a.attributes.get("gen_ai.request.model") == b.attributes.get("gen_ai.request.model")


def test_otlp_round_trip_keeps_errors_and_agents():
    @crashlab.tool()
    def flaky(x: int) -> int:
        raise RuntimeError("boom")

    client = crashlab.wrap(
        FakeOpenAI([tool_call_response(1, "flaky", {"x": 1}), text_response(2, "sorry")]),
        agent="planner",
    )
    with crashlab.record("rt-errors"):
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "go"}])
        with contextlib.suppress(RuntimeError):
            flaky(x=1)
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "go"}])
    src = crashlab.load("last")
    (back,) = from_otlp(to_otlp(src))
    assert _shape(back) == _shape(src)
    assert back.steps[1].error.type == "RuntimeError" and back.steps[1].error.is_tool_error
    assert back.steps[0].agent == "planner"


def test_nested_agent_span_labels_its_children():
    def sp(sid, parent, name, op, extra, t):
        a = {"gen_ai.operation.name": op, **extra}
        return {
            "traceId": "tr",
            "spanId": sid,
            "parentSpanId": parent,
            "name": name,
            "startTimeUnixNano": str(1766000000_000_000_000 + t * 10**9),
            "endTimeUnixNano": str(1766000000_000_000_000 + (t + 1) * 10**9),
            "attributes": [{"key": k, "value": {"stringValue": str(v)}} for k, v in a.items()],
        }

    spans = [
        sp(
            "a1",
            "",
            "invoke_agent orchestrator",
            "invoke_agent",
            {"gen_ai.agent.name": "orchestrator"},
            0,
        ),
        sp(
            "a2",
            "a1",
            "invoke_agent researcher",
            "invoke_agent",
            {"gen_ai.agent.name": "researcher"},
            1,
        ),
        sp("c1", "a2", "chat m", "chat", {"gen_ai.request.model": "m"}, 2),
        sp("c2", "a1", "chat m", "chat", {"gen_ai.request.model": "m"}, 5),
    ]
    (run,) = from_otlp({"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]})
    assert run.name == "orchestrator"
    assert [s.agent for s in run.steps] == ["researcher", None]


def test_otlp_multiple_traces_make_multiple_runs():
    doc = _load("otel_genai.json")
    other = json.loads(json.dumps(_load("otel_legacy_tool_error.json")))
    merged = {"resourceSpans": doc["resourceSpans"] + other["resourceSpans"]}
    runs = from_otlp(merged)
    assert sorted(r.name for r in runs) == ["billing-agent.run", "support-bot"]
    assert len({r.run_id for r in runs}) == 2


# --- Langfuse ---


def test_langfuse_trace_with_retriever_and_generation():
    (run,) = from_langfuse(_load("langfuse_trace.json"))
    assert run.name == "docs-assistant" and run.env["metadata"]["source"] == "langfuse"
    assert run.env["metadata"]["session_id"] == "sess-42"
    assert run.input == {"task": "How do I rotate an API key?"}
    assert [(s.operation, s.name) for s in run.steps] == [
        ("execute_tool", "search_docs"),
        ("chat", "gpt-4o-mini"),
    ]  # the EVENT observation is not a step
    gen = run.steps[1]
    assert gen.request["system"].startswith("Answer only from the context")
    assert gen.request["params"] == {"model": "gpt-4o-mini", "temperature": 0, "max_tokens": 300}
    assert gen.attributes["gen_ai.usage.input_tokens"] == 96
    assert gen.attributes["gen_ai.usage.output_tokens"] == 24
    assert gen.duration_ms == pytest.approx(1900.0)
    # retrieval is visible to crashlab.rag
    docs = rag.retrieved(run, "search_docs")
    assert [d["id"] for _, d in docs] == ["doc-keys", "doc-billing"]
    assert rag.citations_were_retrieved(run, run.output, tool="search_docs") is None


def test_langfuse_tool_error_and_tool_calls():
    lines = (FIX / "langfuse_traces.jsonl").read_text("utf-8").splitlines()
    runs = from_langfuse([json.loads(x) for x in lines])
    assert len(runs) == 2
    run = runs[1]
    chat, tool = run.steps
    assert chat.response["tool_calls"][0]["name"] == "reset_2fa"
    assert chat.attributes["gen_ai.usage.input_tokens"] == 40  # usageDetails
    assert tool.error is not None and tool.error.is_tool_error
    assert "PermissionError" in tool.error.message
    assert tool.attributes["gen_ai.tool.call.id"] == "call_a"


def test_langfuse_api_list_response():
    trace = _load("langfuse_trace.json")
    runs = from_langfuse({"data": [trace], "meta": {"page": 1}})
    assert len(runs) == 1
    with pytest.raises(IngestError, match="observation ids only"):
        from_langfuse({**trace, "observations": ["obs-1", "obs-2"]})


# --- OpenInference / Phoenix ---


def test_openinference_phoenix_spans():
    (run,) = from_openinference(_load("openinference_phoenix.json"))
    assert run.name == "agent" and run.env["metadata"]["source"] == "openinference"
    assert run.output == "It is 18C and sunny in Paris."
    assert [(s.operation, s.name) for s in run.steps] == [
        ("chat", "gpt-4o-mini"),
        ("execute_tool", "get_weather"),
        ("chat", "gpt-4o-mini"),
    ]
    chat, tool, final = run.steps
    assert chat.request["system"] == "You are a weather bot."
    assert chat.request["params"]["temperature"] == 0.1
    assert chat.request["tools"][0]["name"] == "get_weather"
    assert chat.response["tool_calls"][0]["arguments"] == {"city": "Paris"}
    assert chat.attributes["gen_ai.usage.input_tokens"] == 60
    assert tool.args == {"city": "Paris"} and tool.result == {"temp_c": 18, "sky": "sunny"}
    assert tool.attributes["gen_ai.tool.call.id"] == "call_w1"
    tool_msg = final.request["messages"][2]
    assert tool_msg["role"] == "tool" and tool_msg["parts"][0]["id"] == "call_w1"
    assert run.caller_chat(tool.idx) == chat.idx


def test_openinference_flat_dataframe_rows_and_otlp():
    rows = []
    for rec in _load("openinference_phoenix.json"):
        row = {
            "name": rec["name"],
            "parent_id": rec["parent_id"],
            "start_time": rec["start_time"],
            "end_time": rec["end_time"],
            "status_code": rec["status_code"],
            "context.trace_id": rec["context"]["trace_id"],
            "context.span_id": rec["context"]["span_id"],
        }
        row.update({f"attributes.{k}": v for k, v in rec["attributes"].items()})
        rows.append(row)
    (run,) = from_openinference(rows)
    assert [s.operation for s in run.steps] == ["chat", "execute_tool", "chat"]

    # the same spans as OTLP with openinference.* attributes
    flat = {
        "llm.model_name": "gpt-4o-mini",
        "openinference.span.kind": "LLM",
        "llm.input_messages.0.message.role": "user",
        "llm.input_messages.0.message.content": "hi",
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.content": "hello",
        "llm.token_count.prompt": 5,
        "llm.token_count.completion": 2,
    }
    otlp = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "x",
                                "spanId": "y",
                                "name": "ChatCompletion",
                                "attributes": [
                                    {
                                        "key": k,
                                        "value": {"intValue": str(v)}
                                        if isinstance(v, int)
                                        else {"stringValue": v},
                                    }
                                    for k, v in flat.items()
                                ],
                            }
                        ]
                    }
                ]
            }
        ]
    }
    assert detect_format(otlp) == "openinference"
    (run,) = from_otlp(otlp)
    assert run.steps[0].attributes["gen_ai.usage.output_tokens"] == 2
    assert run.output == "hello"


# --- LangSmith ---


def test_langsmith_rag_run_tree():
    (run,) = from_langsmith(_load("langsmith_rag_run.json"))
    assert run.name == "rag_chain" and run.env["metadata"]["source"] == "langsmith"
    assert run.input == {"task": "What is the refund window?"}
    assert run.output == "Refunds are accepted within 90 days. [policy-v1]"
    assert [(s.operation, s.name) for s in run.steps] == [
        ("execute_tool", "VectorStoreRetriever"),
        ("chat", "gpt-4o-mini"),
    ]
    retr, llm = run.steps
    assert retr.args == {"query": "refund window"}
    got = rag.retrieved(run, "VectorStoreRetriever")
    assert [d["id"] for _, d in got] == ["policy-v2", "policy-v1"]
    assert rag.citations_were_retrieved(run, run.output, tool="VectorStoreRetriever") is None
    assert [m["role"] for m in llm.request["messages"]] == ["user"]
    assert llm.request["system"].startswith("Answer from the context")
    assert llm.response["messages"][0]["parts"][0]["content"].endswith("[policy-v1]")
    assert llm.attributes["gen_ai.usage.input_tokens"] == 88
    assert llm.request["params"]["temperature"] == 0.0


def test_langsmith_tool_runs_flat_list_and_errors():
    def lc_ai(calls):
        return {
            "lc": 1,
            "type": "constructor",
            "id": ["langchain", "schema", "messages", "AIMessage"],
            "kwargs": {"content": "", "tool_calls": calls},
        }

    root = {
        "id": "r0",
        "trace_id": "r0",
        "name": "agent",
        "run_type": "chain",
        "inputs": {"input": "go"},
        "outputs": {},
        "error": "ValueError('bad args')",
        "start_time": "2026-11-22T08:00:00",
    }
    llm = {
        "id": "r1",
        "trace_id": "r0",
        "parent_run_id": "r0",
        "name": "ChatOpenAI",
        "run_type": "llm",
        "start_time": "2026-11-22T08:00:01",
        "end_time": "2026-11-22T08:00:02",
        "inputs": {
            "messages": [
                [
                    {
                        "lc": 1,
                        "type": "constructor",
                        "id": ["x", "HumanMessage"],
                        "kwargs": {"content": "go"},
                    }
                ]
            ]
        },
        "outputs": {
            "generations": [
                [{"message": lc_ai([{"name": "add", "args": {"a": 1}, "id": "t1"}]), "text": ""}]
            ]
        },
    }
    tool = {
        "id": "r2",
        "trace_id": "r0",
        "parent_run_id": "r0",
        "name": "add",
        "run_type": "tool",
        "start_time": "2026-11-22T08:00:03",
        "end_time": "2026-11-22T08:00:04",
        "inputs": {"input": "{'a': 1}"},
        "outputs": None,
        "error": "ValueError('missing b')",
    }
    (run,) = from_langsmith([tool, llm, root])  # order does not matter
    assert [s.operation for s in run.steps] == ["chat", "execute_tool"]
    assert run.steps[0].response["tool_calls"] == [
        {"id": "t1", "type": "function", "name": "add", "arguments": {"a": 1}}
    ]
    assert run.steps[1].args == {"a": 1}
    assert run.steps[1].error.type == "ValueError" and run.steps[1].error.message == "'missing b'"
    assert run.steps[1].attributes["gen_ai.tool.call.id"] == "t1"
    assert run.status == "error" and run.error.type == "ValueError"


# --- detect_format and malformed input ---


@pytest.mark.parametrize(
    "name,fmt",
    [
        ("otel_genai.json", "otel"),
        ("otel_legacy_tool_error.json", "otel"),
        ("langfuse_trace.json", "langfuse"),
        ("openinference_phoenix.json", "openinference"),
        ("langsmith_rag_run.json", "langsmith"),
    ],
)
def test_detect_format(name, fmt):
    assert detect_format(_load(name)) == fmt
    assert detect_format((FIX / name).read_text("utf-8")) == fmt  # JSON text too


@pytest.mark.parametrize(
    "bad",
    [{}, [], "[]", {"hello": "world"}, [1, 2], 42],
)
def test_detect_format_rejects_unknown_shapes(bad):
    with pytest.raises(IngestError):
        detect_format(bad)


def test_malformed_input_raises_ingest_error():
    with pytest.raises(IngestError, match="not valid JSON"):
        from_otlp("{not json")
    with pytest.raises(IngestError, match="resourceSpans"):
        from_otlp({"spans": []})
    with pytest.raises(IngestError, match="resourceSpans"):
        from_otlp({"resourceSpans": "nope"})
    with pytest.raises(IngestError, match="no GenAI spans"):
        from_otlp(
            {
                "resourceSpans": [
                    {"scopeSpans": [{"spans": [{"traceId": "t", "spanId": "s", "name": "GET /"}]}]}
                ]
            }
        )
    with pytest.raises(IngestError, match="observations"):
        from_langfuse({"id": "t"})
    with pytest.raises(IngestError, match="run"):
        from_langsmith([{"name": "x"}])
    with pytest.raises(IngestError, match="no llm"):
        from_langsmith({"id": "1", "run_type": "chain", "name": "only a chain"})
    with pytest.raises(IngestError, match="OpenInference"):
        from_openinference(
            [{"name": "x", "context": {"trace_id": "t", "span_id": "s"}, "attributes": {}}]
        )
    assert issubclass(IngestError, ValueError)


def test_load_path_errors(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "x"}\nnot json\n')
    with pytest.raises(IngestError, match="line 2"):
        load_path(bad)
    empty = tmp_path / "empty.json"
    empty.write_text("")
    with pytest.raises(IngestError):
        load_path(empty)
    with pytest.raises(IngestError, match="unknown format"):
        load_path(FIX / "otel_genai.json", "datadog")


# --- storing, redaction, ids ---


def test_save_runs_is_idempotent_and_redacts_secrets():
    secret = "sk-" + "proj-" + "A1b2C3d4E5f6G7h8I9j0K1l2"
    doc = _load("langfuse_trace.json")
    doc["observations"][1]["output"][0]["page_content"] += f" Admin key: {secret}"
    runs = from_langfuse(doc)
    store = Store()
    assert save_runs(store, runs) == [(runs[0].run_id, True)]
    assert save_runs(store, from_langfuse(doc)) == [(runs[0].run_id, False)]  # same trace, same id
    loaded = store.load_run(runs[0].run_id)
    assert secret not in loaded.model_dump_json()
    assert secret not in (store.cassette_path(loaded.run_id)).read_text("utf-8")
    assert len(loaded.steps) == 2 and loaded.steps[0].result[0]["id"] == "doc-keys"
    assert save_runs(store, runs, replace=True)[0][1] is True
    assert len(store.load_run(runs[0].run_id).steps) == 2


# --- CLI ---


def _import(*args: str):
    return runner.invoke(app, ["import", *args])


def test_cli_import_detects_every_format_and_prints_ids():
    for name in (
        "otel_genai.json",
        "langfuse_trace.json",
        "openinference_phoenix.json",
        "langsmith_rag_run.json",
    ):
        res = _import(str(FIX / name))
        assert res.exit_code == 0, _out(res)
        assert "imported:" in _out(res)
    res = _import(str(FIX / "otel_genai.json"))
    assert "already there:" in _out(res)
    res = _import(str(FIX / "otel_genai.json"), "--replace")
    assert "imported:" in _out(res)
    assert len(Store().list_runs(limit=50)) == 4


def test_cli_import_json_explicit_format_and_jsonl(tmp_path):
    res = _import(str(FIX / "langfuse_traces.jsonl"), "--json", "--format", "langfuse")
    assert res.exit_code == 0, _out(res)
    payload = json.loads(res.output)
    assert [r["format"] for r in payload["runs"]] == ["langfuse", "langfuse"]
    assert all(r["added"] for r in payload["runs"]) and payload["runs"][0]["steps"] == 2
    ids = {r["run_id"] for r in payload["runs"]}
    assert ids == {r["run_id"] for r in Store().list_runs(limit=10)}
    # a wrong explicit format is an error, not a silent misparse
    res = _import(str(FIX / "otel_genai.json"), "--format", "langfuse")
    assert res.exit_code == 2 and "observations" in _out(res)
    res = _import(str(FIX / "otel_genai.json"), "--format", "datadog")
    assert res.exit_code == 2 and "--format must be" in _out(res)


def test_cli_import_directory_and_malformed(tmp_path):
    d = tmp_path / "traces"
    d.mkdir()
    for name in ("otel_genai.json", "langsmith_rag_run.json", "langfuse_traces.jsonl"):
        shutil.copy(FIX / name, d / name)
    (d / "notes.txt").write_text("ignored")
    res = _import(str(d), "--json")
    assert res.exit_code == 0, _out(res)
    assert len(json.loads(res.output)["runs"]) == 4
    broken = tmp_path / "broken.json"
    broken.write_text("{oops")
    res = _import(str(broken))
    assert res.exit_code == 2 and "not valid JSON" in _out(res)
    junk = tmp_path / "junk.json"
    junk.write_text('{"hello": 1}')
    res = _import(str(junk))
    assert res.exit_code == 2 and "unrecognized trace format" in _out(res)
    empty = tmp_path / "empty"
    empty.mkdir()
    res = _import(str(empty))
    assert res.exit_code == 2 and "is a directory" in _out(res)


def test_cli_import_keeps_jsonl_exports_and_bundles(tmp_path):
    from examples.refund_bot import agent as refund

    refund.run_agent(
        "Refund my order A17",
        client=crashlab.wrap(FakeOpenAI(refund_script())),
        model="fake-model",
        faults=[],
    )
    src = crashlab.load("last")
    out = tmp_path / "run.jsonl"
    assert runner.invoke(app, ["export", src.run_id, "-o", str(out)]).exit_code == 0
    runner.invoke(app, ["rm", src.run_id, "--yes"])
    res = _import(str(out))
    assert res.exit_code == 0 and f"imported: {src.run_id}" in _out(res)
    assert _import(str(out), "--format", "jsonl").exit_code == 0
    assert len(Store().load_run(src.run_id).steps) == 5


# --- the imported run is a first-class run ---


@pytest.fixture
def imported_tool_error():
    (run,) = from_otlp(_load("otel_legacy_tool_error.json"))
    save_runs(Store(), [run])
    return run


def test_show_check_diagnose_stats_diff_on_imported_runs(imported_tool_error):
    run = imported_tool_error
    ok_run = from_otlp(_load("otel_genai.json"))[0]
    save_runs(Store(), [ok_run])
    show = runner.invoke(app, ["show", run.run_id])
    assert show.exit_code == 0 and "charge_card" in _out(show) and "HTTPError" in _out(show)

    res = runner.invoke(app, ["check", run.run_id, "--json"])
    assert res.exit_code in (0, 1), _out(res)
    kinds = {f["kind"] for f in json.loads(res.output)["findings"]}
    assert "tool_error_streak" in kinds

    res = runner.invoke(app, ["diagnose", run.run_id, "--heuristic", "--json"])
    assert res.exit_code == 0, _out(res)
    diag = json.loads(res.output)
    assert diag["label"] in ("STEP_REPETITION", "TOOL_ERROR_MISHANDLED")

    res = runner.invoke(app, ["stats", "--json"])
    assert res.exit_code == 0 and json.loads(res.output)

    res = runner.invoke(app, ["diff", run.run_id, ok_run.run_id, "--json"])
    assert res.exit_code == 0, _out(res)
    assert json.loads(res.output)


def test_check_with_python_oracle_sees_imported_steps():
    (run,) = from_langfuse(_load("langfuse_trace.json"))
    save_runs(Store(), [run])
    loaded = crashlab.load(run.run_id)
    assert rag.retrieval_non_empty(loaded, tool="search_docs") is None
    assert loaded.tool_steps("search_docs") and loaded.chat_steps()
    from crashlab.budgets import summarize_usage

    assert summarize_usage(loaded).total_tokens == 120


def test_imported_run_replays_offline_with_a_stand_in_entry_point():
    (run,) = from_otlp(_load("otel_genai.json"))
    save_runs(Store(), [run])
    calls: list[str] = []

    @crashlab.tool()
    def get_order(order_id: str) -> dict:
        calls.append(order_id)
        raise AssertionError("tools never run during a replay")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_order",
                "description": "Look up an order",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
            },
        }
    ]
    system = {"role": "system", "content": "You are a support agent. Use tools."}
    user = {"role": "user", "content": "Where is order A17?"}
    asked = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_order", "arguments": json.dumps({"order_id": "A17"})},
            }
        ],
    }

    def main():
        client = crashlab.wrap(FakeOpenAI([]))
        with crashlab.record("support-bot") as rec:
            kw = {"model": "gpt-4o-mini", "tools": tools, "temperature": 0.2, "max_tokens": 512}
            client.chat.completions.create(messages=[system, user], **kw)
            result = get_order(order_id="A17")
            tool_msg = {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": json.dumps(
                    {"order_id": "A17", "status": "shipped", "eta": "2026-12-02"}
                ),
            }
            final = client.chat.completions.create(messages=[system, user, asked, tool_msg], **kw)
            rec.output = final.choices[0].message.content
            return result

    with pytest.raises(crashlab.ReplayError, match="fn="):
        crashlab.replay(run)  # imported runs have no recorded entry point
    rep = crashlab.replay(run.run_id, fn=main)
    assert calls == [] and rep.replay_of == run.run_id
    assert [s.source for s in rep.steps] == ["cassette"] * 3
    assert rep.output == run.output
    assert [s.request_hash for s in rep.steps] == [s.request_hash for s in run.steps]
