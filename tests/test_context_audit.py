"""Prompt context breakdown (``crashlab context``) and tool-schema audits."""

from __future__ import annotations

import json
import re
from datetime import timedelta

from typer.testing import CliRunner

from crashlab.cli import app
from crashlab.context_audit import analyze_context, audit_tool_schemas
from crashlab.models import Run, Usage
from crashlab.store import Store
from tests.debug_fixtures import T0, chat, save, tool

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*m")

GET_ORDER = {
    "type": "function",
    "name": "get_order",
    "description": "Look up an order by id and return its status and total.",
    "parameters": {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "The order identifier to look up.",
            }
        },
        "required": ["order_id"],
    },
}

# Groq-style bug: required keys, but no properties object. Providers reject this
# (the infamous transfer_to_refunds 400).
TRANSFER_BROKEN = {
    "type": "function",
    "name": "transfer_to_refunds",
    "description": "Hand the customer to the refunds agent.",
    "parameters": {
        "type": "object",
        "required": ["reason"],
    },
}

# Valid schema except the parameter has a type and no description.
SEARCH_NO_PARAM_DESC = {
    "type": "function",
    "name": "search_orders",
    "description": "Search orders by a free-text query.",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


def _bloated_run(store: Store) -> Run:
    """Two chat turns: the second prompt repeats history and a huge tool result."""
    rid = "01CONTEXT00000000000000001"
    system = "You are a refund assistant. Never refund twice."
    user = {"role": "user", "parts": [{"type": "text", "content": "Refund my order A17"}]}
    assistant_call = {
        "role": "assistant",
        "parts": [
            {
                "type": "tool_call",
                "id": "call_0",
                "name": "get_order",
                "arguments": {"order_id": "A17"},
            }
        ],
    }
    # A large tool result that will be re-sent on the next turn (context flood).
    bulky = {"status": "delivered", "total": 49, "notes": "x" * 400}
    tool_msg = {
        "role": "tool",
        "name": "get_order",
        "tool_call_id": "call_0",
        "parts": [{"type": "tool_call_response", "id": "call_0", "response": bulky}],
    }
    first = chat(
        rid, 0, "Refund my order A17", tool_call=("get_order", {"order_id": "A17"}),
        tokens=(200, 20), cost=0.001, duration_ms=400.0, at=T0,
    )  # fmt: skip
    first.request["system"] = system
    first.request["tools"] = [GET_ORDER, TRANSFER_BROKEN, SEARCH_NO_PARAM_DESC]
    first.request["messages"] = [user]
    mid = tool(
        rid, 1, "get_order", {"order_id": "A17"}, bulky,
        duration_ms=25.0, at=T0 + timedelta(milliseconds=400),
    )  # fmt: skip
    second = chat(
        rid, 2, None, "Done, refund issued.",
        tokens=(480, 15), cost=0.002, duration_ms=300.0,
        at=T0 + timedelta(milliseconds=425),
    )  # fmt: skip
    second.request["system"] = system
    second.request["tools"] = [GET_ORDER, TRANSFER_BROKEN, SEARCH_NO_PARAM_DESC]
    second.request["messages"] = [user, assistant_call, tool_msg]
    run = Run(
        run_id=rid,
        name="refund-bot",
        status="ok",
        outcome="pass",
        started_at=T0,
        ended_at=T0 + timedelta(milliseconds=800),
        usage=Usage(wall_time_ms=800, input_tokens=680, output_tokens=35, cost_usd=0.003),
    )
    steps = [first, mid, second]
    saved = save(store, run, steps)
    saved.steps = steps
    return saved


def _duplicate_run(store: Store) -> Run:
    """The second chat step appends the same user message a second time."""
    rid = "01CONTEXT00000000000000002"
    user = {"role": "user", "parts": [{"type": "text", "content": "Refund my order A17"}]}
    first = chat(rid, 0, "Refund my order A17", tokens=(80, 10), at=T0)
    first.request["tools"] = [GET_ORDER]
    first.request["messages"] = [user]
    second = chat(
        rid, 1, None, "Still looking.", tokens=(160, 8),
        at=T0 + timedelta(milliseconds=200),
    )  # fmt: skip
    second.request["system"] = first.request["system"]
    second.request["tools"] = [GET_ORDER]
    # Growing history plus a second copy of the same user turn.
    second.request["messages"] = [user, user]
    run = Run(run_id=rid, name="dup-bot", status="ok", outcome="pass", started_at=T0)
    steps = [first, second]
    saved = save(store, run, steps)
    saved.steps = steps
    return saved


def test_analyze_context_section_breakdown_and_scaled_tokens(crashlab_dir):
    run = _bloated_run(Store(crashlab_dir))
    report = analyze_context(run)
    assert report.run_id == run.run_id
    assert [s.idx for s in report.steps] == [0, 2]

    first = report.steps[0]
    by_name = {s.name: s for s in first.sections}
    assert set(by_name) == {
        "system_prompt",
        "tool_definitions",
        "user_messages",
        "assistant_messages",
        "tool_results",
        "other",
    }
    assert by_name["system_prompt"].chars == len("You are a refund assistant. Never refund twice.")
    assert by_name["tool_definitions"].chars > 0
    assert by_name["user_messages"].chars > 0
    assert by_name["assistant_messages"].chars == 0
    assert by_name["tool_results"].chars == 0
    # Recorded input_tokens are distributed across sections and sum exactly.
    assert first.input_tokens == 200
    assert sum(s.estimated_tokens for s in first.sections) == 200
    assert first.estimated_tokens == 200
    assert abs(sum(s.share_pct for s in first.sections) - 100) < 0.05
    assert first.delta_chars is None

    second = report.steps[1]
    by_name = {s.name: s for s in second.sections}
    assert by_name["tool_results"].chars > 400
    assert by_name["assistant_messages"].chars > 0
    assert second.input_tokens == 480
    assert sum(s.estimated_tokens for s in second.sections) == 480
    assert second.delta_chars is not None and second.delta_chars > 0
    assert second.delta_tokens is not None
    # The huge get_order result is the context flood.
    assert report.tool_outputs
    assert report.tool_outputs[0].name == "get_order"
    assert report.tool_outputs[0].chars >= by_name["tool_results"].chars
    assert 2 in report.tool_outputs[0].steps


def test_analyze_context_detects_duplicate_messages(crashlab_dir):
    run = _duplicate_run(Store(crashlab_dir))
    report = analyze_context(run)
    # The second step contains the user message twice (within-step + growth vs turn 0).
    second = next(s for s in report.steps if s.idx == 1)
    assert second.repeats
    assert any(r.role == "user" and r.count >= 2 and r.within_step for r in second.repeats)
    assert any(r.role == "user" and r.count >= 2 for r in report.repeats)

    only_first = analyze_context(run, step=0)
    assert [s.idx for s in only_first.steps] == [0]
    assert only_first.step == 0
    assert not any(r.within_step for r in only_first.repeats)


def test_analyze_context_rejects_a_non_chat_step(crashlab_dir):
    run = _bloated_run(Store(crashlab_dir))
    try:
        analyze_context(run, step=1)
    except ValueError as exc:
        assert "not a chat step" in str(exc)
    else:
        raise AssertionError("expected ValueError")
    try:
        analyze_context(run, step=99)
    except ValueError as exc:
        assert "no step 99" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_audit_tools_catches_groq_required_without_properties(crashlab_dir):
    run = _bloated_run(Store(crashlab_dir))
    report = audit_tool_schemas(run)
    kinds = {i.kind: i for i in report.issues}
    assert "required_missing_properties" in kinds
    issue = kinds["required_missing_properties"]
    assert issue.severity == "error"
    assert issue.tool == "transfer_to_refunds"
    assert "properties" in issue.message
    assert report.ok is False


def test_audit_tools_missing_param_descriptions_and_unused_tools(crashlab_dir):
    run = _bloated_run(Store(crashlab_dir))
    report = audit_tool_schemas(run)
    missing = [i for i in report.issues if i.kind == "param_missing_description"]
    assert any(i.tool == "search_orders" and "query" in i.message for i in missing)
    assert all(i.severity == "warn" for i in missing)

    # get_order was executed; the other two were defined on every chat step and never called.
    assert "get_order" in report.called
    assert "transfer_to_refunds" in report.unused
    assert "search_orders" in report.unused
    unused_issues = [i for i in report.issues if i.kind == "unused_tool"]
    assert {i.tool for i in unused_issues} >= {"transfer_to_refunds", "search_orders"}

    names = [t.name for t in report.tools]
    assert "get_order" in names and "transfer_to_refunds" in names


def test_audit_tools_empty_properties_and_similar_names(crashlab_dir):
    rid = "01CONTEXT00000000000000003"
    empty_required = {
        "type": "function",
        "name": "transfer_to_refunds",
        "description": "Hand the customer to the refunds agent.",
        "parameters": {"type": "object", "properties": {}, "required": ["reason"]},
    }
    almost = {
        "type": "function",
        "name": "transfer_to_refund",
        "description": "Hand the customer to the refunds agent.",
        "parameters": {"type": "object", "properties": {}},
    }
    first = chat(rid, 0, "hi", "ok", tokens=(40, 5), at=T0)
    first.request["tools"] = [empty_required, almost]
    run = Run(run_id=rid, name="similar-bot", status="ok", outcome="pass", started_at=T0)
    saved = save(Store(crashlab_dir), run, [first])
    saved.steps = [first]
    report = audit_tool_schemas(saved)
    assert any(
        i.kind == "required_missing_properties" and i.tool == "transfer_to_refunds"
        for i in report.issues
    )
    assert any(i.kind == "similar_tools" for i in report.issues)
    # Neither tool was called, and both were defined on the only chat step.
    assert set(report.unused) >= {"transfer_to_refunds", "transfer_to_refund"}


def test_context_and_audit_cli_help_json_and_fail_on_error(crashlab_dir):
    help_ctx = _plain(runner.invoke(app, ["context", "--help"]).output)
    assert "Break down a run" in help_ctx
    assert "--step" in help_ctx
    assert "--json" in help_ctx
    assert "\x1b[" not in help_ctx

    help_audit = _plain(runner.invoke(app, ["audit-tools", "--help"]).output)
    assert "Audit recorded tool schemas" in help_audit
    assert "--json" in help_audit
    assert "--fail-on-error" in help_audit
    assert "\x1b[" not in help_audit

    _bloated_run(Store(crashlab_dir))

    res = runner.invoke(app, ["context", "last", "--json"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["run_id"]
    assert len(payload["steps"]) == 2
    sections = {s["name"] for s in payload["steps"][0]["sections"]}
    assert "system_prompt" in sections and "tool_results" in sections
    assert payload["tool_outputs"][0]["name"] == "get_order"

    stepped = runner.invoke(app, ["context", "last", "--step", "2", "--json"])
    assert stepped.exit_code == 0, stepped.output
    only = json.loads(stepped.output)
    assert only["step"] == 2
    assert [s["idx"] for s in only["steps"]] == [2]

    text = _plain(runner.invoke(app, ["context", "last"]).output)
    assert "system_prompt" in text and "tool_definitions" in text
    assert "get_order" in text

    audit = runner.invoke(app, ["audit-tools", "last", "--json"])
    assert audit.exit_code == 0, audit.output
    body = json.loads(audit.output)
    assert body["ok"] is False
    kinds = {i["kind"] for i in body["issues"]}
    assert "required_missing_properties" in kinds
    assert "param_missing_description" in kinds
    assert "transfer_to_refunds" in body["unused"]

    failing = runner.invoke(app, ["audit-tools", "last", "--fail-on-error"])
    assert failing.exit_code == 1, failing.output
    out = _plain(failing.output)
    assert "transfer_to_refunds" in out
    assert "required_missing_properties" in out

    missing = runner.invoke(app, ["context", "no-such-run"])
    assert missing.exit_code == 2
    bad_step = runner.invoke(app, ["context", "last", "--step", "1"])
    assert bad_step.exit_code == 2
    missing_audit = runner.invoke(app, ["audit-tools", "no-such-run"])
    assert missing_audit.exit_code == 2
