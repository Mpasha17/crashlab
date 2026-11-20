"""Terminal waterfall. Offline; no provider SDKs."""

from __future__ import annotations

from datetime import timedelta

from typer.testing import CliRunner

from crashlab.cli import app
from crashlab.models import ErrorInfo, Run
from crashlab.store import Store
from crashlab.timeline import render_timeline, timeline_events
from tests.debug_fixtures import T0, chat, save
from tests.test_otel import _chat_with_tool, _store_linked, _tool

runner = CliRunner()


def test_timeline_bar_tokens_cost_and_errors(crashlab_dir):
    store = Store()
    parent, child = _store_linked(store)
    text = render_timeline(parent, store=store, width=100)
    assert "refund-bot" in text and parent.run_id in text
    assert "chat gpt-4o-mini" in text and "tool get_order" in text
    assert "█" in text
    assert "100/10" in text
    assert "$" in text
    assert "F!" in text  # error-severity finding on the tool step
    # child subtree is indented
    assert "child worker" in text or (text.find("worker") > text.find("refund-bot"))
    child_line = next(ln for ln in text.splitlines() if "worker" in ln and child.run_id in ln)
    assert child_line.startswith("  ")
    # --no-children equivalent
    only = render_timeline(parent, store=store, include_children=False, width=80)
    assert child.run_id not in only
    # narrower still renders
    narrow = render_timeline(parent, width=40)
    assert "refund-bot" in narrow and "█" in narrow


def test_timeline_marks_errors_and_handoffs(crashlab_dir):
    store = Store()
    rid = "01TLHANDOFF000000000000001"
    # agent A calls transfer_to_b; agent B's next chat answers that call
    chat0, call_id = _chat_with_tool(
        rid,
        0,
        "go",
        "transfer_to_writer",
        {},
        agent="researcher",
        at=T0,
    )
    t1 = _tool(
        rid,
        1,
        "transfer_to_writer",
        {},
        "ok",
        call_id=call_id,
        agent="researcher",
        at=T0 + timedelta(milliseconds=20),
    )
    # B continues A's thread by answering the tool call
    msgs = list(chat0.request["messages"]) + [
        {
            "role": "assistant",
            "parts": [
                {"type": "tool_call", "id": call_id, "name": "transfer_to_writer", "arguments": {}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "parts": [{"type": "tool_call_response", "id": call_id, "content": "ok"}],
        },
    ]
    chat2 = chat(
        rid, 2, None, "done", history=msgs, at=T0 + timedelta(milliseconds=40), duration_ms=30
    )
    chat2.attributes["gen_ai.agent.name"] = "writer"
    # also a failing tool later
    chat3, call2 = _chat_with_tool(
        rid,
        3,
        None,
        "boom",
        {},
        agent="writer",
        at=T0 + timedelta(milliseconds=80),
        call_id="call_3",
    )
    # _chat_with_tool appends a user message when user is None? user is None so no extra user
    err = ErrorInfo(type="RuntimeError", message="nope")
    t4 = _tool(
        rid,
        4,
        "boom",
        {},
        None,
        call_id=call2,
        agent="writer",
        error=err,
        at=T0 + timedelta(milliseconds=90),
    )
    run = Run(run_id=rid, name="team", status="ok", outcome="fail", started_at=T0)
    save(store, run, [chat0, t1, chat2, chat3, t4])
    run = store.load_run(rid)
    assert run.handoffs()
    text = render_timeline(run, width=120)
    assert "ERR" in text
    assert "writer" in text and "researcher" in text
    # handoff marker on the via step or the landing step
    assert "↳" in text or "from researcher" in text


def test_timeline_events_and_cli(crashlab_dir):
    store = Store()
    parent, child = _store_linked(store)
    ev = timeline_events(parent, store=store)
    kinds = [e.kind for e in ev]
    assert kinds.count("run") == 2
    assert any(e.kind == "step" and e.idx == 1 and e.operation == "execute_tool" for e in ev)
    child_ev = next(e for e in ev if e.kind == "run" and e.run_id == child.run_id)
    assert child_ev.depth == 1
    res = runner.invoke(app, ["timeline", parent.run_id])
    assert res.exit_code == 0, res.output
    assert "refund-bot" in res.output and "█" in res.output
    res = runner.invoke(app, ["timeline", parent.run_id, "--no-children", "--json"])
    assert res.exit_code == 0
    import json

    data = json.loads(res.output)
    assert all(row["run_id"] == parent.run_id for row in data)
    res = runner.invoke(app, ["timeline", "NOPE"])
    assert res.exit_code == 1
