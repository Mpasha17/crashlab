"""examples/sql_analyst: tools, faults, oracles and ground truth. Offline (fake model)."""

from __future__ import annotations

import pytest
from examples.sql_analyst import agent as sql

import crashlab
from tests.fakes import FakeOpenAI, text_response, tool_call_response

Q = "What was ShopCo's revenue in March 2026?"
GOOD_SQL = (
    "SELECT SUM(total) FROM orders WHERE status = 'paid' "
    "AND order_date BETWEEN '2026-03-01' AND '2026-03-31'"
)
ALL_SQL = "SELECT SUM(total) FROM orders WHERE order_date LIKE '2026-03-%'"


def _run(script, faults=(), question=Q):
    client = crashlab.wrap(FakeOpenAI(script))
    sql.run_agent(question, client=client, model="fake-model", faults=list(faults))
    return crashlab.load("last")


def _script(query, answer):
    return [
        tool_call_response(1, "describe_table", {"table": "orders"}),
        tool_call_response(2, "run_query", {"sql": query}),
        tool_call_response(3, "final_answer", {"answer": answer}),
    ]


def test_reference_answers():
    assert [sql.reference_answer(q) for q in sql.QUESTIONS] == [198.99, 493.5, "Germany"]
    assert sql.reference_answer("unknown?") is None


def test_clean_run_passes_and_replays_offline():
    pytest.importorskip("openai")
    run = _run(_script(GOOD_SQL, "Revenue in March 2026 was $198.99."))
    assert run.outcome == "pass" and run.output == "Revenue in March 2026 was $198.99."
    assert [s.name if s.operation != "chat" else "chat" for s in run.steps] == [
        "chat",
        "describe_table",
        "chat",
        "run_query",
        "chat",
        "final_answer",
    ]
    assert run.steps[3].result == {"columns": ["SUM(total)"], "rows": [[198.99]]}
    assert crashlab.replay(run.run_id).outcome == "pass"


def test_spec_drop_wrong_answer_fails_with_the_step():
    run = _run(_script(ALL_SQL, "848.99"), faults=["spec_drop"])
    assert "Revenue means" not in run.steps[0].request["system"]
    [c] = [c for c in run.checks if c.name == "answer_matches_ground_truth"]
    assert c.status == "fail" and "step 5" in c.message and "198.99" in c.message
    assert sql.ground_truth(run, "spec_drop") == (3, (0, 5))


def test_tool_5xx_returns_html_and_the_window():
    run = _run(
        [
            tool_call_response(1, "run_query", {"sql": GOOD_SQL}),
            text_response(2, "The database is unavailable right now."),
        ],
        faults=["tool_5xx"],
    )
    assert "502 Bad Gateway" in run.steps[1].result and run.outcome == "fail"
    assert sql.ground_truth(run, "tool_5xx") == (1, (1, 2))


def test_schema_drift_renames_the_column():
    run = _run(
        [
            tool_call_response(1, "run_query", {"sql": GOOD_SQL}),
            tool_call_response(2, "final_answer", {"answer": "0"}),
        ],
        faults=["schema_drift"],
    )
    assert run.steps[1].error is not None and "no such column" in run.steps[1].error.message
    assert sql.ground_truth(run, "schema_drift") == (1, (1, 3))
    drifted = sql.make_db(frozenset({"schema_drift"}))
    assert "amount_total" in [c[1] for c in drifted.execute("PRAGMA table_info(orders)")]


def test_misleading_desc_only_changes_the_tool_description():
    plain, bad = sql.tools_spec(), sql.tools_spec(frozenset({"misleading_desc"}))
    assert "cents" in bad[2]["function"]["description"]
    assert "cents" not in plain[2]["function"]["description"]
    run = _run(
        _script(GOOD_SQL.replace("SUM(total)", "SUM(total) / 100"), "1.99"),
        faults=["misleading_desc"],
    )
    assert run.outcome == "fail" and sql.ground_truth(run, "misleading_desc") == (3, (3, 5))


def test_queries_are_read_only():
    with pytest.raises(PermissionError):
        sql.run_query("DELETE FROM orders")
    conn = sql.make_db()
    with pytest.raises(Exception, match="readonly|read-only"):
        conn.execute("UPDATE orders SET total = 0")


def test_country_answer_and_number_matching():
    assert sql.answer_ok("Germany had the most revenue", "Germany")
    assert not sql.answer_ok("France", "Germany")
    assert sql.answer_ok("about $1,033.50 in total", 1033.5)
    assert not sql.answer_ok("198.98", 198.99)


def test_unknown_fault_is_rejected():
    with pytest.raises(ValueError, match="unknown faults"):
        sql.run_agent(Q, client=object(), faults=["nope"])


def test_refund_bot_bench_faults_and_ground_truth():
    from examples.refund_bot import agent as refund

    from tests.test_diagnose import _policy_run

    assert set(refund.BENCH_FAULTS) == set(sql.FAULTS)
    assert "0.9" in refund.tools_for(frozenset({"misleading_desc"}))[0]["function"]["description"]
    assert refund.tools_for() is refund.TOOLS
    assert refund.ground_truth(_policy_run(), "spec_drop") == (3, (0, 3))


def test_an_empty_reply_gets_one_nudge():
    run = _run(
        [
            tool_call_response(1, "run_query", {"sql": GOOD_SQL}),
            text_response(2, ""),
            tool_call_response(3, "final_answer", {"answer": "198.99"}),
        ]
    )
    assert run.outcome == "pass"
    assert sql.NUDGE in str(run.steps[-2].request["messages"][-1])
