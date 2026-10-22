"""sql-analyst: a small text-to-SQL agent over a SQLite copy of ``shop.sql``.

Tools: ``list_tables``, ``describe_table``, ``run_query`` (read-only) and
``final_answer``. ``@crashlab.check`` oracles grade the final answer against the result of
a reference query on the clean database.

Faults (``faults=[...]`` or ``FAULT_<NAME>=1``), recorded in the run input:

- ``spec_drop``: the rule defining revenue (paid orders only) is missing from the prompt.
- ``tool_5xx``: ``run_query`` returns an HTML 502 page instead of rows.
- ``schema_drift``: ``orders.total`` was renamed ``amount_total`` upstream; the prompt
  still says ``orders.total``.
- ``misleading_desc``: ``run_query``'s description wrongly says money is stored in cents.

    OPENAI_BASE_URL=... OPENAI_API_KEY=... python -m examples.sql_analyst.agent \\
        "What was ShopCo's revenue in March 2026?" --model MODEL [--fault-spec-drop]
"""

from __future__ import annotations

import argparse
import contextvars
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

import crashlab

FAULTS = ("spec_drop", "tool_5xx", "schema_drift", "misleading_desc")
SQL_FILE = Path(__file__).with_name("shop.sql")
STEP_BUDGET = 16
MAX_ROWS = 50

RULES = [
    "Revenue means the sum of orders.total over orders with status 'paid' only "
    "(cancelled and refunded orders are not revenue).",
    "Look at the schema with list_tables / describe_table before writing SQL.",
    "Only run read-only SELECT queries.",
    "Finish by calling final_answer with a short answer that includes the number "
    "(2 decimals) or the name that was asked for.",
]
SPEC_DROP_LINE = 0

#: the questions the benchmark asks, with the reference query for the right answer
QUESTIONS: dict[str, str] = {
    "What was ShopCo's revenue in March 2026?": (
        "SELECT ROUND(SUM(total), 2) FROM orders WHERE status = 'paid' "
        "AND order_date BETWEEN '2026-03-01' AND '2026-03-31'"
    ),
    "How much revenue came from customers in Germany in 2026?": (
        "SELECT ROUND(SUM(o.total), 2) FROM orders o JOIN customers c ON c.id = o.customer_id "
        "WHERE o.status = 'paid' AND c.country = 'Germany' AND o.order_date LIKE '2026-%'"
    ),
    "Which country generated the most revenue in Q1 2026 (January to March)?": (
        "SELECT c.country FROM orders o JOIN customers c ON c.id = o.customer_id "
        "WHERE o.status = 'paid' AND o.order_date BETWEEN '2026-01-01' AND '2026-03-31' "
        "GROUP BY c.country ORDER BY SUM(o.total) DESC LIMIT 1"
    ),
}

NUDGE = "Please call final_answer with your answer."
HTML_502 = """<!DOCTYPE html>
<html><head><title>502 Bad Gateway</title></head>
<body><center><h1>502 Bad Gateway</h1></center><hr><center>nginx/1.25.3</center></body>
</html>"""

_active: contextvars.ContextVar[frozenset[str] | None] = contextvars.ContextVar(
    "sql_analyst_faults", default=None
)
_db: contextvars.ContextVar[sqlite3.Connection | None] = contextvars.ContextVar(
    "sql_analyst_db", default=None
)
ANSWERS: list[str] = []


def faults_from_env() -> list[str]:
    on = ("1", "true", "yes", "on")
    return [f for f in FAULTS if os.environ.get(f"FAULT_{f.upper()}", "").lower() in on]


def active_faults() -> frozenset[str]:
    f = _active.get()
    return f if f is not None else frozenset(faults_from_env())


def make_db(faults: frozenset[str] = frozenset()) -> sqlite3.Connection:
    """A fresh in-memory copy of shop.sql (with the schema_drift rename if active),
    opened read-only for queries."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.executescript(SQL_FILE.read_text("utf-8"))
    if "schema_drift" in faults:
        conn.execute("ALTER TABLE orders RENAME COLUMN total TO amount_total")
    conn.commit()
    conn.execute("PRAGMA query_only = ON")
    return conn


def _conn() -> sqlite3.Connection:
    conn = _db.get()
    if conn is None:
        conn = make_db(active_faults())
        _db.set(conn)
    return conn


def reference_answer(question: str) -> Any:
    """The right answer, from the reference query on the clean database."""
    sql = QUESTIONS.get(question.strip())
    if sql is None:
        return None
    return make_db().execute(sql).fetchone()[0]


# --- tools --------------------------------------------------------------------------


@crashlab.tool()
def list_tables() -> list[str]:
    """List the tables in the database."""
    rows = _conn().execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
    return [r[0] for r in rows]


@crashlab.tool()
def describe_table(table: str) -> list[dict[str, str]]:
    """Columns of one table: name and type."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table or ""):
        raise ValueError(f"bad table name {table!r}")
    cols = _conn().execute(f"PRAGMA table_info({table})").fetchall()
    if not cols:
        raise KeyError(f"no table {table!r}")
    return [{"name": c[1], "type": c[2]} for c in cols]


@crashlab.tool()
def run_query(sql: str) -> dict[str, Any] | str:
    """Run one read-only SQL query (SQLite). Returns the column names and rows."""
    if "tool_5xx" in active_faults():
        return HTML_502
    if not re.match(r"\s*(select|with)\b", sql or "", re.I):
        raise PermissionError("only read-only SELECT queries are allowed")
    cur = _conn().execute(sql)
    rows = cur.fetchmany(MAX_ROWS)
    return {"columns": [d[0] for d in cur.description or []], "rows": [list(r) for r in rows]}


@crashlab.tool()
def final_answer(answer: str) -> str:
    """Give the final answer to the user's question. Call this once, at the end."""
    ANSWERS.append(answer)
    return "recorded"


def _tool(name: str, description: str, props: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props, "required": list(props)},
        },
    }


def tools_spec(faults: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    query_desc = "Run one read-only SQL query (SQLite). Returns the column names and rows."
    if "misleading_desc" in faults:
        query_desc += " Note: all money columns are stored in cents; divide by 100 to get dollars."
    return [
        _tool("list_tables", "List the tables in the database.", {}),
        _tool(
            "describe_table",
            "Columns of one table: name and type.",
            {"table": {"type": "string", "description": "table name"}},
        ),
        _tool("run_query", query_desc, {"sql": {"type": "string", "description": "SELECT ..."}}),
        _tool(
            "final_answer",
            "Give the final answer to the user's question. Call this once, at the end.",
            {"answer": {"type": "string"}},
        ),
    ]


TOOL_FUNCS = {
    "list_tables": list_tables,
    "describe_table": describe_table,
    "run_query": run_query,
    "final_answer": final_answer,
}


def system_prompt(faults: frozenset[str] = frozenset()) -> str:
    rules = [r for i, r in enumerate(RULES) if not ("spec_drop" in faults and i == SPEC_DROP_LINE)]
    numbered = "\n".join(f"{i}. {r}" for i, r in enumerate(rules, 1))
    return (
        "You are a data analyst for ShopCo. Answer the user's question from the SQLite "
        f"database using the tools.\n\nRules:\n{numbered}"
    )


# --- the agent loop (plain OpenAI-style tool calling) ---------------------------------


def _run_tool(name: str, arguments: str | dict[str, Any]) -> str:
    fn = TOOL_FUNCS.get(name)
    if fn is None:
        return f"ERROR: unknown tool {name!r}"
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            return f"ERROR: arguments are not valid JSON: {exc}"
    try:
        result = fn(**arguments)
    except Exception as exc:  # the model sees the error and can recover
        return f"ERROR: {type(exc).__name__}: {exc}"
    return result if isinstance(result, str) else json.dumps(result)


def _tool_call_dict(tc: Any) -> dict[str, Any]:
    if hasattr(tc, "model_dump"):
        return tc.model_dump(exclude_none=True)
    return {
        "id": tc.id,
        "type": "function",
        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
    }


def make_client() -> Any:
    from openai import OpenAI

    base_url = os.environ.get("OPENAI_BASE_URL") or None
    api_key = os.environ.get("OPENAI_API_KEY") or ("ollama" if base_url else None)
    return crashlab.wrap(OpenAI(base_url=base_url, api_key=api_key))


@crashlab.agent(name="sql-analyst", ignore_args=("client",))
def run_agent(
    question: str,
    *,
    client: Any = None,
    model: str | None = None,
    max_turns: int = 8,
    faults: list[str] | None = None,
) -> str:
    """Answer one analytics question; returns the final answer text."""
    unknown = set(faults or ()) - set(FAULTS)
    if unknown:
        raise ValueError(f"unknown faults {sorted(unknown)}; choose from {FAULTS}")
    active = frozenset(faults) if faults is not None else frozenset(faults_from_env())
    client = client or make_client()
    model = model or os.environ.get("MODEL") or "gpt-4o-mini"
    tok_f, tok_db = _active.set(active), _db.set(make_db(active))
    try:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt(active)},
            {"role": "user", "content": question},
        ]
        tools = tools_spec(active)
        nudged = False
        for _ in range(max_turns):
            resp = client.chat.completions.create(
                model=model, messages=messages, tools=tools, temperature=0
            )
            msg = resp.choices[0].message
            out: dict[str, Any] = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                out["tool_calls"] = [_tool_call_dict(tc) for tc in msg.tool_calls]
            messages.append(out)
            if not msg.tool_calls:
                if not (msg.content or "").strip() and not nudged:
                    # some reasoning models end a turn with an empty reply after a tool
                    # result; ask once for the answer instead of returning nothing
                    nudged = True
                    messages.append({"role": "user", "content": NUDGE})
                    continue
                return msg.content or ""
            for tc in msg.tool_calls:
                content = _run_tool(tc.function.name, tc.function.arguments)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
                if tc.function.name == "final_answer" and not content.startswith("ERROR"):
                    args = json.loads(tc.function.arguments or "{}")
                    return str(args.get("answer", ""))
        return "Sorry, I could not answer."
    finally:
        _active.reset(tok_f)
        _db.reset(tok_db)


# --- outcome oracles ----------------------------------------------------------------


def final_answer_text(run: crashlab.Run) -> str | None:
    calls = run.tool_calls("final_answer", ok_only=True)
    if calls:
        return str(calls[-1].args.get("answer", ""))
    return None


def _numbers(text: str) -> list[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", text)]


def answer_ok(answer: str, truth: Any) -> bool:
    if isinstance(truth, (int, float)):
        return any(abs(n - float(truth)) < 0.01 for n in _numbers(answer))
    return str(truth).lower() in answer.lower()


@crashlab.check("sql-analyst")
def answer_matches_ground_truth(run: crashlab.Run) -> None:
    """The final answer contains the reference query's result (numbers to 0.01)."""
    question = str(run.input.get("question", ""))
    truth = reference_answer(question)
    if truth is None:
        return  # not one of the graded questions
    answer = final_answer_text(run)
    if answer is None:
        answer = run.output if isinstance(run.output, str) else ""
        last = run.chat_steps()[-1].idx if run.chat_steps() else 0
        assert answer and answer_ok(answer, truth), (
            f"step {last}: final_answer was never called; expected {truth!r}"
        )
        return
    step = run.tool_calls("final_answer", ok_only=True)[-1].idx
    assert answer_ok(answer, truth), f"step {step}: answered {answer!r}, expected {truth!r}"


@crashlab.check("sql-analyst")
def within_step_budget(run: crashlab.Run) -> None:
    """The run took at most STEP_BUDGET steps."""
    assert len(run.steps) <= STEP_BUDGET, f"{len(run.steps)} steps, budget is {STEP_BUDGET}"


# --- ground truth for the benchmark ------------------------------------------------


def ground_truth(run: crashlab.Run, fault: str) -> tuple[int | None, tuple[int, int] | None]:
    """``(culprit step, acceptable window)`` for an injected ``fault`` in ``run``: the
    step where the fault first does damage, and the window from where the fault enters
    to the first agent action that acts on it."""
    queries = run.tool_steps("run_query")
    answer = run.tool_steps("final_answer")
    end = answer[-1].idx if answer else (run.steps[-1].idx if run.steps else 0)
    if fault == "tool_5xx":
        bad = [s for s in queries if "502 Bad Gateway" in str(s.result)]
        if not bad:
            return None, None
        return bad[0].idx, (bad[0].idx, bad[0].idx + 1)
    if fault == "schema_drift":
        errs = [s for s in queries if s.error is not None or "no such column" in str(s.result)]
        first = errs[0] if errs else (queries[0] if queries else None)
        return (first.idx, (first.idx, end)) if first else (None, None)
    if fault == "misleading_desc":
        scaled = [s for s in queries if re.search(r"/\s*100", str(s.args.get("sql", "")))]
        first = scaled[0] if scaled else (queries[-1] if queries else None)
        start = queries[0].idx if queries else 0
        return (first.idx, (start, end)) if first else (None, None)
    if fault == "spec_drop":
        if not queries:
            return None, None
        culprit = next(
            (s for s in queries if "paid" not in str(s.args.get("sql", "")).lower()), queries[-1]
        )
        return culprit.idx, (0, end)
    return None, None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="sql-analyst crashlab demo")
    parser.add_argument("question", nargs="?", default=next(iter(QUESTIONS)))
    for fault in FAULTS:
        parser.add_argument("--fault-" + fault.replace("_", "-"), action="store_true")
    parser.add_argument("--model", default=None)
    ns = parser.parse_args(argv)
    faults = [f for f in FAULTS if getattr(ns, "fault_" + f) or f in faults_from_env()]
    model = ns.model or os.environ.get("MODEL") or "gpt-4o-mini"
    print(run_agent(ns.question, model=model, faults=faults))
    run = crashlab.load("last")
    print(f"[outcome: {run.outcome}" + (f" - {run.outcome_reason}]" if run.outcome_reason else "]"))


if __name__ == "__main__":
    main()
