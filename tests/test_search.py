"""`crashlab search` / crashlab.search.search_runs over hand-built fixture runs."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

import pytest
from typer.testing import CliRunner

from crashlab._runscan import parse_since
from crashlab.cli import app
from crashlab.models import CheckResult, ErrorInfo, Finding
from crashlab.search import FIELDS, search_runs
from crashlab.store import Store
from tests.debug_fixtures import chat, make_run, run_id, tool

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def invoke(args, env=None):
    res = runner.invoke(app, args, env=env or ENV)
    return res.exit_code, _ANSI.sub("", res.output)


ENV = {"COLUMNS": "200", "TERM": "xterm-256color", "NO_COLOR": "1"}


@pytest.fixture
def store(crashlab_dir):
    s = Store()
    make_run(s, 1)  # clean pass
    make_run(  # tool error + failing check + diagnosis
        s, 2, status="error", outcome="fail",
        error=ErrorInfo(type="RuntimeError", message="refund service unavailable"),
        checks=[CheckResult(name="refund_within_policy", status="fail", message="refunded 999")],
        diagnosis_label="TOOL_ERROR_MISHANDLED", diagnosis_step=3,
        steps=[
            chat("x", 0, "Refund order B22", tool_call=("issue_refund", {"order_id": "B22"})),
            tool("x", 1, "issue_refund", {"order_id": "B22", "amount": 999},
                 error=ErrorInfo(type="TimeoutError", message="gateway timed out after 30s",
                                 is_tool_error=True)),
        ],
        user="Refund order B22",
    )  # fmt: skip
    make_run(  # other agent, other model, a finding
        s, 3, name="sql-analyst",
        findings=[Finding(kind="step_repetition", severity="error", step=1, steps=[1],
                          message="run_sql called 4 times with the same args")],
        steps=[chat("x", 0, "How many users signed up?", "42 users", model="claude-3-5-haiku")],
        user="How many users signed up?",
    )  # fmt: skip
    return s


def ids(result):
    return [m.run_id for m in result.matches]


def test_text_hits_errors_messages_tools_with_steps(store):
    r = search_runs("timed out", store=store)
    assert ids(r) == [run_id(2)]
    hit = r.matches[0].hits[0]
    assert (hit.field, hit.step) == ("step.error", 1)
    assert "TimeoutError: gateway timed out after 30s" in hit.snippet

    r = search_runs("b22", store=store)  # case-insensitive, user message + tool args
    fields = {(h.field, h.step) for h in r.matches[0].hits}
    assert ("message.user", 0) in fields and ("tool.args", 1) in fields
    assert ids(r) == [run_id(2)]


def test_repeated_conversation_history_is_reported_once(store):
    # run 1's second chat request repeats the user message: one hit, at step 0
    r = search_runs("Refund my order A17", store=store, fields=["message"])
    assert [(h.field, h.step) for h in r.matches[0].hits] == [("message.user", 0)]
    r = search_runs("Never refund twice", store=store, run_prefix=run_id(1))
    assert [(h.field, h.step) for h in r.matches[0].hits] == [("message.system", 0)]


def test_searches_checks_findings_diagnosis_models_and_run_ids(store):
    assert ids(search_runs("refund_within_policy", store=store)) == [run_id(2)]
    assert ids(search_runs("TOOL_ERROR_MISHANDLED", store=store)) == [run_id(2)]
    assert ids(search_runs("step_repetition", store=store)) == [run_id(3)]
    assert ids(search_runs("haiku", store=store)) == [run_id(3)]
    r = search_runs(run_id(3)[-6:], store=store, fields=["run"])
    assert ids(r) == [run_id(3)] and r.matches[0].hits[0].field == "run.id"


def test_filters_are_anded_and_work_without_a_query(store):
    assert ids(search_runs(store=store)) == [run_id(3), run_id(2), run_id(1)]
    assert ids(search_runs(store=store, errors=True)) == [run_id(2)]
    assert ids(search_runs(store=store, tool="issue_refund")) == [run_id(2)]
    assert ids(search_runs(store=store, tool="issue")) == []  # tool names are exact
    assert ids(search_runs(store=store, model="HAIKU")) == [run_id(3)]
    assert ids(search_runs(store=store, check="refund_within_policy")) == [run_id(2)]
    assert ids(search_runs(store=store, label="tool_error_mishandled")) == [run_id(2)]
    assert ids(search_runs(store=store, name="sql-analyst")) == [run_id(3)]
    assert ids(search_runs(store=store, outcome="fail")) == [run_id(2)]
    assert ids(search_runs(store=store, status="ok")) == [run_id(3), run_id(1)]
    assert ids(search_runs("refund", store=store, errors=True, tool="get_order")) == []
    assert all(not m.hits for m in search_runs(store=store).matches)


def test_passing_check_does_not_match_check_filter(crashlab_dir):
    s = Store()
    make_run(s, 1, checks=[CheckResult(name="c", status="pass")])
    make_run(s, 2, checks=[CheckResult(name="c", status="error", message="boom")])
    assert ids(search_runs(store=s, check="c")) == [run_id(2)]


def test_since_and_run_prefix(crashlab_dir):
    s = Store()
    make_run(s, 1, hours_ago=48)
    make_run(s, 2, hours_ago=1)
    assert ids(search_runs(store=s, since=parse_since("2h"))) == [run_id(2)]
    assert ids(search_runs(store=s, since=parse_since("3d"))) == [run_id(2), run_id(1)]
    assert ids(search_runs(store=s, run_prefix=run_id(1).lower())) == [run_id(1)]


def test_parse_since_formats():
    now = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)
    assert parse_since("30m", now=now) == now - timedelta(minutes=30)
    assert parse_since("1w", now=now) == now - timedelta(weeks=1)
    assert parse_since("2026-11-12T00:00:00Z") == datetime(2026, 11, 12, tzinfo=timezone.utc)
    assert parse_since("2026-11-12").tzinfo is not None  # local midnight, made aware
    with pytest.raises(ValueError, match="--since expects"):
        parse_since("yesterday")


def test_regex_case_sensitivity_limits_and_bad_input(store):
    assert ids(search_runs(r"gateway .* \d+s", store=store, regex=True)) == [run_id(2)]
    assert ids(search_runs("timeouterror", store=store, case_sensitive=True)) == []
    r = search_runs(store=store, limit=2)
    assert len(r.matches) == 2 and r.truncated
    assert not search_runs(store=store, limit=3).truncated
    r = search_runs("refund", store=store, max_hits_per_run=1)
    assert all(len(m.hits) == 1 for m in r.matches)
    assert any(m.hits_truncated > 0 for m in r.matches)
    with pytest.raises(ValueError, match="bad regular expression"):
        search_runs("(", store=store, regex=True)
    with pytest.raises(ValueError, match="unknown search field"):
        search_runs("x", store=store, fields=["bogus"])
    assert "diagnosis" in FIELDS


def test_unreadable_run_is_skipped_not_fatal(store, monkeypatch):
    real = store.load_run

    def load(ref):
        if ref == run_id(1):
            raise OSError("disk gone")
        return real(ref)

    monkeypatch.setattr(store, "load_run", load)
    r = search_runs(store=store)
    assert run_id(1) in r.unreadable and ids(r) == [run_id(3), run_id(2)]


def test_empty_store(crashlab_dir):
    r = search_runs("x", store=Store())
    assert r.matches == [] and r.scanned == 0


def test_cli_table_json_and_exit_codes(store):
    code, out = invoke(["search", "timed out"])
    assert code == 0, out
    assert run_id(2)[:10] in out and "step.error" in out
    assert "gateway timed out after 30s" in out and "1 run(s), 1 hit(s)" in out
    # fixture ids share 25 characters: the shown id must still resolve to one run
    assert run_id(2) in out

    code, out = invoke(["search", "--errors", "--json"])
    assert code == 0
    data = json.loads(out)
    assert [m["run_id"] for m in data["matches"]] == [run_id(2)] and data["query"] is None

    code, out = invoke(["search", "no-such-text"])
    assert code == 1 and "No runs matching 'no-such-text'" in out
    code, out = invoke(["search", "no-such-text", "--json"])
    assert code == 1 and json.loads(out)["matches"] == []

    code, out = invoke(["search", "x", "--since", "soon"])
    assert code == 2
    code, out = invoke(["search", "(", "-E"])
    assert code == 2 and "bad regular expression" in out
    code, out = invoke(["search", "--limit", "1"])
    assert "stopped at --limit" in out


def test_cli_recorded_text_is_not_markup(crashlab_dir):
    s = Store()
    make_run(s, 1, user="please [bold]refund[/] me")
    code, out = invoke(["search", "refund", "--in", "message"])
    assert code == 0 and "[bold]refund[/]" in out


def test_cli_narrow_terminal_fits(store):
    code, out = invoke(["search", "refund"], env={**ENV, "COLUMNS": "80"})
    assert code == 0
    assert max(len(line) for line in out.splitlines()) <= 80
