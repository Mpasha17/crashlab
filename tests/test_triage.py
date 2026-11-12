"""`crashlab triage` / crashlab.triage: cluster failed runs, diagnose each cluster's
representative offline (heuristic), print a ranked summary with next commands."""

from __future__ import annotations

import json
import re

from typer.testing import CliRunner

from crashlab.cli import app
from crashlab.models import CheckResult, Diagnosis, ErrorInfo, Evidence
from crashlab.store import Store
from crashlab.triage import triage_runs
from tests.debug_fixtures import chat, make_run, run_id, tool

runner = CliRunner()
ENV = {"COLUMNS": "200", "TERM": "xterm-256color", "NO_COLOR": "1"}
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def invoke(args):
    res = runner.invoke(app, args, env=ENV)
    return res.exit_code, _ANSI.sub("", res.output)


def _timeout(s, n, order):
    rid = run_id(n)
    steps = [
        chat(rid, 0, f"Refund {order}", tool_call=("get_order", {"order_id": order})),
        tool(rid, 1, "get_order", {"order_id": order},
             error=ErrorInfo(type="TimeoutError", message=f"order {order} timed out after 3.5s",
                             is_tool_error=True)),
    ]  # fmt: skip
    return make_run(
        s, n, status="error", outcome="fail", steps=steps,
        error=ErrorInfo(type="TimeoutError", message=f"order {order} timed out after 3.5s"),
    )  # fmt: skip


def _overrefund(s, n, **kw):
    return make_run(s, n, outcome="fail", checks=[CheckResult(
        name="refund_within_policy", status="fail",
        message="refunded 100 > order total 49")], **kw)  # fmt: skip


def _store():
    s = Store()
    for n, order in ((1, "ORD-1017"), (2, "ORD-99210"), (3, "ORD-3003")):
        _timeout(s, n, order)
    _overrefund(s, 4)
    _overrefund(s, 5)
    make_run(s, 6)  # passes: not triaged
    return s


def test_triage_clusters_and_diagnoses_representatives(crashlab_dir):
    s = _store()
    r = triage_runs(store=s)
    assert (r.scanned, r.failed, len(r.items)) == (6, 5, 2)
    a, b = r.items
    assert (a.size, a.representative) == (3, run_id(3))
    assert (b.size, b.representative) == (2, run_id(5))
    for it in r.items:
        assert it.diagnoser == "heuristic" and it.diagnosis_source == "new"
        assert it.label and it.label != "UNKNOWN"
        assert it.commands == [
            f"crashlab show {it.representative} --failing",
            f"crashlab fix {it.representative}",
            f"crashlab export-test {it.representative}",
        ]
    assert a.failing_step is not None and a.step_description
    assert a.evidence and a.evidence.startswith(f"step {a.failing_step} ")
    # saved with the representative, then reused (not re-diagnosed) next time
    saved = s.load_run(run_id(3)).diagnosis
    assert saved is not None and saved.method.get("backend") == "heuristic"
    again = triage_runs(store=s)
    assert [i.diagnosis_source for i in again.items] == ["stored", "stored"]
    assert s.load_run(run_id(3)).diagnosis.created_at == saved.created_at
    # other runs of the cluster are not diagnosed
    assert s.load_run(run_id(1)).diagnosis is None


def test_stored_llm_diagnosis_reused_unless_rediagnose(crashlab_dir):
    s = _store()
    run = s.load_run(run_id(5))
    ev = [Evidence(step_idx=2, field="args", quote='"amount": 100')]
    llm = Diagnosis(run_id=run.run_id, label="TASK_SPEC_VIOLATION", failing_step=2,
                    root_cause="refund above the order total",
                    model="gpt-x", evidence=ev)  # fmt: skip
    s.save_diagnosis(run, llm)
    item = [i for i in triage_runs(store=s).items if i.representative == run_id(5)][0]
    assert (item.diagnosis_source, item.diagnoser) == ("stored", "gpt-x")
    assert item.root_cause == "refund above the order total"
    assert item.evidence == 'step 2 args: "amount": 100'
    item = [i for i in triage_runs(store=s, rediagnose=True).items
            if i.representative == run_id(5)][0]  # fmt: skip
    assert (item.diagnosis_source, item.diagnoser) == ("new", "heuristic")
    assert s.load_run(run_id(5)).diagnosis.model == "heuristic"


def test_stored_labels_do_not_split_clusters(crashlab_dir):
    """Diagnosing a representative must not split its cluster on the next triage."""
    s = Store()
    _overrefund(s, 1, diagnosis_label="BAD_TOOL_ARGUMENTS")
    _overrefund(s, 2)
    r = triage_runs(store=s)
    assert [i.size for i in r.items] == [2]
    assert "label" not in r.by


def test_no_save_limit_name_and_agent(crashlab_dir):
    s = _store()
    rid = run_id(7)
    steps = [chat(rid, 0, "hi", "Refunded 100.")]
    steps[0].attributes["gen_ai.agent.name"] = "refunds"
    make_run(s, 7, name="desk", outcome="fail", steps=steps, checks=[CheckResult(
        name="answer_ok", status="fail", message="Refunded 100.")])  # fmt: skip
    r = triage_runs(store=s, save=False, limit=1)
    assert len(r.items) == 1 and r.clusters_total == 3
    assert s.load_run(r.items[0].representative).diagnosis is None
    d = triage_runs(store=s, name="desk", save=False)
    assert [i.representative for i in d.items] == [rid]
    assert d.items[0].agent == "refunds"


def test_cli_triage_table_json_and_empty(crashlab_dir):
    _store()
    code, out = invoke(["triage"])
    assert code == 0, out
    assert "heuristic" in out
    assert "crashlab show" in out and "--failing" in out and "crashlab export-test" in out
    assert "TimeoutError" in out and "refund_within_policy" in out
    code, out = invoke(["triage", "--json"])
    data = json.loads(out)
    assert data["failed"] == 5 and len(data["items"]) == 2
    assert {"size", "signature", "label", "failing_step", "agent", "evidence", "diagnoser",
            "diagnosis_source", "commands", "run_ids"} <= set(data["items"][0])  # fmt: skip
    code, out = invoke(["triage", "--since", "1h"])  # fixture runs are days old
    assert code == 0 and "No failed runs" in out
    code, out = invoke(["triage", "--name", "nobody", "--json"])
    assert json.loads(out)["items"] == []
