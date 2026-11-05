"""One CI-safe pass over the RAG failure modes through record -> check (CLI, fresh process)
-> replay -> export (pytest, fresh process). Offline, scripted fake model."""

from __future__ import annotations

import json

import pytest

import crashlab
from crashlab.export import export_test
from tests.test_export import _pytest
from tests.test_rag import GOOD, _cli, _run, _status

CASES = [
    # (label, answer, faults, checks that must fail)
    ("clean", GOOD, [], set()),
    ("wrong_retrieval", "Sorry, I could not find that.", ["wrong_retrieval"],
     {"retrieved_gold_doc"}),
    ("missing_context", GOOD, ["missing_context"],
     {"retrieved_gold_doc", "abstains_without_context", "citations_were_retrieved",
      "answer_grounded", "grounded_in_current_docs"}),
    ("stale", "You have 14 days [refund-policy-v1].", ["stale_docs"],
     {"retrieved_gold_doc", "no_stale_docs", "grounded_in_current_docs"}),
    ("citation_mismatch", "You have 30 days to return an item [refund-policy-v1].", [],
     {"citations_were_retrieved"}),
    ("unsupported_number", "You have 60 days to return an item [refund-policy-v2].", [],
     {"answer_grounded", "grounded_in_current_docs"}),
    ("unsupported_claim", "Refunds are paid only as store credit vouchers [refund-policy-v2].",
     [], {"claims_supported"}),
    ("retrieval_timeout", GOOD, ["retrieval_timeout"],
     {"retrieved_gold_doc", "abstains_without_context", "citations_were_retrieved",
      "answer_grounded", "grounded_in_current_docs"}),
    ("retrieval_error", "I don't know.", ["retrieval_error"], {"retrieved_gold_doc"}),
]  # fmt: skip


def test_rag_failures_record_check_replay_export(crashlab_dir, tmp_path):
    pytest.importorskip("openai")
    runs = {}
    for label, answer, faults, failing in CASES:
        run = _run(answer, faults=faults)
        got = {n for n, s in _status(run).items() if s != "pass"}
        assert got == failing, (label, got, run.checks)
        assert run.outcome == ("pass" if not failing else "fail"), label
        assert crashlab.replay(run.run_id).outcome == run.outcome, label
        runs[label] = run
    # crashlab check in a fresh process re-runs the oracles and agrees
    res = _cli("check", "--all", "--no-save", "--name", "rag-qa", "--json", store=crashlab_dir)
    rows = {r["run_id"]: r for r in json.loads(res.stdout)}
    for label, run in runs.items():
        st = {c["name"]: c["status"] for c in rows[run.run_id]["checks"]}
        assert st == _status(run), label
    # every run exports to a test that passes in a fresh pytest process
    out = tmp_path / "exported"
    for run in runs.values():
        export_test(run.run_id, out_dir=out)
    res = _pytest(out)
    assert res.returncode == 0, res.stdout + res.stderr
    assert f"{len(CASES)} passed" in res.stdout, res.stdout
