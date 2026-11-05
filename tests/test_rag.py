"""RAG testing: crashlab.rag oracle helpers + examples/rag_qa. Offline (fake model, in-memory
retriever); covers wrong retrieval, missing context, stale documents, citation mismatch and
an answer that ignores the retrieved evidence."""

from __future__ import annotations

import pytest
from examples.rag_qa import agent as rag_qa

import crashlab
from crashlab import rag
from crashlab.export import export_test
from tests.fakes import FakeOpenAI, text_response, tool_call_response
from tests.test_export import _pytest

Q = "How many days do I have to return an item?"


def _run(answer, faults=(), query="return policy days"):
    pytest.importorskip("openai")
    script = [tool_call_response(1, "search_docs", {"query": query}), text_response(2, answer)]
    rag_qa.run_agent(Q, client=crashlab.wrap(FakeOpenAI(script)), model="fake-model",
                     faults=list(faults))  # fmt: skip
    return crashlab.load("last")


def _status(run):
    return {c.name: c.status for c in run.checks}


def _msg(run, name):
    return next(c.message for c in run.checks if c.name == name)


GOOD = "You have 30 days from delivery to return an item [refund-policy-v2]."


def test_clean_rag_run_passes_and_replays_offline():
    run = _run(GOOD)
    assert run.outcome == "pass", run.checks
    assert set(_status(run).values()) == {"pass"} and len(run.checks) == 5
    assert [d["id"] for _, d in rag.retrieved(run)][0] == "refund-policy-v2"
    assert crashlab.replay(run.run_id).outcome == "pass"


def test_wrong_retrieval_fails_naming_the_retrieval_step():
    run = _run("Sorry, I could not find that in the documents.", faults=["wrong_retrieval"])
    assert run.outcome == "fail"
    assert _status(run)["retrieved_gold_doc"] == "fail"
    m = _msg(run, "retrieved_gold_doc")
    assert "step 1" in m and "refund-policy-v2" in m and "shipping-eu" in m


def test_missing_context_answer_must_abstain():
    run = _run(GOOD, faults=["missing_context"])
    assert _status(run)["abstains_without_context"] == "fail"
    assert "step 1" in _msg(run, "abstains_without_context")
    # citing a doc that was never retrieved is caught too
    assert _status(run)["citations_were_retrieved"] == "fail"
    # the built-in runtime check already flags the empty retrieval (info)
    assert any(f.kind == "suspicious_tool_result" for f in run.findings)
    ok = _run("I don't know; no documents were found.", faults=["missing_context"])
    assert _status(ok)["abstains_without_context"] == "pass"


def test_stale_document_used_fails():
    run = _run("You have 14 days [refund-policy-v1].", faults=["stale_docs"])
    assert _status(run)["no_stale_docs"] == "fail"
    m = _msg(run, "no_stale_docs")
    assert "refund-policy-v1" in m and "refund-policy-v2" in m and "step 1" in m


def test_citation_mismatch_fails():
    run = _run("You have 30 days to return an item [refund-policy-v1].")
    assert _status(run)["citations_were_retrieved"] == "fail"
    assert "refund-policy-v1" in _msg(run, "citations_were_retrieved")
    uncited = _run("You have 30 days to return an item.")
    assert _status(uncited)["citations_were_retrieved"] == "fail"
    assert "no citation" in _msg(uncited, "citations_were_retrieved")


def test_unsupported_answer_ignoring_evidence_fails():
    run = _run("You have 60 days to return an item [refund-policy-v2].")
    assert _status(run)["answer_grounded"] == "fail"
    m = _msg(run, "answer_grounded")
    assert "60" in m and "step 2" in m and "step 1" in m


def test_failing_rag_run_replays_and_exports(tmp_path):
    run = _run("You have 60 days to return an item [refund-policy-v2].")
    assert crashlab.replay(run.run_id).outcome == "fail"
    out = tmp_path / "exported"
    paths = export_test(run.run_id, out_dir=out)
    src = next(p for p in paths if p.suffix == ".py").read_text()
    assert '"answer_grounded": "fail"' in src
    res = _pytest(out)
    assert res.returncode == 0, res.stdout + res.stderr


# --- helpers on their own -------------------------------------------------------------


def test_cited_ids_and_numbers():
    assert rag.cited_ids("a [x-1] b [y.2][x-1] [not an id]") == ["x-1", "y.2"]
    assert rag.cited_ids("see (doc-3)", pattern=r"\((doc-\d+)\)") == ["doc-3"]


def test_retrieved_accepts_wrapped_results_and_custom_keys():
    @crashlab.tool()
    def lookup(q: str) -> dict:
        return {"results": [{"doc_id": "a", "content": "Price is 12 EUR."}]}

    @crashlab.agent(name="mini-rag")
    def go() -> str:
        lookup("x")
        return "It costs 12 EUR [a]."

    go()
    run = crashlab.load("last")
    docs = rag.retrieved(run, tool="lookup", id_key="doc_id")
    assert [d["doc_id"] for _, d in docs] == ["a"]
    kw = dict(tool="lookup", id_key="doc_id", text_key="content")
    assert rag.answer_grounded(run, run.output, **kw) is None
    assert rag.citations_were_retrieved(run, run.output, tool="lookup", id_key="doc_id") is None
    assert "13" in rag.answer_grounded(run, "It costs 13 EUR [a].", **kw)
    assert rag.retrieved_ids(run, ["a"], tool="lookup", id_key="doc_id") is None
