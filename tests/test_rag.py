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
from tests.test_export import ROOT, _pytest

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
    assert set(_status(run).values()) == {"pass"} and len(run.checks) == 7
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


# --- `crashlab check` runs the RAG oracles in a fresh process --------------------------


def _cli(*args, store):
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if not k.endswith("API_KEY")}
    env["CRASHLAB_DIR"] = str(store)
    return subprocess.run([sys.executable, "-m", "crashlab", *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=120)  # fmt: skip


def test_cli_check_imports_entrypoint_and_runs_rag_oracles(crashlab_dir):
    import json

    run = _run("You have 60 days to return an item [refund-policy-v2].")
    res = _cli("check", run.run_id, "--json", "--no-save", store=crashlab_dir)
    assert res.returncode != 0 or '"fail"' in res.stdout, res.stderr
    out = json.loads(res.stdout)
    assert out["outcome"] == "fail"
    assert {c["name"]: c["status"] for c in out["checks"]}["answer_grounded"] == "fail"


def test_cli_check_all_save_grades_runs_recorded_without_oracles(crashlab_dir):
    import importlib
    import json

    from crashlab.verify.oracles import unregister

    unregister("rag-qa")  # record as if the checks did not exist yet
    try:
        _run("You have 14 days [refund-policy-v1].", faults=["stale_docs"])
        _run(GOOD)
    finally:
        importlib.reload(rag_qa)
    assert all(not r.checks for r in (crashlab.load(r) for r in ("last",)))
    res = _cli("check", "--all", "--save", "--name", "rag-qa", "--json", store=crashlab_dir)
    rows = json.loads(res.stdout)
    assert sorted(r["outcome"] for r in rows) == ["fail", "pass"], res.stdout + res.stderr
    stored = [crashlab.load(r["run_id"]) for r in rows]
    bad = next(r for r in stored if r.outcome == "fail")
    assert {c.name: c.status for c in bad.checks}["no_stale_docs"] == "fail"


# --- retriever return shapes and malformed documents ----------------------------------


def _shape_run(result, name="shape-rag"):
    @crashlab.tool()
    def search_docs(q: str):
        return result

    @crashlab.agent(name=name)
    def go() -> str:
        search_docs("q")
        return "ok"

    go()
    return crashlab.load("last")


@pytest.mark.parametrize(
    "result",
    [
        [{"id": "a", "text": "Fee is 5 EUR."}],
        {"documents": [{"id": "a", "text": "Fee is 5 EUR."}]},
        {"hits": [{"id": "a", "text": "Fee is 5 EUR."}]},
        {"matches": [{"id": "a", "text": "Fee is 5 EUR."}], "total": 1},
        '[{"id": "a", "text": "Fee is 5 EUR."}]',  # JSON text, as many tools return
        '{"results": [{"id": "a", "text": "Fee is 5 EUR."}]}',
        [{"page_content": "Fee is 5 EUR.", "metadata": {"id": "a"}}],  # LangChain-like
        [{"content": "Fee is 5 EUR.", "metadata": {"source": "x"}, "id": "a"}],
    ],
)
def test_common_retriever_shapes(result):
    run = _shape_run(result)
    assert rag.retrieved_ids(run, ["a"]) is None
    assert rag.answer_grounded(run, "The fee is 5 EUR [a].") is None
    assert "6" in rag.answer_grounded(run, "The fee is 6 EUR [a].")
    assert rag.citations_were_retrieved(run, "The fee is 5 EUR [a].") is None


@pytest.mark.parametrize(
    "result",
    [
        None,
        3,
        "plain text, not JSON",
        "[1, 2",
        {"error": "boom"},
        [None, 1, "x", []],
        [{"id": None, "text": None}, {"text": 7}, {"id": ["weird"]}],
    ],  # fmt: skip
)
def test_malformed_results_do_not_crash(result):
    run = _shape_run(result)
    assert rag.retrieved_ids(run, ["a"]) is not None  # nothing usable retrieved
    for fn in (rag.answer_grounded, rag.citations_were_retrieved, rag.claims_supported,
               rag.abstains_without_context):  # fmt: skip
        fn(run, "It costs 5 EUR [a].")  # must not raise
    rag.no_stale_docs(run)


def test_plain_string_documents_are_text_without_ids():
    run = _shape_run(["Fee is 5 EUR.", "Open 9-5."])
    assert [d["text"] for _, d in rag.retrieved(run)] == ["Fee is 5 EUR.", "Open 9-5."]
    assert rag.answer_grounded(run, "The fee is 5 EUR.") is None


# --- word-overlap grounding (non-numeric claims) --------------------------------------


@pytest.fixture
def policy_run():
    return _shape_run([
        {"id": "refund", "text": "Refunds are accepted within 30 days of delivery for unused "
                                 "items. Opened electronics are not refunded; they are repaired."},
        {"id": "ship", "text": "Standard shipping to the EU takes 5-7 business days."},
    ])  # fmt: skip


@pytest.mark.parametrize(
    "answer",
    [
        "Refunds are accepted within 30 days of delivery [refund].",  # verbatim
        "Unused items can be refunded within 30 days after they are delivered [refund].",
        "You can get a refund for unused items within 30 days of delivery [refund].",
        "Opened electronics won't be refunded, they get repaired instead [refund].",
        "Shipping to the EU takes 5-7 business days [ship].",
        "I don't know.",  # abstention: nothing claimed
        "Yes [refund].",  # too short to judge
    ],
)
def test_claims_supported_accepts_paraphrases(policy_run, answer):
    assert rag.claims_supported(policy_run, answer) is None


@pytest.mark.parametrize(
    ("answer", "needle"),
    [
        ("Shipping is free for every order above fifty euros [ship].", "free"),
        ("Refunds are paid as store credit vouchers only [refund].", "voucher"),
        ("Opened electronics are refunded in full [refund].", "negat"),  # flips 'not refunded'
        ("Refunds are not accepted within 30 days of delivery [refund].", "negat"),
        # a supported claim cited to the wrong document
        ("Shipping to the EU takes 5-7 business days [refund].", "[refund]"),
    ],
)
def test_claims_supported_flags_unsupported_and_negated(policy_run, answer, needle):
    msg = rag.claims_supported(policy_run, answer)
    assert msg and needle in msg.lower() and "step" in msg


def test_claims_supported_threshold_is_configurable(policy_run):
    answer = "Refunds need the original packaging and receipt [refund]."
    assert rag.claims_supported(policy_run, answer) is not None
    assert rag.claims_supported(policy_run, answer, min_overlap=0.2) is None


def test_example_flags_non_numeric_unsupported_claim():
    run = _run("Refunds are paid only as store credit vouchers [refund-policy-v2].")
    st = _status(run)
    assert st["claims_supported"] == "fail" and st["answer_grounded"] == "pass"


# --- regressions from the live run (Groq gpt-oss-120b, 5-6 Nov 2026) ------------------


def test_agent_give_up_message_counts_as_abstention():
    # the model searched until max_turns; the agent's own fallback is an abstention
    run = _run("Sorry, I could not answer.", faults=["missing_context"])
    st = _status(run)
    assert st["abstains_without_context"] == "pass"
    assert st["citations_were_retrieved"] == "pass"


def test_live_paraphrase_of_stale_doc_is_not_an_unsupported_claim():
    run = _shape_run([{"id": "v1", "text": "Refunds are accepted within 14 days of delivery."}])
    answer = (
        "You have\u202f14\u202fdays from the date of delivery to return an item for a refund. [v1]"
    )
    assert rag.claims_supported(run, answer) is None
    assert rag.answer_grounded(run, answer) is None


@pytest.mark.parametrize(
    ("answer", "faults", "failing"),
    [
        ("Sorry, I could not find that.", ["wrong_retrieval"], "retrieved_gold_doc"),
        (GOOD, ["missing_context"], "abstains_without_context"),
        ("You have 14 days [refund-policy-v1].", ["stale_docs"], "no_stale_docs"),
        ("Refunds are paid only as store credit vouchers [refund-policy-v2].", [],
         "claims_supported"),
    ],
)  # fmt: skip
def test_exported_rag_failures_pass_in_fresh_process(tmp_path, answer, faults, failing):
    run = _run(answer, faults=faults)
    assert _status(run)[failing] == "fail"
    out = tmp_path / "exported"
    paths = export_test(run.run_id, out_dir=out)
    assert f'"{failing}": "fail"' in next(p for p in paths if p.suffix == ".py").read_text()
    res = _pytest(out)
    assert res.returncode == 0 and "passed" in res.stdout, res.stdout + res.stderr
