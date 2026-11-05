"""Same-id document versions, duplicate ids with different text, LangChain-style documents,
and exported RAG tests keeping their helper configuration. Offline."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

import crashlab
from crashlab import rag
from crashlab.export import export_test
from tests.fakes import FakeOpenAI, text_response, tool_call_response
from tests.test_export import _pytest
from tests.test_rag import _shape_run

OLD = {"id": "policy", "text": "Refunds within 14 days of delivery.", "updated": "2024-01-10"}
NEW = {"id": "policy", "text": "Refunds within 30 days of delivery.", "updated": "2026-09-01"}


# --- (1) stale + current versions under one id ------------------------------------------


@pytest.mark.parametrize("order", [[OLD, NEW], [NEW, OLD]])
def test_versions_default_fails_clearly_instead_of_guessing(order):
    run = _shape_run(order)
    for answer in ("Refunds within 30 days [policy].", "Refunds within 14 days [policy]."):
        msg = rag.answer_grounded(run, answer)
        assert msg and "policy" in msg and "2 versions" in msg and "duplicates=" in msg
    # uncited: same safe failure (ambiguous evidence)
    assert "2 versions" in rag.answer_grounded(run, "Refunds within 30 days.")


@pytest.mark.parametrize("order", [[OLD, NEW], [NEW, OLD]])
def test_versions_latest_uses_current_version(order):
    run = _shape_run(order)
    assert rag.answer_grounded(run, "Refunds within 30 days [policy].", duplicates="latest") is None
    msg = rag.answer_grounded(run, "Refunds within 14 days [policy].", duplicates="latest")
    assert msg and "14" in msg
    msg = rag.claims_supported(run, "Refunds within 14 days [policy].", duplicates="latest")
    assert msg is None  # words match; the number is answer_grounded's job


def test_versions_first_and_any():
    run = _shape_run([OLD, NEW])
    assert rag.answer_grounded(run, "Refunds within 14 days [policy].", duplicates="first") is None
    assert "30" in rag.answer_grounded(run, "Refunds within 30 days [policy].", duplicates="first")
    for n in (14, 30):  # 'any': either version grounds the answer (documented, permissive)
        assert (
            rag.answer_grounded(run, f"Refunds within {n} days [policy].", duplicates="any") is None
        )


def test_superseded_flag_marks_the_old_version_for_latest():
    old = {**OLD, "superseded_by": "policy"}
    old.pop("updated")
    new = {**NEW}
    new.pop("updated")
    run = _shape_run([new, old])
    assert "14" in rag.answer_grounded(run, "Refunds within 14 days [policy].", duplicates="latest")
    assert rag.no_stale_docs(run)  # still reported as stale


def test_version_fields_under_metadata_and_numeric_versions():
    run = _shape_run([
        {"id": "p", "text": "Fee is 5 EUR.", "metadata": {"version": 9}},
        {"id": "p", "text": "Fee is 7 EUR.", "metadata": {"version": 10}},  # 10 > 9 numerically
    ])  # fmt: skip
    assert rag.answer_grounded(run, "Fee is 7 EUR [p].", duplicates="latest") is None
    assert "5" in rag.answer_grounded(run, "Fee is 5 EUR [p].", duplicates="latest")


def test_bad_duplicates_value_is_rejected():
    run = _shape_run([OLD])
    with pytest.raises(ValueError, match="duplicates"):
        rag.answer_grounded(run, "x [policy]", duplicates="newest")


# --- (2) duplicate ids with different text, no version info ---------------------------


def test_chunks_of_one_source_merge_by_default():
    # same id, different text, no version fields: chunks of one document, not versions
    run = _shape_run([
        {"id": "faq.md", "text": "Refunds within 30 days of delivery."},
        {"id": "faq.md", "text": "Shipping to the EU takes 5-7 business days."},
    ])  # fmt: skip
    assert rag.answer_grounded(run, "Refunds take 30 days; EU shipping 5-7 days [faq.md].") is None
    assert "60" in rag.answer_grounded(run, "Refunds take 60 days [faq.md].")


def test_duplicates_fail_mode_flags_any_differing_text():
    run = _shape_run([
        {"id": "faq.md", "text": "Refunds within 30 days."},
        {"id": "faq.md", "text": "Refunds within 14 days."},
    ])  # fmt: skip
    msg = rag.answer_grounded(run, "Refunds within 30 days [faq.md].", duplicates="strict")
    assert msg and "faq.md" in msg and "2 different texts" in msg
    same = _shape_run([{"id": "a", "text": "5 EUR."}, {"id": "a", "text": "5 EUR."}])
    assert rag.answer_grounded(same, "5 EUR [a].", duplicates="strict") is None


def test_unrelated_duplicate_ids_do_not_affect_other_citations():
    run = _shape_run([OLD, NEW, {"id": "fee", "text": "Fee is 5 EUR."}])
    assert rag.answer_grounded(run, "Fee is 5 EUR [fee].") is None
    assert rag.claims_supported(run, "Shipping is free everywhere today [fee].")


# --- (4) LangChain-style documents ----------------------------------------------------


class Document(BaseModel):  # same fields as langchain_core.documents.Document
    page_content: str
    metadata: dict = {}
    id: str | None = None


def _lc_run(doc_cls):
    @crashlab.tool()
    def retrieve(query: str) -> list:
        return [
            doc_cls(page_content="Refunds within 30 days of delivery.",
                    metadata={"source": "refund.md", "updated": "2026-09-01"}),
            doc_cls(page_content="Refunds within 14 days of delivery.",
                    metadata={"source": "refund.md", "updated": "2024-01-10"}),
            doc_cls(page_content="EU shipping takes 5-7 business days.",
                    metadata={"source": "shipping.md"}),
        ]  # fmt: skip

    @crashlab.agent(name="lc-rag")
    def go() -> str:
        retrieve("refund days")
        return "ok"

    go()
    return crashlab.load("last")


@pytest.mark.parametrize("which", ["standin", "langchain_core"])
def test_langchain_documents(which):
    if which == "langchain_core":
        doc_cls = pytest.importorskip("langchain_core.documents").Document
    else:
        doc_cls = Document
    run = _lc_run(doc_cls)
    kw = dict(tool="retrieve", duplicates="latest")
    ids = [d["id"] for _, d in rag.retrieved(run, "retrieve")]
    assert ids == ["refund.md", "refund.md", "shipping.md"]
    assert rag.answer_grounded(run, "Refunds within 30 days [refund.md].", **kw) is None
    assert "14" in rag.answer_grounded(run, "Refunds within 14 days [refund.md].", **kw)
    assert "2 versions" in rag.answer_grounded(run, "30 days [refund.md].", tool="retrieve")
    assert rag.citations_were_retrieved(run, "5-7 days [shipping.md].", tool="retrieve") is None


# --- (3) exported tests keep the helper configuration --------------------------------


def test_exported_langchain_example_keeps_config_in_fresh_process(tmp_path):
    pytest.importorskip("openai")
    from examples.rag_qa import langchain_style as lc

    def go(answer):
        script = [tool_call_response(1, "retrieve", {"query": "refund days"}),
                  text_response(2, answer)]  # fmt: skip
        lc.run_agent(lc.QUESTION, client=crashlab.wrap(FakeOpenAI(script)), model="fake")
        return crashlab.load("last")

    good = go("You can get a refund within 30 days of delivery [refund.md].")
    assert good.outcome == "pass", good.checks
    bad = go("You can get a refund within 14 days of delivery [refund.md].")
    st = {c.name: c.status for c in bad.checks}
    assert st["grounded_in_latest_version"] == "fail"
    for run in (good, bad):
        out = tmp_path / run.run_id
        export_test(run.run_id, out_dir=out)
        res = _pytest(out)
        assert res.returncode == 0 and "passed" in res.stdout, res.stdout + res.stderr
        assert crashlab.replay(run.run_id).outcome == run.outcome


def test_lenticular_bracket_citations_from_live_gpt_oss():
    # Groq gpt-oss-120b cited as 【refund.md】 in 2 of 3 live runs (6 Nov 2026)
    assert rag.cited_ids("30 days 【refund.md】. Also 【a, b】") == ["refund.md", "a", "b"]
    run = _shape_run([{"id": "refund.md", "text": "Refunds within 30 days of delivery."}])
    answer = (
        "You have up to **30\u202fdays** from the date of delivery "
        "to request a refund. 【refund.md】"
    )
    assert rag.citations_were_retrieved(run, answer) is None
    assert rag.answer_grounded(run, answer) is None
    assert "'x'" in rag.citations_were_retrieved(run, "30 days 【x】")


def test_gpt_oss_line_anchor_citations():
    # live run: '【refund.md†L1】' (source + line anchor)
    assert rag.cited_ids("x 【refund.md†L1】 y 【a†L2-L4】") == ["refund.md", "a"]
