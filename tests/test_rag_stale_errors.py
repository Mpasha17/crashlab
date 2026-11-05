"""Distinct stale/current ids (v1/v2) with uncited answers, the opt-in exclude_stale
grounding, retrieval errors/timeouts, empty or malformed citations and citation id
variants (Unicode names, line anchors) - also through `crashlab check` and exported tests.
Offline."""

from __future__ import annotations

import json

import pytest

from crashlab import rag
from crashlab.export import export_test
from tests.test_export import _pytest
from tests.test_rag import _cli, _run, _shape_run, _status

V1 = {"id": "v1", "text": "Refunds within 14 days of delivery.", "superseded_by": "v2"}
V2 = {"id": "v2", "text": "Refunds within 30 days of delivery."}


# --- distinct versioned ids ----------------------------------------------------------


def test_uncited_stale_number_default_is_permissive_and_documented():
    run = _shape_run([V1, V2])
    assert rag.answer_grounded(run, "Refunds within 14 days.") is None  # unchanged default
    assert rag.no_stale_docs(run)  # but the stale retrieval itself is reported
    assert "no citation" in rag.citations_were_retrieved(run, "Refunds within 14 days.")


def test_exclude_stale_fails_stale_numbers_cited_or_not():
    run = _shape_run([V1, V2])
    for answer in ("Refunds within 14 days.", "Refunds within 14 days [v1]."):
        msg = rag.answer_grounded(run, answer, exclude_stale=True)
        assert msg and "14" in msg and "v1" in msg and "stale" in msg
    assert rag.answer_grounded(run, "Refunds within 30 days.", exclude_stale=True) is None
    assert rag.answer_grounded(run, "Refunds within 30 days [v2].", exclude_stale=True) is None


def test_exclude_stale_by_date_and_when_only_stale_docs_exist():
    old = {"id": "a", "text": "Fee is 5 EUR.", "updated": "2023-05-01"}
    new = {"id": "b", "text": "Fee is 7 EUR.", "updated": "2026-05-01"}
    run = _shape_run([old, new])
    kw = dict(exclude_stale=True, date_key="updated", not_before="2025-01-01")
    assert "5" in rag.answer_grounded(run, "Fee is 5 EUR.", **kw)
    assert rag.answer_grounded(run, "Fee is 7 EUR.", **kw) is None
    only_old = _shape_run([V1])
    msg = rag.answer_grounded(only_old, "Refunds within 14 days [v1].", exclude_stale=True)
    assert msg and "stale" in msg


def test_combined_checks_on_v1_v2():
    run = _shape_run([V1, V2])
    answer = "Refunds within 30 days [v2]."
    assert rag.citations_were_retrieved(run, answer) is None
    assert rag.answer_grounded(run, answer, exclude_stale=True) is None
    assert rag.no_stale_docs(run)  # retrieval still surfaced the stale doc: separate signal


# --- retrieval errors / timeouts --------------------------------------------------------


@pytest.mark.parametrize("fault", ["retrieval_timeout", "retrieval_error"])
def test_retrieval_failure_requires_abstention(fault):
    run = _run("You have 30 days [refund-policy-v2].", faults=[fault])
    assert run.status != "error" and run.output
    st = _status(run)
    assert st["abstains_without_context"] == "fail" and st["citations_were_retrieved"] == "fail"
    ok = _run("Sorry, I could not find that in the documents.", faults=[fault])
    st = _status(ok)
    assert st["abstains_without_context"] == "pass" and st["citations_were_retrieved"] == "pass"


def test_retrieval_timeout_through_cli_and_export(crashlab_dir, tmp_path):
    run = _run("You have 30 days [refund-policy-v2].", faults=["retrieval_timeout"])
    assert run.steps[1].error is not None  # the tool raised and was recorded
    res = _cli("check", run.run_id, "--json", "--no-save", store=crashlab_dir)
    out = json.loads(res.stdout)
    st = {c["name"]: c["status"] for c in out["checks"]}
    assert st["abstains_without_context"] == "fail"
    export_test(run.run_id, out_dir=tmp_path / "x")
    res = _pytest(tmp_path / "x")
    assert res.returncode == 0 and "passed" in res.stdout, res.stdout + res.stderr


# --- empty / malformed citations ------------------------------------------------------


@pytest.mark.parametrize("cite", ["[]", "[ ]", "【】", "[,]", "[;]", "【†L1】", "[[x]]", "[x"])
def test_empty_or_malformed_citations_count_as_none(cite):
    assert rag.cited_ids(f"30 days {cite}") in ([], ["x"])
    run = _shape_run([V2])
    msg = rag.citations_were_retrieved(run, f"Refunds within 30 days {cite}.")
    assert msg is None or "no citation" in msg or "'x'" in msg


def test_empty_citation_through_cli_and_export(crashlab_dir, tmp_path):
    run = _run("You have 30 days to return an item [].")
    assert _status(run)["citations_were_retrieved"] == "fail"
    res = _cli("check", run.run_id, "--json", "--no-save", store=crashlab_dir)
    st = {c["name"]: c["status"] for c in json.loads(res.stdout)["checks"]}
    assert st["citations_were_retrieved"] == "fail"
    export_test(run.run_id, out_dir=tmp_path / "x")
    res = _pytest(tmp_path / "x")
    assert res.returncode == 0 and "passed" in res.stdout, res.stdout + res.stderr


# --- citation id variants ------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "ids"),
    [
        ("[Rückgabe-Richtlinie.md]", ["Rückgabe-Richtlinie.md"]),
        ("【返品ポリシー.md】", ["返品ポリシー.md"]),
        ("【返品ポリシー.md†L3-L5】", ["返品ポリシー.md"]),
        ("【refund.md†L1】", ["refund.md"]),
        ("【4†refund.md】", ["refund.md"]),  # cursor†source
        ("【4†L1-L3】", []),  # cursor + lines only: no id
        ("[política_reembolso.pdf]", ["política_reembolso.pdf"]),
    ],
)
def test_citation_id_variants(text, ids):
    assert rag.cited_ids(text) == ids


@pytest.mark.parametrize("suffix", ["#L10", "#L10-L20", ":L3", ":3-5", "#page=2"])
def test_line_suffix_resolves_to_retrieved_id(suffix):
    run = _shape_run([{"id": "Rückgabe.md", "text": "Refunds within 30 days of delivery."}])
    answer = f"Refunds within 30 days [Rückgabe.md{suffix}]."
    assert rag.citations_were_retrieved(run, answer) is None
    assert rag.answer_grounded(run, answer) is None
    assert "60" in rag.answer_grounded(run, f"Refunds within 60 days [Rückgabe.md{suffix}].")
    # a suffix never makes an unretrieved id match
    assert rag.citations_were_retrieved(run, f"30 days [Other.md{suffix}].")


def test_ids_containing_colons_still_match_exactly():
    run = _shape_run([{"id": "kb:12", "text": "Fee is 5 EUR."}])
    assert rag.citations_were_retrieved(run, "Fee is 5 EUR [kb:12].") is None
    assert rag.citations_were_retrieved(run, "Fee is 5 EUR [kb:13].")


@pytest.mark.parametrize(
    "answer",
    [
        "I\u2019m sorry, but I don\u2019t have that information.",  # live gpt-oss, 6 Nov
        "I do not have that information.",
        "I don't have enough information to answer.",
    ],
)
def test_live_refusals_count_as_abstention(answer):
    run = _shape_run([])
    assert rag.abstains_without_context(run, answer) is None


def test_apology_with_an_answer_is_not_an_abstention():
    run = _shape_run([])
    assert rag.abstains_without_context(run, "I'm sorry, refunds take 30 days.")
