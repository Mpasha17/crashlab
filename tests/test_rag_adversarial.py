"""Adversarial inputs for crashlab.rag: negation across sentences, contradictory and
duplicate documents, citation styles, non-English text, long documents, ranking/score
fields, and fuzzed malformed results (helpers must never raise). Offline."""

from __future__ import annotations

import random
import time

import pytest

import crashlab
from crashlab import rag
from tests.test_rag import _shape_run

ALL = (rag.answer_grounded, rag.citations_were_retrieved, rag.claims_supported,
       rag.abstains_without_context)  # fmt: skip


# --- multi-sentence negation -------------------------------------------------------


@pytest.fixture
def mixed_run():
    return _shape_run([{"id": "p", "text": (
        "Unopened items are refunded within 30 days. Opened items are not refunded. "
        "Gift cards are never exchanged for cash.")}])  # fmt: skip


@pytest.mark.parametrize(
    "answer",
    [
        "Unopened items are refunded within 30 days [p]. Opened items are not refunded [p].",
        "Opened items aren't refunded. Unopened items are refunded within 30 days [p].",
        "No, opened items are not refunded [p].",
        "Yes, unopened items are refunded within 30 days [p].",
        "Gift cards are never exchanged for cash [p].",
    ],
)
def test_negation_consistent_answers_pass(mixed_run, answer):
    assert rag.claims_supported(mixed_run, answer) is None


@pytest.mark.parametrize(
    "answer",
    [
        "Unopened items are refunded within 30 days [p]. Opened items are refunded too [p].",
        "Unopened items are not refunded [p].",
        "Gift cards are exchanged for cash [p].",
    ],
)
def test_negation_flip_in_any_sentence_fails(mixed_run, answer):
    msg = rag.claims_supported(mixed_run, answer)
    assert msg and "negation" in msg


# --- contradictory / duplicate / stale ids ------------------------------------------


@pytest.fixture
def conflict_run():
    return _shape_run([
        {"id": "v2", "text": "Refunds are accepted within 30 days of delivery."},
        {"id": "v1", "text": "Refunds are accepted within 14 days of delivery.",
         "superseded_by": "v2"},
        {"id": "v2", "text": "Refunds are accepted within 30 days of delivery."},  # duplicate
    ])  # fmt: skip


def test_contradictory_docs_grounding_follows_citation(conflict_run):
    assert rag.answer_grounded(conflict_run, "Refunds within 30 days [v2].") is None
    msg = rag.answer_grounded(conflict_run, "Refunds within 14 days [v2].")
    assert msg and "14" in msg
    # without a citation either version grounds the number: documented limitation
    assert rag.answer_grounded(conflict_run, "Refunds within 14 days.") is None
    assert "v1" in rag.no_stale_docs(conflict_run)


def test_duplicate_ids_are_reported_once(conflict_run):
    assert rag.retrieved_ids(conflict_run, ["v2", "v1"]) is None
    assert "['v2', 'v1']" in rag.retrieved_ids(conflict_run, ["zz"])
    assert rag.no_stale_docs(conflict_run).count("v1 (superseded") == 1


# --- citation styles ------------------------------------------------------------------


def test_citation_styles():
    assert rag.cited_ids("x [a][b] y [a, b] z [c; d]") == ["a", "b", "c", "d"]
    assert rag.cited_ids("see [the docs](https://x.y/a) and [n1]") == ["n1"]  # links
    assert rag.cited_ids("[not an id] [1] [2]") == ["1", "2"]


def test_mixed_citations_one_bad():
    run = _shape_run([{"id": "a", "text": "Fee is 5 EUR."}, {"id": "b", "text": "Open 9-5."}])
    assert rag.citations_were_retrieved(run, "Fee is 5 EUR [a, b].") is None
    msg = rag.citations_were_retrieved(run, "Fee is 5 EUR [a, c].")
    assert msg and "'c'" in msg and "'a'" not in msg.split("returned")[0]
    assert rag.answer_grounded(run, "Fee is 5 EUR [a, b].") is None


# --- unicode / non-English --------------------------------------------------------------


def test_german_paraphrase_and_unsupported_claim():
    run = _shape_run([{"id": "de", "text": "Rückgaben werden innerhalb von 30 Tagen akzeptiert."}])
    assert rag.claims_supported(run, "Rückgaben sind innerhalb von 30 Tagen möglich [de].") is None
    assert rag.claims_supported(run, "Versand nach Österreich kostet nichts extra [de].")


def test_cjk_without_word_spaces_is_skipped_not_flagged():
    run = _shape_run([{"id": "ja", "text": "返品は配達から30日以内に受け付けます。"}])
    assert rag.claims_supported(run, "返品は30日以内に可能です [ja]。") is None
    assert rag.answer_grounded(run, "返品は30日以内に可能です [ja]。") is None
    assert "60" in rag.answer_grounded(run, "返品は60日以内です [ja]。")


def test_fullwidth_digits_and_thousands_normalize():
    run = _shape_run([{"id": "a", "text": "Limit is 1,000 EUR within 30 days."}])
    assert rag.answer_grounded(run, "Limit １０００ EUR within ３０ days [a].") is None


def test_curly_apostrophe_negation():
    run = _shape_run([{"id": "a", "text": "Opened items are not refunded."}])
    assert rag.claims_supported(run, "Opened items aren\u2019t refunded [a].") is None
    assert "negation" in rag.claims_supported(run, "Opened items are refunded [a].")


# --- long documents ---------------------------------------------------------------------


def test_long_document_is_fast_and_correct():
    filler = " ".join(f"Clause {i} covers topic number {i} in detail." for i in range(20000))
    run = _shape_run([{"id": "big", "text": filler + " Refunds are accepted within 30 days."}])
    t = time.perf_counter()
    assert rag.claims_supported(run, "Refunds are accepted within 30 days [big].") is None
    assert rag.answer_grounded(run, "Refunds within 30 days [big].") is None
    assert "negation" in rag.claims_supported(run, "Refunds are not accepted within 30 days [big].")
    assert time.perf_counter() - t < 3.0


# --- ranking and score fields ----------------------------------------------------------


def test_top_k_checks_ranking_and_scores_are_ignored():
    run = _shape_run([
        {"id": "x", "text": "a", "score": 0.91},
        {"id": "y", "text": "b", "score": "n/a"},
        {"id": "gold", "text": "c", "score": float("nan")},
    ])  # fmt: skip
    assert [d["id"] for _, d in rag.retrieved(run)] == ["x", "y", "gold"]  # order kept
    assert rag.retrieved_ids(run, ["gold"]) is None
    msg = rag.retrieved_ids(run, ["gold"], top_k=2)
    assert msg and "top 2" in msg and "gold" in msg
    assert rag.retrieved_ids(run, ["gold"], top_k=3) is None


# --- fuzz: never raise ------------------------------------------------------------------


def _junk(rng, depth=0):
    pick = rng.randrange(9 if depth < 3 else 5)
    return [
        lambda: None, lambda: rng.random(), lambda: rng.choice(["", "[", "{", "x", "é", "\x00"]),
        lambda: rng.randrange(-5, 5), lambda: True,
        lambda: [_junk(rng, depth + 1) for _ in range(rng.randrange(4))],
        lambda: {rng.choice(["id", "text", "metadata", "documents", "superseded_by", "score"]):
                 _junk(rng, depth + 1) for _ in range(rng.randrange(4))},
        lambda: '[{"id": 1, "text": "5 days"}, 3]', lambda: '{"hits": [null, {"id": [1]}]}',
    ][pick]()  # fmt: skip


@pytest.mark.parametrize("seed", range(25))
def test_fuzzed_results_never_raise(seed):
    rng = random.Random(seed)
    run = _shape_run(_junk(rng))
    answers = [None, "", "[]", "[a] 5 days.", "No. Not [x]! ７ never", "返品 [1]", "x" * 5000]
    for fn in ALL:
        for a in answers:
            fn(run, a)
    rag.no_stale_docs(run, date_key="updated", not_before="2026-01-01")
    rag.retrieved_ids(run, ["a"], top_k=1)
    assert isinstance(rag.retrieved(run), list)
    crashlab.load(run.run_id)
