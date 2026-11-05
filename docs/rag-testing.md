# Testing RAG applications with crashlab

crashlab records your retriever like any other tool, so a RAG run can be replayed offline,
judged by `@crashlab.check` oracles and exported as a pytest regression test.
`crashlab.rag` adds small, deterministic oracle helpers for the common RAG failure modes.

## 1. Record the retriever

```python
import crashlab
from crashlab import rag

@crashlab.tool()
def search_docs(query: str) -> list[dict]:
    return my_retriever(query)  # each document a dict: {"id": ..., "text": ..., ...}
```

The recorded result is what replay serves back, so replay never calls your vector store.
Accepted result shapes: a list of dicts or of plain strings; a dict wrapping that list under
`documents`, `results`, `docs`, `hits` or `matches`; or the same as JSON text. Id/text keys
are parameters (`id_key="id"`, `text_key="text"`); when missing, the id falls back to
`doc_id` / `document_id` / `source` (also under `metadata`, e.g. LangChain-style
`{"page_content": ..., "metadata": {...}}`) and the text to `page_content` / `content` /
`chunk` / `document` / `snippet`. Malformed results (None, numbers, non-JSON text, error
objects, non-dict items) are skipped, never raised.

Two lower-level helpers are public too. `rag.retrieved(run, tool="search_docs")` returns
`(step, document)` pairs with the documents normalized as above. `rag.cited_ids(text)`
returns the citation ids in an answer.

## 2. Add oracles

Each helper returns a failure message (naming the step) or `None`, so it can be returned
from a check:

| Failure mode | Helper |
|---|---|
| Wrong retrieval / ranking | `rag.retrieved_ids(run, ["gold-doc-id"], top_k=3)`. `top_k` is optional and requires the gold document within the first k results of a call. The recorded order is the rank; score fields are not read |
| Missing context (answered anyway) | `rag.abstains_without_context(run, answer)` |
| Stale documents | `rag.no_stale_docs(run)` (`superseded_by` set) or `date_key=..., not_before="2026-01-01"` |
| Citation mismatch | `rag.citations_were_retrieved(run, answer)`. Recognised forms: `[doc-id]`, `[a, b]`, `[a; b]`, `【doc-id】`, `【doc-id†L1】`, `【4†doc-id】`. `[text](url)` links and bracketed text with spaces are ignored. A `#L10`, `#L10-L20`, `:3-5` or `#page=2` suffix resolves to the bare id when that was retrieved (exact ids win). Change the form with `pattern=` |
| Answer ignores evidence (numbers) | `rag.answer_grounded(run, answer)`; with `exclude_stale=True` (opt-in) it also fails numbers that only stale documents support |
| Unsupported / negated claim (words) | `rag.claims_supported(run, answer, min_overlap=0.5, min_words=3)` |

`claims_supported` checks each answer sentence with at least `min_words` content words
against the documents that sentence cites (falls back to the answer's citations, then to
all retrieved documents). It fails when under `min_overlap` of the sentence's content
words (lower-cased, stopwords removed, light suffix stemming) appear there, or when the
only matching document sentences have the opposite negation (`not`, `no`, `never`,
`n't`, ...). For negation, the best-matching document sentence decides, and ties favour the
same polarity, so documents mixing "X is refunded. Y is not refunded." do not flag. A
leading "Yes," / "No," is ignored. Abstentions and short sentences are skipped, and
`stopwords=` (English by default) is configurable. Words are Unicode letters, so German
text works; sentences containing Chinese/Japanese/Korean characters are skipped (no word
spaces). Numbers are NFKC-normalized (full-width digits), and `1,000` equals `1000`.

```python
@crashlab.check("support-rag")
def citations_ok(run):
    return rag.citations_were_retrieved(run, run.output)
```

### Same id, different text (versions and chunks)

`answer_grounded` and `claims_supported` take `duplicates=`, applied to the documents used
as evidence (the cited ones, or all retrieved ones when the answer cites nothing):

| `duplicates` | Same id retrieved with different texts |
|---|---|
| `"versions"` (default) | Fails as ambiguous if they are versions (a differing `version` / `updated` / `updated_at` / `last_modified` / `modified` / `date` field, top level or in `metadata`, or one has `superseded_by`); merges unversioned texts (chunks of one source) |
| `"latest"` | Uses the non-superseded, newest version (numbers compare numerically, other values as strings) |
| `"first"` | Uses the first one retrieved |
| `"any"` | Merges all texts (either version grounds the answer) |
| `"strict"` | Fails on any differing text, versioned or not |

### LangChain-style documents

`examples/rag_qa/langchain_style.py` returns `langchain_core` `Document` objects (or a
pydantic stand-in when langchain-core is not installed). The id comes from
`metadata["source"]`, the text from `page_content`. Its checks pass `duplicates="latest"`.
That configuration is in the check code, so `crashlab check` and exported tests use it.
Annotate the retriever's return type (`-> list[Document]`): replay rebuilds the objects from
it. With `-> list[Any]`, replay hands the agent plain dicts, and an agent that reads
`doc.metadata` breaks (`tests/test_rag_versions.py` covers this).

### Stale vs current documents with different ids

By default `answer_grounded` is permissive. If `v1` (superseded) and `v2` are both
retrieved, an uncited "14 days" from `v1` passes. Combine the checks:

```python
rag.no_stale_docs(run)                       # the stale document was retrieved at all
rag.citations_were_retrieved(run, answer)    # the answer must cite something
rag.answer_grounded(run, answer, exclude_stale=True)  # numbers only from current docs
```

`exclude_stale` uses the same rules as `no_stale_docs` (`superseded_key`, or `date_key` +
`not_before`). A number supported only by a stale document fails, cited or not, and the
message lists the stale documents it ignored.

### Retrieval errors and timeouts

A retriever that raises (e.g. `TimeoutError`) is recorded as a failed tool step. One that
returns an error object (`{"error": ...}`) yields no documents. In both cases
`abstains_without_context` requires the answer to abstain, and `citations_were_retrieved`
fails any citation. `examples/rag_qa` has the `retrieval_timeout` and `retrieval_error`
faults, covered through `crashlab check` and exported tests. Empty or malformed citations
(`[]`, `【】`, `[,]`, `【†L1】`) count as no citation.

## 3. Re-check stored runs from the CLI

`crashlab check` imports the run's recorded entrypoint, which registers the module's
`@crashlab.check` oracles, so RAG checks run on stored runs in a fresh process:

```bash
crashlab check last --json
crashlab check --all --save --name rag-qa   # grade older runs with newly added checks
```

The entrypoint module is imported, so keep the code that starts a run under
`if __name__ == "__main__":` (recommended). If a module starts a run at import time,
crashlab refuses that run while it imports the entry point and reports `could not import
...: ... put that code under if __name__ == "__main__"`. No new run is recorded and nothing
recurses. Checks defined above that call still run. Replay of such a run fails with a
`ReplayError` that names the fix.

## 4. Replay and export

```python
run = crashlab.load("last")
crashlab.replay(run.run_id)                 # offline, same outcome
```

```bash
crashlab export-test last                  # writes tests/crashlab/test_<agent>_<id>.py
```

A complete offline example is `examples/rag_qa/agent.py`. It has an in-memory corpus, the
faults `wrong_retrieval`, `missing_context`, `stale_docs`, `retrieval_timeout` and
`retrieval_error`, and seven checks, including `grounded_in_current_docs`
(`exclude_stale=True`). `tests/test_rag_integration.py` takes every failure mode, with a
scripted fake model, through record, `crashlab check` (fresh process), replay and
export-test (fresh pytest process).

`crashlab.rag` ships in the installed package. The examples, tests and this page are in
the source repository and the sdist, not in the wheel. Run the examples from a checkout or
an extracted sdist (`tests/test_rag_wheel.py` runs the RAG tests from one).

## Live run

`examples/rag_qa` was run once against Groq `openai/gpt-oss-120b` (5 Nov 2026) for
the clean case and each fault. The checks flagged the expected failure modes and each
failing run replayed offline as a fail. Two false positives found there (the agent's
"could not answer" fallback, a stale-doc paraphrase) are now regression tests. Offline
tests remain the source of truth.

Live LangChain-style example runs (Groq `openai/gpt-oss-120b`, 6 runs, 6 Nov 2026) were
not stable. The model's answers and citation style changed between sessions (`[src]`,
`【src】`, `【src†L1】`), and 4 of the 6 answers repeated the stale 14-day version. The
example agent does not pass the `updated` dates to the model, so
`grounded_in_latest_version` was right to flag those answers. The citation forms seen
there are now supported and covered by tests.

Repeated live runs of `examples/rag_qa` (n=3 per case, temperature 0, 6 Nov 2026): 12/12
had the same outcome and failing checks per case and replayed identically. A later n=2
validation (clean, `stale_docs`, `retrieval_timeout`, `retrieval_error`) was less stable.
The model's refusal wording varied ("Sorry, I could not answer." vs "I'm sorry, but I
don't have that information."), and the second phrasing was not recognised as an
abstention. It now is, with a regression test.

## Limitations

- No vector-database integrations are built in; you record your own retriever function.
- `answer_grounded` checks only numbers; `claims_supported` checks word overlap and
  negation. Neither is a semantic entailment judge: paraphrases using different words can
  fail (lower `min_overlap`), and wrong claims that reuse the documents' words can pass.
  Negation handling is sentence-level polarity, not scope-aware.
- Long documents work (20,000 sentences checked in under 3 s in tests). Helpers never raise
  on malformed retrieval results (fuzz-tested).
- Word overlap is not semantic equivalence. In a live run, "You have 14 days from the date
  of delivery to return an item." scored 2/5 against "Refunds are accepted within 14 days
  of delivery." and was flagged at the default `min_overlap=0.5`.
- `abstains_without_context` matches refusal phrases (`rag.ABSTAIN`; pass `phrases=` to
  replace them). `citations_were_retrieved` uses the default phrases to excuse an
  uncited refusal.
- Helpers do not raise on retrieved data. An invalid `duplicates=` value raises
  `ValueError`.
- The built-in runtime checks only flag an empty retrieval (`suspicious_tool_result`, info);
  the other RAG failure modes need these oracles.
