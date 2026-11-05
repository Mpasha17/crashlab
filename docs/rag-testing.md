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

## 2. Add oracles

Each helper returns a failure message (naming the step) or `None`, so it can be returned
from a check:

| Failure mode | Helper |
|---|---|
| Wrong retrieval | `rag.retrieved_ids(run, ["gold-doc-id"])` |
| Missing context (answered anyway) | `rag.abstains_without_context(run, answer)` |
| Stale documents | `rag.no_stale_docs(run)` (`superseded_by` set) or `date_key=..., not_before="2026-01-01"` |
| Citation mismatch | `rag.citations_were_retrieved(run, answer)` (default citation form `[doc-id]`, `pattern=` to change) |
| Answer ignores evidence (numbers) | `rag.answer_grounded(run, answer)` |
| Unsupported / negated claim (words) | `rag.claims_supported(run, answer, min_overlap=0.5, min_words=3)` |

`claims_supported` checks each answer sentence with at least `min_words` content words
against the documents that sentence cites (falls back to the answer's citations, then to
all retrieved documents). It fails when under `min_overlap` of the sentence's content
words (lower-cased, stopwords removed, light suffix stemming) appear there, or when the
only matching document sentences have the opposite negation (`not`, `no`, `never`,
`n't`, ...). Abstentions and short sentences are skipped; `stopwords=` is configurable.

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

## 3. Re-check stored runs from the CLI

`crashlab check` imports the run's recorded entrypoint, which registers the module's
`@crashlab.check` oracles, so RAG checks run on stored runs in a fresh process:

```bash
crashlab check last --json
crashlab check --all --save --name rag-qa   # grade older runs with newly added checks
```

## 4. Replay and export

```python
run = crashlab.load("last")
crashlab.replay(run.run_id)                 # offline, same outcome
```

```bash
crashlab export-test last                  # writes tests/crashlab/test_<agent>_<id>.py
```

A complete offline example is `examples/rag_qa/agent.py` (in-memory corpus, faults
`wrong_retrieval`, `missing_context`, `stale_docs`); `tests/test_rag.py` runs every failure
mode with a scripted fake model, replays a failing run and runs its exported test.

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

## Limitations

- No vector-database integrations are built in; you record your own retriever function.
- `answer_grounded` checks only numbers; `claims_supported` checks word overlap and
  negation. Neither is a semantic entailment judge: paraphrases using different words can
  fail (lower `min_overlap`), and wrong claims that reuse the documents' words can pass.
  Negation handling is sentence-level polarity, not scope-aware.
- `abstains_without_context` matches refusal phrases (`rag.ABSTAIN`, overridable).
- The built-in runtime checks only flag an empty retrieval (`suspicious_tool_result`, info);
  the other RAG failure modes need these oracles.
