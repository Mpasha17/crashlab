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
Results may be a list of dicts or a dict wrapping one under `documents`, `results`, `docs`,
`hits` or `matches`. Id/text keys are parameters (`id_key="id"`, `text_key="text"`).

## 2. Add oracles

Each helper returns a failure message (naming the step) or `None`, so it can be returned
from a check:

| Failure mode | Helper |
|---|---|
| Wrong retrieval | `rag.retrieved_ids(run, ["gold-doc-id"])` |
| Missing context (answered anyway) | `rag.abstains_without_context(run, answer)` |
| Stale documents | `rag.no_stale_docs(run)` (`superseded_by` set) or `date_key=..., not_before="2026-01-01"` |
| Citation mismatch | `rag.citations_were_retrieved(run, answer)` (default citation form `[doc-id]`, `pattern=` to change) |
| Answer ignores evidence | `rag.answer_grounded(run, answer)` |

```python
@crashlab.check("support-rag")
def citations_ok(run):
    return rag.citations_were_retrieved(run, run.output)
```

## 3. Replay and export

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

## Limitations

- No vector-database integrations are built in; you record your own retriever function.
- `answer_grounded` checks only that numbers in the answer appear in the (cited) retrieved
  text. It is not a semantic entailment judge: a wrong non-numeric claim passes.
- `abstains_without_context` matches refusal phrases (`rag.ABSTAIN`, overridable).
- The built-in runtime checks only flag an empty retrieval (`suspicious_tool_result`, info);
  the other RAG failure modes need these oracles.
