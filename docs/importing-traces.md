# Importing traces you already have

You do not have to instrument an agent with crashlab to debug it. If your traces are
already in OpenTelemetry, Langfuse, Arize Phoenix or LangSmith, export them and
`crashlab import` turns each trace into a run. After that, `show`, `diagnose`, `check`,
`eval`, `judge`, `score`, `diff`, `stats` and `crashlab.rag` oracles work on it like on a
recorded run.

```bash
crashlab import trace.json                       # format detected
crashlab import langfuse-export/ --json          # a directory of .json / .jsonl files
crashlab import spans.jsonl --format otel        # force a format
crashlab check last
crashlab diagnose last --heuristic
```

Existing behaviour is unchanged: a crashlab `.jsonl` export (`crashlab export`) and a
`.crashlab.zip` debug bundle import as before.

## Formats

| `--format` | What to export | What it needs |
| --- | --- | --- |
| `otel` | OTLP/JSON (`resourceSpans` → `scopeSpans` → `spans`), one trace per `traceId` | `gen_ai.*` attributes |
| `langfuse` | A trace with its `observations` (one trace per file, per `.jsonl` line, or the API's `{"data": [...]}`) | full observations, not just ids |
| `openinference` | Arize Phoenix spans (JSON records or a DataFrame `to_json(orient="records")`), or OTLP with `openinference.*` attributes | `openinference.span.kind` |
| `langsmith` | A run tree (`child_runs`), a list of trees, or a flat list linked by `parent_run_id` | `run_type`, `inputs`, `outputs` |

`--format auto` (the default) reads the file and picks one. A `.jsonl` file whose first
line is a crashlab export header is a crashlab export; anything else is parsed as traces,
one document per line.

### OpenTelemetry GenAI

This is [`crashlab export --format otel`](observability.md) in reverse. Read:

- `gen_ai.operation.name`: `chat` (also `text_completion`, `generate_content`) → LLM call,
  `execute_tool` (or `gen_ai.tool.name`) → tool call, `embeddings`, `invoke_agent` → the run
  (at the root) or the agent label of its descendants (nested).
- `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.request.temperature` /
  `max_tokens` / `top_p` ..., `gen_ai.usage.input_tokens` / `output_tokens`,
  `gen_ai.response.finish_reasons`, `gen_ai.system` / `gen_ai.provider.name`.
- `gen_ai.input.messages` / `gen_ai.output.messages` (role + `parts`, or plain
  `role` + `content`), `gen_ai.system_instructions`, `gen_ai.tool.definitions`, the same
  attributes on a `gen_ai.client.inference.operation.details` event, the older
  `gen_ai.prompt.N.*` / `gen_ai.completion.N.*` attributes and `gen_ai.*.message` /
  `gen_ai.choice` events.
- Tool arguments and results: `gen_ai.tool.call.arguments` / `gen_ai.tool.call.result`
  (also `gen_ai.tool.arguments` / `result`, OpenLLMetry's `traceloop.entity.input` /
  `output`), and `gen_ai.tool.call.id`.
- Errors: an `exception` event (type, message, stack trace), else an error span status.
- Timing: `startTimeUnixNano` / `endTimeUnixNano`; parents come from `parentSpanId`.

Spans with no GenAI meaning (HTTP, database) are skipped. A tool span without a call id is
matched to the earlier chat call that requested a tool of that name.

A run exported by crashlab and imported again keeps its operations, names, models, tokens,
tool call ids, errors, agents and timing. The OTLP export does not carry message bodies,
so a round trip does not bring them back.

### Langfuse

`GENERATION` observations become LLM calls (`model`, `modelParameters`, `input` as
messages or text, `output`, `usage` / `usageDetails`). `TOOL` and `RETRIEVER` observations,
and leaf `SPAN` observations that have an input or output, become tool calls; `AGENT`
labels its children; `EVENT` and grouping spans are ignored. `level: "ERROR"` marks a failed
call and uses `statusMessage` as the error text. The trace's own `input` / `output` become
the run's. Trace `sessionId` and `release` are kept in the run's metadata (user ids are not).

### OpenInference / Phoenix

`LLM` spans use `llm.input_messages.N.message.*` (role, content, tool calls),
`llm.output_messages`, `llm.model_name`, `llm.invocation_parameters`, `llm.tools`,
`llm.token_count.*`. `TOOL` spans use `tool.name`, `input.value` (the arguments) and
`output.value`. `RETRIEVER` spans use `retrieval.documents.N.document.*`. `AGENT` spans
label their children; `CHAIN` and other kinds only group.

### LangSmith

`llm` runs read LangChain-serialized messages (`HumanMessage`, `AIMessage` with
`tool_calls`, ...) or OpenAI-style messages, `generations` or `choices` outputs, token
counts and `extra.invocation_params`. `tool` runs become tool calls (a stringified-dict
input is parsed), `retriever` runs become tool calls whose result is the retrieved
documents. `chain` runs only group, and the root chain names the run. A run's `error`
becomes the step or run error.

## What an imported run looks like

```python
from crashlab import rag
from crashlab.ingest import from_langsmith, save_runs
from crashlab.store import Store

runs = from_langsmith(open("langsmith_run.json").read())
save_runs(Store(), runs)
run = runs[0]
print(run.env["metadata"]["source"])        # "langsmith"
docs = rag.retrieved(run, "VectorStoreRetriever")
```

- `run.env["metadata"]["source"]` is the format (`otel`, `langfuse`, `openinference`,
  `langsmith`); `trace_id` is the trace's own id. Runs are tagged `imported` and
  `import:<format>`.
- The run id is derived from the trace id, so importing the same file twice finds the run
  it already made (`--replace` overwrites it).
- Retrieval steps hold `[{"id", "text", "score", "metadata"}]`, which is what
  `crashlab.rag` reads: pass the retriever's name as `tool=`.
- Secrets are redacted when a run is saved, exactly like a recording.
- `check` and `diagnose` see the steps and errors but not what was live-only: the runtime
  detectors run when you `crashlab check`, not at import.

Python entry points: `from_otlp`, `from_langfuse`, `from_openinference`, `from_langsmith`
(each takes parsed JSON or JSON text and returns `list[Run]`), `detect_format`,
`load_path`, `save_runs`. Bad input raises `crashlab.ingest.IngestError` (a `ValueError`).

## Replay

An imported run has no recorded entry point, so `crashlab replay RUN` says it cannot
replay. From Python, give `crashlab.replay` a stand-in: a function that makes the same
calls through a crashlab-wrapped client and `@crashlab.tool` functions inside a
`crashlab.record(...)` block. Every call is served from the imported steps, and the replay
raises a divergence error with a diff when the stand-in's requests differ from the trace.

```python
import crashlab

def stand_in():
    client = crashlab.wrap(my_client)
    with crashlab.record("support-bot") as rec:
        ...  # the same LLM and tool calls the traced agent made

replayed = crashlab.replay("RUN_ID", fn=stand_in)
```

Replay is exact only when the trace holds the full request of each call: messages, tools
and model parameters (`gen_ai.input.messages`, Langfuse `input`, `llm.input_messages`,
LangSmith `inputs`). Steps imported without them still show, diagnose and check, but a
stand-in will diverge at the first such call. LLM responses are replayed in OpenAI chat
form (Anthropic message form for Anthropic traces).
