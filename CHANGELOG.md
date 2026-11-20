# Changelog

## Unreleased

### Integrations and bisect (see `docs/integrations.md`)
- `crashlab.integrations.wrap_mcp_session(session)` (and `MCPToolResult`): records MCP `call_tool` calls as `execute_tool` steps; replay-safe, detects side-effectful MCP calls, surfaces `isError` / `structuredContent`
- `crashlab.integrations.CrashlabCallbackHandler` (and `langchain_handler`): LangChain / LangGraph callback handler that records LLM calls, tool executions, and agent boundaries without double-recording `wrap()`
- `crashlab bisect` / `crashlab.bisect.bisect_runs(good, bad)`: compares a passing and failing run to isolate the earliest divergence and minimal causal diff (supports `--fork` single-change revert isolation)

### Fault injection and chaos testing (see `docs/fault-injection.md`)
- `crashlab.faults(...)`: stackable, thread-safe context manager that injects deterministic tool and LLM faults into a live recorded run (or the live steps of a fork). Rules: `tool_error`, `tool_result` (HTML 502, empty, truncated JSON, schema drift), `tool_delay`, `llm_error`, `llm_empty`, `llm_truncate`, `llm_bad_tool_call`. Triggers: `on_calls`, `after_calls`, `every`, `max_triggers`, `probability`+`seed`, `agent=`. Each injection is stored on the step as `crashlab.fault`. Strict replay serves the cassette and does not re-inject.
- `crashlab stress` / `crashlab.stress_run(run)`: built-in chaos scenarios (`tool_502`, `tool_timeout`, `tool_empty`, `tool_schema_drift`, `llm_rate_limit`, `llm_empty_reply`, `llm_bad_tool_args`) targeting the tools and models the run used; reports survived / crashed / failed_checks / skipped / needs_live_agent. `--scenario`, `--at-step`, `--agent`, `--json`, `--fail-on-crash`.

### Runtime checks and store retention
- New runtime checks (on by default, conservative): `context_bloat` (input tokens at 75%/92% of a known model context window, or a 3x prompt jump of at least 20,000 tokens; unknown models skip the ratio check), `tool_fanout` (12+/25+ tool calls in one turn, or the same tool + arguments at least 3 times in a single response), `system_prompt_leak` (a verbatim 100+ character slice of the system prompt in the final reply; short prompts and user-echoes are skipped). Disable with `checks={"context_bloat": false}` (or the same for the others)
- `crashlab tag RUN [+tag / -tag ...]` / `Store.tag_run()`: add or remove tags on a recorded run (`--json`; no tag args prints the current list). `crashlab ls --tag NAME` lists only those runs
- `crashlab rm RUN... --yes` / `Store.delete_run()`: delete one or more recorded runs (id prefixes allowed). Without `--yes`, prints the ids and exits 1
- `crashlab prune --older-than 30d --max-runs N --max-bytes 500MB` / `Store.prune()`: drop old or excess runs. Dry-run unless `--yes`. Failed and tagged runs are kept unless `--include-failed` / `--include-tagged`. `--json`

### Observability and visual debugging (see `docs/observability.md`)
- `crashlab export --format otel|chrome-trace` (and `crashlab.otel.to_otlp` / `to_chrome_trace`): OTLP/JSON with GenAI semantic conventions, root / step / child-run parent links, error and finding events; Chrome `traceEvents` for Perfetto. Linked child runs are included when they are in the store. The default `--format jsonl` is unchanged
- `crashlab stats` / `crashlab.stats.compute_stats`: fleet pass / fail / error rates, cost / token / latency totals and percentiles, breakdowns by agent and model, a tool leaderboard, top findings and failing checks, slowest and most expensive runs; `--since`, `--name`, `--limit`, `--json`
- `crashlab timeline` / `crashlab.timeline.render_timeline`: terminal ASCII waterfall (timing bar, tokens, cost, errors, findings, handoffs) with indented child-run subtrees; `--no-children`, `--json`
- `crashlab ui`: zero-dependency localhost trace explorer (waterfall, chat, tools, findings, diagnosis, child runs, side-by-side diff) and JSON API; `--port`, `--no-open`. Binds 127.0.0.1 only


### Native Google GenAI (see `docs/genai.md`)
- `crashlab.wrap(client)` records a `google.genai.Client` the same way it records OpenAI and Anthropic: `models.generate_content` / `generate_content_stream`, `chats.create` + `send_message` / `send_message_stream`, and the async `client.aio` API. The SDK is optional; replay still exposes `response.text` and `function_calls`

### Declarative policy rules (see `docs/rules.md`)
- `crashlab.assertions` / `crashlab rules`: grade a run from a JSON, TOML or YAML rule file (`must_call`, `must_not_call`, `tool_order`, and other predicates) via `crashlab check`, `crashlab eval --rules`, `crashlab rules test`, or the Python API. Nothing here calls a model

### Ops and profiling (see `docs/ops-and-profiling.md`)
- `crashlab doctor`: inspect the local store, optional packages, redaction and pricing (`--fix`, `--json`)
- `crashlab init`: scaffold a CI eval workflow and a starter `crashlab_checks.py`
- `crashlab profile`: break down a run's time, tokens, cost and prompt-cache opportunities

### Golden datasets and adversarial fuzzing (see `docs/fuzzing-and-datasets.md`)
- `crashlab dataset build` / `inspect`: portable, redacted JSONL golden datasets from recorded runs
- `crashlab fuzz` / `crashlab.fuzz`: seeded mutation libraries (`prompt_injection`, `tool_poison`, `edge_inputs`) against a recorded run or a live entry point

### Alert sinks and webhook watch (see `docs/alerts.md`)
- `crashlab.alerts` and `crashlab watch --webhook` / `--slack-webhook` / `--alert-file`: fire when a run fails, a finding is serious enough, or cost / latency exceeds a cap (stdlib delivery only)

### Zero-config auto-instrumentation (see `docs/auto-instrumentation.md`)
- `crashlab.instrument()` / `uninstrument()` / `instrumented()`: patch OpenAI, Anthropic, LiteLLM and Google GenAI so frameworks that construct a client internally are recorded without a private `_client` hook. Missing SDKs are skipped, not an error

### Multi-turn conversation simulator (see `docs/simulation.md`)
- `crashlab.simulate` / `crashlab simulate`: drive a live agent through a scripted persona or policy scenario; the whole scenario is one recorded run with `crashlab.turn` tags

### Enhanced RAG checks and flake fingerprints (see `docs/rag-testing.md`)
- New `crashlab.rag` oracles: `retrieval_non_empty`, `context_utilization`, `no_prompt_injection_in_docs`, `retrieval_latency_budget`
- `crashlab.flakes` can group runs by a recorded non-git `code_hash` when no git commit is available

### Fleet HTML reports, trajectory diffs and CI baselines (see `docs/observability.md`)
- `crashlab stats --html`: a self-contained HTML fleet report
- `crashlab.diff` reports tool-trajectory edit distance and a normalized trajectory similarity
- `crashlab baseline save` / `check` / `crashlab.baseline`: snapshot fleet metrics and fail CI on pass-rate, cost or latency regressions

### Language-agnostic local HTTP/SSE proxy (see `docs/proxy.md`)
- `crashlab proxy`: a localhost-only HTTP proxy that records OpenAI `/v1/chat/completions` and `/v1/responses` plus Anthropic `/v1/messages` (including `stream: true` SSE) and can later replay the cassette offline, for any language or framework

### Native pytest plugin (see `docs/pytest-plugin.md`)
- Auto-loaded `pytest11` plugin (opt-in): fixtures `crashlab_store`, `crashlab_record`, `crashlab_replay`, `@pytest.mark.crashlab`, and `--crashlab-save-failures`. pytest remains an optional extra

### Prompt context debugger and tool-schema auditor (see `docs/context-and-tools.md`)
- `crashlab context`: break a run's prompt into sections (system, tools, messages, tool results) with estimated tokens and growth
- `crashlab audit-tools`: flag provider-breaking tool schemas (required fields missing from properties, vague or missing descriptions, duplicates, unused tools)

### Weighted rubric scoring (see `docs/rubrics.md`)
- `crashlab.rubrics` / `crashlab score` / `crashlab eval --rubric`: grade recorded runs 0–100 across weighted criteria (outcome, safety, tool efficiency, cost, latency), with a built-in default rubric

### Portable debug bundles and secret/PII audit (see `docs/bundles-and-security.md`)
- `crashlab bundle` / `unbundle`: pack a run into a shareable `.crashlab.zip` (JSONL, diagnosis, prompt / tool-schema summary, HTML report; secrets redacted by default)
- `crashlab audit-secrets` / `redact-run`: scan or clean leaked API keys, tokens, private keys, or optional PII already in the store

### Interactive HTML and PR Markdown diffs
- `crashlab diff --html` / `--markdown` / `--fail-on-regression`: interactive side-by-side HTML, a GitHub-flavored Markdown summary for PR comments, and a CI gate when run B failed, grew a new finding, or a check flipped from pass to fail

### Framework adapters (see `docs/integrations.md`)
- First-class, duck-typed adapters in `crashlab.integrations`: OpenAI Agents SDK (`crashlab_tracing_processor`), PydanticAI (`wrap_pydantic_ai_agent`), and CrewAI (`wrap_crew` / `CrewAICallback`). Those packages are never imported, so a missing extra never breaks `import crashlab`

## 0.0.5 (pre-alpha) - 2026-11-13

Fifth public cut; matches `main` at release. `pip install crashlab`.

- `crashlab export-test`: if pytest is not installed, print a hint to `pip install pytest` or `pip install "crashlab[pytest]"` (the export still succeeds). New optional extra `crashlab[pytest]` (`pytest>=8`); pytest is not a hard dependency

## 0.0.4 (pre-alpha) - 2026-11-13

Fourth public cut; matches `main` at release. `pip install crashlab`.

### Debugging many runs (see `docs/debugging.md`)
- `crashlab search` / `crashlab.search.search_runs`: offline text search across runs (messages, tool names / args / results, errors, models, checks, findings, diagnosis labels, run ids) with `--tool`, `--model`, `--check`, `--label`, `--errors`, `--since`, `--run`, `--in`, regex and JSON output; exits 1 when nothing matched
- `crashlab cluster` / `crashlab.cluster.cluster_runs`: group failed runs by a conservative signature (agent, normalized exception, first failing step, failing checks, diagnosis label, serious findings); `--by` to loosen, stable cluster ids, JSON output
- `crashlab triage` / `crashlab.triage.triage_runs`: cluster failed runs, diagnose each cluster's newest run offline (heuristic; a stored diagnosis is reused unless `--rediagnose`), and print them ranked with size, signature, label, blamed step and agent, one line of evidence and the next commands (`show --failing`, `fix`, `export-test`); `--json`, `--since`, `--name`, `--limit`, `--no-save`. Heuristic labels are marked as such
- `crashlab show` gains `--errors`, `--tools`, `--llm`, `--failing` and `--step N` (full detail of one step), and a failed run's output ends with a line pointing at the first failing step and failing checks; existing output is otherwise unchanged
- `crashlab watch`: print runs as they start and finish (polls the store), with a first-failure pointer; `--failed`, `--existing`, `--json`, `--timeout`
- HTML report: per-step timing / cost waterfall from recorded durations, tokens and prices; unknown prices shown as `?`, no chart when no durations were recorded

### Budgets and flakes
- `crashlab.budgets.within_budget()`: `@crashlab.check` budget oracle for cost, tokens (total / input / output / largest prompt), wall time, slowest call and step counts, from recorded usage. Missing usage and unpriced models fail with "cannot check" unless `unknown="pass"`; time limits are not evaluated on replays / forks (`replay_time`). `summarize_usage()` exposes the numbers
- Runtime `budget` check: opt-in `max_call_input_tokens` and `max_call_seconds` per-call limits; an `info` finding when a token limit is set but some LLM calls reported no usage (the limit is then a lower bound)
- `crashlab flakes` / `crashlab.flakes.find_flakes()`: same agent + canonical input + code version, signals (raised, outcome, checks, findings) that differ between runs, with counts and a Wilson 95% interval; `--json`, `--fail-on-flaky`. See `docs/budgets-and-flakes.md`

### Fixes
- A cassette or export saved by an editor that adds a UTF-8 byte-order mark (and CRLF line endings) lost its first step on read, and `crashlab import` rejected it; both now read normally
- Security: `crashlab import` / `import_run` of a crafted export could write `cassette.jsonl` and `run.json` outside the store (a run id like `../x` or an absolute path). A run id must now be one plain name (letters, digits, `.`, `_`, `-`). Imported steps are always stored under the run's own id
- Recording no longer crashes the agent (`OverflowError`) or stops (`sqlite3` binding error) when a provider reports a token count beyond 64 bits, or sends a non-string finish reason or tool-call id, or a NaN duration. The SQLite index clamps or stringifies the value, and the cassette and `run.json` keep the exact value
- Test suite: refuses to start (with the fix in the message) when `import crashlab` resolves outside this checkout's `src/`, so tests can no longer silently run against another worktree or an old install; `CRASHLAB_TEST_ANY_INSTALL=1` opts out
- Experimental compact writer (`CRASHLAB_EXPERIMENTAL_COMPACT=1`): the shared prefix is now found by comparing the JSON text of the whole list (one encode per step), not per-message encodes or fingerprints. A 1,500-message chat records in about 3.4 s instead of 5.5 s through the store (full format: 3.0 s), and typical chats now cost about the same as the full format. Same on-disk format, byte-identical to 0.0.3 for the same input. Writing stays opt-in; measured numbers are in `docs/compatibility.md`
- `crashlab check` import guard: tasks an entry point submits to a `ProcessPoolExecutor` at import time (or from threads it started) are now refused in the worker for fork, spawn and forkserver. Before, spawn and forkserver workers recorded a run, and fork workers created inside that `submit` stayed tainted and refused later legitimate tasks. `multiprocessing.Process`/`Pool` with spawn and custom queues are still not covered (pinned by tests)
- `crashlab fork` (and `verify --patch`-style flags) failed with "None is not a valid string" on typer 0.29+; tuple options now default to `None`
- Dependency floors raised to `typer>=0.16` and `pydantic>=2.6`: versions inside the old bounds broke on Python 3.10 (typer < 0.12.4: every command; typer < 0.16 with click 8.2+: "Secondary flag"; pydantic < 2.6: nested dataclass tool results came back as dicts on replay). CI now also tests the floors on Python 3.10
- Python 3.10: `crashlab ls` / `show` / `search` / `report` crashed with "Invalid isoformat string" on timestamps with other than 3 or 6 fraction digits (written by pydantic < 2.3, about one run in ten) or a trailing `Z`
- `crashlab flakes` is listed under "Inspect" in `crashlab --help`
- Replay, exported tests and `crashlab check` no longer need a real `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` when the agent module builds its client at import; a module that reads another key at import (`KeyError: 'GROQ_API_KEY'`) now gets a hint that any value works there
- `crashlab show --failing` on a run that failed only a check now shows the step the check's message cites ("step N ...") instead of the whole run
- `crashlab diagnose` and `crashlab verify` print what to run next (fix / verify / export-test / report); diagnose no longer suggests diagnosing again after a confident verdict
- `examples/rag_qa/agent.py` takes a question and `--fault` from the command line
- Docs: a test parses every `crashlab` command and checks every `crashlab` Python name and keyword in README, docs and example READMEs, and runs the README's no-API-key block
- `scripts/smoke.sh`: every CLI command end to end, offline (fake OpenAI server), run in CI
- Diagnose: a `@crashlab.check` that crashed on a bug in its own code (`'Step' object has no attribute ...`, `NameError`, a missing import) is no longer blamed on the agent; the heuristic verdict is UNKNOWN with "the check itself is broken", and the LLM diagnoser's view marks it

### Multi-agent
- Steps record which agent made them (`gen_ai.agent.name`, `step.agent`): `crashlab.wrap(client, agent=...)`, `crashlab.agent_scope(name)`, nested `@crashlab.agent`; tool steps inherit the requesting chat step's agent. `run.agents`, `run.tool_calls(..., agent=)`, `run.chat_steps(agent=)`; `crashlab show` prints agents. Labels only; replay unchanged. See `docs/multi-agent.md`
- Fix: `step_repetition` judges reply / request repetition per agent in labelled runs (a writer repeating a researcher's answer is not a loop)
- Fix: a tool taking PydanticAI's `RunContext` (or any dataclass holding a lock / client) crashed the agent with `TypeError: cannot pickle '_thread.RLock'`; context objects are no longer recorded, dataclass args are converted without deep-copying, and an argument that cannot be converted is recorded as a placeholder. `@crashlab.tool(ignore_args=...)`
- Fix: replaying a run with a sub-agent used as a tool dropped the sub-agent's steps, so oracles on them passed on the recording and failed on the replay / exported test; the served tool's recorded inner steps are now copied into the replay
- Fix: concurrent sub-agents in one run no longer lose tool-call ids; `describe_step` / diagnose blame the chat step that requested a tool call (`run.caller_chat`), not just the nearest one
- New runtime check `empty_reply` (warn): a chat turn that ended with no text and no tool call (e.g. reasoning only). Found live: a LlamaIndex AgentWorkflow agent on gpt-oss-120b "decided" to hand off in its reasoning but emitted no handoff call, and the workflow returned "" with status ok. Disable with `checks={"empty_reply": False}`
- Fix: replaying an agent used as a tool whose result holds models / dataclasses inside a plain dict or list (Haystack `AgentTool` returns `{"messages": [ChatMessage, ...]}`) served plain dicts, so the framework code reading `messages[-1].text` failed and the replay diverged; their paths and types are recorded (`crashlab.result.types`) and revived on replay. Older recordings replay as before
- Fix: parallel tool calls streamed under one index (Gemini's OpenAI endpoint sends them without an index) are merged by clients into one broken call (`researchresearch`, `{..}{..}`); the record keeps what the client saw, adds `crashlab.stream.tool_calls_merged`, `bad_tool_args` says why the call is broken, and replay now streams each delta with its own id (it reused the last id). Found live with the OpenAI Agents SDK `as_tool` + `Runner.run_streamed` on Gemini
- Fix (heuristic diagnose): a failing check whose message quotes a reply now blames the chat step that first wrote that text (and names its agent), not a tool whose result shares its numbers (Haystack: the orchestrator turned "7 October" into "July 7" and the lookup was blamed); an `empty_reply` outranks a vague end-of-run check; heuristic evidence names the field the quote came from (a critic's "APPROVE" was attributed to its system prompt). An LLM diagnoser's `field` is honoured when the quote is really there
- Concurrent runs of the same agent are told apart: `run.agent_instances()` / `run.agent_labels` label them `researcher#1`, `researcher#2` from their conversations (found live: OpenAI Agents SDK `as_tool` called twice in parallel); `chat_steps` / `tool_calls(agent="researcher#2")`, `crashlab show` and `--agent`, and the diagnoser view use the labels. Works on existing recordings; `run.agents` unchanged
- Per-agent budgets: `within_budget(run, agent="researcher", ...)` and `summarize_usage(run, agent=...)` count only one agent's steps (or one concurrent instance, `"researcher#2"`); an agent with no steps fails; `max_seconds` is refused with `agent=`
- Fork patches for one agent: request patches (system prompt, model, tools, ...) take `agent=` and edit only that agent's LLM calls; a fork with no call of that agent from the fork step on is refused and lists the run's agents; descriptions end with `[agent X]`. Patches without `agent=` and their JSON are unchanged
- Handoffs: `run.handoffs()` finds where one agent handed the conversation to another (the next agent's first request answers the previous agent's handoff tool call; found live on OpenAI Agents SDK `transfer_to_*` and LlamaIndex AgentWorkflow `handoff`); `crashlab show` prints a `handoffs` row and marks the step `⇢ from X`, `show --json` adds `handoffs` when there are any, the diagnoser view notes them. Derived from the recording: no new steps, replay and step numbers unchanged, works on older runs
- Runs in other processes are linked: `crashlab.child_env()` gives a subprocess `CRASHLAB_PARENT_RUN` / `CRASHLAB_TRACE_ID` / `CRASHLAB_DIR`, and any run it records links to the parent; `crashlab.record(..., parent=, trace=)` for HTTP / queues. `run.parent_run`, `run.trace_id`; `crashlab show` lists child runs and shows the parent and trace; `crashlab ls --trace RUN`, `crashlab search --trace RUN`, `Store().child_runs()`. Stored in `run.env["link"]`: no schema change, 0.0.3 reads linked runs; replays and forks are not linked

## 0.0.3 (pre-alpha)

Third public cut; matches `main` at release. `pip install crashlab`.

### Diagnose / fix / verify
- Diagnosis blames the model's decision step (not the tool step) when that is the cause, names missing rules, and keeps vague failures `UNKNOWN`
- Better categories for external errors, HTML error pages, recovered tool errors and false success claims
- Secondary causes: primary + capped, deduplicated secondary causes in JSON, CLI, HTML report, fix prompts and exported tests
- Weak or passive fix proposals (inform-only, ask-a-person) are rejected; fixes can't claim to resolve a remaining secondary failure
- Verify records which patch actually ran (kind, operation, digest); VERIFIED reflects trial outcomes under that patch, not causal proof of a wording

### RAG
- `crashlab.rag` oracle helpers: wrong retrieval, missing context, stale documents, citation mismatch, numeric / claim grounding (heuristic; no vector-DB integrations). See `docs/rag-testing.md`

### Reliability
- Import-time guard: agents started at module import no longer recurse or record during `check` / `replay` / `export-test`; clear `__main__` hint, clean errors on import failure
- Secret-redaction fixes (JSON output, warnings, import errors)
- Generator tools, timeout replay (incl. Python 3.10), parallel tool calls, `export-test last` after replay
- Crashing checks surface as warnings; storage errors don't crash your agent
- Reads 0.0.2 recordings (compat fixtures) and the compact cassette format; writing compact is experimental opt-in (`CRASHLAB_EXPERIMENTAL_COMPACT=1`), exports stay in the full format

## 0.0.2 (pre-alpha)

Second public cut. Published to PyPI: `pip install crashlab`.

### Record / replay
- Generator and async-generator agents; replay under trio / non-asyncio loops
- Stream fidelity: OpenAI chat / Responses and Anthropic keep recorded chunk boundaries and usage
- `responses.retrieve`, `embeddings.create`, Anthropic `messages.count_tokens`
- Unfinished recordings replay up to where they end; rebuilt SDK errors keep the provider body
- `@crashlab.agent(env=[...])` records non-secret env the agent reads and restores it on replay/fork
- Tool results keep richer Python types on replay (tuples/sets/datetimes/Decimals/UUIDs/paths/enums; NaN/inf)

### Diagnose / fix / verify
- Restore proposals from past good runs; drop ask-a-person patches; fix prompt v5
- `fix --verify` tries the next fix after REJECTED (`--attempts`); `--patch` accepts a fix id
- Provider setup errors (401/403/404, unknown model) and short 429s handled honestly in verify/diagnose
- Heuristics name provider-rejected LLM calls (e.g. "Tool choice is required")

### Runtime checks / CLI / docs
- New checks: `tool_result_instructions`, `unhandled_tool_error`, `failed_side_effect_claim`; stronger `step_repetition`
- CLI next-step hints when a failing run was diagnosed as no failure (try a stronger model / `--heuristic`)
- Framework wrap one-liners in the README quickstart, linking to `examples/integrations/`
- CI on Python 3.10–3.14; redaction ~15× faster on large strings

## 0.0.1 (pre-alpha)

First public-shaped cut. Published to PyPI: `pip install crashlab`.

### Record / replay
- `crashlab.wrap` for OpenAI, Anthropic, LiteLLM (sync + async + streaming assembled)
- OpenAI `chat.completions.with_raw_response.create` and `responses.with_raw_response.create` recorded (CrewAI path)
- `crashlab.wrap` accepts responses-only OpenAI-compatible clients (no chat.completions)
- Anthropic `messages.with_raw_response.create` recorded
- `@crashlab.agent`, `@crashlab.tool`, `@crashlab.check`
- Strict offline replay and fork-at-step with typed patches
- Thread fallback: unbound worker threads attach to the only live run (smolagents CodeAgent)

### Diagnose / fix / verify
- Heuristic + LLM diagnosis (prompt `diagnose-v3`) with verbatim evidence
- Fix proposals (`fix-v2`) with typed patches; `system_prompt.when` scopes multi-agent edits
- `crashlab verify` (k patched vs control + regression); Wilson CIs; VERIFIED / LIKELY /
  INCONCLUSIVE / REJECTED; errored provider trials excluded; thin arms → INCONCLUSIVE

### Export / report / bench
- `export-test` + pytest plugin (`--crashlab`)
- Self-contained HTML report
- Seed fault-injection bench (`crashlab bench`) with offline rescoring
- Export scrub of cwd / install paths from cassette stack traces (editable `$CRASHLAB_REPO` vs wheel `$SITE_PACKAGES`)

### Examples / packaging / hygiene
- `sql_analyst`: recover answer from Groq gpt-oss `tool_use_failed` / `output_parse_failed` / `failed_generation` (and empty-reply nudge)
- `scripts/check.sh` / `scripts/scan.sh` for local pre-push (key-shaped suffixes only)
- SECURITY.md + CONTRIBUTING secret-scan notes
- `py.typed` shipped (`Typing :: Typed`); sdist includes docs/, CHANGELOG, scripts/
- Redactor value patterns include Groq `gsk_` and DashScope-shaped `AQ.Ab8…`

### Evidence (Day 10–11)
- Post-sqlv4 Groq clean smoke: **3/3 pass** after failed_generation recovery
- Seed bench on held-out `sql-analyst` (Gemini flash-lite): LLM label ok 6/6 triggered
  (CI 61–100%), small n — see `bench/README.md`
- Groq sqlv4 (after agent nudges): LLM label ok 3/11; 1/3 clean pass; 0/3 fixes verified
- Real bug in huggingface/smolagents text-to-sql (empty `:memory:` SQLite across threads);
  StaticPool fix verified with no-LLM A/B — `docs/case-studies/`
- OpenAI Agents SDK customer_service: genuine triage follow-up failure; two prompt fixes
  correctly REJECTED for regression

### Not in 0.0.1
- LangGraph/CrewAI first-class adapters, MCP proxy, hosted UI (PyPI publish landed with 0.0.1)
