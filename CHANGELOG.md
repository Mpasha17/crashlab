# Changelog

## Unreleased

### Debugging many runs (see `docs/debugging.md`)
- `crashlab search` / `crashlab.search.search_runs`: offline text search across runs (messages, tool names / args / results, errors, models, checks, findings, diagnosis labels, run ids) with `--tool`, `--model`, `--check`, `--label`, `--errors`, `--since`, `--run`, `--in`, regex and JSON output; exits 1 when nothing matched
- `crashlab cluster` / `crashlab.cluster.cluster_runs`: group failed runs by a conservative signature (agent, normalized exception, first failing step, failing checks, diagnosis label, serious findings); `--by` to loosen, stable cluster ids, JSON output
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

- Diagnose: a `@crashlab.check` that crashed on a bug in its own code (`'Step' object has no attribute ...`, `NameError`, a missing import) is no longer blamed on the agent; the heuristic verdict is UNKNOWN with "the check itself is broken", and the LLM diagnoser's view marks it

### Multi-agent
- Steps record which agent made them (`gen_ai.agent.name`, `step.agent`): `crashlab.wrap(client, agent=...)`, `crashlab.agent_scope(name)`, nested `@crashlab.agent`; tool steps inherit the requesting chat step's agent. `run.agents`, `run.tool_calls(..., agent=)`, `run.chat_steps(agent=)`; `crashlab show` prints agents. Labels only; replay unchanged. See `docs/multi-agent.md`
- Fix: `step_repetition` judges reply / request repetition per agent in labelled runs (a writer repeating a researcher's answer is not a loop)
- Fix: a tool taking PydanticAI's `RunContext` (or any dataclass holding a lock / client) crashed the agent with `TypeError: cannot pickle '_thread.RLock'`; context objects are no longer recorded, dataclass args are converted without deep-copying, and an argument that cannot be converted is recorded as a placeholder. `@crashlab.tool(ignore_args=...)`
- Fix: replaying a run with a sub-agent used as a tool dropped the sub-agent's steps, so oracles on them passed on the recording and failed on the replay / exported test; the served tool's recorded inner steps are now copied into the replay
- Fix: concurrent sub-agents in one run no longer lose tool-call ids; `describe_step` / diagnose blame the chat step that requested a tool call (`run.caller_chat`), not just the nearest one

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
