# Changelog

## 0.0.1 (pre-alpha, unreleased on PyPI)

First public-shaped cut. Not on PyPI yet.

### Record / replay
- `crashlab.wrap` for OpenAI, Anthropic, LiteLLM (sync + async + streaming assembled)
- OpenAI `chat.completions.with_raw_response.create` recorded (CrewAI path)
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

- `sql_analyst`: recover answer from Groq gpt-oss `tool_use_failed` / `failed_generation` (and empty-reply nudge)

### Evidence (Day 10–11)
- Seed bench on held-out `sql-analyst` (Gemini flash-lite): LLM label ok 6/6 triggered
  (CI 61–100%), small n — see `bench/README.md`
- Groq sqlv4 (after agent nudges): LLM label ok 3/11; 1/3 clean pass; 0/3 fixes verified
- Real bug in huggingface/smolagents text-to-sql (empty `:memory:` SQLite across threads);
  StaticPool fix verified with no-LLM A/B — `docs/case-studies/`
- OpenAI Agents SDK customer_service: genuine triage follow-up failure; two prompt fixes
  correctly REJECTED for regression

### Not in 0.0.1
- PyPI publish, LangGraph/CrewAI first-class adapters, MCP proxy, hosted UI
