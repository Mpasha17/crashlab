# Day 11 report (for Mahaboob)

**Window:** 2026-10-22 → 2026-10-23 ~11:42 IST (continuing toward 14:00)  
**HEAD at write time:** see `git log -1` on main

## Projects tried

| Project | Why | Result |
|---|---|---|
| **huggingface/smolagents** `examples/text_to_sql.py` | High stars, tools, clear oracle | **Flagship:** empty `:memory:` SQLite across CodeAgent threads. StaticPool fix verified no-LLM. DRAFT issue ready, not filed. |
| **openai/openai-agents-python** `examples/customer_service` | Official multi-agent handoff example | Genuine triage follow-up failure on qwen. Two prompt fixes **REJECTED** (regression). Do not file yet. |
| **CrewAI** (tiny custom crew) | Third framework | No logic bug; found crashlab gap (`with_raw_response`). Fixed; 3/3 live scenarios pass. |
| LlamaIndex | Planned third | Deferred: openai 2.x vs agents 3.x pin in shared venv |

## Genuine failures (evidence)

1. **smolagents empty DB** — `docs/case-studies/smolagents-text-to-sql.md`, demo GIF, `DRAFT_ISSUE.md`
2. **OpenAI Agents triage invents FAQ answers** — `docs/case-studies/openai-agents-customer-service.md`; verify REJECTED ×2
3. **crashlab missed CrewAI LLM calls** — fixed in `feac59e`; `docs/case-studies/crewai-integration.md`

## Verify verdicts
- smolagents: verified via no-LLM A/B (code fix, not prompt patch)
- openai-agents fix-triage: **REJECTED** (2/5 vs 0/5, bags regression)
- openai-agents wifi-only: **REJECTED** (0/3, 2 regression fails)
- No fake VERIFIED

## Bench (sqlv4 Groq test split)
- 15/15 episodes; LLM label ok **3/11 (10–57%)**; fix VERIFIED 0/3
- Clean smoke after `failed_generation` recovery: **3/3 pass**
- Details: `bench/results/seed_groq-gpt-oss-120b-sqlv4.md`

## Integration bugs fixed in crashlab
- `system_prompt.when` (multi-agent)
- verify INCONCLUSIVE for thin arms
- thread fallback (smolagents CodeAgent)
- AsyncOpenAI close note across verify trials
- export path scrub (cwd/install)
- sql_analyst empty-reply + failed_generation recovery
- **`with_raw_response.create` recording (CrewAI)**

## Launch-ready vs open
**Ready:** flagship story + demo + drafts in `docs/launch/` + CHANGELOG + SECURITY + integrations + CrewAI case study  
**Open:** Mahaboob review of copy; decide on filing smolagents issue; no openai-agents issue until non-regressing fix; confirm CI green on soft-launch HEAD; no PyPI

## Case study / draft paths
- `docs/case-studies/smolagents-text-to-sql.md` + `smolagents-DRAFT_ISSUE.md`
- `docs/case-studies/openai-agents-customer-service.md` + `openai-agents-DRAFT_ISSUE.md`
- `docs/case-studies/crewai-integration.md`
- `/workspace/real-agents/*/CASE_STUDY.md` (local notes)
