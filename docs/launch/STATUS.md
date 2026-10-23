**Last updated:** 2026-10-23 11:45 Asia/Calcutta

# Launch readiness (updated live)

**Target:** Thu 29 Oct 17:30 IST · **Now building continuously** · Not on PyPI

## Ready
- CrewAI integration: `with_raw_response` recording fixed; 3/3 live scenarios pass
  (`examples/integrations/crewai.md`)
- Flagship story: smolagents empty-`:memory:` bug — case study, DRAFT_ISSUE (not filed),
  demo GIF + terminal.html + demoscript, README section
- Seed bench + sqlv4 Groq test-split documented with CIs and caveats
- OpenAI Agents triage failure documented; both prompt fixes **REJECTED** (honest)
- Launch drafts: announcement, Show HN, one-pager, checklist, STATUS, Day 11 report
- Clean-venv offline quickstart verified (`docs/demo/quickstart-offline.sh`)
- CHANGELOG 0.0.1, CONTRIBUTING, SECURITY.md, LICENSE (Apache-2.0)
- CI: last fully green on `5e8943b`; tip `9631a6a`; confirm CI green before soft launch
- Offline suite: 450 tests (see `pytest` locally; CI 3.10–3.13)

Full Day 11 narrative: [DAY11_REPORT.md](DAY11_REPORT.md).

## Open before soft launch
- [ ] Mahaboob review of `docs/launch/*` copy (announcement, Show HN, one-pager)
- [ ] Decide whether to file smolagents DRAFT_ISSUE upstream
- [ ] Do **not** file openai-agents issue until a non-regressing fix exists
- [x] Groq clean smoke 2026-10-23 10:57 IST: **3/3 pass** after failed_generation recovery
  (`b31ca7e`). sqlv4 clean 1/3 was pre-fix; denominator can be treated as healthy.
- [ ] Confirm CI green on the soft-launch HEAD (3.10–3.13)
- [x] Worker C live 2026-10-23 ~11:41–11:45 IST Asia/Calcutta (Groq):
  - refund_bot clean A17: **pass** (4/4 checks) `01M5KP4VCRTZ1D4DYRE7B6QQT7`
  - sql_analyst clean smoke: **3/3 pass** (gpt-oss-120b)
  - CrewAI demo (gpt-oss-20b): **3/3 pass** (crashlab/sqlite/invent); steps recorded
  - policy_drop C03 parent fail → SystemPromptPatch fork: patched **2/2 pass**, control **2/2 fail**
    (parent `01M5KPB26YGDFPFNR9C4M1A3V0`). Anecdote (k=2); **not** labeled VERIFIED.
  - Quota after smokes: gpt-oss-120b **747/1000** req remaining (reset 6h4m19.2s); Gemini flash alive (no RPM headers via OpenAI-compat).
- [ ] No PyPI; no force-push; secret-scan still required on every push

## Do not claim
- "VERIFIED fix on OpenAI Agents SDK" — false (REJECTED)
- Groq sqlv1 test-split numbers — invalid (empty-reply)
- Large-n statistical superiority of any model — intervals are wide
- That `with_raw_response` is still unintercepted — false (`create` path is recorded)
