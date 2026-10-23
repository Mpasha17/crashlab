# Launch readiness (updated live)

**Target:** Thu 29 Oct 17:30 IST · **Now building continuously** · Not on PyPI

## Ready
- CrewAI integration: `with_raw_response` recording fixed; 3/3 live scenarios pass
  (`examples/integrations/crewai.md`)
- Flagship story: smolagents empty-`:memory:` bug — case study, DRAFT_ISSUE (not filed),
  demo GIF + terminal.html + demoscript, README section
- Seed bench + sqlv4 Groq test-split documented with CIs and caveats
- OpenAI Agents triage failure documented; both prompt fixes **REJECTED** (honest)
- Launch drafts: announcement, Show HN, one-pager, checklist
- CHANGELOG 0.0.1, CONTRIBUTING, CI green on recent main
- Offline suite: 443 tests (as of c612162)

## Open before soft launch
- [ ] Mahaboob review of `docs/launch/*` copy
- [ ] Decide whether to file smolagents DRAFT_ISSUE upstream
- [ ] Do **not** file openai-agents issue until a non-regressing fix exists
- [x] Groq clean smoke 2026-10-23 10:57 IST: **3/3 pass** after failed_generation recovery
  (`b31ca7e`). sqlv4 clean 1/3 was pre-fix; denominator can be treated as healthy.
- [ ] No PyPI; no force-push; secret-scan still required on every push

## Do not claim
- "VERIFIED fix on OpenAI Agents SDK" — false (REJECTED)
- Groq sqlv1 test-split numbers — invalid (empty-reply)
- Large-n statistical superiority of any model — intervals are wide
