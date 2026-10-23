# Launch readiness (updated live)

**Target:** Thu 29 Oct 17:30 IST · **Now building continuously** · Not on PyPI

## Ready
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
- [ ] Optional: re-run 2–3 Groq sql-analyst **clean** episodes with tool_use_failed nudge
  to thicken the false-alarm denominator (currently n=1)
- [ ] No PyPI; no force-push; secret-scan still required on every push

## Do not claim
- "VERIFIED fix on OpenAI Agents SDK" — false (REJECTED)
- Groq sqlv1 test-split numbers — invalid (empty-reply)
- Large-n statistical superiority of any model — intervals are wide
