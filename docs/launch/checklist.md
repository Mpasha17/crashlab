# Pre-launch checklist (Thu 29 Oct 17:30 IST)

_Last checked: 2026-10-23 12:47 Asia/Calcutta_

## Must
- [ ] CI green on origin/main (3.10–3.13) — recent greens `020a467` / `9f12d8f`; tip may still be in flight; re-check soft-launch HEAD
- [x] README offline quickstart works from a **clean** venv with no keys
  (`docs/demo/quickstart-offline.sh`: import 5 demos, `check --all` exits 1 on expected
  policy fail, `replay` of policy run reproduces). Verified 2026-10-23 ~11:43 IST.
- [x] Flagship case study linked from README (`docs/case-studies/smolagents-text-to-sql.md`)
- [x] Demo opens: `docs/demo/terminal.html`, GIF, and `demoscript.sh`
- [x] Secret-scan clean on tip `a136809` (`./scripts/scan.sh` green at 2026-10-23 12:47 Asia/Calcutta); re-run before every push
- [ ] Decide: file smolagents DRAFT_ISSUE.md upstream or not (user decision)
- [ ] Decide: file openai-agents triage follow-up issue or not (user decision — default **no**
  until a non-regressing fix exists)
- [x] LICENSE (Apache-2.0) + CONTRIBUTING + SECURITY.md present
- [ ] Tag not required for soft launch; do **not** publish to PyPI yet

## Should
- [x] smolagents empty-`:memory:` fix verified (no-LLM A/B + demo)
- [ ] OpenAI Agents triage: two prompt fixes **REJECTED**; needs a non-regressing remedy before filing
- [x] refund-bot policy_drop SystemPromptPatch **VERIFIED** (3/3 vs control 0/3 Groq) — see case-studies
- [x] refund-bot html_502 fix-1 **LIKELY** (3/3 vs control 1/3) — not VERIFIED
- [x] refund-bot misleading_desc **VERIFIED** on gpt-oss-20b (3/3 vs 0/3); 120b INCONCLUSIVE (TPD)
- [x] Groq sqlv4 complete (15/15): label 3/11, clean 1/3 during run; post-recovery clean smoke **3/3**
- [x] Integration notes linked: openai-agents, smolagents, crewai (`examples/integrations/`)
- [ ] Show HN + X / LinkedIn drafts in `docs/launch/` — Mahaboob review still needed

## Nice
- [x] Demo GIF: `docs/demo/smolagents-text-to-sql.gif` (+ terminal.html / demoscript)
- [x] Second real-agent case study: CrewAI (`docs/case-studies/crewai-integration.md`)
- [ ] `pip install crashlab` from a test index (still no public PyPI)
