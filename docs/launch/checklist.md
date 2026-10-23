# Pre-launch checklist (Thu 29 Oct 17:30 IST)

_Last checked: 2026-10-23 11:46 Asia/Calcutta_

## Must
- [ ] CI green on origin/main (3.10–3.13) — last full green `5e8943b`; re-check at soft launch
- [x] README offline quickstart works from a **clean** venv with no keys
  (`docs/demo/quickstart-offline.sh`: import 5 demos, `check --all` exits 1 on expected
  policy fail, `replay` of policy run reproduces). Verified 2026-10-23 ~11:43 IST.
- [x] Flagship case study linked from README (`docs/case-studies/smolagents-text-to-sql.md`)
- [x] Demo opens: `docs/demo/terminal.html`, GIF, and `demoscript.sh`
- [x] Secret-scan clean on tip `32b67e5` (`./scripts/scan.sh` green at 2026-10-23 11:46 Asia/Calcutta); re-run before every push
- [ ] Decide: file smolagents DRAFT_ISSUE.md upstream or not (user decision)
- [ ] Decide: file openai-agents triage follow-up issue or not (user decision — default **no**
  until a non-regressing fix exists)
- [x] LICENSE (Apache-2.0) + CONTRIBUTING + SECURITY.md present
- [ ] Tag not required for soft launch; do **not** publish to PyPI yet

## Should
- [x] smolagents empty-`:memory:` fix verified (no-LLM A/B + demo)
- [ ] OpenAI Agents triage: two prompt fixes **REJECTED**; needs a non-regressing remedy before filing
- [x] Groq sqlv4 complete (15/15): label 3/11, clean 1/3 during run; post-recovery clean smoke **3/3**
- [x] Integration notes linked: openai-agents, smolagents, crewai (`examples/integrations/`)
- [ ] Show HN + X / LinkedIn drafts in `docs/launch/` — Mahaboob review still needed

## Nice
- [x] Demo GIF: `docs/demo/smolagents-text-to-sql.gif` (+ terminal.html / demoscript)
- [x] Second real-agent case study: CrewAI (`docs/case-studies/crewai-integration.md`)
- [ ] `pip install crashlab` from a test index (still no public PyPI)
