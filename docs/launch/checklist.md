# Pre-launch checklist (Thu 29 Oct 17:30 IST)

_Last checked: 2026-10-28 20:05 Asia/Calcutta_

## Must
- [x] CI green on origin/main `5a2028f` (3.10–3.13 + lint,
  [run 40706038326](https://github.com/Mpasha17/crashlab/actions/runs/40706038326)) — re-check if HEAD moves
- [x] README offline quickstart works from a **clean** venv with no keys
  (`docs/demo/quickstart-offline.sh`: import 5 demos, `check --all` exits 1 on expected
  policy fail, `replay` of policy run reproduces). Verified 2026-10-23 ~11:43 IST; re-verified
  2026-10-28 ~20:00 IST with `pip install git+https://github.com/Mpasha17/crashlab`.
- [x] Flagship case study linked from README (`docs/case-studies/smolagents-text-to-sql.md`)
- [x] Demo opens: `docs/demo/terminal.html`, GIF, and `demoscript.sh`
- [x] Secret-scan green on this push (`./scripts/scan.sh`); re-run before every push
- [x] smolagents issue: **not filed** (duplicate of open PR #2475)
- [x] PR #2475: commented with repro: https://github.com/huggingface/smolagents/pull/2475#issuecomment-6472807342
- [ ] Decide: file openai-agents triage follow-up — default **no** until non-regressing fix
- [x] LICENSE (Apache-2.0) + CONTRIBUTING + SECURITY.md present
- [ ] Tag not required for soft launch; do **not** publish to PyPI yet

## Should
- [x] smolagents empty-`:memory:` fix verified (no-LLM A/B + demo) — flagship; known bug, open PR #2475
- [ ] OpenAI Agents triage: two prompt fixes **REJECTED**; needs non-regressing remedy before filing
- [x] policy_drop SystemPromptPatch **VERIFIED** on Groq 120b, Groq 20b, Gemini flash-lite (3/3 vs 0/3)
- [x] misleading_desc **VERIFIED** on Groq 20b + Gemini; 120b verify **INCONCLUSIVE** (TPD)
- [x] html_502 fix-1 **LIKELY** only (3/3 vs control 1/3) — **not** VERIFIED
- [x] Groq sqlv4 complete (15/15): label 3/11, clean 1/3 during run; post-recovery clean smoke **3/3**
- [x] Integration notes linked: openai-agents, smolagents, crewai (`examples/integrations/`)
- [ ] Show HN + X / LinkedIn drafts — Mahaboob review ([REVIEW.md](REVIEW.md))

## Nice
- [x] Demo GIF: `docs/demo/smolagents-text-to-sql.gif` (+ terminal.html / demoscript)
- [x] Second real-agent case study: CrewAI (`docs/case-studies/crewai-integration.md`)
- [ ] `pip install crashlab` from a test index (still no public PyPI)
