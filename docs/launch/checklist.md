# Pre-launch checklist (Thu 29 Oct 17:30 IST)

## Must
- [ ] CI green on origin/main (3.10–3.13)
- [ ] README quickstart works from a clean venv with no keys (offline path)
- [ ] Flagship case study linked from README (`docs/case-studies/smolagents-text-to-sql.md`)
- [ ] Demo opens: `docs/demo/terminal.html` and `demoscript.sh`
- [ ] Secret-scan clean on HEAD; no keys in history of launch commits
- [ ] Decide: file smolagents DRAFT_ISSUE.md upstream or not (user decision)
- [ ] Decide: file openai-agents triage follow-up issue or not (user decision)
- [ ] LICENSE + CONTRIBUTING present and accurate
- [ ] Tag not required for soft launch; do **not** publish to PyPI yet

## Should
- [ ] One VERIFIED fix on a real OSS agent (openai-agents fix-triage.yaml — blocked on quota earlier)
- [ ] Groq sql-analyst test-split numbers valid after empty-reply fix (quota)
- [ ] Integration notes linked: openai-agents, smolagents
- [ ] Show HN + X drafts reviewed by Mahaboob (`docs/launch/`)

## Nice
- [ ] Short demo GIF/mp4 exported from terminal.html or asciinema
- [ ] Second real-agent case study (CrewAI / LangGraph) if time
- [ ] `pip install crashlab` from a test index (still no public PyPI)
