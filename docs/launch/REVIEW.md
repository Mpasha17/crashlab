# Mahaboob review — soft launch decisions

Drafts only until you say otherwise. Target soft launch: **Thu 29 Oct 17:30 IST**.

## Must decide

- [ ] **Approve launch copy** — `announcement.md` (X ≤280 + LinkedIn), `show-hn.md`, `one-pager.md`
- [ ] **Post Show HN?** When (29 Oct vs later)? Title + body ready in `show-hn.md`
- [ ] **Post X / LinkedIn?** Use ≤280 paste-ready block in `announcement.md` (or alt with REJECTED)
- [ ] **File smolagents upstream issue?** Draft: `docs/case-studies/smolagents-DRAFT_ISSUE.md` — **not filed**
- [ ] **openai-agents upstream?** Default **no** until a non-regressing fix exists (both prompt patches **REJECTED**)
- [x] **CI green** on `5a2028f` (3.10–3.13 + lint, [run 40706038326](https://github.com/Mpasha17/crashlab/actions/runs/40706038326)) — re-check only if HEAD moves before announcing
- [ ] **Still no PyPI** for this soft launch?

## Claims you can make (honest)

- Flagship: smolagents empty-`:memory:` bug; StaticPool fix; no-LLM A/B verified
- OpenAI Agents triage failure real; verify **REJECTED** both prompt patches (that is the point)
- refund-bot **policy_drop** restore-policy: **VERIFIED** on Groq 120b, Groq 20b, Gemini flash-lite (3/3 vs control 0/3)
- refund-bot **misleading_desc**: **VERIFIED** on Groq 20b + Gemini; 120b formal verify **INCONCLUSIVE** (TPD)
- refund-bot **html_502** fix-1: **LIKELY** only (3/3 vs control 1/3; not VERIFIED)
- CrewAI: crashlab `with_raw_response` gap fixed; 3/3 live scenarios pass
- Offline quickstart works from a clean venv (no keys)

## Do not claim

- VERIFIED on OpenAI Agents SDK fix
- html_502 as VERIFIED
- misleading_desc VERIFIED on Groq 120b (TPD / INCONCLUSIVE)
- Large-n statistical superiority / PyPI availability

## Pointers

- Live readiness: [STATUS.md](STATUS.md) · Checklist: [checklist.md](checklist.md)
- Scoreboard: [../case-studies/worker-c-live-scoreboard.md](../case-studies/worker-c-live-scoreboard.md)
