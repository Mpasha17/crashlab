**Last updated:** 2026-10-23 12:39 Asia/Calcutta

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
- CI: greens include `320be73`, `020a467`, `9f12d8f`; tip `9ea84ad`; intermediate fail `d8ccd2e` (responses-only test) later superseded — confirm soft-launch HEAD green
- Offline suite: 450 tests (see `pytest` locally; CI 3.10–3.13)

Full Day 11 narrative: [DAY11_REPORT.md](DAY11_REPORT.md).

## Open before soft launch
- [ ] Mahaboob review of `docs/launch/*` copy (announcement, Show HN, one-pager)
- [ ] Decide whether to file smolagents DRAFT_ISSUE upstream
- [ ] Do **not** file openai-agents issue until a non-regressing fix exists
- [x] Groq clean smoke 2026-10-23 10:57 IST: **3/3 pass** after failed_generation recovery
  (`b31ca7e`). sqlv4 clean 1/3 was pre-fix; denominator can be treated as healthy.
- [ ] Confirm CI green on the soft-launch HEAD (3.10–3.13)
- [x] Worker C live 2026-10-23 ~11:41–12:39 IST Asia/Calcutta:
  - Groq refund_bot clean A17: **pass** (4/4) `01M5KP4VCRTZ1D4DYRE7B6QQT7`
  - Groq sql_analyst clean: **3/3 pass** (gpt-oss-120b)
  - CrewAI demo (gpt-oss-20b): **3/3 pass**; steps recorded
  - policy_drop → SystemPromptPatch: **VERIFIED** patched **3/3** control **0/3**
    (parent `01M5KPB26YGDFPFNR9C4M1A3V0`; earlier k=2 was anecdote). See
    `docs/case-studies/refund-policy-fork-note.md`.
  - Gemini `flash-lite-latest`: refund clean A17 **pass**; C03 full-policy refuse **pass**;
    sql_analyst clean **3/3** after one Q2 free-tier 429 retry (first pass 2/3).
  - Groq html_502 A17 loop: parent fail (step budget); fix-1 **LIKELY** patched 3/3 control 1/3
    (+ 1 clean regression pass). See `docs/case-studies/refund-html502-verify-note.md`.
  - Groq schema_drift A17: **pass** (refunded correct $49; oracle green).
  - Groq misleading_desc: parent fail ($44.10); 120b `fix --verify` **INCONCLUSIVE** (TPD 429s);
    20b manual k=3+control+regression: **VERIFIED** patched 3/3 control 0/3 (fix from 120b diagnoser).
    See `docs/case-studies/refund-misleading-desc-note.md`.
  - Groq html_502 on **gpt-oss-20b** ×3: all **pass** (stops); 120b TPD nearly exhausted (~199k/200k) — pause 120b.
  - CrewAI recheck ~12:02 IST: still **3/3 pass**.
  - Groq sql_analyst on **gpt-oss-20b**: **0/3** — Groq `output_parse_failed` (recovery only handles
    `tool_use_failed`; report for Worker A). Gemini lite refund clean recheck **pass**.
  - Groq sql_analyst **gpt-oss-120b** reconfirm ~12:17 IST: Q0 **pass**; Q1–Q2 **TPD 429** (not agent fail; earlier 3/3 stands).
  - Scoreboard: `docs/case-studies/worker-c-live-scoreboard.md`.
  - sql_analyst: extended Groq recovery for `output_parse_failed` (tests + live 20b **3/3**).
  - Quota after: gpt-oss-120b **688/1000** req (TPD rolling; was near cap earlier).
- [ ] No PyPI; no force-push; secret-scan still required on every push

## Do not claim
- "VERIFIED fix on OpenAI Agents SDK" — false (REJECTED)
- Groq sqlv1 test-split numbers — invalid (empty-reply)
- Large-n statistical superiority of any model — intervals are wide
- That `with_raw_response` is still unintercepted — false (`create` path is recorded)
- policy_drop fork **k=2 alone** as VERIFIED — false; k=3+control **is** VERIFIED (see case-study note)
- misleading_desc **120b** formal verify as VERIFIED — false (INCONCLUSIVE/TPD); **20b** k=3+control **is** VERIFIED
- html_502 fix-1 as VERIFIED — false (**LIKELY**; CIs overlap)
