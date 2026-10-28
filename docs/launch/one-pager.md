# crashlab one-pager (draft)

**One line.** Crash-test your AI agents: record → replay → diagnose → verify a fix →
export a regression test.

**Who.** People shipping LLM agents (support bots, tool-using ReAct loops, multi-agent
handoffs) who are tired of "it worked in the demo".

**What you do.**
```python
client = crashlab.wrap(OpenAI())          # or Anthropic / LiteLLM
@crashlab.tool()
def search(...): ...
@crashlab.agent(name="support")
def handle(msg: str) -> str: ...
@crashlab.check("support")
def refund_within_policy(run): ...
```
Then: `crashlab ls` · `check` · `diagnose` · `fix` · `verify -k 5` · `export-test` ·
`report --html`.

**Why trust a fix.** `crashlab verify` reruns the failure with the patch, reruns it
without (control), and reruns past good runs (regression). Verdicts: VERIFIED /
LIKELY / INCONCLUSIVE / REJECTED, with Wilson CIs. Errored provider trials (429/5xx)
are not counted as fails.

**Evidence so far.**
- Seed fault-injection bench on a held-out SQL agent (Gemini flash-lite): LLM label
  ok 6/6 triggered (CI 61–100%); heuristic 2/6. Small n — see bench/README.md.
- **Flagship:** known smolagents text-to-sql bug (empty `:memory:` DB across threads;
  open fix PR [#2475](https://github.com/huggingface/smolagents/pull/2475)) reproduced and diagnosed independently; crashlab adds a
  no-LLM repro and the root cause. Not filed as an issue (would duplicate the PR).
- OpenAI Agents customer_service: genuine triage follow-up failure on a smaller open
  model; two prompt patches correctly **REJECTED** (regression / no gain).
- CrewAI path: crashlab now records `with_raw_response.create`; 3/3 live scenarios pass.
- refund-bot **policy_drop** restore-policy: **VERIFIED** on Groq gpt-oss-120b, gpt-oss-20b,
  and Gemini flash-lite (3/3 vs control 0/3 each).
- refund-bot **misleading_desc**: **VERIFIED** on Groq 20b + Gemini flash-lite; Groq 120b
  formal verify **INCONCLUSIVE** (TPD).
- refund-bot **html_502** no-retry patch: **LIKELY** only (3/3 vs control 1/3 — not VERIFIED).
- Groq sql-analyst clean smoke: 3/3 after `failed_generation` / `output_parse_failed` recovery
  (sqlv4 test-split still documented separately with CIs).

**Status.** Pre-alpha (0.0.1). Not on PyPI. Apache-2.0. Works on Python 3.10–3.13.

**Links.** github.com/Mpasha17/crashlab · docs/demo/ · docs/case-studies/ · docs/launch/REVIEW.md
