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
- Real bug in smolagents text-to-sql (empty `:memory:` DB across threads). Fix
  verified with a no-LLM A/B. Upstream issue not filed yet.
- Groq sql-analyst clean smoke: 3/3 pass after recovering gpt-oss `failed_generation` answers.

**Status.** Pre-alpha. Not on PyPI. Apache? check LICENSE. Works on Python 3.10–3.13.

**Links.** github.com/Mpasha17/crashlab · docs/demo/ · docs/case-studies/
