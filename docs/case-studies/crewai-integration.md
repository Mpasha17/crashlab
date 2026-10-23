# Case study: CrewAI recording via `with_raw_response`

**Not an upstream CrewAI bug** — a crashlab gap found while running a tiny CrewAI crew
on Groq.

## Symptom
`@crashlab.agent` runs finished with correct answers but **0 recorded steps** and
`usage.tokens = 0`. Oracles that required a `lookup` tool call failed with `saw []`.

## Cause
CrewAI 1.15's OpenAI provider calls
`client.chat.completions.with_raw_response.create(...)`. crashlab only wrapped plain
`chat.completions.create`, so the raw path hit the inner client and never recorded.
When we later wrapped raw without parsing `LegacyAPIResponse`, steps appeared but
responses were empty.

## Fix (in crashlab)
Record `with_raw_response.create` and parse the LegacyAPIResponse for the cassette
while returning the raw object to CrewAI. Offline test:
`tests/test_openai_raw_response.py`.

## Live check (2026-10-23)
Three Groq scenarios (`crashlab`, `sqlite`, `invent`) all **pass** with steps recorded
after the fix. Demo harness: `/workspace/real-agents/crewai-demo/` (not in this repo).
