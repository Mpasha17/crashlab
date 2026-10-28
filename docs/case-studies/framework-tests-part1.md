# Live framework tests, part 1 (28 Oct 2026)

crashlab 0.0.1 from PyPI, used as a real user would, on seven agent stacks: LangGraph, PydanticAI, LlamaIndex, smolagents ToolCallingAgent, OpenAI Agents SDK, CrewAI and a raw LiteLLM loop. Each table shows the result on the published 0.0.1 and on main after the fixes (commit ids in the notes). Integration guides: [examples/integrations/](../../examples/integrations/README.md).


Setup: a fresh venv per framework, `crashlab` from PyPI
(0.0.1) + framework at current release; Python 3.12. Agent model: Groq
`openai/gpt-oss-20b` (diagnose/fix model: Gemini `gemini-flash-lite-latest` via its
OpenAI-compatible endpoint). Scenario: a small shop assistant with tools
`get_price`, `place_order` (side_effects=True) and the rule "never order more than 5
units; order 5 and tell the limit". Faults are part of the entry function's input
(`faults=["drop_rule"]`), so forks/regressions see exactly the recorded ones.
Offline stages run with `env -i` (no keys) inside `unshare -rn` (no network).
After a bug fix, the venv is switched to the local checkout (`uv pip install -e`) and the
stage is re-run; "0.0.1" below means the PyPI release as published.

## 1. LangGraph 1.2.18 (langchain-openai, prebuilt create_react_agent)

| stage | 0.0.1 | after fixes | notes |
|---|---|---|---|
| a wrap ≤5 lines | friction | pass | `crashlab.wrap(ChatOpenAI(...))` → TypeError. Works with `oc = crashlab.wrap(OpenAI(...)); ChatOpenAI(..., client=oc.chat.completions, root_client=oc)` — not in README. Tools: `@tool` over `@crashlab.tool()` keeps the schema. |
| b recording complete | pass | pass | 3 chat + 2 tool steps = LangGraph's 3 AIMessage + 2 ToolMessage |
| c show | pass | pass | readable; narrow terminals wrap heavily |
| d offline replay | **FAIL** | pass | `TypeError: 'NoneType' object is not callable` at `raw_response.parse()` (replay returned no raw-response wrapper), then `ValueError: Unexpected response type ... got ReplayObject` (Groq `service_tier="on_demand"` fails ChatCompletion.model_validate). Fixed 51c603b. Also: entry fn must not require keys (`os.environ["GROQ_API_KEY"]` → KeyError on replay) |
| e failing run | pass | pass | drop_rule → ordered 10 (2/2 runs) |
| f diagnose | partial | partial | TASK_SPEC_VIOLATION, conf 1.0, culprit = step 3 (the tool call) not step 2 (the LLM decision); root cause = symptom ("ordered 10") not cause (rule missing from prompt) |
| g fix + verify -k 3 | **FAIL (wrong verdict)** | pass | fix-1 appends the limit. 0.0.1: REJECTED/INCONCLUSIVE-worthy runs graded as fail because the correct behaviour (`place_order(qty=5)`) is a side-effect call the parent never made → SideEffectBlockedError → counted as fail. After 5c65879: without flag INCONCLUSIVE (3 blocked, hint); with `--allow-live-side-effects`: **VERIFIED** 3/3 vs control 0/3, 2/2 regressions pass |
| h export-test + pytest --crashlab | pass | pass | 2 passed offline (no keys, no network) |
| i report --html | pass | pass | 8.8 KB, no scripts |

Follow-up found on PydanticAI (applies to every agent): a *regression* rerun whose
side-effect call differed trivially from the recording (`place_order(item="widgets")` vs
`"widget"`) was blocked and counted as "good run now fails" → false REJECTED. Fixed
116c924: a blocked side effect is "unknown" (errored) in every arm.

## 2. PydanticAI (pydantic-ai-slim 2.63.0, OpenAIChatModel + AsyncOpenAI, async + run_stream)

| stage | 0.0.1 | after fixes | notes |
|---|---|---|---|
| a wrap ≤5 lines | pass | pass | `OpenAIChatModel(m, provider=OpenAIProvider(openai_client=crashlab.wrap(AsyncOpenAI(...))))`; `@crashlab.tool()` functions passed as `tools=[...]` keep their schema. 3 lines. |
| b recording complete | pass | pass | 5 steps (chat, get_price, chat, place_order, chat) = pydantic-ai's messages; run_stream also recorded (5 steps) |
| c show | pass | pass | |
| d offline replay | **FAIL** | pass | `UnexpectedModelBehavior: ... expected JSON data` — pydantic-ai checks `isinstance(resp, ChatCompletion)`; Groq `service_tier="on_demand"` made rebuild return a ReplayObject (same bug B as LangGraph, fixed 51c603b). Stream and non-stream replays reproduce. |
| e failing run | pass | pass | drop_rule → `place_order(widgets, 10)` without a price lookup |
| f diagnose | pass | pass | TASK_SPEC_VIOLATION, step 0 (right decision step), conf 1.0; root cause again states the symptom, not "rule missing from prompt" |
| g fix + verify -k 3 | **FAIL (wrong verdict)** | pass | fix-1 appends the rule. 0.0.1: REJECTED (blocked trials = fails, regression blocked on "widgets" ≠ "widget"). Now: INCONCLUSIVE + hint by default; `--allow-live-side-effects`: **VERIFIED** 3/3 vs 0/3, 2/2 regressions pass, $0.001 |
| h export-test | (blocked by d) | pass | 3 passed offline (failing run, fixed run, good run) |
| i report --html | pass | pass | 8.7 KB, no scripts |

Note: with a fix that changes a side-effect call, the default `verify` can never be better
than INCONCLUSIVE (the fork starts at step 0, every patched trial wants a new
`place_order`). Documented in the README verify section (116c924); worth surfacing the
flag in `fix --verify` output too.

## 3. LlamaIndex (llama-index-core 0.14.25, FunctionAgent workflow + OpenAILike, async, streams by default)

| stage | 0.0.1 | after fixes | notes |
|---|---|---|---|
| a wrap ≤5 lines | pass | pass | `OpenAILike(..., async_openai_client=crashlab.wrap(AsyncOpenAI(...)))`; `@crashlab.tool()` functions passed straight to `FunctionAgent(tools=[...])` (schema from the wrapped signature). Not documented in README. |
| b recording complete | pass | pass | FunctionAgent always streams: chat steps recorded as streams; 3–5 steps = agent's tool calls |
| c show | pass | pass | |
| d offline replay | pass | pass | streamed responses replay fine even on 0.0.1 (5/5, identical requests) |
| e failing run | pass | pass | drop_rule → `place_order(widgets, 10)` |
| f diagnose | pass/flaky | pass/flaky | 1st: TASK_SPEC_VIOLATION with the right root cause ("system prompt lacked the limit"). A `--no-cache` re-run was downgraded to UNKNOWN (no verbatim quote) by flash-lite — guardrail works, model is flaky; 2 more runs: TASK_SPEC_VIOLATION step 3 |
| g fix + verify -k 3 | pass (correct REJECT) | pass | fix-1 "Note: the maximum is 5" made the agent ask instead of ordering → REJECTED 0/3, and a regression good run stopped ordering — a correct rejection. Found: the regression note was blank ("fail:") → 95538cc; `fix --no-cache` re-diagnosed and wiped the good diagnosis → f7851fe; a new proposal reused the id fix-1, deleted the old file and showed the old REJECTED next to the new patch → d8ff9df. fix-2 ("inform them and only order up to 5") with `--allow-live-side-effects`: **VERIFIED** 3/3 vs 0/3, 2/2 regressions pass, $0.0011 |
| h export-test | pass | pass | 3 passed offline (252 pydantic deprecation warnings come from llama-index, not crashlab) |
| i report --html | pass | pass | 7.7 KB, no scripts, shows fix-1 REJECTED and fix-2 VERIFIED |

## 4. smolagents 1.26.0 (ToolCallingAgent + OpenAIServerModel; Gemini flash-lite as the agent model)

Groq gpt-oss-20b does not work with ToolCallingAgent: smolagents sends `tool_choice: required`
and the model answers in text → Groq 400 "Tool choice is required, but model did not
call a tool" (recorded fine as an errored run). Streaming (`stream_outputs=True`) fails
on both providers: Gemini sends tool-call deltas without an index (smolagents raises
"Tool call index is not provided"), Groq returns "Parsing failed". Both are upstream /
provider issues, not crashlab — but they make good natural failures.

| stage | 0.0.1 | after fixes | notes |
|---|---|---|---|
| a wrap ≤5 lines | friction | pass | Guide pattern works: `m.client = crashlab.wrap(m.client)`, `tool.forward = crashlab.tool(...)(tool.forward)` (2 + 1 per tool). |
| b recording complete | **FAIL** | pass | Tool args were recorded as `{"args": [], "kwargs": {...}}` (smolagents puts `self` in the staticmethod's `__signature__`), so `step.args["quantity"]` was missing and the `quantity<=5` check **silently passed a run that ordered 10**. Fixed c03bde2. The streaming failure recorded **0 steps** (stream abandoned by a raising consumer was finalized after the run was saved) → fixed a07f1e3 (partial step recorded). |
| c show | pass | pass | |
| d offline replay | partial | pass | non-stream runs 3/3. The stream failure replayed *past* the failure (replay added `index=0` to tool deltas, so smolagents no longer choked) → fixed 1b1ccf0; now "same error as recorded". |
| e failing run | (masked by b) | pass | drop_rule → ordered 10; natural failure: Gemini stream delta without index |
| f diagnose | pass / downgraded | pass | drop_rule: TASK_SPEC_VIOLATION step 0. Natural failure: right root cause (missing tool-call index) but downgraded to UNKNOWN because the quote came from the run's exception, not a step → fixed ca0a916; now EXTERNAL_FAILURE, conf 1.0, quote from run.error |
| g fix + verify -k 3 | (b) | pass | fix-1 appends the limit; `--allow-live-side-effects`: **VERIFIED** 3/3 vs 0/3, 2/2 regressions |
| h export-test | pass | pass | 4 tests offline incl. the stream failure. Gotcha (my harness, worth a README line): anything that changes behaviour must be an argument of the `@crashlab.agent` function — my `STREAM` env var was not recorded, so replaying without it served the recorded stream to a non-streaming agent (crashlab treats `stream=` as transport, by design) and diverged |
| i report --html | pass | pass | 6 KB |

## 5. OpenAI Agents SDK 0.24.0 (single shop agent, `@function_tool` over `@crashlab.tool()`, Runner.run + run_streamed)

Groq 20b hit its 200k tokens/day limit mid-verify (429 → trials "errored", verdict
INCONCLUSIVE — handled correctly); the rest ran with Gemini flash-lite as the agent model.

| stage | 0.0.1 | after fixes | notes |
|---|---|---|---|
| a wrap ≤5 lines | pass | pass | guide pattern (wrapped AsyncOpenAI + OpenAIChatCompletionsModel, close the client). **New:** `@function_tool` stacked over `@crashlab.tool()` keeps the SDK's schema (descriptions from the docstring) and records execute_tool steps — the guide said "untested" |
| b recording complete | pass | pass | 3–5 steps; streamed run recorded |
| c show | pass | pass | |
| d offline replay | **FAIL** (failing run) | pass | 0.0.1: `AttributeError: 'tuple' object has no attribute 'choices'` on the failing Groq run (Groq `service_tier` → ReplayObject, bug B, fixed 51c603b); passing run replayed |
| e failing run | pass | pass | drop_rule → ordered 10 (Groq and Gemini) |
| f diagnose | pass | pass | TASK_SPEC_VIOLATION, step 0 (Gemini) / step 3 (Groq) |
| g fix + verify -k 3 | pass | pass | fix-1 ("inform them of the limit, don't exceed 5") → agent asked instead of ordering → REJECTED 0/3 (correct). `fix --no-cache` → fix-2 (numbered after the verified fix-1, d8ff9df) → **VERIFIED** 3/3 vs 0/3, 2/2 regressions |
| h export-test | pass | pass | 3 passed offline |
| i report --html | pass | pass | 7 KB |

## 6. CrewAI 1.15.32 (one agent + task, `@tool` over `@crashlab.tool()`, Groq gpt-oss-120b; Gemini for the natural failure)

| stage | 0.0.1 | after fixes | notes |
|---|---|---|---|
| a wrap ≤5 lines | pass | pass | guide pattern (`llm._client = crashlab.wrap(llm._client)`, restore `llm.model`). `crewai.tools.tool` over `@crashlab.tool()` keeps the args schema and records execute_tool steps |
| b recording complete | pass | pass | |
| c show | pass | pass | findings are useful: see e |
| d offline replay | **FAIL** | pass | 0.0.1 diverged at step 1 (with_raw_response replay, bug A, fixed 51c603b); now 3/3, 5/5 and 9/9 "same error as recorded" |
| e failing run | pass | pass | drop_rule → ordered 10. **Natural failure with Gemini:** CrewAI drops Gemini's `thought_signature` when it sends the tool call back → 400; CrewAI retries the whole step, so **place_order ran 3 times**. crashlab flagged it (`step_repetition`: same LLM request 3×, `place_order` 3× with the same args). Upstream CrewAI/Gemini issue, same without crashlab |
| f diagnose | pass | pass | natural failure: EXTERNAL_FAILURE step 2, quotes the 400 thought_signature error, secondary STEP_REPETITION. drop_rule: TASK_SPEC_VIOLATION |
| g fix + verify -k 3 | (d) | pass | `--allow-live-side-effects`: **VERIFIED** 3/3 vs 0/3, 2/2 regressions |
| h export-test | pass | pass | 4 offline; the Gemini-recorded test needs `AGENT_PROVIDER=gemini` (model chosen by env in my app — the divergence now says so, 47d3d46) |
| i report --html | pass | pass | 6 KB |

## 7. Raw LiteLLM loop (litellm 1.106.5, `crashlab.wrap(litellm)`, native `gemini/gemini-flash-lite-latest`, acompletion, plain + streamed)

| stage | 0.0.1 | after fixes | notes |
|---|---|---|---|
| a wrap ≤5 lines | pass | pass | `crashlab.wrap(litellm)` + `@crashlab.tool` on the two functions |
| b recording complete | pass | pass | 3–4 steps per run; streamed run recorded (cost shows blank for the streamed call — no usage chunk) |
| c show | pass | pass | |
| d offline replay | pass | pass | 3/3 for plain, streamed and failing runs, no cost-map download needed offline |
| e failing run | pass | pass | drop_rule → ordered 10. Realistic fault `bad_price` (get_price returns an nginx 502 page): the agent ordered anyway and quoted the price from the order total; **all checks passed**, but the `suspicious_tool_result` finding flagged step 1. Natural failure: litellm 1.106.5 raises `No module named 'mcp'` whenever `tools=` is passed (upstream packaging bug; `pip install mcp` fixes it; recorded fine) |
| f diagnose | pass | pass | drop_rule: TASK_SPEC_VIOLATION step 0. bad_price: TOOL_ERROR_MISHANDLED step 1, conf 0.95, "ignored the 502 page and ordered without the price" — right |
| g fix + verify -k 3 | pass (correct REJECT) | pass | fix-1 (0.0.1 fix prompt) made the agent ask instead of order → REJECTED 0/3 (correct). On main (fix-v3 prompt) fix-2 → **VERIFIED** 3/3 vs 0/3, 2/2 regressions (`--allow-live-side-effects`) |
| h export-test | pass | pass | 4 passed offline |
| i report --html | pass | pass | 7 KB |

## LangGraph async + streaming (follow-up)

`AsyncOpenAI` wrapped, `ChatOpenAI(streaming=True, async_client=..., root_async_client=...)`,
`agent.astream(stream_mode="messages")`, Gemini: recorded (3 steps); every run hit a Gemini
400 on the second turn (LangChain drops Gemini's thought_signature; upstream). Replaying
that failure diverged: `'BadRequestError' object has no attribute 'response'` (LangChain
re-wraps provider errors from e.response/e.body; the replayed error had none) → fixed
914f23b, now "same error as recorded". Fix/verify not exercised on this path (no passing
Gemini runs; Groq quota spent).

## Offline sweep on main (22:55 IST)

All exported tests, every agent, `env -i` + `unshare -rn`, editable crashlab at main:
langgraph 2/2, pydanticai 3/3, llamaindex 3/3, openai-agents 3/3, litellm 4/4,
smolagents 3/4 and crewai 3/4 — the two failures are my harness: those runs chose
streaming / the model from an env var that was not set the same way (the replay now says
so for the model case, 47d3d46).

## Fix-quality pattern (not a crashlab bug, but the launch story should say it)

The fixer's first proposal (Gemini flash-lite) was too weak in 4 of 7 agents ("tell the
customer the limit" without "order 5" → the agent asks instead of ordering, the
`placed_an_order` check fails). `verify` REJECTED every one of them correctly; the next
proposal was VERIFIED each time. Diagnoses: right label in 7/7; culprit step sometimes
the tool step instead of the LLM decision; root cause usually describes the symptom
("ordered 10, limit is 5") rather than "the rule is missing from the prompt".

## README / docs friction (for real users)

1. Frameworks that build their own client need a per-framework incantation
   (`ChatOpenAI(client=, root_client=)`, `OpenAIProvider(openai_client=)`,
   `OpenAILike(async_openai_client=)`, `m.client = wrap(m.client)`, `llm._client`);
   `crashlab.wrap(ChatOpenAI(...))` just raises TypeError. Guides now exist for all 7
   (d2edb33), but the README quickstart only shows the raw client.
2. Everything that changes behaviour (model/provider, streaming, faults) must be an
   argument of the `@crashlab.agent` function or set identically at replay; env-driven
   config silently changes replays. Keys must be read with `os.environ.get`.
3. A fix that changes a side-effect call needs `--allow-live-side-effects` (sandboxed
   tools) to be graded; otherwise INCONCLUSIVE. Documented (116c924) but easy to miss.
4. `fix --no-cache` used to re-diagnose (fixed f7851fe); fix ids used to be reused
   (fixed d8ff9df).
5. `crashlab --help` / command help keep hard line breaks from docstrings (ragged wrapping);
   `show` is cramped at 80 columns.
6. Provider quirks hit on the way (not crashlab): Groq gpt-oss + smolagents
   `tool_choice=required` 400; Gemini OpenAI endpoint drops/needs thought_signature
   (CrewAI, LangChain streaming); Gemini stream tool deltas without index; litellm needs
   `mcp`; Gemini latency 30–60 s per call tonight; Groq 20b 200k tokens/day shared.
