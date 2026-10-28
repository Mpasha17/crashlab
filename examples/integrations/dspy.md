# crashlab with DSPy (ReAct)

Tested with `dspy` 3.4.1, `litellm` 1.106, Gemini `gemini-flash-lite-latest` and Groq
`openai/gpt-oss-20b` (OpenAI-compatible endpoints). Record, replay, diagnose, export-test
and the HTML report work once LLM calls go through LiteLLM.

```python
import crashlab, crashlab.integrations.litellm, dspy
crashlab.integrations.litellm.patch()

@crashlab.agent(name="dspy-shop")
def run_agent(request: str) -> str:
    lm = dspy.LM("openai/" + MODEL, api_base=BASE_URL, api_key=KEY, temperature=0,
                 cache=False,        # a cache hit never reaches the model: forks would not run live
                 engine="litellm")   # REQUIRED: the default engine bypasses LiteLLM
    react = dspy.ReAct(Sig.with_instructions(PROMPT), tools=[get_price, place_order])
    with dspy.context(lm=lm):
        return react(request=request).reply
```

Gotchas

- **DSPy 3.4's default `engine="auto"` uses its native `lm15` HTTP engine, not LiteLLM**, so
  `crashlab.integrations.litellm.patch()` sees nothing: only the tool calls are recorded and
  replay/fork/diagnose miss the model. crashlab now flags such runs with a `no_llm_calls`
  runtime finding. Pass `engine="litellm"`.
- Pass `cache=False`: DSPy's request cache would answer forks and verify trials from disk.
- ReAct does not use native tool calling: the model writes `next_tool_name` / `next_tool_args`
  as text and DSPy calls the Python function, so tool calls appear as `execute_tool` steps
  (from `@crashlab.tool`) and **not** as `tool_calls` on the chat steps. Oracles should use
  `run.tool_calls(name)`. ReAct ends with one extra "extract" LLM call.
- DSPy prompts are long (~1.5-2k tokens per call); Groq free tier hit its 8k TPM limit
  within one run. crashlab recorded the 429s as errored steps.
