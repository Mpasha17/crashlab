# Zero-config auto-instrumentation

Frameworks such as AutoGen, Haystack, CrewAI, LlamaIndex, PydanticAI and the
OpenAI Agents SDK construct an OpenAI, Anthropic, LiteLLM or Google GenAI
client *inside* their own classes — often during `warm_up()` or `run()`, and
often behind a private `_client`. `crashlab.wrap(client)` cannot see that
constructor. `crashlab.instrument()` can: it patches the provider class so
every client created afterwards is recorded automatically.

Optional SDKs are never imported until you call `instrument()`. A missing
`openai`, `anthropic`, `litellm` or `google-genai` package is skipped, not an
error. `import crashlab` itself stays lightweight.

## `crashlab.instrument()`

Call it once at process start, before the framework builds its client:

```python
import crashlab

crashlab.instrument()

@crashlab.agent(name="shop-bot")
def run_agent(msg: str) -> str:
    # Haystack (and similar) construct openai.OpenAI() inside warm_up().
    # There is no wrap() and no gen.client = ... assignment.
    agent = build_framework_agent()
    return agent.run(msg)
```

Or as a context manager, which restores the original constructors on exit:

```python
import crashlab

with crashlab.instrument():
    run_agent("what is the price of A17?")

run = crashlab.load("last")
print(len(run.chat_steps()))
```

Nested blocks are reference-counted: an inner `with crashlab.instrument():`
exiting does not unpatch an outer one. `handle = crashlab.instrument()` and
`handle.close()` are the same as the context manager.

### Providers

By default every supported provider is requested:

```python
import crashlab

crashlab.instrument(providers=["openai", "anthropic"])
print(crashlab.instrumented())
crashlab.uninstrument(providers=["openai"])
```

Supported names are `openai`, `anthropic`, `litellm` and `google_genai`.
`instrumented()` returns a `dict[str, bool]` for all four, including ones you
did not ask for (they stay `False`).

`openai` patches `OpenAI` and `AsyncOpenAI` (and `AzureOpenAI` /
`AsyncAzureOpenAI` when the SDK defines them). `anthropic` patches `Anthropic`
and `AsyncAnthropic`. `google_genai` patches `google.genai.Client`. `litellm`
is the same hook as `crashlab.integrations.litellm.patch()`:
`litellm.completion` / `litellm.acompletion`.

### Agent labels

`agent=` names every client constructed while that `instrument()` call is
active, the same way `crashlab.wrap(client, agent=...)` does. If you omit it,
an active `crashlab.agent_scope(...)` still labels the recorded steps:

```python
import crashlab

crashlab.instrument(agent="researcher")

with crashlab.instrument():
    with crashlab.agent_scope("writer"):
        run_writer()
```

`store=` is accepted and kept on the returned handle. Recording still starts
with `crashlab.record(...)` or `@crashlab.agent`, which choose the store
(`CRASHLAB_DIR` or `./.crashlab`) as usual.

## `crashlab.wrap()` together with `instrument()`

Wrapping an already auto-instrumented client is a no-op: you get the same
client back, and a call is recorded once, not twice.

```python
import crashlab
from openai import OpenAI

with crashlab.instrument():
    client = OpenAI()
    wrapped = crashlab.wrap(client)
    with crashlab.record("once"):
        wrapped.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )
```

Clients constructed *before* `instrument()` are not retroactively patched;
`crashlab.wrap(client)` still works for those.

## `crashlab.uninstrument()`

```python
import crashlab

crashlab.instrument()
crashlab.uninstrument()
```

Restores the original `__init__` (and LiteLLM hooks) when the last matching
`instrument()` call has been closed. Clients already constructed stay
recorded: only *new* constructors go back to being unpatched. `uninstrument()`
is safe to call when nothing is instrumented.

## Limits

- `from openai import OpenAI` *before* `instrument()`, then `OpenAI()`, is
  still patched: the hook mutates the class object, which is the same one the
  alias points at. `from litellm import completion` *before* `instrument()`
  is not: that alias is the original function. Call `litellm.completion(...)`
  or patch first (the same limit as `crashlab.integrations.litellm.patch()`).
- A framework that talks HTTP without going through one of the supported
  client classes is not captured; wrap the client it *does* expose, or use
  `CrashlabCallbackHandler` for LangChain / LangGraph.
- Auto-instrumentation records into the active `crashlab.record()` /
  `@crashlab.agent` run. It does not start a run on its own.
