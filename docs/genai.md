# Native Google GenAI (`google.genai`)

`crashlab.wrap(client)` records a `google.genai.Client` (the `google-genai` SDK) the
same way it records OpenAI and Anthropic clients. The SDK is **not** a crashlab
dependency: wrap detects the client by shape (`models.generate_content`), and a
replayed response is a lightweight object so `response.text`,
`response.function_calls` and `candidates[0].content.parts` still work when
`google-genai` is not installed.

Intercepted:

- `client.models.generate_content(...)` and `client.models.generate_content_stream(...)`
- `client.chats.create(...)`, then `chat.send_message(...)` / `chat.send_message_stream(...)`
- the async API on `client.aio` (`aio.models.*`, `aio.chats.create`)

Each call is a `chat` step with `gen_ai.provider.name == "google_genai"`. Requests
normalize `model`, `contents` (strings, `Content` / `Part` dicts or objects,
multi-turn history) and `config` (`system_instruction`, `temperature`,
`max_output_tokens`, `tools` / `function_declarations`). Responses record output
text, `function_calls`, `usage_metadata` (prompt / candidate / total / cached token
counts), `finish_reason`, and a USD estimate from `pricing.py`.

```python
import crashlab
from google import genai
from google.genai import types

client = crashlab.wrap(genai.Client())


@crashlab.agent(name="gemini-bot")
def answer(prompt: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction="Be concise.",
            temperature=0.2,
            max_output_tokens=256,
        ),
    )
    return response.text
```

Streaming and chat sessions are recorded too. History is included on every
`send_message`, so a later turn still replays offline.

```python
import crashlab


@crashlab.agent(name="gemini-chat")
def chat_once(prompt: str) -> str:
    session = client.chats.create(model="gemini-2.5-flash")
    first = session.send_message(prompt)
    pieces = [chunk.text or "" for chunk in session.send_message_stream("And then?")]
    return (first.text or "") + "".join(pieces)
```

Async code uses the same wrap: call `client.aio.models.generate_content` or
`await client.aio.chats.create(...)`.

```python
import crashlab


@crashlab.agent(name="gemini-async")
async def answer_async(prompt: str) -> str:
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text
```

## Replay, forks, and faults

A recorded run replays from the cassette with no network and no SDK:

```python
import crashlab

recorded = crashlab.load("last")
crashlab.replay(recorded)
branch = crashlab.fork(recorded, at_step=0)
```

`crashlab.faults(...)` applies to live Google GenAI calls (`llm_error` raises
before the request; `llm_empty`, `llm_truncate`, and `llm_bad_tool_call` edit the
response the agent sees). A strict replay of that run serves the recorded fault
and does not re-inject it.

```python
import crashlab
from crashlab.faults import llm_empty, llm_error, llm_truncate, llm_bad_tool_call

with crashlab.faults(llm_error(status=429, on_calls=[1])):
    answer("hello")

with crashlab.faults(llm_empty(), llm_truncate(max_chars=20), llm_bad_tool_call()):
    answer("hello")
```

## Limits

- Only the methods listed above are recorded. Files, caches, batches, tunings and
  embeddings are forwarded unrecorded.
- Streaming mutation faults are not applied (`llm_error` still raises before the
  request), the same as for OpenAI and Anthropic.
- `google-generativeai` (the older `genai.GenerativeModel` SDK) is not wrapped;
  use `google.genai.Client` or the provider's OpenAI-compatible endpoint.

```bash
crashlab show last
crashlab replay last
```
