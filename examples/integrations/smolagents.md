# crashlab with smolagents

Tested with `smolagents` 1.26 (`CodeAgent` and `ToolCallingAgent`, `OpenAIServerModel`) on
Gemini and Groq.

```python
import crashlab
from smolagents import CodeAgent, OpenAIServerModel

@crashlab.agent(name="my-sql-agent")
def ask(question: str, model: str = "gemini-flash-latest") -> str:
    m = OpenAIServerModel(model_id=model, api_base=BASE_URL, api_key=KEY)
    m.client = crashlab.wrap(m.client)            # record every LLM call
    sql_engine.forward = crashlab.tool(name="sql_engine")(sql_engine.forward)  # tool steps
    return str(CodeAgent(tools=[sql_engine], model=m).run(question))
```

Notes:

- **`CodeAgent` runs each code block in a fresh `ThreadPoolExecutor`**, which does not copy
  context variables. Before crashlab 419235a, every tool call made from agent code was
  silently not recorded. crashlab now falls back to the only live run for such unbound
  worker threads (`CRASHLAB_THREAD_FALLBACK=0` turns this off). If you record several runs
  concurrently in one process, bind workers explicitly with `crashlab.bind_context`.
- The same thread hop breaks thread-affine resources. See the smolagents case study:
  `examples/text_to_sql.py` builds its database on `sqlite:///:memory:`, so SQLAlchemy's
  `SingletonThreadPool` gives the code thread a fresh, empty database (`no such table:
  receipts`).
- **gpt-oss on Groq does not suit `CodeAgent`.** The model emits native tool calls while
  `CodeAgent` sends `tool_choice: none`, and Groq answers 400 `tool_use_failed`. Use
  `ToolCallingAgent` or another model.

- **Tool arguments are recorded by name** (after c03bde2). smolagents' `@tool` gives
  `forward` a signature that starts with `self`; crashlab 0.0.1 recorded
  `{"args": [], "kwargs": {...}}`, so a check on `step.args["quantity"]` saw nothing.
- **`ToolCallingAgent` + gpt-oss on Groq:** smolagents sends `tool_choice: required`; when
  the model answers in text Groq returns 400 "Tool choice is required". Gemini works.
- **`stream_outputs=True`:** Gemini streams tool-call deltas without an `index` and
  smolagents raises "Tool call index is not provided"; Groq returns "Parsing failed".
  crashlab records the partial stream (a07f1e3) and replays it without an index
  (1b1ccf0), so the failure reproduces offline.
