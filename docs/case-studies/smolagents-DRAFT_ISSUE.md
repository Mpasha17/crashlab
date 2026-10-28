# DRAFT upstream issue — not filed

File only if Mahaboob decides to. Do not open this against huggingface/smolagents
from the agent.

**Repo:** https://github.com/huggingface/smolagents
**Example:** `examples/text_to_sql.py`
**Observed with:** smolagents 1.26.0, SQLAlchemy 2.1.6, upstream commit `96f33fa`

---

## Title

`examples/text_to_sql.py`: CodeAgent sees an empty database ("no such table: receipts") because `:memory:` SQLite is per-thread

## Body

Running `examples/text_to_sql.py` with a current smolagents (1.26.0, SQLAlchemy 2.1.6),
the agent's first query fails with:

```
OperationalError: (sqlite3.OperationalError) no such table: receipts
```

and the agent then reports that the table doesn't exist — even though the example just
created and filled it.

### Cause

`LocalPythonExecutor` runs each code block under `timeout()`, which uses a new
`ThreadPoolExecutor` (`smolagents/local_python_executor.py`). SQLAlchemy's default pool
for `sqlite:///:memory:` is `SingletonThreadPool`, so the worker thread opens a new,
empty in-memory database. The `receipts` table exists only on the main thread's
connection. Calling `sql_engine` directly on the main thread works, which is why this
is easy to miss in a unit test that never goes through the executor.

### Minimal repro (no LLM)

```python
from smolagents.local_python_executor import LocalPythonExecutor
# ... the example's engine / receipts / sql_engine setup unchanged ...
ex = LocalPythonExecutor(additional_authorized_imports=[])
ex.send_tools({"sql_engine": sql_engine, "final_answer": lambda x: x})
ex('print(sql_engine(query="SELECT * FROM receipts"))')
# -> OperationalError: no such table: receipts
```

With the engine change below, the same call returns `('Woodrow Wilson',)` (or the
receipts rows).

### Suggested fix

```diff
-engine = create_engine("sqlite:///:memory:")
+from sqlalchemy.pool import StaticPool
+
+# one shared connection: CodeAgent runs tool calls in worker threads, and a
+# per-thread :memory: database would be empty there
+engine = create_engine(
+    "sqlite:///:memory:",
+    poolclass=StaticPool,
+    connect_args={"check_same_thread": False},
+)
```

Please also check whether the text-to-SQL docs example uses the same pattern.

### How we found it

We ran the unmodified example under [crashlab](https://github.com/Mpasha17/crashlab)
with a `@crashlab.check` oracle that expects "Woodrow Wilson". The oracle failed; a
no-LLM reproduction through `LocalPythonExecutor` confirmed the empty database. Full
write-up (not an upstream PR): https://github.com/Mpasha17/crashlab/blob/main/docs/case-studies/smolagents-text-to-sql.md
