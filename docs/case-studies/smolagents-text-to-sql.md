# Case study: smolagents `examples/text_to_sql.py` — CodeAgent queries an empty database

**Project:** [huggingface/smolagents](https://github.com/huggingface/smolagents) 1.26.0
(SQLAlchemy 2.1.6). The example file was used unmodified.

**Status:** Finding is real and the fix is verified with a no-LLM A/B. No upstream
issue or PR has been opened from this work; draft text lives with the Day 11 notes.

## Task
The example's question: *"Can you give me the name of the client who got the most
expensive receipt?"* Expected answer: **Woodrow Wilson**.

## Failure (genuine, not injected)
```
OperationalError: (sqlite3.OperationalError) no such table: receipts
```
The agent then reports that the table does not exist.

**Root cause.** The example builds `create_engine("sqlite:///:memory:")` on the main
thread. SQLAlchemy's default pool for `:memory:` is `SingletonThreadPool` (one
connection per thread). smolagents' `LocalPythonExecutor` runs every code block under
`timeout()` in a **new `ThreadPoolExecutor`**. Every `sql_engine` call from agent code
therefore hits a fresh, empty database. Direct calls on the main thread work, which is
why a unit test that never goes through the executor misses this.

## Fix (verified)
```diff
-engine = create_engine("sqlite:///:memory:")
+from sqlalchemy.pool import StaticPool
+engine = create_engine(
+    "sqlite:///:memory:",
+    poolclass=StaticPool,
+    connect_args={"check_same_thread": False},
+)
```

No-LLM A/B through smolagents' own executor:

```
upstream   -> ERROR ... no such table: receipts
engine_fix -> ('Woodrow Wilson',)
```

See [docs/demo/](../demo/README.md) to replay the terminal capture.

## What crashlab did
Recorded the live failing run, failed a `@crashlab.check` oracle, and (separately)
fixed a crashlab bug: tool calls from CodeAgent's unbound worker thread were silently
not recorded. That is `CRASHLAB_THREAD_FALLBACK` (commit `419235a`).
