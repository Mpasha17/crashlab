# DRAFT comment for huggingface/smolagents#2475 — not posted

Post only if Mahaboob decides to: https://github.com/huggingface/smolagents/pull/2475

---

## Comment

Thanks for this fix! I hit the same bug independently while running the unmodified
`examples/text_to_sql.py`, and can confirm the change works on smolagents 1.26.0 /
SQLAlchemy 2.1.6 (upstream `96f33fa`).

**Root cause, for reviewers:** `LocalPythonExecutor` runs each code block under
`timeout()`, which uses a fresh `ThreadPoolExecutor`. SQLAlchemy's default pool for
`sqlite:///:memory:` is `SingletonThreadPool`, so the worker thread opens a new, empty
in-memory database and `receipts` doesn't exist there. Calling `sql_engine` on the main
thread works, which is why it's easy to miss.

**Minimal repro, no LLM** (the example's engine / `receipts` / `sql_engine` setup unchanged):

```python
from smolagents.local_python_executor import LocalPythonExecutor

ex = LocalPythonExecutor(additional_authorized_imports=[])
ex.send_tools({"sql_engine": sql_engine, "final_answer": lambda x: x})
result = ex('print(sql_engine(query="SELECT customer_name FROM receipts ORDER BY price DESC LIMIT 1"))')
print(result.logs)
```

Before this PR:

```
smolagents.local_python_executor.InterpreterError: Code execution failed at line 'print(sql_engine(query="SELECT customer_name FROM receipts ORDER BY price DESC LIMIT 1"))' due to: OperationalError: (sqlite3.OperationalError) no such table: receipts
[SQL: SELECT customer_name FROM receipts ORDER BY price DESC LIMIT 1]
```

With `StaticPool` + `check_same_thread=False` (this PR), the same call prints:

```
('Woodrow Wilson',)
```

which is the answer to the example's question. Write-up:
https://github.com/Mpasha17/crashlab/blob/main/docs/case-studies/smolagents-text-to-sql.md
