# Recording compatibility between crashlab versions

**Newer crashlab reads older runs.** Current `main` loads and replays runs recorded by
crashlab 0.0.2: exported cassettes (`crashlab export-test` / `export_run`) and whole
0.0.2 store directories, index included. `tests/test_compat_v0_0_2.py` checks this
against fixtures recorded with the released 0.0.2 code (`tests/fixtures/v0_0_2/`).

**Older crashlab may not read newer runs.** crashlab 0.0.2 rejects fields it does not
know about (`extra="forbid"`) and skips such a step with a "damaged line" warning, so
replaying it there reports the step as missing. After 0.0.2, recorded errors may have
one new field, `error.state`. It holds the exception's `args`, its public JSON
attributes, and whether the caller cancelled the call. It is written **only** when there
is something to store, so most runs stay readable by 0.0.2. A run whose tool or LLM
errors carry state has to be replayed with a crashlab newer than 0.0.2.

In practice, pin crashlab in CI to at least the version that recorded your exported
cassettes. Re-recording or re-exporting with the newest version is always safe.

## Python versions and cancellation

When a caller cancels an async tool or LLM call (an `asyncio.wait_for` timeout, a
failing TaskGroup sibling), replay waits to be cancelled again, so the agent's timeout
handling runs the same way. Telling an outside cancel apart from a `CancelledError` the
call raised itself needs `asyncio.Task.cancelling()` (Python 3.11+). A run recorded on
3.10 stores the cancel as "origin unknown". Replay then waits briefly (at least 0.5 s,
or 3x the recorded time) so an outer timeout can fire again, and otherwise raises the
recorded `CancelledError`. A cancellation is never swallowed; the worst case is that
short delay.

## `error.state` on disk

Main writes a replayed exception's state as the step attribute `crashlab.error.state`
instead of an `error.state` key. crashlab 0.0.2 therefore loads every step of a new
recording. It cannot rebuild the exception's args, so it may report a final-output
divergence (for example `missing 'b'` vs `missing b`), but it never drops a step. Main
still reads inline `error.state` lines written by earlier main builds. Keys from a newer
crashlab are ignored with a "written by a newer crashlab" warning rather than skipping the step.
