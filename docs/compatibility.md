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

## Compact cassette lines (experimental, off by default)

Long chat runs repeat the whole conversation in every step, so the size of a recording
grows quadratically. Compact format v1 (`src/crashlab/store/compact.py`) stores only the
new tail of each message list and refers to an earlier step for the prefix. It carries a
required digest of the full list.

- **Reading is always on.** Old and compact lines can be mixed in one file. They load
  identically in store reads, `import`, replay, checks and `export`.
- **Writing is opt-in:** `CRASHLAB_EXPERIMENTAL_COMPACT=1`. The default writer is unchanged.
- crashlab 0.0.2 **cannot** read compact lines. It skips them, and older main builds ignore
  the `compact` key with a warning. Only enable compact writing for recordings that are
  replayed by this version or newer. `crashlab export` always writes the full format, so
  exported files stay readable by 0.0.2.
- If a step cannot be expanded (damaged or missing base, cycle, bad digest, unknown
  version), it is **kept**, not dropped. Its stored tail is all that survives, it is
  marked with the attribute `crashlab.compact.unresolved`, and a RuntimeWarning names it.
  The recorded response is still served on replay.

Write failures (for example a full disk) raise as before. The partial line is truncated
away, and compact state rolls back, so the next step starts a clean line and never refers
to a step that was not written. This applies to the default format too. Warnings and the
`crashlab.compact.unresolved` flag redact secrets and shorten names taken from the file.

The experimental writer caches JSON encodings of the shared message-object prefix, so an
append-only chat is serialised once per message (not once per step). Digests are taken
from that cache and match ``digest(full_messages)``. A failed write still rolls the cache
back. Default format writing is unchanged.

Cache encodings are content-based. Reusing an encoding because a message object is
identical (``is``) is unsafe when callers mutate nested fields; compact recordings always
encode the messages passed to each append so a deep-copied full baseline matches. Mapping
subclasses in message lists are coerced to dicts on every append (compact on or off).
