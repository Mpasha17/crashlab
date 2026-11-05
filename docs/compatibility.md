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
crashlab are ignored with a "written by a newer crashlab" warning rather than skipping
the step.

## Compact cassette lines (experimental, off by default)

Long chat runs repeat the whole conversation in every step, so the size of a recording
grows quadratically. Compact format v1 (`src/crashlab/store/compact.py`) stores only the
new tail of each message list and refers to an earlier step for the prefix. It carries a
required digest of the full list.

- **Reading is always on.** Old and compact lines can be mixed in one file. They load
  identically in store reads, `import`, replay, checks and `export`.
- **Writing is opt-in only:** `CRASHLAB_EXPERIMENTAL_COMPACT=1`. With the env unset, the
  writer is the unchanged full format. Do not enable compact writing for recordings that
  must be replayed by crashlab 0.0.2 (it skips compact lines).
- **`crashlab export` / `export_run` always write the full format**, so exported files stay
  readable by 0.0.2 even when the on-disk cassette used compact lines.
- If a step cannot be expanded (damaged or missing base, cycle, bad digest, unknown
  version), it is **kept**, not dropped. Its stored tail is all that survives; it is
  marked with the attribute `crashlab.compact.unresolved`, and a RuntimeWarning names it.
  The recorded response is still served on replay.
- Write failures (for example a full disk) raise as before. The partial line is truncated
  away and compact state rolls back, so the next step starts a clean line and never refers
  to a step that was not written. This applies to the default format too. Warnings and the
  unresolved attribute redact secrets and shorten names taken from the file.
- Mapping subclasses in message lists are coerced to dicts on every append (compact on or
  off) so a cassette never stores a `repr` address via `default=str`.

### Measured write tradeoff (main, Nov 2026)

Encodings are **content-based**, never keyed by object identity. An earlier identity-based
cache made 1500 short-message appends look ~14× faster; that path silently dropped nested
mutations and was removed. Measured on this tree:

| Workload | Compact write | Full write | Compact size | Full size |
|---|---|---|---|---|
| 1500 short messages (~40 B each) | ~2.4 s | ~0.4 s | ~0.4 MB | ~54 MB |
| 200 messages × ~50 KB content | ~2.9 s | ~3.1 s | ~10 MB | ~1006 MB |

So for typical short chats, compact writing is **slower** than full (still O(n²) JSON work
on the growing prefix) but **much smaller on disk**. For large unchanged prefixes, a
content fingerprint lets the writer reuse an immutable cached encoding when the fingerprint
still matches (`FP_THRESHOLD` characters); a nested mutation changes the fingerprint and
forces a fresh serialize. Small messages re-serialize and compare (cheaper than hashing
them). Compact's practical win is size (and large-attachment prefixes), not short-chat
wall-clock. Prefer the default full writer unless disk growth is the problem, and only
enable the experimental writer for recordings replayed by this version or newer.
