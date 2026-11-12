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
- A message holding a value whose `str()` changes between calls (written via `default=str`)
  is written in full rather than referenced, so it can never produce a digest mismatch.

### Measured write cost (Nov 13 2026)

The writer compares **JSON text**, never object identity and never a fingerprint. Each step
encodes the whole current message list once (one `json.dumps`, the same text the required
digest is computed from), keeps the longest run of leading messages whose text is
byte-identical to the previous step's, and encodes only the new messages one by one to find
their boundaries. In-place and nested edits change the text, so they are always written. An
earlier identity-based cache looked ~14× faster and silently lost nested mutations; it was
removed, and no speed claim from it stands. The on-disk format is unchanged (still v1), and
for the same input the writer produces byte-identical lines to 0.0.3's
(`tests/fixtures/compact_v1_0_0_3`).

Reproduce with `PYTHONPATH=src python bench/compact_write.py --store --repeat 5`. Each workload
is a chat that grows by a user message and an assistant message per step, recorded through
`Store.add_step`, which includes redaction and the SQLite index. Both `request.messages` and
`raw_request.messages` are recorded. Median of 5, one 8-core Linux box, Python 3.12; your
numbers will differ:

| Workload | Full write | Compact write (0.0.3) | Compact write (now) | Full size | Compact size |
|---|---|---|---|---|---|
| 5 steps, 40 B messages | 0.009 s | 0.007 s | 0.007 s | 0.01 MB | <0.01 MB |
| 20 steps, 200 B messages | 0.028 s | 0.027 s | 0.027 s | 0.20 MB | 0.03 MB |
| 50 steps, 1 KB messages | 0.095 s | 0.119 s | 0.093 s | 5.3 MB | 0.23 MB |
| 50 steps, 1 KB, first message changes every step | 0.086 s | 0.117 s | 0.126 s | 5.3 MB | 5.3 MB |
| 750 steps (1,500 messages), 40 B | 3.0 s | 5.5 s | 3.4 s | 85 MB | 0.62 MB |
| 100 steps, 50 KB messages | 4.0 s | 2.0 s | 3.3 s | 1,011 MB | 20 MB |

What this means:

- For typical chats, compact writing now costs about the same as full writing (0.8–1.2×
  through the store), and the files are much smaller. With the writer alone (no redaction
  or index, `bench/compact_write.py` without `--store`), compact is still 1.3–2× slower
  on short chats, but that difference is under a few milliseconds per run.
- When the first message changes every step (for example a system prompt that carries
  the current time), nothing can be shared. Compact gives no size win and is about 1.5×
  slower. Leave it off for agents like that.
- With very large messages (50 KB each), compact is faster than full but slower than
  0.0.3's fingerprint path (3.3 s against 2.0 s), because every step now re-encodes the
  whole list instead of trusting a hash of it. That trade was made on purpose.

Compact **writing stays opt-in** (`CRASHLAB_EXPERIMENTAL_COMPACT=1`). The speed concern is
gone, but crashlab 0.0.2 cannot read compact lines (it skips them), so turning it on by
default would break replay with 0.0.2. Reading compact lines has been on since 0.0.3, and
`export` always writes the full format.
