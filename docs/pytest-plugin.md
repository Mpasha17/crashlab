# pytest plugin

Write agent tests in ordinary pytest. The plugin is auto-loaded from the
`pytest11` entry point when pytest is installed (`pip install pytest` or
`pip install 'crashlab[pytest]'`). pytest is not a hard dependency:
`import crashlab` works in environments that do not have pytest, and this
plugin module itself imports cleanly if pytest is absent.

The plugin is opt-in. Unmarked tests that do not request a crashlab fixture
are left alone. `--crashlab-save-failures` only copies runs recorded by the
`crashlab_record` fixture.

## Fixtures

`crashlab_store` is an isolated `Store` backed by `tmp_path / ".crashlab"`
(and `CRASHLAB_DIR` is pointed at it for the duration of the test).

`crashlab_record` opens `crashlab.record(name=...)` around the test body,
using that store. The recording name is `@pytest.mark.crashlab(agent=...)`
when set, otherwise the test name. After the test, `crashlab_record.run` is
the persisted `Run`.

`crashlab_replay(run_or_path, **kwargs)` imports an exported cassette (or
replays a `Run` already in the store) strictly offline. Extra keyword
arguments go to `crashlab.replay`. A missing provider SDK skips the test
instead of failing it.

```python
import crashlab

def test_records_the_body(crashlab_record, crashlab_store):
    crashlab_record.run.output = "hello"
    assert crashlab_record.run.name == "test_records_the_body"
    assert crashlab_store.root.name == ".crashlab"


def test_context_sugar(crashlab_record):
    with crashlab_record as run:
        run.output = "hello"
    assert crashlab_record.run is run
```

Replay a cassette, or a run you just recorded, in one line:

```python
import crashlab
from pathlib import Path

def toy() -> str:
    with crashlab.record("toy") as run:
        run.output = "hello"
    return "hello"


def test_replay_from_run_or_path(crashlab_replay, crashlab_store):
    toy()
    src = crashlab_store.load_run("last")
    assert crashlab_replay(src, fn=toy).output == "hello"
    cassette = Path("toy.jsonl")
    crashlab.store.export_run(crashlab_store, src.run_id, cassette)
    assert crashlab_replay(cassette, fn=toy).output == "hello"
```

`export-test` still emits `@pytest.mark.crashlab` tests that call
`crashlab_replay(path)` the same way as before.

## Marker constraints

`@pytest.mark.crashlab(...)` still tags exported regression tests (and
`pytest --crashlab` still runs only those). When the test also uses
`crashlab_record`, the optional kwargs are enforced at teardown and fail the
test with an `AssertionError` if they are violated:

- `agent`: name of the recorded run
- `rules`: a rules file path, or an in-memory list / mapping accepted by
  `crashlab.assertions.load_rules`
- `max_cost_usd` / `max_tokens`: budget gates (`crashlab.budgets.within_budget`)
- `no_findings_above`: a severity (`info`, `warn`, `error`, `critical`);
  findings at or above that level fail the test

```python
import pytest

@pytest.mark.crashlab(
    agent="refund-bot",
    rules="policies/refund.yaml",
    max_cost_usd=0.05,
    max_tokens=20_000,
    no_findings_above="error",
)
def test_refund_policy(crashlab_record):
    crashlab_record.run.output = "refund issued"
```

The same rules file works with `crashlab check --rules` and
`crashlab.assertions`. See [rules.md](rules.md).

## Saving failures

`--crashlab-store PATH` sets the persistent store used for saved failures
(default `.crashlab/`). `--crashlab-save-failures` copies each failing
`crashlab_record` run there and prints pointers in the pytest terminal
summary:

```bash
pytest --crashlab-save-failures --crashlab-store .crashlab
# then:
#   crashlab show <run_id>
#   crashlab diagnose <run_id>
```

Ordinary test failures that never requested `crashlab_record` are not copied.

## Exported regression tests

The existing `--crashlab` flag and `crashlab_replay(path)` fixture used by
`crashlab export-test` are unchanged:

```bash
pytest --crashlab
pytest --crashlab tests/crashlab
```
