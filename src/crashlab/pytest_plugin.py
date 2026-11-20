"""pytest plugin (auto-loaded via the ``pytest11`` entry point).

Safe to import when pytest is not installed: hook and fixture registration are
skipped. ``import crashlab`` never imports this module.

- marker ``@pytest.mark.crashlab`` for exported regression tests, and optional
  record-time constraints (``agent``, ``rules``, ``max_cost_usd``, ``max_tokens``,
  ``no_findings_above``) enforced by the ``crashlab_record`` fixture;
- ``pytest --crashlab`` runs only those marked tests;
- ``--crashlab-store PATH`` / ``--crashlab-save-failures`` copy failing
  ``crashlab_record`` runs to a persistent store and print ``crashlab show`` /
  ``crashlab diagnose`` pointers in the terminal summary;
- fixtures ``crashlab_store`` (an isolated :class:`~crashlab.store.Store`),
  ``crashlab_record`` (records the test body) and ``crashlab_replay(run_or_path)``.

The plugin does not wrap, record, or filter tests unless they opt into these
fixtures / the marker, or the process is invoked with ``--crashlab`` /
``--crashlab-save-failures``.
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

try:
    import pytest
except ImportError:  # pragma: no cover - CI / users without the pytest extra
    pytest = None  # type: ignore[assignment]

_SAVE_KEY = None if pytest is None else pytest.StashKey[list[dict[str, str]]]()


def pytest_addoption(parser: Any) -> None:
    group = parser.getgroup("crashlab")
    group.addoption(
        "--crashlab",
        action="store_true",
        default=False,
        help="run only @pytest.mark.crashlab tests (exported agent regression tests)",
    )
    group.addoption(
        "--crashlab-store",
        action="store",
        default=None,
        metavar="PATH",
        help="persistent store path for saved failures (default: .crashlab/)",
    )
    group.addoption(
        "--crashlab-save-failures",
        action="store_true",
        default=False,
        help=(
            "copy failing crashlab_record runs to --crashlab-store or .crashlab/ "
            "and print crashlab show / diagnose pointers in the terminal summary"
        ),
    )


def pytest_configure(config: Any) -> None:
    config.addinivalue_line(
        "markers",
        "crashlab(agent=..., rules=..., max_cost_usd=..., max_tokens=..., "
        "no_findings_above=...): exported crashlab regression test, or record-time "
        "constraints enforced by the crashlab_record fixture",
    )
    if _SAVE_KEY is not None:
        config.stash[_SAVE_KEY] = []


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    if not config.getoption("--crashlab"):
        return
    keep = [i for i in items if i.get_closest_marker("crashlab")]
    drop = [i for i in items if not i.get_closest_marker("crashlab")]
    if drop:
        config.hook.pytest_deselected(items=drop)
    items[:] = keep


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: Any) -> None:
    if _SAVE_KEY is None:
        return
    saved = config.stash.get(_SAVE_KEY, [])
    if not saved:
        return
    terminalreporter.section("crashlab failures")
    for item in saved:
        terminalreporter.write_line(f"saved run {item['run_id']} to {item['store']}")
        terminalreporter.write_line(f"  crashlab show {item['run_id']}")
        terminalreporter.write_line(f"  crashlab diagnose {item['run_id']}")


def _import_ancestors(store: Any, path: Path, run: Any) -> None:
    """Also import sibling cassettes this run was forked / replayed from: a fix cassette
    recorded by crashlab<=0.0.1 names its script only via its parent run."""
    from crashlab.capture._util import REPLAY_MODULE_PREFIX
    from crashlab.store import import_run

    if not (run.entrypoint or "").startswith(REPLAY_MODULE_PREFIX):
        return
    wanted = {run.forked_from, run.replay_of} - {None}
    for sibling in sorted(path.parent.glob("*.jsonl")):
        if sibling == path or not wanted:
            continue
        try:
            with sibling.open(encoding="utf-8") as f:
                head = json.loads(f.readline() or "{}")
        except (OSError, ValueError):
            continue
        sib = (head.get("run") or {}) if isinstance(head, dict) else {}
        if sib.get("run_id") in wanted:
            wanted.discard(sib["run_id"])
            import_run(store, sibling)
            wanted |= {sib.get("forked_from"), sib.get("replay_of")} - {None}


def _as_store(value: Any) -> Any:
    from crashlab.store import Store

    return value if isinstance(value, Store) else Store(value)


def _marker_kwargs(request: Any) -> dict[str, Any]:
    mark = request.node.get_closest_marker("crashlab")
    return dict(mark.kwargs) if mark is not None else {}


def _constraints(kwargs: dict[str, Any]) -> dict[str, Any]:
    keys = ("rules", "max_cost_usd", "max_tokens", "no_findings_above")
    return {key: kwargs[key] for key in keys if key in kwargs}


def _findings_above(run: Any, threshold: str) -> str | None:
    from crashlab.models import SEVERITY_RANK

    if threshold not in SEVERITY_RANK:
        known = ", ".join(SEVERITY_RANK)
        raise AssertionError(
            f"crashlab marker no_findings_above={threshold!r} is not a severity ({known})"
        )
    floor = SEVERITY_RANK[threshold]
    hits = [f for f in (run.findings or []) if SEVERITY_RANK.get(f.severity, 0) >= floor]
    if not hits:
        return None
    shown = ", ".join(f"{f.severity}:{f.kind}" + (f" ({f.message})" if f.message else "")
                      for f in hits[:5])  # fmt: skip
    extra = "" if len(hits) <= 5 else f" and {len(hits) - 5} more"
    return f"no_findings_above: {len(hits)} finding(s) at or above {threshold} ({shown}{extra})"


def _enforce_constraints(run: Any, constraints: dict[str, Any]) -> None:
    if not constraints:
        return
    from crashlab.budgets import within_budget

    problems: list[str] = []
    cost = constraints.get("max_cost_usd")
    tokens = constraints.get("max_tokens")
    if cost is not None or tokens is not None:
        msg = within_budget(run, max_cost_usd=cost, max_tokens=tokens)
        if msg:
            problems.append(msg)
    if "no_findings_above" in constraints:
        msg = _findings_above(run, constraints["no_findings_above"])
        if msg:
            problems.append(msg)
    if "rules" in constraints:
        from crashlab.assertions import evaluate_rules, load_rules

        for result in evaluate_rules(run, load_rules(constraints["rules"])):
            if result.status != "pass":
                problems.append(f"rule {result.name}: {result.message or result.status}")
    if problems:
        raise AssertionError(
            f"crashlab marker constraints failed for {run.name} ({run.run_id}):\n"
            + "\n".join(f"- {p}" for p in problems)
        )


def _persistent_store(config: Any) -> Any:
    from crashlab.store import Store

    raw = config.getoption("--crashlab-store")
    path = Path(raw) if raw else Path(".crashlab")
    path = path.expanduser()
    if not path.is_absolute():
        start = getattr(getattr(config, "invocation_params", None), "dir", None)
        path = Path(start or Path.cwd()) / path
    return Store(path)


def _save_failure(config: Any, store: Any, run: Any) -> dict[str, str] | None:
    from crashlab.store import export_run, import_run

    dest = _persistent_store(config)
    if dest.root.resolve() != store.root.resolve():
        with tempfile.TemporaryDirectory(prefix="crashlab-pytest-") as tmp:
            exported = export_run(store, run.run_id, Path(tmp) / f"{run.run_id}.jsonl")
            import_run(dest, exported, replace=True)
    return {"run_id": run.run_id, "store": str(dest.root)}


def _reload(store: Any, run: Any) -> Any:
    try:
        return store.load_run(run.run_id)
    except Exception:  # pragma: no cover - store unusable / not recorded
        return run


def _ensure_in_store(store: Any, run: Any) -> None:
    from crashlab.store import RunNotFoundError

    try:
        store.load_run(run.run_id)
        return
    except (RunNotFoundError, FileNotFoundError, ValueError, OSError):
        pass
    store.begin_run(run)
    for step in run.steps:
        store.add_step(step)
    store.end_run(run)


class CrashlabRecord:
    """Helper yielded by the ``crashlab_record`` fixture.

    The test body is already inside ``crashlab.record``; :attr:`run` is that
    recording. ``with crashlab_record as run:`` is sugar for the same object.
    """

    def __init__(
        self,
        run: Any,
        store: Any,
        rec: Any,
        request: Any,
        constraints: dict[str, Any],
    ) -> None:
        self.run = run
        self.store = store
        self._rec = rec
        self._request = request
        self._constraints = constraints
        self.closed = False

    def __enter__(self) -> Any:
        return self.run

    def __exit__(self, *exc: Any) -> None:
        return None

    def close(self, excinfo: tuple[Any, ...] | None) -> None:
        if self.closed:
            return
        self.closed = True
        typ, val, tb = excinfo if excinfo and excinfo[0] else (None, None, None)
        self._rec.__exit__(typ, val, tb)
        self.run = _reload(self.store, self.run)

    def save_if_needed(self) -> None:
        if not self._request.config.getoption("--crashlab-save-failures", default=False):
            return
        saved = _save_failure(self._request.config, self.store, self.run)
        if saved is not None and _SAVE_KEY is not None:
            self._request.config.stash[_SAVE_KEY].append(saved)

    def enforce(self) -> None:
        _enforce_constraints(self.run, self._constraints)


def _is_skip_or_xfail(excinfo: tuple[Any, ...] | None) -> bool:
    if not excinfo or excinfo[0] is None:
        return False
    typ = excinfo[0]
    if typ.__name__ in {"Skipped", "XFailed"}:
        return True
    try:
        from _pytest.outcomes import Skipped, XFailed

        return issubclass(typ, (Skipped, XFailed))
    except Exception:
        return False


def _finish_record_call(item: Any, outcome: Any) -> None:
    if "crashlab_record" not in getattr(item, "fixturenames", ()):
        return
    helper = item.funcargs.get("crashlab_record")
    if helper is None or helper.closed:
        return
    excinfo = getattr(outcome, "excinfo", None)
    if _is_skip_or_xfail(excinfo):
        helper.close((None, None, None))
        return
    if excinfo is not None:
        helper.close(excinfo)
        helper.save_if_needed()
        return
    helper.close((None, None, None))
    try:
        helper.enforce()
    except AssertionError:
        helper.save_if_needed()
        raise


if pytest is not None:

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(item: Any) -> Iterator[None]:
        outcome = yield
        _finish_record_call(item, outcome)

    @pytest.fixture
    def crashlab_store(tmp_path: Path, monkeypatch: Any) -> Any:
        from crashlab.store import Store

        root = tmp_path / ".crashlab"
        monkeypatch.setenv("CRASHLAB_DIR", str(root))
        return Store(root)

    @pytest.fixture
    def crashlab_record(request: Any, crashlab_store: Any) -> Iterator[CrashlabRecord]:
        from crashlab.capture import record as record_run

        store = _as_store(crashlab_store)
        kw = _marker_kwargs(request)
        name = kw.pop("agent", None) or request.node.name
        constraints = _constraints(kw)
        rec = record_run(name, store_dir=store.root)
        helper = CrashlabRecord(rec.__enter__(), store, rec, request, constraints)
        try:
            yield helper
        finally:
            if not helper.closed:
                info = sys.exc_info()
                helper.close(info if info[0] is not None else (None, None, None))

    @pytest.fixture
    def crashlab_replay(crashlab_store: Any) -> Any:
        from crashlab.models import Run
        from crashlab.replay import replay
        from crashlab.store import import_run

        store = _as_store(crashlab_store)

        def _replay(run_or_path: Any, **kw: Any) -> Any:
            if isinstance(run_or_path, Run):
                _ensure_in_store(store, run_or_path)
                try:
                    return replay(run_or_path, store_dir=store.root, **kw)
                except ModuleNotFoundError as e:  # pragma: no cover - depends on extras
                    pytest.skip(f"replay needs {e.name!r} installed")
            path = Path(run_or_path)
            run_id, _ = import_run(store, path, replace=True)
            _import_ancestors(store, path, store.load_run(run_id))
            try:
                return replay(run_id, store_dir=store.root, **kw)
            except ModuleNotFoundError as e:  # pragma: no cover - depends on installed extras
                pytest.skip(f"replay needs {e.name!r} installed")
            except Exception as e:
                cause = e.__cause__ or e.__context__
                if isinstance(cause, ModuleNotFoundError) or "No module named" in str(e):
                    missing = getattr(cause, "name", None) or ""
                    entry = (store.load_run(run_id).entrypoint or "").split(":", 1)[0]
                    top = entry.split(".", 1)[0]
                    if missing and top and missing.split(".", 1)[0] == top:
                        # the agent's own code is not importable: a regression test must not
                        # quietly skip (CI would stay green without testing anything)
                        pytest.fail(
                            f"cannot import the agent's module {entry!r} to replay {run_id}: "
                            f"{cause}. Run pytest from the project root, or add the directory "
                            "that contains it to pytest's `pythonpath` setting.",
                            pytrace=False,
                        )
                    pytest.skip(f"replay needs a provider SDK: {e}")
                raise

        return _replay
