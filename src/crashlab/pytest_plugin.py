"""pytest plugin (auto-loaded via the ``pytest11`` entry point).

- marker ``@pytest.mark.crashlab`` for exported regression tests;
- ``pytest --crashlab`` runs only those tests;
- fixtures ``crashlab_store`` (a throwaway store dir, set as ``CRASHLAB_DIR``) and
  ``crashlab_replay(path)``: import an exported run and replay it strictly offline
  (network blocked, placeholder credentials); returns the replay :class:`~crashlab.Run`.
  A missing provider SDK skips the test instead of failing it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def pytest_addoption(parser: Any) -> None:
    group = parser.getgroup("crashlab")
    group.addoption(
        "--crashlab",
        action="store_true",
        default=False,
        help="run only @pytest.mark.crashlab tests (exported agent regression tests)",
    )


def pytest_configure(config: Any) -> None:
    config.addinivalue_line(
        "markers", "crashlab: an exported crashlab regression test (offline cassette replay)"
    )


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    if not config.getoption("--crashlab"):
        return
    keep = [i for i in items if i.get_closest_marker("crashlab")]
    drop = [i for i in items if not i.get_closest_marker("crashlab")]
    if drop:
        config.hook.pytest_deselected(items=drop)
    items[:] = keep


@pytest.fixture
def crashlab_store(tmp_path: Path, monkeypatch: Any) -> Path:
    store = tmp_path / "crashlab-store"
    monkeypatch.setenv("CRASHLAB_DIR", str(store))
    return store


@pytest.fixture
def crashlab_replay(crashlab_store: Path) -> Any:
    from crashlab.replay import replay
    from crashlab.store import Store, import_run

    def _replay(path: str | Path, **kw: Any) -> Any:
        store = Store(crashlab_store)
        run_id, _ = import_run(store, path, replace=True)
        try:
            return replay(run_id, store_dir=crashlab_store, **kw)
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
