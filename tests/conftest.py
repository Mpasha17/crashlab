from __future__ import annotations

import os
import sys
from pathlib import Path

# never let litellm fetch its model cost map from the network during tests
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def crashlab_dir(tmp_path, monkeypatch):
    """Every test gets its own store directory and a clean redactor / demo DB."""
    from crashlab.store import redact

    store = tmp_path / "store"
    monkeypatch.setenv("CRASHLAB_DIR", str(store))
    for fault in ("FAULT_HTML_502", "FAULT_SCHEMA_DRIFT", "FAULT_POLICY_DROP"):
        monkeypatch.delenv(fault, raising=False)
    # the README tells users to export these; a developer shell must not leak into tests
    for var in ("CRASHLAB_DIAGNOSER_MODEL", "CRASHLAB_DIAGNOSER_BASE_URL",
                "CRASHLAB_DIAGNOSER_API_KEY"):  # fmt: skip
        monkeypatch.delenv(var, raising=False)
    redact.reset()
    from examples.refund_bot import agent as refund_agent

    refund_agent.reset_db()
    yield store
    redact.reset()
    from crashlab.integrations import litellm as cl_litellm

    cl_litellm.unpatch()


# ---------------------------------------------------------------------------
# Optional provider SDKs: the CI job "Test without provider SDKs" installs only
# the core package. Mark any test that imports openai / anthropic / httpx /
# litellm / ... with ``@pytest.mark.requires_providers("openai", "httpx")`` so
# it is skipped there automatically. Prefer this over a late importorskip after
# the import has already run.
# ---------------------------------------------------------------------------
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_providers(*modules): skip unless every named package imports",
    )


def pytest_runtest_setup(item):
    mark = item.get_closest_marker("requires_providers")
    if mark is None:
        return
    for mod in mark.args:
        pytest.importorskip(mod)
