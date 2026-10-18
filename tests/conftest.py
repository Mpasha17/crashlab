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
    monkeypatch.delenv("FAULT_HTML_502", raising=False)
    redact.reset()
    from examples.refund_bot import agent as refund_agent

    refund_agent.reset_db()
    yield store
    redact.reset()
    from crashlab.integrations import litellm as cl_litellm

    cl_litellm.unpatch()
