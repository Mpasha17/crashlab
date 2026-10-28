"""Packaging guards (pyproject extras)."""

from __future__ import annotations

import re
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _extra(name: str) -> str:
    m = re.search(rf"^{name} = \[(.*?)\]", PYPROJECT.read_text("utf-8"), re.M | re.S)
    assert m, name
    return m.group(1)


def test_litellm_extra_brings_mcp():
    # litellm 1.106 imports mcp whenever tools= is passed, but only its proxy/mcp extras
    # depend on it: `pip install crashlab[litellm]` then failed every tool-calling agent
    extra = _extra("litellm")
    assert '"litellm>=' in extra and '"mcp>=' in extra
