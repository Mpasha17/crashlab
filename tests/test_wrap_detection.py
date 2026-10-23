"""``crashlab.wrap`` detection edges (responses-only, unknown clients)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import crashlab
from crashlab.capture.openai import looks_like_openai


def test_looks_like_openai_responses_only():
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **k: None))
    assert looks_like_openai(client) is True


def test_looks_like_openai_rejects_empty():
    assert looks_like_openai(SimpleNamespace()) is False


def test_wrap_unknown_client_mentions_responses():
    with pytest.raises(TypeError, match="responses.create"):
        crashlab.wrap(object())
