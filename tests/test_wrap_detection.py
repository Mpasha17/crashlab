"""``crashlab.wrap`` detection edges (responses-only, unknown clients, idempotency)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import crashlab
from crashlab.capture.anthropic import looks_like_anthropic
from crashlab.capture.openai import looks_like_openai


def test_looks_like_openai_responses_only():
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **k: None))
    assert looks_like_openai(client) is True


def test_looks_like_openai_rejects_empty():
    assert looks_like_openai(SimpleNamespace()) is False


def test_looks_like_anthropic_messages_only():
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **k: None))
    assert looks_like_anthropic(client) is True
    assert looks_like_anthropic(SimpleNamespace()) is False


def test_wrap_unknown_client_mentions_responses():
    with pytest.raises(TypeError, match="responses.create"):
        crashlab.wrap(object())


def test_wrap_is_idempotent():
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **k: None)),
        base_url="https://api.example.test/v1",
    )
    first = crashlab.wrap(client)
    second = crashlab.wrap(first)
    assert second is first
