"""A client that passes as OpenAI-compatible only through responses.create but has its
own chat methods (Mistral's chat.complete / chat.stream) warns that those go unrecorded."""

from __future__ import annotations

import warnings

import pytest

import crashlab


class _Chat:
    def complete(self, **kw):
        return {"ok": True}

    def stream(self, **kw):
        return iter(())


class _Responses:
    def create(self, **kw):
        return {}


class MistralLike:
    def __init__(self):
        self.chat = _Chat()
        self.responses = _Responses()


class ResponsesOnly:
    def __init__(self):
        self.responses = _Responses()


def test_chat_methods_without_completions_warn_at_wrap():
    with pytest.warns(UserWarning, match=r"chat\.complete\(\), chat\.stream\(\) are not"):
        wrapped = crashlab.wrap(MistralLike())
    assert wrapped.__crashlab_wrapped__


def test_openai_shaped_and_responses_only_clients_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        crashlab.wrap(ResponsesOnly())
