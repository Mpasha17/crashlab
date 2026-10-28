"""Responses of SDKs generated like openai's (groq, together, cerebras ...) rebuild as
their own classes on replay, also when they do not validate (a mocked or extended body),
as the SDK itself builds them with ``construct``; not as a ReplayObject."""

from __future__ import annotations

import sys
import textwrap

import pytest

from crashlab.replay import rebuild as R


@pytest.fixture
def fakesdk(tmp_path, monkeypatch):
    pytest.importorskip("openai")
    pkg = tmp_path / "fakesdk"
    (pkg / "types").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "_models.py").write_text("from openai._models import BaseModel  # noqa: F401\n")
    (pkg / "types" / "__init__.py").write_text("")
    (pkg / "types" / "chat.py").write_text(
        textwrap.dedent("""
        from typing import List, Optional
        from .._models import BaseModel

        class Delta(BaseModel):
            content: Optional[str] = None

        class Choice(BaseModel):
            index: float
            delta: Delta
            finish_reason: Optional[str] = None

        class ChatCompletionChunk(BaseModel):
            id: str
            choices: List[Choice]
            prompt: List[str]  # required: the recorded body has none, so validation fails
    """)
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    yield "fakesdk.types.chat.ChatCompletionChunk"
    for name in [m for m in sys.modules if m.startswith("fakesdk")]:
        del sys.modules[name]


def test_stainless_style_sdk_model_rebuilds_as_its_class(fakesdk):
    data = {"id": "c1", "choices": [{"index": 0, "delta": {"content": "hi"}}]}
    obj = R.rebuild(data, fakesdk)
    assert type(obj).__qualname__ == "ChatCompletionChunk"
    assert type(obj.choices[0]).__qualname__ == "Choice"
    assert obj.choices[0].index == 0.0 and isinstance(obj.choices[0].index, float)
    assert obj.choices[0].delta.content == "hi"


def test_other_classes_that_do_not_validate_stay_replay_objects(tmp_path, monkeypatch):
    (tmp_path / "plainsdk.py").write_text(
        textwrap.dedent("""
        from pydantic import BaseModel

        class Thing(BaseModel):
            required: int
    """)
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        assert isinstance(R.rebuild({"x": 1}, "plainsdk.Thing"), R.ReplayObject)
    finally:
        sys.modules.pop("plainsdk", None)
