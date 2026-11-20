"""Zero-config auto-instrumentation: ``crashlab.instrument()`` / ``uninstrument()``.

Uses duck-typed fake ``openai`` / ``anthropic`` / ``litellm`` / ``google.genai``
modules in ``sys.modules`` so the suite runs with no provider SDKs installed.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

import crashlab
from crashlab.integrations.instrument import PROVIDERS, _reset
from tests.fakes import (
    FakeStream,
    a_text,
    chunks_for_text,
    text_response,
)
from tests.test_genai import text_response as genai_text_response

_WATCHED = ("openai", "anthropic", "litellm", "google.genai", "google")


@pytest.fixture(autouse=True)
def _clean_instrumentation():
    """Each test starts unpatched and must not leak fake modules into later tests."""
    saved = {name: sys.modules[name] for name in _WATCHED if name in sys.modules}
    _reset()
    yield
    _reset()
    for name in _WATCHED:
        if name in saved:
            sys.modules[name] = saved[name]
        else:
            sys.modules.pop(name, None)
    # a leftover litellm patch would also break later tests
    from crashlab.integrations import litellm as cl_litellm

    cl_litellm.unpatch()


# --- fake provider modules ---------------------------------------------------


def _install_openai() -> types.ModuleType:
    mod = types.ModuleType("openai")

    class _Completions:
        def __init__(self, owner: Any) -> None:
            self.owner = owner

        def create(self, **kwargs: Any) -> Any:
            self.owner.calls.append(kwargs)
            if kwargs.get("stream"):
                return FakeStream(chunks_for_text("hello", model=kwargs.get("model", "m")))
            return text_response(1, "hello from openai", model=kwargs.get("model", "fake-model"))

    class _Chat:
        def __init__(self, owner: Any) -> None:
            self.completions = _Completions(owner)

    class OpenAI:
        base_url = "http://fake.local/v1"

        def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
            self.api_key = api_key
            self.calls: list[dict[str, Any]] = []
            self.chat = _Chat(self)

    class AsyncOpenAI(OpenAI):
        def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
            super().__init__(api_key=api_key, **kwargs)

            class _AsyncCompletions(_Completions):
                async def create(self, **kwargs: Any) -> Any:  # type: ignore[override]
                    return super().create(**kwargs)

            self.chat = types.SimpleNamespace(completions=_AsyncCompletions(self))

    mod.OpenAI = OpenAI
    mod.AsyncOpenAI = AsyncOpenAI
    sys.modules["openai"] = mod
    return mod


def _install_anthropic() -> types.ModuleType:
    mod = types.ModuleType("anthropic")

    class _Messages:
        def __init__(self, owner: Any) -> None:
            self.owner = owner

        def create(self, **kwargs: Any) -> Any:
            self.owner.calls.append(kwargs)
            return a_text(1, "hello from anthropic")

    class Anthropic:
        base_url = "https://api.anthropic.fake"

        def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
            self.api_key = api_key
            self.calls: list[dict[str, Any]] = []
            self.messages = _Messages(self)

    class AsyncAnthropic(Anthropic):
        def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
            super().__init__(api_key=api_key, **kwargs)

            class _AsyncMessages(_Messages):
                async def create(self, **kwargs: Any) -> Any:  # type: ignore[override]
                    return super().create(**kwargs)

            self.messages = _AsyncMessages(self)

    mod.Anthropic = Anthropic
    mod.AsyncAnthropic = AsyncAnthropic
    sys.modules["anthropic"] = mod
    return mod


def _install_litellm() -> types.ModuleType:
    mod = types.ModuleType("litellm")
    mod.calls = []

    def completion(model, messages, stream=False, **kwargs):
        mod.calls.append((model, messages, kwargs))
        if stream:
            return FakeStream(chunks_for_text("lite stream", model=model))
        return text_response(1, "lite hello", model=model)

    async def acompletion(model, messages, stream=False, **kwargs):
        return text_response(2, "lite async", model=model)

    mod.completion = completion
    mod.acompletion = acompletion
    sys.modules["litellm"] = mod
    return mod


def _install_genai() -> types.ModuleType:
    from tests.test_genai import FakeClient

    pkg = sys.modules.get("google")
    if pkg is None:
        pkg = types.ModuleType("google")
        pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules["google"] = pkg
    mod = types.ModuleType("google.genai")

    class Client(FakeClient):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(script=[genai_text_response("hello from genai")])

    mod.Client = Client
    sys.modules["google.genai"] = mod
    pkg.genai = mod
    return mod


def _install_all() -> dict[str, types.ModuleType]:
    return {
        "openai": _install_openai(),
        "anthropic": _install_anthropic(),
        "litellm": _install_litellm(),
        "google_genai": _install_genai(),
    }


# --- status / lifecycle ------------------------------------------------------


def test_instrument_without_sdks_does_not_raise():
    status = crashlab.instrumented()
    assert status == {name: False for name in PROVIDERS}
    handle = crashlab.instrument()
    assert set(crashlab.instrumented()) == set(PROVIDERS)
    # nothing to patch if the packages are absent and not faked; still "on"
    assert crashlab.instrumented()["openai"] is True
    handle.close()
    assert crashlab.instrumented() == {name: False for name in PROVIDERS}


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="unknown provider"):
        crashlab.instrument(providers=["openai", "not-a-provider"])
    assert crashlab.instrumented()["openai"] is False


def test_empty_or_blank_agent_is_rejected():
    with pytest.raises(ValueError, match="agent="):
        crashlab.instrument(agent="")
    with pytest.raises(ValueError, match="agent="):
        crashlab.instrument(agent="   ")


def test_context_manager_restores_init():
    openai = _install_openai()
    orig = openai.OpenAI.__init__
    with crashlab.instrument(providers=["openai"]):
        assert crashlab.instrumented()["openai"] is True
        assert openai.OpenAI.__init__ is not orig
        assert getattr(openai.OpenAI.__init__, "__crashlab_instrumented_init__", False)
    assert openai.OpenAI.__init__ is orig
    assert crashlab.instrumented()["openai"] is False
    client = openai.OpenAI(api_key="x")
    with crashlab.record("after"):
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
    assert crashlab.load("last").steps == []


def test_nested_instrumentation_keeps_patch_until_outer_exit():
    openai = _install_openai()
    orig = openai.OpenAI.__init__
    with crashlab.instrument(providers="openai") as outer:
        patched = openai.OpenAI.__init__
        with crashlab.instrument(providers=["openai"]):
            assert openai.OpenAI.__init__ is patched
            client = openai.OpenAI(api_key="x")
        assert openai.OpenAI.__init__ is patched
        assert crashlab.instrumented()["openai"] is True
        with crashlab.record("nested"):
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
        assert len(crashlab.load("last").steps) == 1
        outer.close()  # explicit close is idempotent with __exit__
    assert openai.OpenAI.__init__ is orig
    assert crashlab.instrumented()["openai"] is False


def test_uninstrument_and_instrumented_subset():
    _install_all()
    crashlab.instrument()
    assert crashlab.instrumented() == {name: True for name in PROVIDERS}
    crashlab.uninstrument(providers=["openai"])
    status = crashlab.instrumented()
    assert status["openai"] is False
    assert status["anthropic"] is True
    assert status["litellm"] is True
    assert status["google_genai"] is True
    crashlab.uninstrument()
    assert crashlab.instrumented() == {name: False for name in PROVIDERS}
    crashlab.uninstrument()  # idempotent


def test_handle_close_is_idempotent():
    _install_openai()
    handle = crashlab.instrument(providers=["openai"], agent="shop-bot", store=None)
    assert handle.agent == "shop-bot" and handle.store is None
    handle.close()
    handle.close()
    assert crashlab.instrumented()["openai"] is False


# --- recording --------------------------------------------------------------


def test_openai_constructor_is_recorded():
    openai = _install_openai()
    with crashlab.instrument(providers=["openai"]):
        client = openai.OpenAI(api_key="x")
        with crashlab.record("oai"):
            resp = client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}]
            )
    assert resp.choices[0].message.content == "hello from openai"
    step = crashlab.load("last").steps[0]
    assert step.operation == "chat"
    assert step.attributes["gen_ai.provider.name"] == "openai"
    assert step.request["messages"][0]["parts"][0]["content"] == "hi"


def test_anthropic_constructor_is_recorded():
    anthropic = _install_anthropic()
    with crashlab.instrument(providers=["anthropic"]):
        client = anthropic.Anthropic(api_key="x")
        with crashlab.record("ant"):
            msg = client.messages.create(
                model="claude-x", max_tokens=16, messages=[{"role": "user", "content": "hi"}]
            )
    assert msg.content[0].text == "hello from anthropic"
    step = crashlab.load("last").steps[0]
    assert step.attributes["gen_ai.provider.name"] == "anthropic"


def test_litellm_completion_is_recorded():
    litellm = _install_litellm()
    orig = litellm.completion
    with crashlab.instrument(providers=["litellm"]):
        assert litellm.completion is not orig
        with crashlab.record("lite"):
            r = litellm.completion("gpt-4o-mini", [{"role": "user", "content": "hi"}])
        assert r.choices[0].message.content == "lite hello"
        step = crashlab.load("last").steps[0]
        assert step.attributes["crashlab.api"] == "litellm.completion"
    assert litellm.completion is orig


def test_genai_constructor_is_recorded():
    genai = _install_genai()
    with crashlab.instrument(providers=["google_genai"]):
        client = genai.Client()
        with crashlab.record("gem"):
            resp = client.models.generate_content(model="gemini-2.5-flash", contents="hi")
    assert resp.text == "hello from genai"
    step = crashlab.load("last").steps[0]
    assert step.attributes["gen_ai.provider.name"] == "google_genai"


def test_wrap_on_auto_instrumented_client_does_not_double_record():
    openai = _install_openai()
    with crashlab.instrument(providers=["openai"]):
        client = openai.OpenAI(api_key="x")
        wrapped = crashlab.wrap(client)
        assert wrapped is client
        with crashlab.record("once"):
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
            wrapped.chat.completions.create(
                model="m", messages=[{"role": "user", "content": "again"}]
            )
    steps = crashlab.load("last").steps
    assert len(steps) == 2  # two calls, one step each — not four


async def test_async_openai_constructor_is_recorded():
    openai = _install_openai()
    with crashlab.instrument(providers=["openai"]):
        client = openai.AsyncOpenAI(api_key="x")
        with crashlab.record("aoai"):
            resp = await client.chat.completions.create(
                model="m", messages=[{"role": "user", "content": "hi"}]
            )
    assert resp.choices[0].message.content == "hello from openai"
    assert crashlab.load("last").steps[0].attributes["gen_ai.provider.name"] == "openai"


# --- frameworks, scopes, labels --------------------------------------------


class _FakeHaystackGenerator:
    """A framework that builds ``openai.OpenAI()`` lazily inside ``warm_up()``."""

    def __init__(self) -> None:
        self.client = None

    def warm_up(self) -> None:
        import openai

        if self.client is None:
            self.client = openai.OpenAI(api_key="framework-secret")

    def run(self, msg: str) -> str:
        self.warm_up()
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": msg}]
        )
        return resp.choices[0].message.content


def test_lazy_framework_warmup_is_captured():
    _install_openai()
    crashlab.instrument(providers=["openai"])
    gen = _FakeHaystackGenerator()
    with crashlab.record("haystack"):
        answer = gen.run("what is the price of A17?")
    assert answer == "hello from openai"
    run = crashlab.load("last")
    assert len(run.steps) == 1
    assert run.steps[0].request["messages"][0]["parts"][0]["content"] == "what is the price of A17?"
    # the framework still holds a plain instance, not a crashlab proxy class
    assert type(gen.client).__name__ == "OpenAI"
    assert getattr(gen.client, "__crashlab_wrapped__", False) is True


def test_agent_kwarg_labels_constructed_clients():
    openai = _install_openai()
    with crashlab.instrument(providers=["openai"], agent="shop-bot"):
        client = openai.OpenAI(api_key="x")
        with crashlab.record("named"):
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
    assert crashlab.load("last").steps[0].agent == "shop-bot"


def test_agent_scope_labels_when_instrument_has_no_agent():
    openai = _install_openai()
    with crashlab.instrument(providers=["openai"]):
        client = openai.OpenAI(api_key="x")
        with crashlab.record("scoped"):
            with crashlab.agent_scope("writer"):
                client.chat.completions.create(
                    model="m", messages=[{"role": "user", "content": "hi"}]
                )
    assert crashlab.load("last").steps[0].agent == "writer"


def test_from_crashlab_integrations_import():
    from crashlab.integrations import instrument, instrumented, uninstrument

    assert instrument is crashlab.instrument
    assert uninstrument is crashlab.uninstrument
    assert instrumented is crashlab.instrumented


def test_late_import_is_patched_by_the_hook(tmp_path, monkeypatch):
    """``instrument()`` first, ``import openai`` later, still records."""
    pkg = tmp_path / "fakepkgs"
    pkg.mkdir()
    (pkg / "openai.py").write_text(
        "class OpenAI:\n"
        "    base_url = 'http://fake.local/v1'\n"
        "    def __init__(self, api_key=None, **kwargs):\n"
        "        self.api_key = api_key\n"
        "        class C:\n"
        "            def create(self, **kw):\n"
        "                from tests.fakes import text_response\n"
        "                return text_response(1, 'late hello', model=kw.get('model', 'm'))\n"
        "        self.chat = type('Chat', (), {'completions': C()})()\n"
    )
    # hide a real (or previously faked) openai so the hook imports ours
    monkeypatch.syspath_prepend(str(pkg))
    sys.modules.pop("openai", None)
    with crashlab.instrument(providers=["openai"]):
        import openai

        client = openai.OpenAI(api_key="x")
        with crashlab.record("late"):
            resp = client.chat.completions.create(
                model="m", messages=[{"role": "user", "content": "hi"}]
            )
    assert resp.choices[0].message.content == "late hello"
    assert crashlab.load("last").steps[0].operation == "chat"
