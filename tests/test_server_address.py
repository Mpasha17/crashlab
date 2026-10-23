"""``server_address`` extracts hostnames from client base_url variants."""

from __future__ import annotations

from types import SimpleNamespace

from crashlab.capture._proxy import server_address


def test_server_address_https_and_schemeless():
    assert server_address(SimpleNamespace(base_url="https://api.openai.com/v1")) == "api.openai.com"
    assert server_address(SimpleNamespace(base_url="http://localhost:8080/v1")) == "localhost"
    assert server_address(SimpleNamespace(base_url="api.groq.com/openai/v1")) == "api.groq.com"
    assert server_address(SimpleNamespace(base_url="")) is None
    assert server_address(SimpleNamespace()) is None


def test_responses_only_client_records_with_raw_response(crashlab_dir):
    """OpenAI-compatible clients that only expose ``responses`` still wrap."""
    import crashlab
    from crashlab.capture.openai import WrappedOpenAI

    class _Responses:
        def __init__(self) -> None:
            self.calls = 0

            def raw_create(**kwargs):
                self.calls += 1

                class _Raw:
                    status_code = 200
                    headers = {}

                    def parse(self):
                        return SimpleNamespace(
                            output_text="hi",
                            model="m",
                            id="resp-x",
                            output=[],
                            usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
                        )

                return _Raw()

            self.with_raw_response = SimpleNamespace(create=raw_create)

        def create(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(output_text="hi", model="m", id="resp-x", output=[])

    class _Client:
        # no chat.completions — responses only
        def __init__(self):
            self.responses = _Responses()
            self.base_url = "api.example.test/v1"

    inner = _Client()
    wrapped = crashlab.wrap(inner)
    assert isinstance(wrapped, WrappedOpenAI)
    assert wrapped.responses.with_raw_response  # attribute present

    @crashlab.agent(name="responses-only")
    def run() -> str:
        return wrapped.responses.with_raw_response.create(model="m", input="x").parse().output_text

    assert run() == "hi"
    rec = crashlab.load("last")
    assert len(rec.steps) == 1
    assert rec.steps[0].attributes.get("server.address") == "api.example.test"
    assert rec.steps[0].attributes.get("crashlab.api") == "openai.responses"


def test_wrap_responses_only_custom_provider_name(crashlab_dir):
    import crashlab
    from types import SimpleNamespace

    class _Responses:
        def create(self, **kwargs):
            return SimpleNamespace(
                output_text="ok",
                model="m",
                id="resp-x",
                output=[],
                usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
            )

    class _Client:
        def __init__(self):
            self.responses = _Responses()
            self.base_url = "https://api.groq.com/openai/v1"

    @crashlab.agent(name="prov")
    def run() -> str:
        c = crashlab.wrap(_Client(), provider="groq")
        return c.responses.create(model="m", input="x").output_text

    assert run() == "ok"
    step = crashlab.load("last").steps[0]
    assert step.attributes.get("gen_ai.provider.name") == "groq"
    assert step.attributes.get("server.address") == "api.groq.com"
