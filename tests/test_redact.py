"""Redaction: nothing secret reaches the cassette, run.json or the SQLite index."""

from __future__ import annotations

import json
import sqlite3

import crashlab
from crashlab.store.redact import REDACTED, Redactor
from tests.fakes import FakeAnthropic, FakeOpenAI, a_text, text_response

SECRETS = {
    "openai": "sk-proj-AAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "anthropic": "sk-ant-api03-BBBBBBBBBBBBBBBBBBBBBBBBBB",
    "bearer": "Bearer eyJhbGciOiJIUzI1NiJ9.CCCCCCCCCCCC",
    "github": "ghp_DDDDDDDDDDDDDDDDDDDDDDDDDD",
    "aws": "AKIAEEEEEEEEEEEEEEEE",
    "groq": "gsk_" + "F" * 30,
    "dashscope": "AQ.Ab8" + "G" * 16,
}


def _everything_on_disk(store) -> str:
    files = "".join(p.read_text() for p in store.rglob("*") if p.suffix in (".json", ".jsonl"))
    db = "\n".join(sqlite3.connect(store / "index.sqlite").iterdump())
    return files + db


def _assert_clean(store):
    blob = _everything_on_disk(store)
    for value in SECRETS.values():
        token = value.split()[-1]
        assert token not in blob, f"leaked {token[:12]}..."


def test_header_variants_and_nested_keys():
    r = Redactor()
    data = {
        "extra_headers": {
            "x-api-key": "k1",
            "X-API-KEY": "k2",
            "anthropic-api-key": "k3",
            "Proxy-Authorization": "Basic Zm9vOmJhcg==",
            "Cookie": "session=abc",
            "x-trace-id": "keep-me",
        },
        "nested": {"deep": [{"client_secret": "s", "password": "p", "token_count": 3}]},
        "header_list": [["Authorization", "Bearer abcdefghijk"], ["Accept", "json"]],
        "max_tokens": 10,
        "api_key": None,
    }
    out = r.redact(data)
    h = out["extra_headers"]
    assert all(h[k] == REDACTED for k in ("x-api-key", "X-API-KEY", "anthropic-api-key",
                                          "Proxy-Authorization", "Cookie"))  # fmt: skip
    assert h["x-trace-id"] == "keep-me"
    assert out["nested"]["deep"][0] == {"client_secret": REDACTED, "password": REDACTED,
                                        "token_count": 3}  # fmt: skip
    assert out["header_list"] == [["Authorization", REDACTED], ["Accept", "json"]]
    assert out["max_tokens"] == 10 and out["api_key"] is None
    assert data["extra_headers"]["x-api-key"] == "k1"  # input untouched


def test_value_patterns():
    r = Redactor()
    text = " | ".join(SECRETS.values())
    out = r.redact_str(text)
    for value in SECRETS.values():
        assert value.split()[-1] not in out
    assert r.redact_str("max_tokens=100 and sk-short") == "max_tokens=100 and sk-short"


def test_user_hooks_patterns_and_keys(crashlab_dir):
    crashlab.redaction.add_pattern(r"\b\d{3}-\d{2}-\d{4}\b")  # SSN-like
    crashlab.redaction.add_key(r"^customer_email$")
    crashlab.redaction.add_hook(lambda s: s.replace("Alice Smith", "<customer>"))

    @crashlab.tool()
    def lookup(customer_email: str, note: str) -> dict:
        return {"note": note, "ssn": "123-45-6789"}

    with crashlab.record("hooks", input={"who": "Alice Smith"}):
        lookup("alice@example.com", "Alice Smith called")
    run = crashlab.load("last")
    step = run.steps[0]
    assert run.input == {"who": "<customer>"}
    assert step.args == {"customer_email": REDACTED, "note": "<customer> called"}
    assert step.result == {"note": "<customer> called", "ssn": REDACTED}
    blob = _everything_on_disk(crashlab_dir)
    assert "alice@example.com" not in blob and "123-45-6789" not in blob
    assert "Alice Smith" not in blob


def test_raw_request_and_response_never_contain_secrets(crashlab_dir):
    openai_fake = FakeOpenAI([text_response(1, f"your key {SECRETS['openai']} was rotated")])
    anthropic_fake = FakeAnthropic([a_text(1, f"token {SECRETS['github']} found")])
    o, a = crashlab.wrap(openai_fake), crashlab.wrap(anthropic_fake)
    with crashlab.record("raw", input={"aws": SECRETS["aws"]}):
        o.chat.completions.create(
            model="m",
            messages=[{"role": "user", "content": f"use {SECRETS['openai']}"}],
            extra_headers={"Authorization": SECRETS["bearer"], "OpenAI-Organization": "org"},
            extra_body={"metadata": {"api_key": SECRETS["openai"]}},
        )
        a.messages.create(
            model="c",
            max_tokens=5,
            messages=[{"role": "user", "content": "hi"}],
            extra_headers={"x-api-key": SECRETS["anthropic"], "anthropic-beta": "x"},
        )
    run = crashlab.load("last")
    o_step, a_step = run.steps
    assert o_step.raw_request["extra_headers"]["Authorization"] == REDACTED
    assert o_step.raw_request["extra_body"]["metadata"]["api_key"] == REDACTED
    assert "[REDACTED]" in o_step.raw_response["choices"][0]["message"]["content"]
    assert a_step.raw_request["extra_headers"] == {"x-api-key": REDACTED, "anthropic-beta": "x"}
    assert "[REDACTED]" in a_step.raw_response["content"][0]["text"]
    _assert_clean(crashlab_dir)
    # the live calls still received the real values
    assert anthropic_fake.calls[0]["extra_headers"]["x-api-key"] == SECRETS["anthropic"]


def test_error_messages_redacted(crashlab_dir):
    class AuthError(Exception):
        status_code = 401

    fake = FakeOpenAI([AuthError(f"Incorrect API key provided: {SECRETS['openai']}")])
    client = crashlab.wrap(fake)
    try:
        with crashlab.record("err"):
            client.chat.completions.create(model="m", messages=[])
    except AuthError:
        pass
    run = crashlab.load("last")
    assert run.error.message == f"Incorrect API key provided: {REDACTED}"
    assert run.steps[0].error.status_code == 401
    assert SECRETS["openai"] not in json.dumps(run.model_dump(mode="json"))
    _assert_clean(crashlab_dir)


def test_groq_and_dashscope_shaped_values_redacted_but_prefix_names_kept():
    r = Redactor()
    assert SECRETS["groq"] not in r.redact_str(f"key={SECRETS['groq']}")
    assert SECRETS["dashscope"] not in r.redact_str(f"key={SECRETS['dashscope']}")
    # docs that only name the prefix must stay readable
    assert r.redact_str("look for gsk_ and AQ.Ab8 in scan.sh") == (
        "look for gsk_ and AQ.Ab8 in scan.sh"
    )


def test_groq_shaped_token_in_nested_message_content():
    r = Redactor()
    key = SECRETS["groq"]
    out = r.redact({"messages": [{"role": "user", "content": f"use {key} please"}]})
    assert key not in str(out)
    assert "use " in out["messages"][0]["content"]


def test_dashscope_shaped_token_in_nested_message_content():
    r = Redactor()
    key = SECRETS["dashscope"]
    out = r.redact({"messages": [{"role": "user", "content": f"token {key} end"}]})
    assert key not in str(out)
    assert out["messages"][0]["content"].startswith("token ")
    assert out["messages"][0]["content"].endswith(" end")
