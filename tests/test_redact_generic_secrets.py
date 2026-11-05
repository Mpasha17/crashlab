"""Secrets that are not provider-prefixed keys: suffixed / camelCase header and argument
names, credentials in URLs, ``api_key=`` in error strings and env-style lines."""

from __future__ import annotations

import contextlib

import pytest

from crashlab.store.redact import REDACTED, Redactor

SECRET = "plainsecretvalue123"


@pytest.mark.parametrize(
    "key",
    [
        "X-Auth-Token", "auth_token", "api_token", "bearer_token", "aws_session_token",
        "aws_secret_access_key", "secret_key", "AWS_SECRET_ACCESS_KEY", "token",
        "apiKey", "authToken", "accessToken", "clientSecret", "openaiApiKey",
        "db_password", "x-api-token", "service_account_private_key", "session-cookie",
    ],
)  # fmt: skip
def test_sensitive_key_variants(key):
    assert Redactor().redact({key: SECRET}) == {key: REDACTED}


@pytest.mark.parametrize(
    "key",
    ["max_tokens", "token_count", "input_tokens", "next_page_token", "tokenizer",
     "aws_access_key_id", "password_hint", "secretary", "keyword", "monkey"],
)  # fmt: skip
def test_non_secret_keys_kept(key):
    assert Redactor().redact({key: "keep-me"}) == {key: "keep-me"}


def test_booleans_under_secret_names_kept():
    assert Redactor().redact({"has_password": True, "secret": False}) == {
        "has_password": True,
        "secret": False,
    }


@pytest.mark.parametrize(
    ("text", "kept"),
    [
        ("https://user:" + SECRET + "@db.example.com/x", "https://user:"),
        ("GET https://api.x.com/v1?api_key=" + SECRET + "&a=1 failed", "&a=1 failed"),
        ("https://h/v1/models?key=" + SECRET, "?key="),
        ("https://h/cb?access_token=" + SECRET + "#x", "#x"),
        ("OPENAI_API_KEY=" + SECRET, "OPENAI_API_KEY="),
        ("export GITHUB_TOKEN='" + SECRET + "'", "GITHUB_TOKEN="),
        ("DB_PASSWORD: " + SECRET, "DB_PASSWORD"),
        ('{"api_key": "' + SECRET + '", "n": 1}', '"n": 1'),
        ('{"password":"' + SECRET + '"}', '"password"'),
        ("Authorization: Basic dXNlcjpodW50ZXIycGFzcw==", "Authorization"),
    ],
)
def test_secret_values_in_free_text(text, kept):
    out = Redactor().redact_str(text)
    assert SECRET not in out and "dXNlcjpodW50ZXIycGFzcw" not in out, out
    assert REDACTED in out and kept in out, out


@pytest.mark.parametrize(
    "text",
    [
        "a basic understanding of the problem",
        "the password field is required",
        "set max_tokens=100 and top_k=5",
        "https://example.com/search?q=secret+recipes&page=2",
        "MY_TOKEN_COUNT=12",
    ],
)
def test_prose_not_redacted(text):
    assert Redactor().redact_str(text) == text


def test_url_secret_split_across_stream_pieces():
    pieces = ["see https://api.x.com/v1?api_", "key=plainsecret", "value123 now"]
    out = Redactor().redact(pieces)
    assert "plainsecret" not in "".join(out) and "value123" not in "".join(out), out


def test_tool_args_results_and_errors_on_disk(crashlab_dir):
    """End to end: a tool taking ``api_token``, returning a DSN with a password and
    failing with a URL carrying ``?api_key=`` leaves none of them in the store."""
    import sqlite3

    import crashlab

    @crashlab.tool()
    def connect(dsn_host: str, api_token: str) -> dict:
        return {"dsn": f"postgres://app:{SECRET}@{dsn_host}/db", "ok": True}

    @crashlab.tool()
    def fetch(url: str) -> str:
        raise RuntimeError(f"GET {url}?api_key={SECRET} -> 500")

    @crashlab.agent(name="secrets-bot")
    def bot() -> str:
        connect("db.local", api_token=SECRET)
        with contextlib.suppress(RuntimeError):
            fetch("https://api.example.com/v1/items")
        return "done"

    bot()
    blob = "".join(
        p.read_text() for p in crashlab_dir.rglob("*") if p.suffix in (".json", ".jsonl")
    )
    blob += "\n".join(sqlite3.connect(crashlab_dir / "index.sqlite").iterdump())
    assert SECRET not in blob
    run = crashlab.load("last")
    assert run.steps[0].request["args"]["api_token"] == REDACTED
    assert "postgres://app:" + REDACTED + "@db.local/db" in str(run.steps[0].response)
    crashlab.replay(run, fn=bot)  # redaction is applied the same way on replay: no divergence


def test_recorded_env_dsn_password_redacted(crashlab_dir, monkeypatch):
    """``@crashlab.agent(env=["DATABASE_URL"])`` passes the name check (no KEY/TOKEN)
    but the value carries a password."""
    import crashlab

    monkeypatch.setenv("DATABASE_URL", f"postgres://app:{SECRET}@db.local/shop")

    @crashlab.agent(name="env-dsn", env=["DATABASE_URL"])
    def bot() -> int:
        return 1

    bot()
    env = crashlab.load("last").env["vars"]
    assert env["DATABASE_URL"] == f"postgres://app:{REDACTED}@db.local/shop"
