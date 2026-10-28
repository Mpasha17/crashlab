"""Default redaction patterns started with ``\\b``, which keeps ``re`` from scanning for
their literal prefix: a 20 MB tool result took ~1.9 s to redact (10 patterns x 0.19 s).
They now check the word boundary after the literal (``KEY(?<!\\wKEY)``): same matches,
~15x faster."""

from __future__ import annotations

import time

import pytest

from crashlab.store.redact import REDACTED, Redactor

AWS = "AKIA" + "ABCDEFGHIJKLMNOP"
GROQ = "gsk_" + "a1B2c3D4e5F6g7H8i9J0k1"


@pytest.mark.parametrize(
    ("text", "redacted"),
    [
        (AWS, True),  # start of string
        (f"key={AWS}", True),
        (f"-{AWS} ", True),
        (f"x{AWS}", False),  # inside a longer word: not a key (as with \b)
        (f"é{AWS}", False),  # unicode word character
        (f"_{GROQ}", False),
        (f"({GROQ})", True),
        ("Authorization: BEARER abcdefgh12345", True),
        ("xbearer abcdefgh12345", False),
        ("ghp_" + "A" * 24, True),
        ("aghp_" + "A" * 24, False),
        ("AQ.Ab8" + "z" * 10, True),
    ],
)
def test_word_boundary_kept(text, redacted):
    out = Redactor().redact_str(text)
    assert (REDACTED in out) is redacted, out


def test_large_string_redacts_fast():
    r = Redactor()
    big = "x" * (20 * 1024 * 1024)
    t = time.perf_counter()
    assert r.redact_str(big + " " + GROQ).endswith(REDACTED)
    assert time.perf_counter() - t < 1.5  # was ~1.9 s; now ~0.15 s
