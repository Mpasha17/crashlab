"""A secret split across consecutive strings of a list (a generator agent's streamed
pieces) is redacted in every piece, so the cassette cannot be pieced back together."""

from __future__ import annotations

from crashlab.store.redact import REDACTED, Redactor

KEY = "sk-proj-" + "Ab3dE6gH9jK2mN5pQ8sT1vW4yZ7bC0eF3hJ6kM9nP2rS5uV8xA1dG4"


def test_secret_split_over_pieces_is_redacted_in_each():
    pieces = ["Your key is ", KEY[:15], KEY[15:40], KEY[40:], " ok"]
    out = Redactor().redact({"output": pieces})["output"]
    assert out == ["Your key is ", REDACTED, REDACTED, REDACTED, " ok"]
    assert not any(KEY[15:40] in p or KEY[40:] in p for p in out)


def test_partial_pieces_keep_their_other_text():
    pieces = ["a " + KEY[:20], KEY[20:] + " b", "c"]
    assert Redactor().redact(pieces) == ["a " + REDACTED, REDACTED + " b", "c"]


def test_non_string_items_break_runs_and_plain_lists_are_untouched():
    pieces = [KEY[:10], 3, KEY[10:20]]  # not adjacent: not joined
    assert Redactor().redact(pieces) == pieces[:1] + [3] + pieces[2:]
    assert Redactor().redact(["x", "y", KEY]) == ["x", "y", REDACTED]
