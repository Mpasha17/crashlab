"""Helpers for driving real SDK clients through a mocked HTTP transport."""

from __future__ import annotations

import json

import pytest


def httpx_module():
    try:  # openai>=3 / anthropic>=1 ship on httpx2; older releases use httpx
        import httpx2 as httpx
    except ImportError:
        httpx = pytest.importorskip("httpx")
    return httpx


def sse(events: list[dict], *, named: bool, done: bool = False) -> bytes:
    out = []
    for e in events:
        if named:
            out.append(f"event: {e['type']}\n")
        out.append(f"data: {json.dumps(e)}\n\n")
    if done:
        out.append("data: [DONE]\n\n")
    return "".join(out).encode()


class Script:
    """Scripted HTTP responses: each item is a JSON dict or (bytes, content_type)."""

    def __init__(self, items: list) -> None:
        self.items = list(items)
        self.requests: list = []

    def handler(self, httpx):
        def handle(request):
            self.requests.append(request)
            item = self.items.pop(0)
            if isinstance(item, tuple):
                body, ctype = item
                return httpx.Response(200, headers={"content-type": ctype}, content=body)
            if isinstance(item, int):
                return httpx.Response(item, json={"type": "error", "error": {
                    "type": "api_error", "message": "boom"}})  # fmt: skip
            return httpx.Response(200, json=item)

        return handle
