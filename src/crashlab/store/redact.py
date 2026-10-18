"""Redaction applied to everything before it is written to disk.

Defaults strip API keys, bearer tokens and auth headers. Add your own rules with
``add_pattern(regex)`` (matches are replaced) or ``add_hook(fn)`` (``fn(str) -> str``
applied to every string value).
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from typing import Any

REDACTED = "[REDACTED]"

DEFAULT_SENSITIVE_KEYS = re.compile(
    r"^(authorization|proxy[-_]authorization|x[-_]api[-_]key|api[-_]?key|"
    r"openai[-_]api[-_]key|anthropic[-_]api[-_]key|x[-_]goog[-_]api[-_]key|"
    r"access[-_]?token|refresh[-_]?token|id[-_]?token|client[-_]?secret|secret|"
    r"password|passwd|cookie|set[-_]cookie|private[-_]?key)$",
    re.IGNORECASE,
)

DEFAULT_VALUE_PATTERNS: list[str] = [
    r"sk-(?:proj-|ant-(?:api\d+-)?|or-v1-)?[A-Za-z0-9_\-]{16,}",  # OpenAI / Anthropic / OpenRouter
    r"(?i)\bbearer\s+[A-Za-z0-9._\-~+/]{8,}=*",
    r"\bAKIA[0-9A-Z]{16}\b",  # AWS access key id
    r"\bgh[pousr]_[A-Za-z0-9]{20,}\b",  # GitHub tokens
    r"\bgithub_pat_[A-Za-z0-9_]{20,}\b",
    r"\bxai-[A-Za-z0-9]{20,}\b",
    r"\bAIza[0-9A-Za-z_\-]{35}\b",  # Google API key
    r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b",  # Slack
    r"\bhf_[A-Za-z0-9]{20,}\b",  # Hugging Face
]

Hook = Callable[[str], str]


class Redactor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.key_patterns: list[re.Pattern[str]] = [DEFAULT_SENSITIVE_KEYS]
            self.value_patterns: list[re.Pattern[str]] = [
                re.compile(p) for p in DEFAULT_VALUE_PATTERNS
            ]
            self.hooks: list[Hook] = []

    def add_pattern(self, pattern: str | re.Pattern[str]) -> None:
        with self._lock:
            self.value_patterns.append(re.compile(pattern) if isinstance(pattern, str) else pattern)

    def add_key(self, pattern: str | re.Pattern[str]) -> None:
        """Redact the whole value of any dict key matching this regex."""
        compiled = re.compile(pattern, re.IGNORECASE) if isinstance(pattern, str) else pattern
        with self._lock:
            self.key_patterns.append(compiled)

    def add_hook(self, hook: Hook) -> None:
        with self._lock:
            self.hooks.append(hook)

    def _sensitive_key(self, key: str) -> bool:
        return any(p.search(key) for p in self.key_patterns)

    def redact_str(self, value: str) -> str:
        for pat in self.value_patterns:
            value = pat.sub(REDACTED, value)
        for hook in self.hooks:
            value = hook(value)
        return value

    def redact(self, obj: Any) -> Any:
        """Return a redacted deep copy of a JSON-like structure."""
        if isinstance(obj, str):
            return self.redact_str(obj)
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if isinstance(k, str) and self._sensitive_key(k) and v not in (None, ""):
                    out[k] = REDACTED
                else:
                    out[k] = self.redact(v)
            return out
        if isinstance(obj, (list, tuple)):
            # header lists like [("Authorization", "Bearer ..."), ...]
            if (
                len(obj) == 2
                and isinstance(obj[0], str)
                and isinstance(obj[1], str)
                and self._sensitive_key(obj[0])
                and obj[1]
            ):
                return [obj[0], REDACTED]
            return [self.redact(v) for v in obj]
        return obj


_default = Redactor()


def get_redactor() -> Redactor:
    return _default


def redact(obj: Any) -> Any:
    return _default.redact(obj)


def add_pattern(pattern: str | re.Pattern[str]) -> None:
    _default.add_pattern(pattern)


def add_key(pattern: str | re.Pattern[str]) -> None:
    _default.add_key(pattern)


def add_hook(hook: Hook) -> None:
    _default.add_hook(hook)


def reset() -> None:
    _default.reset()
