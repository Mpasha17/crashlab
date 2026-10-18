"""The quotable text of a step, by field. The compactor shows these texts (truncated) and
the evidence validator searches the full ones, so a quote the diagnoser copied from the
trace is found again."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from ..models import Step


def as_text(value: Any) -> str:
    """Strings as-is, everything else as compact JSON (the way the trace shows them)."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def part_text(part: dict[str, Any]) -> str:
    kind = part.get("type")
    if kind in ("text", "reasoning", "refusal"):
        return str(part.get("content") or "")
    if kind == "tool_call":
        return f"{part.get('name')}({as_text(part.get('arguments'))})"
    if kind == "tool_call_response":
        return as_text(part.get("response"))
    return as_text({k: v for k, v in part.items() if k != "type"})


def message_text(msg: dict[str, Any]) -> str:
    parts = msg.get("parts")
    if isinstance(parts, list):
        return "\n".join(part_text(p) for p in parts if isinstance(p, dict))
    return as_text(msg.get("content"))


def reply_parts(step: Step, kind: str) -> list[str]:
    out = []
    for msg in (step.response or {}).get("messages") or []:
        for part in msg.get("parts") or [] if isinstance(msg, dict) else []:
            if isinstance(part, dict) and part.get("type") == kind and part.get("content"):
                out.append(str(part["content"]))
        if isinstance(msg, dict) and kind == "text" and isinstance(msg.get("content"), str):
            out.append(msg["content"])
    return out


def tool_calls(step: Step) -> list[dict[str, Any]]:
    return [c for c in (step.response or {}).get("tool_calls") or [] if isinstance(c, dict)]


def step_fields(step: Step) -> dict[str, str]:
    """Field name -> full text, for everything a diagnosis may quote from ``step``."""
    out: dict[str, str] = {}
    if step.operation == "chat":
        req = step.request or {}
        if req.get("system"):
            out["request.system"] = as_text(req["system"])
        for i, msg in enumerate(req.get("messages") or []):
            if isinstance(msg, dict):
                out[f"request.messages[{i}]"] = message_text(msg)
        text = "\n".join(reply_parts(step, "text"))
        if text:
            out["response.text"] = text
        reasoning = "\n".join(reply_parts(step, "reasoning"))
        if reasoning:
            out["response.reasoning"] = reasoning
        for i, call in enumerate(tool_calls(step)):
            out[f"response.tool_calls[{i}]"] = (
                f"{call.get('name')}({as_text(call.get('arguments'))})"
            )
    else:
        out["args"] = as_text(step.args)
        out["call"] = f"{step.name}({out['args']})"  # how the trace shows the call
        if step.response is not None and step.error is None:
            out["result"] = as_text(step.result)
    if step.error is not None:
        out["error"] = f"{step.error.type}: {step.error.message}"
    return out


_QUOTES = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }
)
_ELLIPSIS = re.compile(r"^(\.\.\.|…)+|(\.\.\.|…)+$")
# the labels compact() puts in front of step content ("reply: ", "calls tool: " ...): they
# are crashlab's, not the trace's, so a quote that starts with one is matched without it
_LABEL = re.compile(
    r"^\s*(?:-\s*)?(?:\[\d+\]\s*)?(?:calls tool|reply|reasoning|result|args|error"
    r"|new (?:user|assistant|tool|system) message|system prompt changed)\s*:\s*",
    re.IGNORECASE,
)
_ESCAPES = re.compile(r"\\[ntr]")


def normalize(text: str) -> str:
    """Whitespace collapsed, NFKC, typographic quotes and dashes made plain; JSON escapes
    of quotes undone. Applied to both sides of a verbatim comparison."""
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES).replace('\\"', '"')
    return " ".join(_ESCAPES.sub(" ", text).split())


def find_quote(step: Step, quote: str) -> str | None:
    """The field of ``step`` that contains ``quote`` verbatim (after normalization), or
    None. Leading / trailing ellipses in the quote are ignored."""
    q = normalize(_ELLIPSIS.sub("", normalize(_LABEL.sub("", quote))).strip())
    if len(q) < 3:
        return None
    for name, text in step_fields(step).items():
        if q in normalize(text):
            return name
    return None
