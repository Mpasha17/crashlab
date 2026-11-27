"""Memory and state debugging for long-running and multi-turn agents.

Five deterministic detectors (``memory.*`` findings, run by ``crashlab check`` and while
recording; configure under ``[tool.crashlab.checks.memory]``) catch an agent that forgets,
contradicts or loses context:

``memory.context_truncation``  earlier messages silently missing from a later request
``memory.forgotten_fact``      a fact the user gave is asked for again or contradicted
``memory.self_contradiction``  conflicting numbers / dates / yes-no across replies
``memory.stale_tool_result``   a reply uses a value a newer tool call replaced
``memory.unbounded_growth``    prompt (or state) size growing faster every turn

plus :func:`snapshot` (record your agent's state mid-run), :func:`snapshots` and
:func:`state_diff` to read it back, and :func:`analyze` / :func:`confirm_with_judge` behind
``crashlab memory``. Everything is heuristic and English-centric by design: findings carry
verbatim evidence so you can judge them, and each detector can be switched off.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .checks.detectors import Detector, History, as_text, canonical, excerpt, response_text
from .context_audit import _is_tool_result, _message_body
from .models import Evidence, Finding, Run, Step

SNAPSHOT_STEP = "crashlab.snapshot"
SNAPSHOT_NAME_ATTR = "crashlab.snapshot.name"
MAX_SNAPSHOT_CHARS = 20_000
_LOOKBACK = 40  # earlier requests of an agent compared with each new one
TURN_ATTR = "crashlab.turn"

# --- conversation view ------------------------------------------------------------------


@dataclass
class Msg:
    fp: str
    role: str  # system / user / assistant / tool / other
    text: str
    idx: int  # position in request.messages
    texts: list[tuple[str, str]]  # (field suffix, text)
    raw: dict[str, Any] | None


@dataclass
class ChatView:
    step: Step
    msgs: list[Msg]
    turn: int | None
    system: str
    fps: set[str] = field(default_factory=set)

    @property
    def idx(self) -> int:
        return self.step.idx

    @property
    def agent(self) -> str | None:
        return self.step.agent


def _texts(msg: dict[str, Any]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    content = msg.get("content")
    if isinstance(content, str):
        out.append((".content", content))
    elif isinstance(content, list):
        for j, p in enumerate(content):
            if isinstance(p, str):
                out.append((f".content[{j}]", p))
            elif isinstance(p, dict) and isinstance(p.get("text"), str):
                out.append((f".content[{j}].text", p["text"]))
    for j, p in enumerate(msg.get("parts") or []):
        if isinstance(p, dict) and p.get("type") == "text":
            t = p.get("content", p.get("text"))
            if isinstance(t, str):
                out.append((f".parts[{j}].content", t))
    return [(f, t) for f, t in out if t]


def _role(msg: dict[str, Any]) -> str:
    role = str(msg.get("role") or "other").strip().lower() or "other"
    if role == "developer":
        role = "system"
    return "tool" if _is_tool_result(msg, role) else role


def _view(step: Step) -> ChatView:
    req = step.request or {}
    msgs: list[Msg] = []
    for i, m in enumerate(req.get("messages") or []):
        if not isinstance(m, dict):
            continue
        role = _role(m)
        texts = _texts(m)
        text = "\n".join(t for _, t in texts)
        body = _message_body(m)
        msgs.append(Msg(f"{role}\n{body}", role, text, i, texts, m))
    system = req.get("system")
    sys_text = as_text(system) if system else ""
    sys_text += "\n".join(m.text for m in msgs if m.role == "system")
    turn_attr = step.attributes.get(TURN_ATTR)
    turn = turn_attr if isinstance(turn_attr, int) and not isinstance(turn_attr, bool) else None
    if turn is None:
        users = sum(1 for m in msgs if m.role == "user")
        turn = users or None
    view = ChatView(step, msgs, turn, sys_text)
    view.fps = {m.fp for m in msgs if m.role != "system"}
    return view


def _views(h: History) -> list[ChatView]:
    cached = getattr(h, "_memory_views", None)
    if cached is not None and cached[0] == len(h.steps):
        return cached[1]
    out = [_view(s) for s in sorted(h.steps, key=lambda s: s.idx)
           if s.operation == "chat" and s.error is None]  # fmt: skip
    h._memory_views = (len(h.steps), out)  # type: ignore[attr-defined]
    return out


def _by_agent(views: list[ChatView]) -> dict[str | None, list[ChatView]]:
    out: dict[str | None, list[ChatView]] = defaultdict(list)
    for v in views:
        out[v.agent].append(v)
    return out


def _ev_msg(view: ChatView, m: Msg, start: int = 0) -> Evidence:
    if m.texts:
        suffix, text = m.texts[0]
        return Evidence(step_idx=view.idx, field=f"request.messages[{m.idx}]{suffix}",
                        quote=excerpt(text, start))  # fmt: skip
    return Evidence(step_idx=view.idx, field=f"request.messages[{m.idx}]",
                    quote=excerpt(as_text(m.raw), 0))  # fmt: skip


def _short(text: str, width: int = 70) -> str:
    one = " ".join(text.split())
    return one if len(one) <= width else one[: width - 1].rstrip() + "…"


def _words(text: str) -> frozenset[str]:
    return frozenset(text.split())


# --- facts the user gave ------------------------------------------------------------------

_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
_MONTH_RE = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
             r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")  # fmt: skip
_DATE = re.compile(
    rf"(?P<iso>\b\d{{4}}-\d{{2}}-\d{{2}}\b)|(?P<slash>\b\d{{1,2}}/\d{{1,2}}/\d{{2,4}}\b)|"
    rf"(?P<md>\b(?P<m1>{_MONTH_RE})\.?\s+(?P<d1>\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(?P<y1>\d{{4}}))?\b)|"
    rf"(?P<dm>\b(?P<d2>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<m2>{_MONTH_RE})\b(?:,?\s+(?P<y2>\d{{4}}))?)",
    re.I,
)
_DATE_LABELS = ("birthday|birth date|date of birth|dob|check-in|check in|check-out|check out|"
                "arrival|departure|appointment|deadline|due date|delivery|start date|end date|"
                "flight|travel|booking|meeting|event|wedding|anniversary|move-in|pickup"
                )  # fmt: skip
_DATE_LABEL = re.compile(rf"\b({_DATE_LABELS})\b", re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"(?<![\w.])\+?\d[\d\s().-]{7,}\d(?![\w])")
_ID_LABELS = ("order|ticket|booking|reservation|invoice|account|case|customer|reference|"
              "confirmation|tracking|policy|claim|shipment|subscription|member|patient|"
              "transaction|payment")  # fmt: skip
_ID = re.compile(
    rf"\b(?P<label>{_ID_LABELS})(?:\s+(?:number|no\.?|id|ref(?:erence)?|code))?\s*"
    rf"(?:is|are|:|=|#|-)?\s*(?:is\s+)?#?(?P<v>(?=[\w-]*\d)[A-Za-z0-9][\w-]{{2,}})",
    re.I,
)
_NAME_USER = re.compile(
    r"(?i:my name is|my name's|call me|name:|i am|i'm|this is)\s+"
    r"(?P<v>[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)?)"
)
_NAME_BAD = re.compile(
    r"^(?:fine|good|great|sorry|not|looking|trying|interested|calling|writing|having|unable|"
    r"happy|sure|ok|okay|here|back|afraid|just|new|the|a|an|your|so|very|really|still|"
    r"currently|also|using|wondering|asking|reaching|hoping|thinking|wanting|getting|"
    r"waiting|curious|confused|upset|angry|glad|ready|done|late|stuck|from|in|at|on|with|"
    r"not|yes|no|please|thanks|hello|hi)\b|(?:ing|ed)$",
    re.I,
)
_NAME_ASSISTANT = re.compile(
    r"\b(?i:hi|hello|hey|thanks|thank you|dear|welcome|sure|ok|okay|great|got it),?\s+"
    r"(?P<v>[A-Z][a-z]+)\b"
)
_NAME_STOP = _words(
    "For You The I Your Order Please We It That This Let Great Sure Got Understood Certainly "
    "Absolutely Of No Yes Again Sir Madam Ok Okay There Everyone All Team Here One Can May"
)
_MONEY = re.compile(
    r"\b(?P<label>budget|price limit|spending limit)\b\D{0,15}?"
    r"(?P<v>[$€£]\s?\d(?:[\d,]*\d)?(?:\.\d+)?k?|\d(?:[\d,]*\d)?(?:\.\d+)?\s?(?:usd|eur|dollars|euros|k))",
    re.I,
)
_UNITS = ("people|guests|adults|children|kids|passengers|travellers|travelers|nights|rooms|"
          "seats|tickets|items|units|pieces")  # fmt: skip
_COUNT = re.compile(
    rf"(?:\b(?P<v>\d{{1,3}})\s+(?P<u>{_UNITS})\b|\bparty of\s+(?P<v2>\d{{1,3}})\b)", re.I
)
_PARTY_UNITS = _words(
    "people guests adults children kids passengers travellers travelers nights rooms seats"
)


@dataclass
class Fact:
    kind: str  # email / phone / id / name / date / money / count
    label: str  # "order", "email", "birthday", ...
    value: str
    norm: str
    start: int
    end: int


def _date_norm(m: re.Match[str]) -> str:
    g = m.groupdict()
    if g["iso"]:
        return g["iso"]
    if g["slash"]:
        return g["slash"]
    mon = (g["m1"] or g["m2"] or "")[:3].lower()
    day = int(g["d1"] or g["d2"])
    year = g["y1"] or g["y2"]
    mm = _MONTHS.index(mon) + 1 if mon in _MONTHS else 0
    return f"{year}-{mm:02d}-{day:02d}" if year else f"--{mm:02d}-{day:02d}"


def _same_date(a: str, b: str) -> bool:
    if a == b:
        return True
    if a.startswith("--") or b.startswith("--"):
        return a[-5:] == b[-5:] and a[-5:-4] == "-" and "/" not in a + b
    return False


def _norm_num(text: str) -> str:
    t = text.lower().replace(",", "").replace(" ", "")
    mult = 1.0
    if t.endswith("k"):
        mult, t = 1000.0, t[:-1]
    digits = re.sub(r"[^\d.]", "", t)
    try:
        return str(round(float(digits) * mult, 2)).removesuffix(".0")
    except ValueError:
        return digits


def extract_facts(text: str, *, speaker: str = "user") -> list[Fact]:
    """Facts in ``text``: emails, phones, labelled ids, names (user: "my name is ...";
    agent: a greeting), dates, budgets and party sizes. Heuristic, English."""
    out: list[Fact] = []
    for m in _EMAIL.finditer(text):
        out.append(Fact("email", "email", m.group(0), m.group(0).lower(), m.start(), m.end()))
    spans = [(f.start, f.end) for f in out]
    for m in _PHONE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if (not 9 <= len(digits) <= 15 or _DATE.search(m.group(0))
                or any(a <= m.start() < b for a, b in spans)):  # fmt: skip
            continue
        out.append(Fact("phone", "phone", m.group(0).strip(), digits[-10:], m.start(), m.end()))
    for m in _ID.finditer(text):
        v = m.group("v")
        out.append(Fact("id", m.group("label").lower(), v, re.sub(r"\W", "", v).lower(),
                        m.start("v"), m.end("v")))  # fmt: skip
    name_re = _NAME_USER if speaker == "user" else _NAME_ASSISTANT
    for m in name_re.finditer(text):
        v = m.group("v")
        first = v.split()[0]
        if speaker == "user" and _NAME_BAD.search(first):
            continue
        if speaker != "user" and first in _NAME_STOP:
            continue
        out.append(Fact("name", "name", v, v.lower(), m.start("v"), m.end("v")))
    for m in _DATE.finditer(text):
        before = text[max(0, m.start() - 40) : m.start()]
        lab = _DATE_LABEL.findall(before)
        out.append(Fact("date", lab[-1].lower() if lab else "date", m.group(0),
                        _date_norm(m), m.start(), m.end()))  # fmt: skip
    for m in _MONEY.finditer(text):
        out.append(Fact("money", "budget", m.group("v"), _norm_num(m.group("v")),
                        m.start("v"), m.end("v")))  # fmt: skip
    for m in _COUNT.finditer(text):
        v = m.group("v") or m.group("v2")
        unit = (m.group("u") or "people").lower()
        out.append(Fact("count", unit, v, v.lstrip("0") or "0", m.start(), m.end()))
    return out


_CUE = re.compile(
    r"\b(?:what|which|when|where|how many|how much|could you|can you|would you|please|may i|"
    r"do you have|provide|share|confirm|repeat|remind|tell me|enter|send|give|verify)\b",
    re.I,
)
_ASK = {
    "email": re.compile(r"\be-?mail(?:\s+address)?\b", re.I),
    "phone": re.compile(
        r"\b(?:phone|mobile|telephone|cell)(?:\s+number)?\b|\bcontact number\b", re.I
    ),
    "name": re.compile(
        r"\byour\s+(?:full\s+|first\s+|last\s+)?name\b|\bwho am i (?:speaking|talking)\b|"
        r"\bwhat (?:should|can|may) i call you\b", re.I),
}  # fmt: skip
_ASK_NEEDS_CUE = {"email", "phone", "date", "money", "count"}


def _asks_for(fact_kind: str, label: str, sentence: str) -> bool:
    if not (sentence.rstrip().endswith("?") or re.match(r"\s*please\b", sentence, re.I)):
        return False
    if fact_kind in _ASK:
        pat = _ASK[fact_kind]
        return bool(pat.search(sentence)) and (fact_kind == "name" or bool(_CUE.search(sentence)))
    word = re.escape(label)
    if fact_kind == "id":
        return bool(re.search(
            rf"\b{word}\s+(?:number|id|no\.?|#|reference|code)\b|\bwhich\s+{word}\b|"
            rf"\b{word}\s+(?:you(?:'re| are)|that you)\b", sentence, re.I))  # fmt: skip
    if fact_kind == "count":
        return bool(re.search(rf"\bhow many\b[^?]*\b{word}\b|\bnumber of {word}\b", sentence, re.I))
    if fact_kind == "money":
        return bool(re.search(r"\bbudget\b", sentence, re.I)) and bool(_CUE.search(sentence))
    if fact_kind == "date":
        has_label = label == "date" or re.search(rf"\b{word}\b", sentence, re.I)
        when = re.search(r"\b(?:what|which)\s+(?:date|day)\b|\bdate\b|\bwhen\b", sentence, re.I)
        return bool(has_label and when and _CUE.search(sentence))
    return False


_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def _sentences(text: str) -> list[tuple[str, int]]:
    out, pos = [], 0
    for part in _SENT.split(text):
        at = text.find(part, pos)
        if part.strip():
            out.append((part, at))
        pos = max(pos, at + len(part))
    return out


def _value_in(text: str, f: Fact) -> bool:
    low = text.lower()
    if f.value.lower() in low:
        return True
    squashed = re.sub(r"\W", "", low)
    return bool(f.norm) and f.kind in ("id", "phone", "email") and f.norm in squashed


# --- detectors -------------------------------------------------------------------------------


class _MemoryDetector(Detector):
    def on_end(self, run: Run, h: History) -> list[Finding]:
        return self.scan(run, h)

    def scan(self, run: Run, h: History) -> list[Finding]:  # pragma: no cover - abstract
        return []


# memory.context_truncation


_COMPACTED = re.compile(
    r"\b(?:summary of|summari[sz]ed|conversation so far|previous (?:conversation|messages)|"
    r"earlier in the conversation|conversation summary|compacted)\b",
    re.I,
)


def _user_ordinal(view: ChatView, m: Msg) -> int:
    return 1 + sum(1 for x in view.msgs if x.role == "user" and x.idx < m.idx)


def _describe_dropped(view: ChatView, m: Msg) -> str:
    if m.role == "user":
        return f"user turn {_user_ordinal(view, m)} ({_short(m.text, 50)!r})"
    if m.role == "tool":
        return "a tool result"
    return f"{m.role} message ({_short(m.text, 40)!r})" if m.text else f"{m.role} message"


class ContextTruncation(_MemoryDetector):
    kind = "memory.context_truncation"

    def scan(self, run: Run, h: History) -> list[Finding]:
        out: list[Finding] = []
        for agent, views in _by_agent(_views(h)).items():
            reported: set[str] = set()
            for i, cur in enumerate(views):
                prev, shared = None, 0
                for cand in reversed(views[max(0, i - _LOOKBACK) : i]):
                    ov = len(cand.fps & cur.fps)
                    if ov > shared:
                        prev, shared = cand, ov
                if prev is None:
                    f = self._reset(i, views, cur)
                    if f is not None:
                        out.append(f)
                    continue
                if not (cur.fps - prev.fps):
                    continue  # nothing new: a shorter / repeated request, not progress
                if any(_COMPACTED.search(m.text) for m in cur.msgs if m.role != "user") or (
                    _COMPACTED.search(cur.system)
                ):
                    continue  # the agent says it summarised the history
                last_shared = max((m.idx for m in prev.msgs if m.fp in cur.fps), default=-1)
                dropped = [m for m in prev.msgs if m.role != "system"
                           and m.fp not in cur.fps and m.idx < last_shared
                           and m.fp not in reported]  # fmt: skip
                if len(dropped) < self.cfg.min_dropped:
                    continue
                reported.update(m.fp for m in dropped)
                out.append(self._finding(agent, prev, cur, dropped))
        return out

    def _finding(self, agent: Any, prev: ChatView, cur: ChatView, dropped: list[Msg]) -> Finding:
        lost = _lost_facts(dropped, cur)
        what = "; ".join(_describe_dropped(prev, m) for m in dropped[:3])
        more = f" (+{len(dropped) - 3} more)" if len(dropped) > 3 else ""
        facts = f"; facts lost: {', '.join(lost[:4])}" if lost else ""
        return self.finding(
            severity="error" if lost else "warn",
            step=cur.idx, steps=[prev.idx, cur.idx], key=f"drop:{cur.idx}",
            message=f"step {cur.idx} silently dropped {len(dropped)} earlier message(s) that "
                    f"step {prev.idx} still had: {what}{more}{facts}",
            evidence=[_ev_msg(prev, m) for m in dropped[:3]],
        )  # fmt: skip

    def _reset(self, i: int, views: list[ChatView], cur: ChatView) -> Finding | None:
        """A turn (``crashlab.simulate``) whose request has nothing of the previous turn."""
        if not self.cfg.reset or i == 0 or cur.step.attributes.get(TURN_ATTR) is None:
            return None
        prev_turn = [v for v in views[:i] if isinstance(v.step.attributes.get(TURN_ATTR), int)
                     and v.step.attributes[TURN_ATTR] < cur.step.attributes[TURN_ATTR]]  # fmt: skip
        if not prev_turn:
            return None
        prev = prev_turn[-1]
        if prev.turn == cur.turn:
            return None
        mine = [m for m in prev.msgs if m.role in ("user", "assistant") and m.text]
        if not mine or any(m.fp in cur.fps for m in prev.msgs):
            return None
        lost = _lost_facts(mine, cur)
        facts = f"; facts lost: {', '.join(lost[:4])}" if lost else ""
        return self.finding(
            severity="error" if lost else "warn",
            step=cur.idx, steps=[prev.idx, cur.idx], key=f"reset:{cur.turn}",
            message=f"turn {cur.turn} (step {cur.idx}) carries nothing of turn {prev.turn} "
                    f"(step {prev.idx}): the conversation history was not passed on{facts}",
            evidence=[_ev_msg(prev, m) for m in mine[:2]],
        )  # fmt: skip


def _lost_facts(dropped: list[Msg], cur: ChatView) -> list[str]:
    seen = " ".join(m.text for m in cur.msgs).lower() + " " + cur.system.lower()
    out: list[str] = []
    for m in dropped:
        if m.role != "user":
            continue
        for f in extract_facts(m.text):
            if f.value.lower() in seen:
                continue
            out.append(f"{f.kind} {f.value}" if f.kind in ("email", "phone", "name")
                       else f"{f.label} {f.value}")  # fmt: skip
    return list(dict.fromkeys(out))


# memory.forgotten_fact


@dataclass
class _Given:
    fact: Fact
    view: ChatView
    msg: Msg
    turn: int


def _given_facts(views: list[ChatView], skip: set[str]) -> list[_Given]:
    """Facts from each distinct user message, at the first request that contained it."""
    seen: set[str] = set()
    out: list[_Given] = []
    for v in views:
        users = [m for m in v.msgs if m.role == "user"]
        for n, m in enumerate(users, 1):
            if m.fp in seen:
                continue
            seen.add(m.fp)
            attr = v.step.attributes.get(TURN_ATTR)
            turn = attr if (m is users[-1] and isinstance(attr, int)) else n
            out += [_Given(f, v, m, turn) for f in extract_facts(m.text) if f.kind not in skip]
    return out


def _corpus(h: History) -> str:
    """Everything the agent was legitimately told: user text, system prompts, tool I/O."""
    parts: list[str] = []
    for s in h.steps:
        if s.operation == "execute_tool":
            parts += [as_text(s.args), as_text(s.result)]
        elif s.operation == "chat":
            v = _view(s)
            parts.append(v.system)
            parts += [m.text for m in v.msgs if m.role in ("user", "system", "tool")]
    return " ".join(parts).lower()


class ForgottenFact(_MemoryDetector):
    kind = "memory.forgotten_fact"

    def scan(self, run: Run, h: History) -> list[Finding]:
        out: list[Finding] = []
        skip = set(self.cfg.ignore_kinds)
        corpus = squashed = None
        for _agent, views in _by_agent(_views(h)).items():
            given = _given_facts(views, skip)
            if not given:
                continue
            for v in views:
                reply = response_text(v.step)
                if not reply:
                    continue
                for g in given:
                    if g.view.idx > v.idx:
                        continue
                    for sent, at in _sentences(reply):
                        if self.cfg.ask_again and _asks_for(g.fact.kind, g.fact.label, sent) \
                                and not _value_in(sent, g.fact):  # fmt: skip
                            out.append(self._asked(g, v, sent, at))
                if not self.cfg.contradicted:
                    continue
                if corpus is None:
                    corpus = _corpus(h)
                    squashed = re.sub(r"\W", "", corpus)
                out += self._contradictions(given, v, reply, corpus, squashed or "")
        return out

    def _asked(self, g: _Given, v: ChatView, sent: str, at: int) -> Finding:
        f = g.fact
        what = f.label if f.kind in ("id", "date", "count") else f.kind
        return self.finding(
            severity="warn", step=v.idx, steps=[g.view.idx, v.idx],
            key=f"ask:{f.kind}:{f.label}:{f.norm}",
            message=f"the agent asked for the user's {what} again at step {v.idx} although the "
                    f"user already gave it ({f.value}) in turn {g.turn} (step {g.view.idx})",
            evidence=[_ev_msg(g.view, g.msg, max(0, f.start - 40)),
                      Evidence(step_idx=v.idx, field="response.text", quote=excerpt(sent.strip()))],
        )  # fmt: skip

    def _contradictions(self, given: list[_Given], v: ChatView, reply: str, corpus: str,
                        squashed: str) -> list[Finding]:  # fmt: skip
        out: list[Finding] = []
        for sent, _at in _sentences(reply):
            for m in extract_facts(sent, speaker="assistant"):
                mine = [g for g in given if g.fact.kind == m.kind and g.fact.label == m.label
                        and g.view.idx <= v.idx]  # fmt: skip
                if not mine:
                    continue
                if any(_same(g.fact, m) for g in given if g.fact.kind == m.kind):
                    continue  # it is a value the user gave (possibly of another label)
                if m.value.lower() in corpus or (m.kind in ("id", "phone") and m.norm in squashed):
                    continue  # grounded elsewhere: system prompt, tool result, other message
                if m.kind in ("email", "phone", "count", "money", "date") and not re.search(
                    r"\b(?:your|you|you're|for you)\b", sent, re.I
                ):
                    continue
                if m.kind == "count" and m.label not in _PARTY_UNITS:
                    continue
                g = mine[-1]
                out.append(self.finding(
                    severity="error", step=v.idx, steps=[g.view.idx, v.idx],
                    key=f"contradict:{m.kind}:{m.label}:{m.norm}",
                    message=f"the agent said the user's {m.label if m.kind != 'name' else 'name'} "
                            f"is {m.value!r} at step {v.idx}, but the user gave "
                            f"{g.fact.value!r} (turn {g.turn}, step {g.view.idx})",
                    evidence=[_ev_msg(g.view, g.msg, max(0, g.fact.start - 40)),
                              Evidence(step_idx=v.idx, field="response.text",
                                       quote=excerpt(sent.strip()))],
                ))  # fmt: skip
        return out


def _same(a: Fact, b: Fact) -> bool:
    if a.kind == "date":
        return _same_date(a.norm, b.norm)
    return a.norm == b.norm


# memory.self_contradiction

_KEYS = ("total|price|cost|amount|balance|fee|refund|deadline|date|eta|quantity|limit|rate|"
         "discount|score|age|number|stock|capacity|fare|premium|allowance")  # fmt: skip
_HEDGE = re.compile(r"\b(?:if|unless|assuming|when|once|would|could|might|may|should|depends|"
                    r"approximately|estimate[ds]?|around)\b", re.I)  # fmt: skip
_ASSERT_NUM = re.compile(
    rf"\b(?:(?:your|the|our|this|that|a|its)\s+)?(?P<key>(?:[a-z][\w'-]*\s+){{0,2}}(?:{_KEYS}))\s+"
    r"(?:is|was|will be|comes to|equals|totals|are|:)\s*(?:now\s+|currently\s+|only\s+)?"
    r"(?P<val>\d{4}-\d{2}-\d{2}|" + _MONTH_RE + r"\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?"
    r"|[$€£]\s?\d(?:[\d,]*\d)?(?:\.\d+)?"
    r"|\d(?:[\d,]*\d)?(?:\.\d+)?(?:\s?%|\s?(?:days?|hours?|minutes?|units?))?)",
    re.I,
)  # fmt: skip
_PREDS = ("eligible|available|refundable|supported|allowed|possible|covered|cancelled|canceled|"
          "shipped|delivered|approved|confirmed|active|free")  # fmt: skip
_ASSERT_YN = re.compile(
    rf"\b(?P<subj>(?:[a-z][\w'-]*\s+){{1,3}}?)"
    rf"(?:(?P<be>is|are|was|will be)(?P<neg> not| no longer)?|"
    rf"(?P<neg2>isn't|aren't|wasn't|won't be))\s+(?P<pred>{_PREDS})\b",
    re.I,
)  # fmt: skip
_VERBS = "refund|cancel|ship|exchange|return|change|modify|process|book|upgrade|waive"
_ASSERT_CAN = re.compile(
    rf"\bI(?:'m|\s+am)?\s*(?P<neg>cannot|can't|can not|unable to|am not able to|won't be able to|"
    rf"can|am able to|will)\s+(?P<verb>{_VERBS})\b",
    re.I,
)  # fmt: skip
_PRONOUN = _words("it this that there which here they he she we i you")
_DET = re.compile(r"^(?:your|the|this|that|our|a|an|its|that's)\s+", re.I)


@dataclass
class _Assertion:
    key: str
    value: str
    shown: str
    step: Step
    sentence: str
    kind: str  # num / date / yn


def _num_value(raw: str) -> str:
    unit = re.search(r"%|days?|hours?|minutes?|units?", raw, re.I)
    n = _norm_num(re.sub(r"[%a-z]+", "", raw, flags=re.I))
    return n + (unit.group(0).lower().rstrip("s") if unit else "")


def _assertions(step: Step) -> list[_Assertion]:
    text = response_text(step)
    out: list[_Assertion] = []
    for sent, _at in _sentences(text):
        if _HEDGE.search(sent) or sent.rstrip().endswith("?"):
            continue
        for m in _ASSERT_NUM.finditer(sent):
            raw = m.group("val")
            key = re.sub(r"\s+", " ", _DET.sub("", m.group("key").lower())).strip()
            dm = _DATE.fullmatch(raw.strip())
            if dm:
                out.append(_Assertion(key, _date_norm(dm), raw, step, sent, "date"))
            else:
                out.append(_Assertion(key, _num_value(raw), raw, step, sent, "num"))
        for m in _ASSERT_YN.finditer(sent):
            subj = _DET.sub("", " ".join(m.group("subj").lower().split())).strip()
            if not subj or subj.split()[-1] in _PRONOUN or subj in _PRONOUN:
                continue
            neg = bool(m.group("neg") or m.group("neg2"))
            out.append(_Assertion(f"{subj} | {m.group('pred').lower()}", "no" if neg else "yes",
                                  m.group(0), step, sent, "yn"))  # fmt: skip
        for m in _ASSERT_CAN.finditer(sent):
            neg = m.group("neg").lower() in ("cannot", "can't", "can not", "unable to",
                                             "am not able to", "won't be able to")  # fmt: skip
            out.append(_Assertion(f"can {m.group('verb').lower()}", "no" if neg else "yes",
                                  m.group(0), step, sent, "yn"))  # fmt: skip
    return out


def _backed(a: _Assertion, text: str) -> bool:
    """``text`` (a tool result, a user message) contains ``a``'s value (numbers compared by
    value: ``150.0`` backs ``$150.00``; dates by calendar day)."""
    if a.kind == "num":
        want = re.sub(r"[a-z%]+$", "", a.value)
        return want in {_norm_num(n) for n in re.findall(r"\d(?:[\d,]*\d)?(?:\.\d+)?", text)}
    if a.kind == "date":
        return any(_same_date(a.value, _date_norm(m)) for m in _DATE.finditer(text))
    return a.shown.lower() in text.lower()


def _same_assertion(a: _Assertion, b: _Assertion) -> bool:
    return _same_date(a.value, b.value) if a.kind == "date" else a.value == b.value


class SelfContradiction(_MemoryDetector):
    kind = "memory.self_contradiction"

    def scan(self, run: Run, h: History) -> list[Finding]:
        steps = sorted(h.steps, key=lambda s: s.idx)
        per_agent: dict[str | None, dict[str, list[_Assertion]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for s in steps:
            if s.operation == "chat" and s.error is None:
                for a in _assertions(s):
                    per_agent[s.agent][a.key].append(a)
        out: list[Finding] = []
        views = _views(h)
        for agent, keys in per_agent.items():
            for key, items in keys.items():
                for old, new in zip(items, items[1:], strict=False):
                    if old.step.idx == new.step.idx or _same_assertion(old, new):
                        continue
                    if self._explained(old, new, steps, views, agent):
                        continue
                    out.append(self._finding(key, old, new))
                    break
        return out

    def _explained(self, old: _Assertion, new: _Assertion, steps: list[Step],
                   views: list[ChatView], agent: str | None) -> bool:  # fmt: skip
        lo, hi = old.step.idx, new.step.idx
        for s in steps:
            if not lo < s.idx < hi:
                continue
            if s.operation == "execute_tool":
                if s.attributes.get("crashlab.tool.side_effects"):
                    return True
                if _backed(new, as_text(s.result) + " " + as_text(s.args)):
                    return True
        if new.kind == "yn":
            return False
        old_fps = {fp for v in views if v.idx <= lo and v.agent == agent for fp in v.fps}
        for v in views:
            if v.agent != agent or not lo < v.idx <= hi:
                continue
            for m in v.msgs:
                if m.role == "user" and m.fp not in old_fps and _backed(new, m.text):
                    return True
        return False

    def _finding(self, key: str, old: _Assertion, new: _Assertion) -> Finding:
        label = key.replace(" | ", " ")
        return self.finding(
            severity="warn", step=new.step.idx, steps=[old.step.idx, new.step.idx],
            key=f"{key}:{old.value}->{new.value}",
            message=f"the agent said {label} {old.shown!r} at step {old.step.idx} but "
                    f"{new.shown!r} at step {new.step.idx}, with no tool result or user message "
                    "in between that explains the change",
            evidence=[Evidence(step_idx=old.step.idx, field="response.text",
                               quote=excerpt(old.sentence.strip())),
                      Evidence(step_idx=new.step.idx, field="response.text",
                               quote=excerpt(new.sentence.strip()))],
        )  # fmt: skip


# memory.stale_tool_result

_VOLATILE = re.compile(r"(?:^|[._\[])(?:timestamp|ts|time|now|requested_at|request_id|trace_id|"
                       r"etag|nonce|latency|elapsed|uuid|retrieved_at|fetched_at)(?:$|[._\]])",
                       re.I)  # fmt: skip


def _flatten(value: Any, path: str = "", out: dict[str, Any] | None = None) -> dict[str, Any]:
    out = {} if out is None else out
    if len(out) > 400:
        return out
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(v, f"{path}.{k}" if path else str(k), out)
    elif isinstance(value, list):
        for i, v in enumerate(value[:60]):
            _flatten(v, f"{path}[{i}]", out)
    elif path:
        out[path] = value
    return out


def _significant(v: Any) -> str | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        s = str(int(v)) if float(v).is_integer() else str(v)
        return s if len(s.lstrip("-")) >= 2 else None
    if isinstance(v, str) and len(v.strip()) >= 4:
        return v.strip()
    return None


def _mentions(text: str, value: str) -> int:
    pat = rf"(?<![\w.]){re.escape(value)}(?![\w])" if value[0].isalnum() or value[0] == "-" \
        else re.escape(value)  # fmt: skip
    m = re.search(pat, text, re.I)
    return m.start() if m else -1


class StaleToolResult(_MemoryDetector):
    kind = "memory.stale_tool_result"

    def scan(self, run: Run, h: History) -> list[Finding]:
        groups: dict[tuple[str, str], list[Step]] = defaultdict(list)
        for s in sorted(h.tool_steps(), key=lambda s: s.idx):
            if s.error is None and s.name not in self.cfg.ignore_tools and s.result is not None:
                groups[(s.name, canonical(s.args))].append(s)
        chats = [s for s in sorted(h.steps, key=lambda s: s.idx)
                 if s.operation == "chat" and s.error is None]  # fmt: skip
        out: list[Finding] = []
        for (name, _a), calls in groups.items():
            for old, new in zip(calls, calls[1:], strict=False):
                if canonical(old.result) == canonical(new.result):
                    continue
                out += self._compare(name, old, new, chats)
        return out

    def _compare(self, name: str, old: Step, new: Step, chats: list[Step]) -> list[Finding]:
        a, b = _flatten(old.result), _flatten(new.result)
        new_blob = as_text(new.result).lower()
        out: list[Finding] = []
        for path, ov in a.items():
            if _VOLATILE.search(path) or path in b and canonical(b[path]) == canonical(ov):
                continue
            sig = _significant(ov)
            if sig is None or sig.lower() in new_blob:
                continue
            nv = b.get(path)
            nsig = _significant(nv)
            for c in chats:
                if c.idx <= new.idx:
                    continue
                reply = response_text(c)
                at = _mentions(reply, sig) if reply else -1
                if at < 0 or (nsig is not None and _mentions(reply, nsig) >= 0):
                    continue
                out.append(self.finding(
                    severity="error", step=c.idx, steps=[old.idx, new.idx, c.idx],
                    key=f"stale:{name}:{path}:{sig}",
                    message=f"step {c.idx} states {path}={sig!r} from {name} at step {old.idx}, "
                            f"but the same call at step {new.idx} returned "
                            f"{(nv if nv is not None else '(no such field)')!r}",
                    evidence=[Evidence(step_idx=old.idx, field=f"result.{path}",
                                       quote=excerpt(as_text(ov))),
                              Evidence(step_idx=new.idx, field=f"result.{path}",
                                       quote=excerpt(as_text(nv if nv is not None else b))),
                              Evidence(step_idx=c.idx, field="response.text",
                                       quote=excerpt(reply, max(0, at - 40)))],
                ))  # fmt: skip
                break
        return out


# memory.unbounded_growth


def _est_tokens(step: Step) -> int:
    used = step.attributes.get("gen_ai.usage.input_tokens")
    if isinstance(used, (int, float)) and not isinstance(used, bool) and used > 0:
        return int(used)
    req = step.request or {}
    return len(as_text(req.get("system") or "") + as_text(req.get("messages") or [])) // 4


def _superlinear(values: list[int], cfg: Any) -> tuple[float, float] | None:
    """(early, late) mean per-step growth when ``values`` grow faster and faster."""
    if len(values) < max(cfg.min_steps, 4):
        return None
    if any(b < a * 0.9 for a, b in zip(values, values[1:], strict=False)):
        return None  # shrinks: history is being compacted
    deltas = [b - a for a, b in zip(values, values[1:], strict=False)]
    half = len(deltas) // 2
    early, late = sum(deltas[:half]) / half, sum(deltas[half:]) / (len(deltas) - half)
    if values[-1] - values[0] < cfg.min_growth_tokens or late <= 0:
        return None
    rising = sum(1 for a, b in zip(deltas, deltas[1:], strict=False) if b >= a) / (len(deltas) - 1)
    if late >= max(early, 1) * cfg.acceleration and rising >= 0.6 and values[-1] >= 2 * values[0]:
        return early, late
    return None


class UnboundedGrowth(_MemoryDetector):
    kind = "memory.unbounded_growth"

    def scan(self, run: Run, h: History) -> list[Finding]:
        out: list[Finding] = []
        for agent, views in _by_agent(_views(h)).items():
            per_turn: dict[Any, ChatView] = {}
            for v in views:  # the last request of each turn (or every request)
                per_turn[v.step.attributes.get(TURN_ATTR, f"s{v.idx}")] = v
            series = list(per_turn.values())
            vals = [_est_tokens(v.step) for v in series]
            res = _superlinear(vals, self.cfg)
            if res is not None:
                out.append(self._finding(f"prompt:{agent}", "prompt", [v.idx for v in series],
                                         vals, res, series[-1].idx))  # fmt: skip
        snaps: dict[str, list[Snapshot]] = defaultdict(list)
        for sn in snapshots(Run(run_id=run.run_id, name=run.name, steps=h.steps)):
            snaps[sn.name].append(sn)
        for name, items in snaps.items():
            vals = [max(1, sn.chars // 4) for sn in items]
            res = _superlinear(vals, self.cfg)
            if res is not None:
                idxs = [sn.idx for sn in items]
                what = f"state snapshot {name!r}"
                out.append(self._finding(f"state:{name}", what, idxs, vals, res, idxs[-1]))
        return out

    def _finding(self, key: str, what: str, idxs: list[int], vals: list[int],
                 res: tuple[float, float], last: int) -> Finding:  # fmt: skip
        shown = idxs if len(idxs) <= 12 else idxs[:: max(1, len(idxs) // 12)] + [idxs[-1]]
        return self.finding(
            severity="warn", step=last, steps=sorted(set(shown)), key=key,
            message=f"the {what} grows faster every turn: ~{vals[0]:,} -> ~{vals[-1]:,} tokens "
                    f"over {len(vals)} samples (+{res[0]:,.0f}/turn early, +{res[1]:,.0f}/turn "
                    "late): memory is not bounded (no trimming, summarising or dedup)",
        )  # fmt: skip


def build_detectors(cfg: Any) -> list[Detector]:
    """The enabled ``memory.*`` detectors for a :class:`~crashlab.checks.config.MemoryConfig`."""
    pairs = (
        (cfg.context_truncation, ContextTruncation),
        (cfg.forgotten_fact, ForgottenFact),
        (cfg.self_contradiction, SelfContradiction),
        (cfg.stale_tool_result, StaleToolResult),
        (cfg.unbounded_growth, UnboundedGrowth),
    )
    return [cls(c) for c, cls in pairs if c.enabled]


# --- state snapshots ----------------------------------------------------------------------


def _shrink(value: Any, depth: int = 0) -> Any:
    if depth > 8:
        return _clip_text(repr(value), 200)
    if isinstance(value, str):
        return _clip_text(value, 2000)
    if isinstance(value, dict):
        items = list(value.items())
        out = {str(k): _shrink(v, depth + 1) for k, v in items[:200]}
        if len(items) > 200:
            out["…"] = f"(+{len(items) - 200} more keys)"
        return out
    if isinstance(value, list):
        out_l = [_shrink(v, depth + 1) for v in value[:100]]
        if len(value) > 100:
            out_l.append(f"…(+{len(value) - 100} more items)")
        return out_l
    return value


def _clip_text(text: str, cap: int) -> str:
    return text if len(text) <= cap else f"{text[:cap]}…[+{len(text) - cap} chars]"


def snapshot(name: str, state: Any) -> bool:
    """Record ``state`` (anything JSON-serialisable: a dict of memory, scratchpad, plan,
    cart ...) as a step of the run being recorded, labelled ``name``.

    The state is converted to JSON, redacted like every other step, and capped
    (long strings, lists and dicts are cut; a snapshot over 20,000 characters keeps only a
    preview). Replays serve the recorded snapshot, so they stay offline. Returns False and
    does nothing when no run is being recorded. See ``crashlab state`` and ``crashlab memory``.
    """
    from .capture._steps import begin_step
    from .capture.normalize import to_jsonable
    from .store.redact import get_redactor

    if not isinstance(name, str) or not name.strip():
        raise ValueError("crashlab.snapshot(): name must be a non-empty string")
    name = name.strip()
    pending = begin_step(
        "custom",
        SNAPSHOT_STEP,
        {"name": name},
        attributes={SNAPSHOT_NAME_ATTR: name},
        hash_input={"snapshot": name},
    )
    if pending is None:
        return False
    src = pending.replay_source()
    if src is not None:
        pending.finish_from(src)
        return True
    data = get_redactor().redact(_shrink(to_jsonable(state)))
    text = json.dumps(data, ensure_ascii=False, default=str)
    truncated = len(text) > MAX_SNAPSHOT_CHARS
    if truncated:
        data = {"_truncated": True, "_chars": len(text), "preview": text[:2000]}
    pending.finish(
        {"state": data},
        attributes={"crashlab.snapshot.chars": len(text), "crashlab.snapshot.truncated": truncated},
    )
    return True


@dataclass
class Snapshot:
    idx: int
    name: str
    state: Any
    chars: int
    truncated: bool
    turn: int | None
    agent: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.idx, "name": self.name, "turn": self.turn, "agent": self.agent,
                "chars": self.chars, "truncated": self.truncated, "state": self.state}  # fmt: skip


def snapshots(run: Run, name: str | None = None) -> list[Snapshot]:
    """The :func:`snapshot` steps of ``run`` in order (optionally only those named ``name``)."""
    out: list[Snapshot] = []
    for s in sorted(run.steps, key=lambda s: s.idx):
        if s.operation != "custom" or s.name != SNAPSHOT_STEP:
            continue
        label = s.attributes.get(SNAPSHOT_NAME_ATTR) or s.request.get("name") or "state"
        if name is not None and label != name:
            continue
        state = (s.response or {}).get("state")
        chars = s.attributes.get("crashlab.snapshot.chars")
        if not isinstance(chars, int):
            chars = len(json.dumps(state, default=str))
        turn = s.attributes.get(TURN_ATTR)
        out.append(Snapshot(s.idx, str(label), state, chars,
                            bool(s.attributes.get("crashlab.snapshot.truncated")),
                            turn if isinstance(turn, int) else None, s.agent))  # fmt: skip
    return out


@dataclass
class Change:
    path: str
    op: str  # added / removed / changed
    old: Any = None
    new: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "op": self.op, "old": self.old, "new": self.new}


def state_diff(old: Any, new: Any, *, limit: int = 200) -> list[Change]:
    """Key-level differences between two JSON states: dict keys added / removed / changed,
    list items by index (``items[2]``). Capped at ``limit`` changes."""
    out: list[Change] = []

    def walk(a: Any, b: Any, path: str) -> None:
        if len(out) >= limit:
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for k in a:
                p = f"{path}.{k}" if path else str(k)
                if k not in b:
                    out.append(Change(p, "removed", a[k]))
                else:
                    walk(a[k], b[k], p)
            for k in b:
                if k not in a:
                    out.append(Change(f"{path}.{k}" if path else str(k), "added", None, b[k]))
        elif isinstance(a, list) and isinstance(b, list):
            for i in range(max(len(a), len(b))):
                p = f"{path}[{i}]"
                if i >= len(a):
                    out.append(Change(p, "added", None, b[i]))
                elif i >= len(b):
                    out.append(Change(p, "removed", a[i]))
                else:
                    walk(a[i], b[i], p)
        elif canonical(a) != canonical(b):
            out.append(Change(path or "(root)", "changed", a, b))

    walk(old, new, "")
    return out[:limit]


# --- reports (crashlab memory) --------------------------------------------------------------


def memory_findings(run: Run, config: Any = None) -> list[Finding]:
    """Evaluate only the ``memory.*`` detectors on ``run`` (config: the run's stored check
    config, with every other detector off)."""
    from .checks.config import _SECTIONS, ChecksConfig, resolve_config
    from .checks.engine import evaluate

    base = (
        config
        if isinstance(config, ChecksConfig)
        else resolve_config(config, base=run.check_config)
    )
    cfg = base.merged(
        {s: False for s in _SECTIONS if s != "memory"} | {"memory": {"enabled": True}}
    )
    return [f for f in evaluate(run, cfg) if f.kind.startswith("memory.")]


def confirm_with_judge(run: Run, findings: list[Finding], *, judge_fn: Any = None,
                       model: str | None = None, store: Any = None,
                       limit: int = 8) -> list[dict[str, Any]]:  # fmt: skip
    """Ask a model (``crashlab.judges``) whether each finding is a real memory failure.

    One custom judge per finding, shown the finding and its verbatim evidence. ``confirmed``
    is True when the judge scores it below its threshold (the agent did fail). Raises
    :class:`crashlab.judges.JudgeError` if no model is configured; ``judge_fn`` replaces the
    model call (offline use and tests)."""
    from . import judges

    out: list[dict[str, Any]] = []
    for f in findings[:limit]:
        quotes = "\n".join(f"- step {e.step_idx} {e.field}: {e.quote!r}" for e in f.evidence)
        j = judges.custom(
            f"A heuristic detector flagged a memory failure in this agent run: [{f.kind}] "
            f"{f.message}\nEvidence:\n{quotes}\nScore 1.0 if the agent behaved correctly (a "
            "false alarm: the information was legitimately updated, summarised or not needed), "
            "0.0 if the agent really forgot, lost, contradicted or misused earlier context.",
            name=f.kind.replace(".", "_"),
            threshold=0.5,
            sections=("question", "answer", "tool_calls"),
        )
        r = j.evaluate(run, judge_fn=judge_fn, model=model, store=store)
        out.append({"kind": f.kind, "key": f.key, "step": f.step, "confirmed": not r.passed,
                    "score": r.score, "reasoning": r.reasoning, "model": r.model,
                    "cached": r.cached})  # fmt: skip
    return out


__all__ = [
    "Change",
    "Snapshot",
    "build_detectors",
    "confirm_with_judge",
    "extract_facts",
    "memory_findings",
    "snapshot",
    "snapshots",
    "state_diff",
]
