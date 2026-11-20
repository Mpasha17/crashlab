# ruff: noqa: E501
"""Zero-dependency local web trace explorer: ``crashlab ui``.

A ``http.server.ThreadingHTTPServer`` bound to localhost serves a self-contained
HTML/CSS/JS viewer and a small JSON API. No extra packages, no external assets.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .compare import diff as diff_runs
from .models import Run, Step
from .otel import to_otlp
from .pricing import run_cost
from .stats import compute_stats
from .store import RunNotFoundError, Store

__all__ = ["create_server", "make_handler"]

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    return str(value)


def _step_brief(step: Step) -> dict[str, Any]:
    a = step.attributes or {}
    return {
        "idx": step.idx,
        "span_id": step.span_id,
        "operation": step.operation,
        "name": step.name,
        "agent": step.agent,
        "started_at": step.started_at.isoformat() if step.started_at else None,
        "duration_ms": step.duration_ms,
        "source": step.source,
        "error": _jsonable(step.error) if step.error else None,
        "request": _jsonable(step.request),
        "response": _jsonable(step.response),
        "attributes": {
            k: _jsonable(v)
            for k, v in a.items()
            if k.startswith("gen_ai.") or k.startswith("crashlab.")
        },
    }


def _run_summary(run: Run) -> dict[str, Any]:
    cost = None
    try:
        cost = run_cost(run)
    except Exception:
        cost = None
    duration_s = None
    if run.usage.wall_time_ms is not None:
        duration_s = max(run.usage.wall_time_ms, 0.0) / 1000.0
    elif run.ended_at is not None and run.started_at is not None:
        duration_s = max((run.ended_at - run.started_at).total_seconds(), 0.0)
    return {
        "run_id": run.run_id,
        "name": run.name,
        "status": run.status,
        "outcome": run.outcome,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "ended_at": run.ended_at.isoformat() if run.ended_at else None,
        "duration_s": duration_s,
        "steps": len(run.steps) or run.usage.steps,
        "tokens": run.usage.total_tokens,
        "cost_usd": cost,
        "findings": len(run.findings),
        "error": run.error.message if run.error is not None else None,
        "parent_run": run.parent_run,
        "trace_id": run.trace_id,
    }


def _run_detail(store: Store, run: Run) -> dict[str, Any]:
    data = _run_summary(run)
    data.update(
        {
            "input": _jsonable(run.input),
            "output": _jsonable(run.output),
            "outcome_reason": run.outcome_reason,
            "checks": _jsonable(run.checks),
            "findings": _jsonable(run.findings),
            "diagnosis": _jsonable(run.diagnosis),
            "steps": [_step_brief(s) for s in sorted(run.steps, key=lambda s: s.idx)],
            "agents": list(run.agents),
            "handoffs": _jsonable(run.handoffs()) if run.agents else [],
            "children": [
                {
                    "run_id": r["run_id"],
                    "name": r.get("name"),
                    "status": r.get("status"),
                    "outcome": r.get("outcome"),
                }
                for r in store.child_runs(run.run_id)
            ],
        }
    )
    return data


def _index_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": row.get("run_id"),
        "name": row.get("name"),
        "status": row.get("status"),
        "outcome": row.get("outcome"),
        "started_at": row.get("started_at"),
        "ended_at": row.get("ended_at"),
        "steps": row.get("steps") or 0,
        "tokens": row.get("total_tokens") or 0,
        "cost_usd": row.get("cost_usd"),
        "findings": row.get("flag_count") or 0,
        "error": None,
        "replay_of": row.get("replay_of"),
        "forked_from": row.get("parent_run_id"),
    }


def _diff_payload(a: Run, b: Run) -> dict[str, Any]:
    d = diff_runs(a, b)
    return {
        "a": _run_summary(a),
        "b": _run_summary(b),
        "summary": d.summary(),
        "relation": d.relation,
        "first_divergence": d.first_divergence,
        "steps": [
            {
                "idx": s.idx,
                "status": s.status,
                "fields": list(s.fields),
                "detail": s.detail(),
                "a": None
                if s.a is None
                else {"operation": s.a.operation, "name": s.a.name, "agent": s.a.agent},
                "b": None
                if s.b is None
                else {"operation": s.b.operation, "name": s.b.name, "agent": s.b.agent},
            }
            for s in d.steps
        ],
        "checks": [
            {
                "name": c.name,
                "changed": c.changed,
                "a": None if c.a is None else {"status": c.a.status, "message": c.a.message},
                "b": None if c.b is None else {"status": c.b.status, "message": c.b.message},
            }
            for c in d.checks
        ],
        "findings": [
            {
                "kind": f.kind,
                "changed": f.changed,
                "worst_a": f.worst_a,
                "worst_b": f.worst_b,
                "a": [{"severity": x.severity, "message": x.message, "step": x.step} for x in f.a],
                "b": [{"severity": x.severity, "message": x.message, "step": x.step} for x in f.b],
            }
            for f in d.findings
        ],
    }


def _load(store: Store, ref: str) -> Run:
    return store.load_run(ref)


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>crashlab trace explorer</title>
<style>
:root {
  --bg: #f6f7fb; --fg: #1d2330; --muted: #667085; --line: #dde1ea;
  --card: #fff; --accent: #4f46e5; --pass: #137333; --fail: #b3261e;
  --warn: #a15c00; --bar: #5b6ad0; --bar-tool: #2a9d8f; --bar-err: #b3261e;
  --mono: ui-monospace, SFMono-Regular, Menlo, monospace;
  --sans: system-ui, -apple-system, Segoe UI, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #12141a; --fg: #e8eaf0; --muted: #9aa3b2; --line: #2a2e3a;
    --card: #1a1d26; --accent: #8b85ff;
  }
}
:root[data-theme="dark"] {
  --bg: #12141a; --fg: #e8eaf0; --muted: #9aa3b2; --line: #2a2e3a;
  --card: #1a1d26; --accent: #8b85ff;
}
* { box-sizing: border-box; }
body { margin: 0; font: 14px/1.45 var(--sans); background: var(--bg); color: var(--fg); }
header { display: flex; gap: .8rem; align-items: center; padding: .7rem 1rem;
  border-bottom: 1px solid var(--line); background: var(--card); position: sticky; top: 0; z-index: 2; }
header h1 { font-size: 1rem; margin: 0; letter-spacing: .02em; }
header input { flex: 1; min-width: 8rem; padding: .35rem .6rem; border: 1px solid var(--line);
  border-radius: 6px; background: var(--bg); color: var(--fg); font: inherit; }
header button, .btn { border: 1px solid var(--line); background: var(--bg); color: var(--fg);
  border-radius: 6px; padding: .3rem .6rem; cursor: pointer; font: inherit; }
header button:hover, .btn:hover { border-color: var(--accent); }
#layout { display: grid; grid-template-columns: minmax(16rem, 22rem) 1fr; min-height: calc(100vh - 52px); }
#runs { border-right: 1px solid var(--line); overflow: auto; max-height: calc(100vh - 52px); }
#main { overflow: auto; max-height: calc(100vh - 52px); padding: 1rem 1.2rem 3rem; }
.run { padding: .55rem .8rem; border-bottom: 1px solid var(--line); cursor: pointer; }
.run:hover, .run.on { background: var(--card); }
.run .id { font: 12px var(--mono); color: var(--muted); }
.run .meta { color: var(--muted); font-size: 12px; }
.pass { color: var(--pass); font-weight: 600; }
.fail, .error, .critical { color: var(--fail); font-weight: 600; }
.unknown, .warn, .running { color: var(--warn); font-weight: 600; }
h2 { font-size: 1.05rem; margin: 1.3rem 0 .4rem; }
h3 { font-size: .95rem; margin: 1rem 0 .3rem; }
code, pre { font-family: var(--mono); font-size: 12.5px; }
pre { background: var(--card); border: 1px solid var(--line); padding: .6rem; overflow: auto;
  white-space: pre-wrap; word-break: break-word; }
table { border-collapse: collapse; width: 100%; }
td, th { border: 1px solid var(--line); padding: .28rem .5rem; text-align: left; vertical-align: top; }
th { background: var(--card); font-weight: 600; }
.wf { position: relative; height: .85rem; min-width: 80px; background: var(--bg);
  border: 1px solid var(--line); border-radius: 2px; }
.wf i { position: absolute; top: 0; bottom: 0; border-radius: 2px; min-width: 2px; background: var(--bar); }
.wf i.tool { background: var(--bar-tool); }
.wf i.err { background: var(--bar-err); }
.muted { color: var(--muted); }
.row { display: flex; gap: .5rem; flex-wrap: wrap; align-items: baseline; }
.badge { display: inline-block; padding: .05rem .45rem; border-radius: 999px; background: var(--card);
  border: 1px solid var(--line); font-size: 12px; }
.hidden { display: none !important; }
#diffbox { margin-top: 1rem; }
.pane { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
.same { color: var(--muted); }
@media (max-width: 800px) {
  #layout { grid-template-columns: 1fr; }
  #runs { max-height: 40vh; border-right: 0; border-bottom: 1px solid var(--line); }
}
</style>
</head>
<body>
<header>
  <h1>crashlab</h1>
  <input id="q" type="search" placeholder="Filter runs by id, name, outcome…" autocomplete="off">
  <button type="button" id="statsBtn" title="Fleet stats">Stats</button>
  <button type="button" id="diffBtn" title="Compare two selected runs">Diff</button>
  <button type="button" id="themeBtn" title="Toggle theme">Theme</button>
</header>
<div id="layout">
  <aside id="runs"></aside>
  <main id="main"><p class="muted" id="placeholder">Select a run, or open Stats.</p></main>
</div>
<script>
(function () {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
  const el = (html) => { const d = document.createElement("div"); d.innerHTML = html; return d.firstElementChild; };
  const fmtMoney = (v) => (v == null ? "?" : (v === 0 ? "$0" : (v >= 0.0001 ? "$" + v.toFixed(4) : "$" + v.toFixed(6))));
  const fmtDur = (ms) => (ms == null ? "-" : (ms >= 1000 ? (ms/1000).toFixed(2) + "s" : Math.round(ms) + "ms"));
  const cls = (w) => '<span class="' + esc(w) + '">' + esc(w) + "</span>";
  const state = { runs: [], selected: new Set(), view: null, theme: null };

  function applyTheme(t) {
    state.theme = t;
    if (t) document.documentElement.setAttribute("data-theme", t);
    else document.documentElement.removeAttribute("data-theme");
  }
  $("themeBtn").onclick = () => {
    const cur = document.documentElement.getAttribute("data-theme");
    const sysDark = matchMedia("(prefers-color-scheme: dark)").matches;
    if (!cur) applyTheme(sysDark ? "light" : "dark");
    else if (cur === "dark") applyTheme("light");
    else applyTheme("dark");
  };

  async function get(path) {
    const r = await fetch(path, { headers: { "Accept": "application/json" } });
    if (!r.ok) {
      let msg = r.status + " " + r.statusText;
      try { const j = await r.json(); if (j.error) msg = j.error; } catch (e) {}
      throw new Error(msg);
    }
    return r.json();
  }

  function matches(row, q) {
    if (!q) return true;
    const hay = [row.run_id, row.name, row.outcome, row.status, row.error].join(" ").toLowerCase();
    return hay.includes(q.toLowerCase());
  }

  function renderList() {
    const q = $("q").value.trim();
    const box = $("runs");
    box.textContent = "";
    const rows = state.runs.filter((r) => matches(r, q));
    if (!rows.length) { box.appendChild(el('<p class="muted" style="padding:1rem">No runs.</p>')); return; }
    for (const r of rows) {
      const item = el(
        '<div class="run' + (state.view === r.run_id ? " on" : "") + '" role="button" tabindex="0">' +
        '<label class="muted" style="float:right"><input type="checkbox" data-id="' + esc(r.run_id) + '"' +
        (state.selected.has(r.run_id) ? " checked" : "") + "> diff</label>" +
        '<div><strong>' + esc(r.name) + "</strong> " + cls(r.outcome) + " " + cls(r.status) + "</div>" +
        '<div class="id">' + esc(r.run_id) + "</div>" +
        '<div class="meta">' + esc((r.steps||0) + " steps · " + (r.tokens||0) + " tok · " + fmtMoney(r.cost_usd)) +
        (r.findings ? " · " + r.findings + " findings" : "") + "</div></div>"
      );
      item.addEventListener("click", (e) => {
        if (e.target && e.target.closest("input")) return;
        openRun(r.run_id);
      });
      const cb = item.querySelector("input");
      cb.addEventListener("change", () => {
        if (cb.checked) state.selected.add(r.run_id); else state.selected.delete(r.run_id);
        if (state.selected.size > 2) {
          const first = state.selected.values().next().value;
          state.selected.delete(first);
          renderList();
        }
      });
      box.appendChild(item);
    }
  }

  function waterfall(steps) {
    if (!steps.length) return '<p class="muted">No steps.</p>';
    const t0 = Math.min(...steps.map((s) => Date.parse(s.started_at) || 0));
    const span = Math.max(1, ...steps.map((s) => {
      const off = (Date.parse(s.started_at) || t0) - t0;
      return off + (s.duration_ms || 0);
    }));
    let rows = "";
    for (const s of steps) {
      const off = Math.max(0, (Date.parse(s.started_at) || t0) - t0);
      const left = (off / span) * 100;
      const w = Math.max(((s.duration_ms || 0) / span) * 100, 0.3);
      const kind = s.error ? "err" : (s.operation === "execute_tool" || s.operation === "invoke_agent" ? "tool" : "");
      const toks = s.attributes && (s.attributes["gen_ai.usage.input_tokens"] != null)
        ? (s.attributes["gen_ai.usage.input_tokens"] + "/" + (s.attributes["gen_ai.usage.output_tokens"] || 0))
        : "";
      const cost = s.attributes && s.attributes["crashlab.usage.cost_usd"] != null
        ? fmtMoney(s.attributes["crashlab.usage.cost_usd"]) : "";
      rows += "<tr><td>" + esc(s.idx) + "</td><td>" + esc(s.operation) + "</td><td>" + esc(s.name) + "</td>" +
        "<td>" + esc(s.agent || "") + "</td><td>" + esc(fmtDur(s.duration_ms)) + "</td>" +
        "<td>" + esc(toks) + "</td><td>" + esc(cost) + "</td>" +
        '<td><div class="wf" title="' + esc("+" + Math.round(off) + " ms") + '"><i class="' + kind +
        '" style="left:' + left.toFixed(2) + "%;width:" + w.toFixed(2) + '%"></i></div></td>' +
        "<td>" + (s.error ? '<span class="fail">' + esc(s.error.message || s.error.type) + "</span>" : "") +
        "</td></tr>";
    }
    return "<table><thead><tr><th>#</th><th>op</th><th>name</th><th>agent</th><th>dur</th>" +
      "<th>tok</th><th>cost</th><th>waterfall</th><th></th></tr></thead><tbody>" + rows + "</tbody></table>";
  }

  function messages(step) {
    const out = [];
    const req = step.request || {};
    if (req.system) out.push({ role: "system", content: req.system });
    for (const m of (req.messages || [])) {
      const parts = m.parts || [];
      const text = parts.filter((p) => p && p.type === "text").map((p) => p.content).join("\n")
        || (typeof m.content === "string" ? m.content : JSON.stringify(m.content || m));
      out.push({ role: m.role || "user", content: text });
    }
    const res = step.response || {};
    for (const m of (res.messages || [])) {
      const parts = m.parts || [];
      const text = parts.map((p) => {
        if (!p) return "";
        if (p.type === "text") return p.content;
        if (p.type === "tool_call") return "tool " + (p.name || "") + " " + JSON.stringify(p.arguments || {});
        return JSON.stringify(p);
      }).join("\n");
      out.push({ role: m.role || "assistant", content: text });
    }
    return out;
  }

  function renderRun(r) {
    state.view = r.run_id;
    renderList();
    const kids = (r.children || []).map((c) =>
      '<li><a href="#run/' + esc(c.run_id) + '">' + esc(c.name) + " " + esc(c.run_id) +
      "</a> " + cls(c.outcome || "") + "</li>"
    ).join("");
    const findings = (r.findings || []).map((f) =>
      "<tr><td>" + cls(f.severity) + "</td><td>" + esc(f.kind) + "</td><td>" + esc(f.step ?? (f.steps||[]).join(",")) +
      "</td><td>" + esc(f.message) + "</td></tr>"
    ).join("");
    const checks = (r.checks || []).map((c) =>
      "<tr><td>" + cls(c.status) + "</td><td>" + esc(c.name) + "</td><td>" + esc(c.message || "") + "</td></tr>"
    ).join("");
    let diag = "";
    if (r.diagnosis) {
      diag = "<h2>Diagnosis</h2><p>" + cls(r.diagnosis.label) + " · step " +
        esc(r.diagnosis.failing_step) + "</p><p>" + esc(r.diagnosis.root_cause) + "</p>";
      if (r.diagnosis.evidence && r.diagnosis.evidence.length) {
        diag += "<ul>" + r.diagnosis.evidence.map((e) =>
          "<li>step " + esc(e.step_idx) + " " + esc(e.field) + ": <code>" + esc(e.quote) + "</code></li>"
        ).join("") + "</ul>";
      }
    }
    let chats = "";
    for (const s of (r.steps || []).filter((s) => s.operation === "chat")) {
      const msgs = messages(s);
      chats += "<h3>step " + esc(s.idx) + " chat " + esc(s.name) + "</h3>";
      chats += msgs.map((m) => "<p><strong>" + esc(m.role) + "</strong></p><pre>" + esc(m.content) + "</pre>").join("");
    }
    let tools = "";
    for (const s of (r.steps || []).filter((s) => s.operation === "execute_tool")) {
      tools += "<h3>step " + esc(s.idx) + " tool " + esc(s.name) + (s.agent ? " · " + esc(s.agent) : "") + "</h3>";
      tools += "<p class='muted'>args</p><pre>" + esc(JSON.stringify((s.request || {}).args ?? s.request, null, 2)) + "</pre>";
      if (s.error) tools += "<p class='fail'>" + esc(s.error.message) + "</p>";
      else tools += "<p class='muted'>result</p><pre>" + esc(JSON.stringify((s.response || {}).result ?? s.response, null, 2)) + "</pre>";
    }
    $("main").innerHTML =
      "<div class='row'><h2 style='margin:0'>" + esc(r.name) + "</h2>" + cls(r.outcome) + cls(r.status) +
      '<a class="btn" href="/api/runs/' + esc(r.run_id) + '/otel" download>OTLP</a></div>' +
      "<p class='muted'><code>" + esc(r.run_id) + "</code> · " + esc(r.steps.length) + " steps · " +
      esc(r.tokens) + " tok · " + esc(fmtMoney(r.cost_usd)) +
      (r.parent_run ? " · parent <a href='#run/" + esc(r.parent_run) + "'>" + esc(r.parent_run) + "</a>" : "") +
      (r.trace_id && r.trace_id !== r.run_id ? " · trace " + esc(r.trace_id) : "") + "</p>" +
      (r.outcome_reason ? "<p>" + esc(r.outcome_reason) + "</p>" : "") +
      (r.error ? "<p class='fail'>" + esc(r.error) + "</p>" : "") +
      "<h2>Waterfall</h2>" + waterfall(r.steps || []) +
      (checks ? "<h2>Checks</h2><table><tr><th>status</th><th>name</th><th>message</th></tr>" + checks + "</table>" : "") +
      (findings ? "<h2>Findings</h2><table><tr><th>sev</th><th>kind</th><th>step</th><th>message</th></tr>" + findings + "</table>" : "") +
      diag +
      (kids ? "<h2>Child runs</h2><ul>" + kids + "</ul>" : "") +
      (chats ? "<h2>Chat</h2>" + chats : "") +
      (tools ? "<h2>Tools</h2>" + tools : "") +
      (r.input ? "<h2>Input</h2><pre>" + esc(JSON.stringify(r.input, null, 2)) + "</pre>" : "") +
      (r.output != null ? "<h2>Output</h2><pre>" + esc(typeof r.output === "string" ? r.output : JSON.stringify(r.output, null, 2)) + "</pre>" : "");
    location.hash = "run/" + r.run_id;
  }

  async function openRun(id) {
    try {
      const r = await get("/api/runs/" + encodeURIComponent(id));
      renderRun(r);
    } catch (e) {
      $("main").innerHTML = "<p class='fail'>" + esc(e.message) + "</p>";
    }
  }

  async function openStats() {
    state.view = "stats";
    renderList();
    try {
      const s = await get("/api/stats");
      const agents = (s.by_agent || []).map((a) =>
        "<tr><td>" + esc(a.name) + "</td><td>" + esc(a.runs) + "</td><td>" + esc(a.pass_n) + "</td><td>" +
        esc(a.fail_n) + "</td><td>" + esc(fmtMoney(a.total_cost_usd)) + "</td></tr>"
      ).join("");
      const tools = (s.tools || []).map((t) =>
        "<tr><td>" + esc(t.name) + "</td><td>" + esc(t.calls) + "</td><td>" + esc(t.error_n) + "</td><td>" +
        esc((t.error_rate * 100).toFixed(1) + "%") + "</td><td>" + esc(t.p95_duration_s == null ? "-" : t.p95_duration_s + "s") + "</td></tr>"
      ).join("");
      const findings = (s.findings || []).map((f) =>
        "<tr><td>" + esc(f.kind) + "</td><td>" + cls(f.severity) + "</td><td>" + esc(f.count) + "</td></tr>"
      ).join("");
      const checks = (s.failing_checks || []).map((c) =>
        "<tr><td>" + esc(c.name) + "</td><td>" + esc(c.fail_n) + "</td><td>" + esc(c.error_n) + "</td></tr>"
      ).join("");
      $("main").innerHTML =
        "<h2>Fleet stats</h2><p>" + esc(s.n) + " runs · pass " + esc((s.pass_rate*100).toFixed(1)) +
        "% · fail " + esc((s.fail_rate*100).toFixed(1)) + "% · error " + esc((s.error_rate*100).toFixed(1)) +
        "% · " + esc(s.total_tokens) + " tok · " + esc(fmtMoney(s.total_cost_usd)) + "</p>" +
        "<p class='muted'>mean cost " + esc(fmtMoney(s.mean_cost_usd)) + " · p95 " + esc(fmtMoney(s.p95_cost_usd)) +
        " · mean duration " + esc(s.mean_duration_s == null ? "-" : s.mean_duration_s + "s") +
        " · p95 " + esc(s.p95_duration_s == null ? "-" : s.p95_duration_s + "s") + "</p>" +
        (agents ? "<h3>By agent</h3><table><tr><th>agent</th><th>runs</th><th>pass</th><th>fail</th><th>cost</th></tr>" + agents + "</table>" : "") +
        (tools ? "<h3>Tools</h3><table><tr><th>tool</th><th>calls</th><th>errors</th><th>err rate</th><th>p95</th></tr>" + tools + "</table>" : "") +
        (findings ? "<h3>Findings</h3><table><tr><th>kind</th><th>sev</th><th>n</th></tr>" + findings + "</table>" : "") +
        (checks ? "<h3>Failing checks</h3><table><tr><th>name</th><th>fail</th><th>error</th></tr>" + checks + "</table>" : "");
      location.hash = "stats";
    } catch (e) {
      $("main").innerHTML = "<p class='fail'>" + esc(e.message) + "</p>";
    }
  }

  async function openDiff() {
    const ids = Array.from(state.selected);
    if (ids.length !== 2) {
      $("main").innerHTML = "<p class='muted'>Tick two runs in the list, then click Diff.</p>";
      return;
    }
    try {
      const d = await get("/api/diff?a=" + encodeURIComponent(ids[0]) + "&b=" + encodeURIComponent(ids[1]));
      const rows = (d.steps || []).map((s) =>
        "<tr><td>" + esc(s.idx) + "</td><td>" + esc(s.status) + "</td><td>" + esc((s.fields||[]).join(", ")) +
        "</td><td><pre>" + esc(s.detail || "") + "</pre></td></tr>"
      ).join("");
      const checks = (d.checks || []).map((c) =>
        "<tr><td>" + esc(c.name) + "</td><td>" + esc(c.a ? c.a.status : "-") + "</td><td>" +
        esc(c.b ? c.b.status : "-") + "</td></tr>"
      ).join("");
      $("main").innerHTML =
        "<h2>Diff</h2><p>" + esc(d.summary) + (d.relation ? " · " + esc(d.relation) : "") + "</p>" +
        "<div class='pane'><div><h3>A " + esc(d.a.name) + "</h3><p class='muted'><code>" + esc(d.a.run_id) +
        "</code></p><p>" + cls(d.a.outcome) + " · " + esc(fmtMoney(d.a.cost_usd)) + "</p></div>" +
        "<div><h3>B " + esc(d.b.name) + "</h3><p class='muted'><code>" + esc(d.b.run_id) +
        "</code></p><p>" + cls(d.b.outcome) + " · " + esc(fmtMoney(d.b.cost_usd)) + "</p></div></div>" +
        "<h3>Steps</h3><table><tr><th>#</th><th>status</th><th>fields</th><th>detail</th></tr>" + rows + "</table>" +
        (checks ? "<h3>Checks</h3><table><tr><th>name</th><th>A</th><th>B</th></tr>" + checks + "</table>" : "");
      location.hash = "diff/" + ids[0] + "/" + ids[1];
    } catch (e) {
      $("main").innerHTML = "<p class='fail'>" + esc(e.message) + "</p>";
    }
  }

  $("q").addEventListener("input", renderList);
  $("statsBtn").onclick = openStats;
  $("diffBtn").onclick = openDiff;

  function route() {
    const h = (location.hash || "").replace(/^#/, "");
    if (h === "stats") return openStats();
    const m = /^run\/(.+)$/.exec(h);
    if (m) return openRun(decodeURIComponent(m[1]));
    const d = /^diff\/([^/]+)\/(.+)$/.exec(h);
    if (d) {
      state.selected = new Set([decodeURIComponent(d[1]), decodeURIComponent(d[2])]);
      return openDiff();
    }
  }

  get("/api/runs").then((data) => {
    state.runs = data.runs || data;
    renderList();
    if (location.hash) route();
  }).catch((e) => { $("runs").innerHTML = "<p class='fail'>" + esc(e.message) + "</p>"; });
  window.addEventListener("hashchange", route);
})();
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    store: Store

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        # keep the test / terminal output quiet; the UI is a local debug tool
        return

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._handle_get()
        except Exception as exc:  # pragma: no cover - unexpected bugs still return JSON
            self._error(500, f"{type(exc).__name__}: {exc}")

    def _handle_get(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        qs = parse_qs(parsed.query)
        if path in ("/", "/index.html"):
            self._send(200, HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/runs":
            self._list_runs(qs)
            return
        if path == "/api/stats":
            since = (qs.get("since") or [None])[0]
            name = (qs.get("name") or [None])[0]
            limit_s = (qs.get("limit") or [None])[0]
            limit = int(limit_s) if limit_s and str(limit_s).isdigit() else None
            self._json(
                200, compute_stats(self.store, since=since, name=name, limit=limit).to_dict()
            )
            return
        if path == "/api/diff":
            a = (qs.get("a") or [None])[0]
            b = (qs.get("b") or [None])[0]
            if not a or not b:
                self._error(400, "query parameters a and b (run ids) are required")
                return
            try:
                ra, rb = _load(self.store, a), _load(self.store, b)
            except (RunNotFoundError, ValueError) as exc:
                self._error(404, str(exc))
                return
            self._json(200, _diff_payload(ra, rb))
            return
        m = re.fullmatch(r"/api/runs/([^/]+)/otel", path)
        if m:
            rid = m.group(1)
            try:
                run = _load(self.store, rid)
            except (RunNotFoundError, ValueError) as exc:
                self._error(404, str(exc))
                return
            include = (qs.get("children") or ["1"])[0] not in {"0", "false", "no"}
            self._json(200, to_otlp(run, store=self.store, include_children=include))
            return
        m = re.fullmatch(r"/api/runs/([^/]+)", path)
        if m:
            rid = m.group(1)
            if not _SAFE_ID.fullmatch(rid):
                self._error(404, f"no run matching {rid!r}")
                return
            try:
                run = _load(self.store, rid)
            except (RunNotFoundError, ValueError) as exc:
                self._error(404, str(exc))
                return
            self._json(200, _run_detail(self.store, run))
            return
        self._error(404, "not found")

    def _list_runs(self, qs: dict[str, list[str]]) -> None:
        name = (qs.get("name") or [None])[0]
        limit_s = (qs.get("limit") or [None])[0]
        limit = int(limit_s) if limit_s and str(limit_s).isdigit() else None
        q = (qs.get("q") or [""])[0].lower()
        rows = [_index_summary(r) for r in self.store.list_runs(limit=limit, name=name)]
        if q:
            rows = [
                r
                for r in rows
                if q
                in " ".join(
                    str(r.get(k) or "") for k in ("run_id", "name", "outcome", "status")
                ).lower()
            ]
        self._json(200, {"runs": rows, "n": len(rows)})

    def do_HEAD(self) -> None:  # noqa: N802
        # used by health checks; honour the same paths as GET without a body
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send(200, b"", "text/html; charset=utf-8")
            return
        if parsed.path.startswith("/api/"):
            self._send(200, b"", "application/json; charset=utf-8")
            return
        self._error(404, "not found")


def make_handler(store: Store) -> type[BaseHTTPRequestHandler]:
    """A ``BaseHTTPRequestHandler`` subclass closed over ``store``."""

    class Handler(_Handler):
        pass

    Handler.store = store
    Handler.protocol_version = "HTTP/1.1"
    return Handler


def create_server(store: Store, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """A localhost HTTP server for the trace explorer. ``port=0`` picks an ephemeral port
    (read it back from ``server.server_address[1]``). Call ``serve_forever`` to start.
    """
    if host not in _LOOPBACK:
        raise ValueError("crashlab ui binds only to localhost (127.0.0.1, ::1, or localhost)")
    server = ThreadingHTTPServer((host, port), make_handler(store))
    server.daemon_threads = True
    return server
