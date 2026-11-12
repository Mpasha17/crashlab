"""The README and docs cannot rot silently.

- every ``crashlab ...`` command line in a fenced block parses against the real CLI
  (command exists, options exist, values have the right type);
- every Python block compiles, every ``crashlab.X`` / ``from crashlab... import`` name it
  uses exists, and keyword arguments passed to crashlab functions are real parameters;
- every TOML block parses;
- the README's "Try it with no API key" block runs offline, command by command.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
import shlex
import sys
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from crashlab.cli import app

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

# typer < 0.2x uses click; newer typer vendors it (typer._click) and does not install it
_usage_errors: list[type[Exception]] = []
try:
    import click

    _usage_errors.append(click.UsageError)
except ImportError:  # pragma: no cover
    pass
try:
    from typer._click.exceptions import UsageError as _TyperUsageError

    _usage_errors.append(_TyperUsageError)
except ImportError:  # pragma: no cover
    pass
USAGE_ERRORS: tuple[type[Exception], ...] = tuple(_usage_errors)

ROOT = Path(__file__).resolve().parents[1]
DOCS = sorted(
    [ROOT / "README.md", ROOT / "examples/refund_bot/README.md", ROOT / "bench/README.md"]
    + [p for p in (ROOT / "docs").glob("*.md")]
    + list((ROOT / "examples/integrations").glob("*.md"))
)
FENCE = re.compile(r"^```(\w*)[^\n]*\n(.*?)^```", re.S | re.M)


def _blocks(lang: str):
    for path in DOCS:
        text = path.read_text("utf-8")
        for m in FENCE.finditer(text):
            if m.group(1) == lang:
                line = text.count("\n", 0, m.start()) + 2
                yield f"{path.relative_to(ROOT)}:{line}", m.group(2)


def _cli_lines():
    for where, body in _blocks("bash"):
        for line in body.replace("\\\n", " ").splitlines():
            for part in line.split("&&"):
                part = part.strip().removeprefix("$ ")
                if part.startswith("crashlab "):
                    yield pytest.param(part, id=f"{where}: {part[:60]}")


def _parse_error(line: str) -> str | None:
    args = shlex.split(line, comments=True)[1:]
    group = typer.main.get_command(app)
    with group.make_context("crashlab", ["--help"], resilient_parsing=True) as ctx:
        cmd = group.get_command(ctx, args[0])
        if cmd is None:
            return f"no command {args[0]!r}"
        try:
            cmd.make_context(args[0], args[1:], parent=ctx)
        except USAGE_ERRORS as exc:
            return exc.format_message()
    return None


@pytest.mark.parametrize("line", list(_cli_lines()))
def test_documented_command_parses(line):
    assert _parse_error(line) is None, line


def test_the_cli_check_catches_stale_commands():
    assert _parse_error("crashlab show last --failing") is None
    assert "no command" in _parse_error("crashlab shw last")
    assert _parse_error("crashlab show last --no-such-flag")
    assert _parse_error("crashlab fork last --at notanumber")


def _resolve(dotted: str):
    """crashlab.a.b.c -> the object, importing submodules on the way."""
    parts = dotted.split(".")
    obj = importlib.import_module(parts[0])
    for i, part in enumerate(parts[1:], 1):
        if not hasattr(obj, part):
            try:
                obj = importlib.import_module(".".join(parts[: i + 1]))
                continue
            except ImportError:
                raise AttributeError(f"{'.'.join(parts[:i])} has no {part!r}") from None
        obj = getattr(obj, part)
    return obj


def _dotted(node: ast.AST) -> str | None:
    names = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return ".".join([node.id, *reversed(names)])
    return None


def _check_python(where: str, source: str) -> list[str]:
    tree = ast.parse(source, where)  # SyntaxError fails the test with the location
    names: dict[str, str] = {}  # local name -> dotted crashlab path
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] == "crashlab":
                    names[(a.asname or a.name).split(".")[0]] = (a.asname and a.name) or "crashlab"
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("crashlab"):
            for a in node.names:
                names[a.asname or a.name] = f"{node.module}.{a.name}"
    if "crashlab" not in source:
        return []
    names.setdefault("crashlab", "crashlab")
    problems = []

    def lookup(expr: ast.AST):
        dotted = _dotted(expr)
        if not dotted:
            return None
        head, _, rest = dotted.partition(".")
        if head not in names:
            return None
        full = names[head] + ("." + rest if rest else "")
        try:
            return _resolve(full)
        except (AttributeError, ImportError) as exc:
            problems.append(f"{where}: {dotted}: {exc}")
            return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and not isinstance(node.ctx, ast.Store):
            lookup(node)
        elif isinstance(node, ast.Call):
            fn = lookup(node.func)
            kws = [k.arg for k in node.keywords if k.arg]
            if fn is None or not callable(fn) or not kws:
                continue
            try:
                sig = inspect.signature(fn)
            except (TypeError, ValueError):
                continue
            if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
                continue
            bad = [k for k in kws if k not in sig.parameters]
            if bad:
                problems.append(f"{where}: {_dotted(node.func)}() has no parameter(s) {bad}")
    return problems


@pytest.mark.parametrize("where,source", [pytest.param(w, s, id=w) for w, s in _blocks("python")])
def test_documented_python_uses_real_api(where, source):
    assert _check_python(where, source) == []


@pytest.mark.parametrize("where,source", [pytest.param(w, s, id=w) for w, s in _blocks("toml")])
def test_documented_toml_parses(where, source):
    tomllib.loads(source)


def test_the_checker_catches_stale_snippets():
    bad = "import crashlab\ncrashlab.no_such_thing()\ncrashlab.fork(run, at_stepp=0)\n"
    problems = _check_python("x", bad)
    assert any("no_such_thing" in p for p in problems)
    assert any("at_stepp" in p for p in problems)


def _no_key_block() -> str:
    text = (ROOT / "README.md").read_text("utf-8")
    section = text.split("### Try it with no API key", 1)[1]
    return FENCE.search(section).group(2)


def test_readme_try_it_with_no_api_key_runs(tmp_path, monkeypatch):
    """Copy-paste of the README block, minus the clone / venv lines (this is that
    checkout). Each command must exit as its comment says (0 unless "exit 1")."""
    pytest.importorskip("openai")  # the demo agent replays through the OpenAI SDK
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("CRASHLAB_DIR", str(tmp_path / "demo"))
    for var in ("OPENAI_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    runner = CliRunner()
    ran = 0
    for line in _no_key_block().replace("\\\n", " ").splitlines():
        line = line.strip()
        if line.startswith(
            "for f in "
        ):  # for f in examples/.../*.jsonl; do crashlab import "$f"; done
            glob = line.split(" in ", 1)[1].split(";", 1)[0].strip()
            for f in sorted(ROOT.glob(glob)):
                res = runner.invoke(app, ["import", str(f)])
                assert res.exit_code == 0, res.output
            continue
        if not line.startswith("crashlab "):
            continue
        want = 1 if re.search(r"#.*exit 1", line) else 0
        res = runner.invoke(app, shlex.split(line, comments=True)[1:])
        assert res.exit_code == want, f"{line}\n{res.output}"
        ran += 1
    assert ran >= 6
