"""The built wheel ships crashlab.rag (not the examples, tests or docs); the sdist ships
examples/rag_qa and docs/rag-testing.md. Slow-ish: builds once, offline install."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dist(tmp_path_factory):
    if importlib.util.find_spec("build") is None and importlib.util.find_spec("hatchling") is None:
        pytest.skip("needs `build` or hatchling to build the wheel")
    out = tmp_path_factory.mktemp("dist")
    if importlib.util.find_spec("build"):
        cmd = [sys.executable, "-m", "build", "--wheel", "--sdist", "--no-isolation",
               "--outdir", str(out), str(ROOT)]  # fmt: skip
    else:
        pytest.skip("needs `build`")
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if res.returncode != 0:
        pytest.skip(f"build unavailable here: {res.stderr[-300:]}")
    return out


def test_wheel_contains_rag_but_not_examples(dist):
    [whl] = dist.glob("*.whl")
    names = zipfile.ZipFile(whl).namelist()
    assert "crashlab/rag.py" in names
    assert not any(n.startswith(("examples/", "tests/", "docs/")) for n in names)


def test_sdist_contains_rag_example_and_docs(dist):
    [sdist] = dist.glob("*.tar.gz")
    with tarfile.open(sdist) as tar:
        names = {n.split("/", 1)[1] for n in tar.getnames() if "/" in n}
    for path in ("src/crashlab/rag.py", "examples/rag_qa/agent.py",
                 "examples/rag_qa/langchain_style.py", "docs/rag-testing.md",
                 "tests/test_rag.py"):  # fmt: skip
        assert path in names, path


def test_wheel_installs_and_rag_works_in_clean_venv(dist, tmp_path):
    [whl] = dist.glob("*.whl")
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, timeout=120)
    py = venv / ("Scripts" if sys.platform == "win32" else "bin") / "python"
    # deps from the current environment's wheels cache are not guaranteed offline: reuse
    # this interpreter's site-packages for dependencies, install crashlab itself --no-deps
    res = subprocess.run([str(py), "-m", "pip", "install", "-q", "--no-deps", str(whl)],
                         capture_output=True, text=True, timeout=300)  # fmt: skip
    assert res.returncode == 0, res.stderr
    site = next(p for p in sys.path if p.endswith("site-packages"))
    code = (
        f"import sys; sys.path.append({site!r})\n"
        "import crashlab, crashlab.rag as rag\n"
        f"assert 'site-packages' in crashlab.__file__ and {str(ROOT)!r} not in crashlab.__file__\n"
        "assert rag.cited_ids('x [a] [b, c]') == ['a', 'b', 'c']\n"
        "import importlib.util; assert importlib.util.find_spec('examples') is None\n"
        "print('ok', crashlab.__file__)\n"
    )
    res = subprocess.run([str(py), "-c", code], capture_output=True, text=True, timeout=120,
                         cwd=tmp_path)  # fmt: skip
    assert res.returncode == 0 and res.stdout.startswith("ok"), res.stdout + res.stderr


def test_sdist_rag_examples_and_tests_run_from_extracted_archive(dist, tmp_path):
    pytest.importorskip("openai")
    [sdist] = dist.glob("*.tar.gz")
    with tarfile.open(sdist) as tar:
        tar.extractall(tmp_path, filter="data")
    [src] = [p for p in tmp_path.iterdir() if p.is_dir()]
    env = {k: v for k, v in os.environ.items() if not k.endswith("API_KEY")}
    env.pop("CRASHLAB_DIR", None)
    env["PYTHONPATH"] = str(src / "src")  # the archive's crashlab, not this checkout's
    tests = ["tests/test_rag.py", "tests/test_rag_integration.py", "tests/test_rag_docs.py"]
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
         "-k", "not wheel", *tests],
        cwd=src, env=env, capture_output=True, text=True, timeout=600,
    )  # fmt: skip
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-2000:]
    assert " passed" in res.stdout and "failed" not in res.stdout
    check = subprocess.run(
        [sys.executable, "-c", "import crashlab, sys; print(crashlab.__file__)"],
        cwd=src, env=env, capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    assert check.stdout.strip().startswith(str(src)), check.stdout
