"""Launch demo assets stay present and honest."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_smolagents_demo_assets_exist_and_show_the_ab():
    demo = ROOT / "docs" / "demo"
    assert (demo / "transcript.txt").is_file()
    assert (demo / "terminal.html").is_file()
    assert (demo / "smolagents-text-to-sql.gif").is_file()
    assert (demo / "demoscript.sh").is_file()
    text = (demo / "transcript.txt").read_text()
    assert "no such table" in text or "ERROR" in text
    assert "Woodrow Wilson" in text
    assert "engine_fix" in text
    html = (demo / "terminal.html").read_text()
    assert "http://" not in html and "https://" not in html  # self-contained


def test_case_studies_and_drafts_exist():
    cs = ROOT / "docs" / "case-studies"
    assert (cs / "smolagents-text-to-sql.md").is_file()
    assert (cs / "smolagents-DRAFT_ISSUE.md").is_file()
    assert "not filed" in (cs / "smolagents-DRAFT_ISSUE.md").read_text().lower()
    assert (cs / "openai-agents-customer-service.md").is_file()


def test_package_ships_py_typed_and_version():
    import crashlab
    from crashlab._version import __version__

    assert __version__ == "0.0.1"
    assert (Path(crashlab.__file__).parent / "py.typed").is_file()
    assert '"Typing :: Typed"' in (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()


def test_scripts_check_and_scan_exist_and_scan_is_quiet_on_docs():
    assert (ROOT / "scripts" / "check.sh").is_file()
    scan = ROOT / "scripts" / "scan.sh"
    assert scan.is_file() and scan.stat().st_mode & 0o111
    import subprocess

    res = subprocess.run(["bash", str(scan)], cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "pattern hits: 0" in res.stdout


def test_sdist_include_lists_docs_and_hygiene():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.hatch.build.targets.sdist]" in text
    for name in ("docs", "scripts", "CHANGELOG.md", "SECURITY.md", "CONTRIBUTING.md"):
        assert name in text, f"sdist include missing {name}"


def test_launch_docs_and_changelog_present():
    for rel in (
        "docs/launch/STATUS.md",
        "docs/launch/checklist.md",
        "docs/launch/DAY11_REPORT.md",
        "SECURITY.md",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
    ):
        assert (ROOT / rel).is_file(), rel
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## 0.0.1" in text
    assert "with_raw_response" in text


def test_crewai_case_study_exists():
    assert (ROOT / "docs" / "case-studies" / "crewai-integration.md").is_file()
