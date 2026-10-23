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
