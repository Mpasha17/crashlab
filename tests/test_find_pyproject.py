from __future__ import annotations

from crashlab.checks import config


def test_find_pyproject_cache_is_per_directory(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "pyproject.toml").write_text("[tool.crashlab.prices]\n", "utf-8")
    assert config.find_pyproject(a) == a / "pyproject.toml"
    assert config.find_pyproject(b) != a / "pyproject.toml"
    assert config.find_pyproject(a) == a / "pyproject.toml"  # cached
