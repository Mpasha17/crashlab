"""The requires_providers marker must skip before the test body imports anything."""

from __future__ import annotations

import pytest


@pytest.mark.requires_providers("definitely_not_installed_crashlab_pkg_xyz")
def test_missing_provider_is_skipped_before_body():
    raise AssertionError("marker should have skipped this test")


@pytest.mark.requires_providers("json")  # stdlib always present
def test_present_provider_runs():
    import json

    assert json.dumps({"a": 1}) == '{"a": 1}'
