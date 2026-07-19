from __future__ import annotations

import pytest

from src.settings.config import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    """Prevent environment-dependent settings from leaking between tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
