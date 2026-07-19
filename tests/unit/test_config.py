from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.settings.config import Settings


@pytest.mark.unit
def test_settings_parse_and_validate_runtime_values(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        app_env="test",
        index_dir=tmp_path,
        top_k="7",
        use_bm25="true",
        llm_provider="ollama",
    )

    assert settings.index_dir == tmp_path
    assert settings.top_k == 7
    assert settings.use_bm25 is True


@pytest.mark.unit
def test_production_requires_api_secret() -> None:
    with pytest.raises(ValidationError, match="API_SECRET is required"):
        Settings(_env_file=None, app_env="production", api_secret=None)


@pytest.mark.unit
def test_invalid_retrieval_limit_has_informative_error() -> None:
    with pytest.raises(ValidationError, match="less than or equal to 100"):
        Settings(_env_file=None, top_k=101)
