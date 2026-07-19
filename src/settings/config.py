from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Validated runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    api_secret: SecretStr | None = None

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "moscow_kb"
    qdrant_timeout: int = Field(default=30, gt=0, le=300)
    top_k: int = Field(default=5, ge=1, le=100)
    dense_candidates_k: int = Field(default=50, ge=1, le=1000)
    sparse_candidates_k: int = Field(default=50, ge=1, le=1000)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    reranker_candidates_k: int = Field(default=50, ge=1, le=1000)
    reranker_device: str = "cpu"
    max_context_chars: int = Field(default=14_000, ge=1_000, le=200_000)

    llm_provider: Literal["ollama", "openai_compatible", "openai", "vllm"] = "ollama"
    llm_timeout: float = Field(default=180.0, gt=0.0, le=1800.0)
    openai_base_url: str = "http://localhost:8000/v1"
    openai_api_key: SecretStr | None = None
    openai_model: str = "customer-model"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"

    es_url: str = "http://localhost:9200"
    es_index: str = "kb_chunks"
    chunks_dir: Path = BASE_DIR / "data" / "chunks_json"
    es_search_api_url: str = "http://localhost:8010"

    rag_api_port: int = Field(default=8000, ge=1, le=65535)
    es_api_port: int = Field(default=8010, ge=1, le=65535)
    gradio_host: str = "127.0.0.1"
    gradio_port: int = Field(default=7860, ge=1, le=65535)
    telegram_bot_token: SecretStr | None = None

    @model_validator(mode="after")
    def validate_production_secrets(self) -> Settings:
        if self.app_env == "production" and not self.api_secret:
            raise ValueError("API_SECRET is required when APP_ENV=production")
        if self.llm_provider in {
            "openai",
            "openai_compatible",
            "vllm",
        } and (not self.openai_base_url.strip() or not self.openai_model.strip()):
            raise ValueError(
                "OPENAI_BASE_URL and OPENAI_MODEL are required for an OpenAI-compatible LLM"
            )
        return self

    @property
    def api_secret_value(self) -> str:
        return self.api_secret.get_secret_value() if self.api_secret else ""

    @property
    def qdrant_api_key_value(self) -> str | None:
        if self.qdrant_api_key is None:
            return None
        return self.qdrant_api_key.get_secret_value() or None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
