from __future__ import annotations

import httpx
import pytest

from src.settings.config import Settings
from src.utils.generator import LLM


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openai_compatible_client_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-token"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Ответ из контекста."}}]},
        )

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def client_factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    settings = Settings(
        _env_file=None,
        llm_provider="openai_compatible",
        openai_base_url="https://llm.test/v1",
        openai_api_key="test-token",  # pragma: allowlist secret
        openai_model="test-model",
        llm_timeout=5,
    )

    answer = await LLM(config=settings).arun("Контекст", max_tokens=50)

    assert answer == "Ответ из контекста."
