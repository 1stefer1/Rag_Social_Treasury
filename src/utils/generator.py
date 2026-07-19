from __future__ import annotations

import logging
from typing import Any

import httpx

from src.settings.config import Settings, get_settings

logger = logging.getLogger(__name__)


class LLM:
    """
    LLM client for RAG generation.

    Supported providers:
    - ollama: Ollama /api/chat
    - openai_compatible: OpenAI-compatible /v1/chat/completions (vLLM, gateways)
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        system_prompt: str = "Ты помощник. Отвечай строго по предоставленному контексту.",
        config: Settings | None = None,
    ) -> None:
        config = config or get_settings()
        self.provider = config.llm_provider

        if self.provider in ("openai", "openai_compatible", "vllm"):
            self.provider = "openai_compatible"
            self.model = (
                config.openai_model.strip() if "openai_model" in config.model_fields_set else model
            )
            self.base_url = (
                config.openai_base_url.rstrip("/")
                if "openai_base_url" in config.model_fields_set
                else "http://localhost:8000/v1"
            )
            self.api_key = config.openai_api_key.get_secret_value() if config.openai_api_key else ""
        else:
            self.provider = "ollama"
            self.model = (
                config.ollama_model.strip() if "ollama_model" in config.model_fields_set else model
            )
            self.base_url = (
                config.ollama_base_url.rstrip("/")
                if "ollama_base_url" in config.model_fields_set
                else base_url.rstrip("/")
            )
            self.api_key = ""

        self.timeout = config.llm_timeout if "llm_timeout" in config.model_fields_set else timeout
        self.system_prompt = system_prompt

    async def arun(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 300,
        extra_options: dict[str, Any] | None = None,
    ) -> str:
        if self.provider == "openai_compatible":
            return await self._run_openai_compatible(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_options=extra_options,
            )
        return await self._run_ollama(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_options=extra_options,
        )

    async def _run_ollama(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        extra_options: dict[str, Any] | None,
    ) -> str:
        url = f"{self.base_url}/api/chat"

        options: dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": 2048,
        }
        if extra_options:
            options.update(extra_options)

        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "options": options,
        }

        logger.info(
            "LLM request: provider=%s model=%s url=%s prompt_len=%d",
            self.provider,
            self.model,
            url,
            len(prompt),
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
                r = await client.post(url, json=payload)
                r.raise_for_status()
                data = r.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                "LLM HTTP error: status=%s",
                e.response.status_code if e.response else "unknown",
            )
            raise
        except httpx.HTTPError:
            logger.exception("LLM connection error")
            raise

        msg = data.get("message", {})
        content = (msg.get("content") or "").strip()

        if not content:
            logger.warning(
                "LLM returned empty content. provider=%s model=%s",
                self.provider,
                self.model,
            )

        return content

    async def _run_openai_compatible(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        extra_options: dict[str, Any] | None,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        if extra_options:
            payload.update(extra_options)

        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        logger.info(
            "LLM request: provider=%s model=%s url=%s prompt_len=%d",
            self.provider,
            self.model,
            url,
            len(prompt),
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
                r = await client.post(url, json=payload, headers=headers)
                r.raise_for_status()
                data = r.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                "LLM HTTP error: status=%s",
                e.response.status_code if e.response else "unknown",
            )
            raise
        except httpx.HTTPError:
            logger.exception("LLM connection error")
            raise

        choices = data.get("choices") or []
        content = ""
        if choices:
            message = choices[0].get("message") or {}
            content = (message.get("content") or "").strip()

        if not content:
            logger.warning(
                "LLM returned empty content. provider=%s model=%s",
                self.provider,
                self.model,
            )

        return content
