from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import httpx

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
    ) -> None:
        self.provider = os.environ.get("LLM_PROVIDER", "ollama").strip().lower()
        env_timeout = os.environ.get("LLM_TIMEOUT") or os.environ.get("OLLAMA_TIMEOUT")

        if self.provider in ("openai", "openai_compatible", "vllm"):
            self.provider = "openai_compatible"
            self.model = (
                os.environ.get("OPENAI_MODEL")
                or os.environ.get("VLLM_MODEL")
                or model
            ).strip()
            self.base_url = (
                os.environ.get("OPENAI_BASE_URL")
                or os.environ.get("VLLM_BASE_URL")
                or "http://localhost:8000/v1"
            ).rstrip("/")
            self.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("VLLM_API_KEY") or ""
        else:
            self.provider = "ollama"
            self.model = (os.environ.get("OLLAMA_MODEL") or model).strip()
            self.base_url = (os.environ.get("OLLAMA_BASE_URL") or base_url).rstrip("/")
            self.api_key = ""

        if env_timeout:
            try:
                timeout = float(env_timeout)
            except ValueError:
                logger.warning(
                    "Invalid LLM timeout=%r, using default timeout=%s",
                    env_timeout,
                    timeout,
                )

        self.timeout = timeout
        self.system_prompt = system_prompt

    async def arun(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 300,
        extra_options: Optional[Dict[str, Any]] = None,
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
        extra_options: Optional[Dict[str, Any]],
    ) -> str:
        url = f"{self.base_url}/api/chat"

        options: Dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": 2048,
        }
        if extra_options:
            options.update(extra_options)

        payload: Dict[str, Any] = {
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
            body = e.response.text if e.response is not None else "<no body>"
            logger.error(
                "LLM HTTP error: status=%s body=%s",
                e.response.status_code if e.response else "unknown",
                body,
            )
            raise
        except httpx.HTTPError:
            logger.exception("LLM connection error")
            raise

        msg = data.get("message", {})
        content = (msg.get("content") or "").strip()

        if not content:
            logger.warning("LLM returned empty content. provider=%s model=%s", self.provider, self.model)

        return content

    async def _run_openai_compatible(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        extra_options: Optional[Dict[str, Any]],
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
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

        headers: Dict[str, str] = {}
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
            body = e.response.text if e.response is not None else "<no body>"
            logger.error(
                "LLM HTTP error: status=%s body=%s",
                e.response.status_code if e.response else "unknown",
                body,
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
            logger.warning("LLM returned empty content. provider=%s model=%s", self.provider, self.model)

        return content
