from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class LLM:
    """
    Требования:
    - установлен и запущен Ollama
    - модель заранее скачана: ollama pull <model>

    По умолчанию:
    - model = "qwen2.5:7b-instruct"
    - base_url = "http://localhost:11434"
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        system_prompt: str = "Ты помощник. Отвечай строго по предоставленному контексту.",
    ) -> None:
        env_model = os.environ.get("OLLAMA_MODEL")
        env_base_url = os.environ.get("OLLAMA_BASE_URL")
        env_timeout = os.environ.get("OLLAMA_TIMEOUT")

        self.model = (env_model or model).strip()
        self.base_url = (env_base_url or base_url).rstrip("/")

        if env_timeout:
            try:
                timeout = float(env_timeout)
            except ValueError:
                logger.warning(
                    "Invalid OLLAMA_TIMEOUT=%r, using default timeout=%s",
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
        max_tokens: int = 700,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Args:
            prompt: финальный промпт (уже с контекстом RAG)
            temperature: 0.0 для детерминированности в RAG
            max_tokens: ограничение на длину ответа (у Ollama это num_predict)
            extra_options: дополнительные параметры Ollama options
        """
        url = f"{self.base_url}/api/chat"

        options: Dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens,
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

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()

        # Ollama отдаёт ответ в data["message"]["content"]
        msg = data.get("message", {})
        content = (msg.get("content") or "").strip()

        if not content:
            logger.warning("LLM вернула пустой ответ. payload.model=%s", self.model)

        return content
