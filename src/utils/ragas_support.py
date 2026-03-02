from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import PromptValue
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms.base import BaseRagasLLM

from src.utils.embedder import Embedder
from src.utils.retriever import RetrievedChunk, Retriever


def _prompt_to_str(prompt: PromptValue) -> str:
    # PromptValue in langchain_core exposes to_string in most versions.
    if hasattr(prompt, "to_string"):
        return prompt.to_string()
    # Fallback
    return str(prompt)


@dataclass
class OllamaRagasLLM(BaseRagasLLM):
    """Ragas LLM wrapper around Ollama /api/chat."""

    model: str = "qwen2.5:7b-instruct"
    base_url: str = "http://localhost:11434"
    timeout: float = 600.0
    system_prompt: str = (
        "You are a strict evaluator. Follow the user's instructions exactly. "
        "If the user asks for JSON, output ONLY valid JSON with no extra text."
    )

    # ragas will set run_config later
    multiple_completion_supported: bool = field(default=False, repr=False)

    def _chat_once(
        self,
        user_prompt: str,
        *,
        temperature: float,
        stop: Optional[List[str]] = None,
    ) -> str:
        url = f"{self.base_url.rstrip('/')}/api/chat"
        options: Dict[str, Any] = {"temperature": float(temperature)}
        if stop:
            options["stop"] = stop

        payload: Dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": options,
        }
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
        msg = data.get("message", {})
        return (msg.get("content") or "").strip()

    async def _achat_once(
        self,
        user_prompt: str,
        *,
        temperature: float,
        stop: Optional[List[str]] = None,
    ) -> str:
        url = f"{self.base_url.rstrip('/')}/api/chat"
        options: Dict[str, Any] = {"temperature": float(temperature)}
        if stop:
            options["stop"] = stop

        payload: Dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": options,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
        msg = data.get("message", {})
        return (msg.get("content") or "").strip()

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.01,
        stop: Optional[List[str]] = None,
        callbacks=None,
    ) -> LLMResult:
        text = _prompt_to_str(prompt)
        gens = [
            Generation(text=self._chat_once(text, temperature=temperature, stop=stop))
            for _ in range(n)
        ]
        return LLMResult(generations=[gens])

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: Optional[float] = 0.01,
        stop: Optional[List[str]] = None,
        callbacks=None,
    ) -> LLMResult:
        if temperature is None:
            temperature = self.get_temperature(n)
        text = _prompt_to_str(prompt)
        # Ollama doesn't support n completions in one request; run sequentially.
        outs: List[str] = []
        for _ in range(n):
            outs.append(
                await self._achat_once(text, temperature=float(temperature), stop=stop)
            )
        gens = [Generation(text=o) for o in outs]
        return LLMResult(generations=[gens])

    def is_finished(self, response: LLMResult) -> bool:
        return True


class E5RagasEmbeddings(BaseRagasEmbeddings):
    """Ragas embeddings wrapper around src.utils.embedder.Embedder (E5)."""

    def __init__(self, embedder: Embedder):
        super().__init__()
        self._embedder = embedder

    def embed_query(self, text: str) -> List[float]:
        vec = self._embedder.embed([text], input_type="query")[0]
        return vec.astype("float32").tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vecs = self._embedder.embed(texts, input_type="document")
        return vecs.astype("float32").tolist()

    async def aembed_query(self, text: str) -> List[float]:
        vec = await self._embedder.aembed(text, input_type="query")
        return vec.astype("float32").tolist()

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        vecs = await self._embedder.aembed(texts, input_type="document")
        return vecs.astype("float32").tolist()


def format_locator(idx: int, meta: Dict[str, Any]) -> str:
    source_file = meta.get("source_file", "unknown_file")
    clause = meta.get("clause") or "no_clause"
    clause_span = meta.get("clause_span")
    appendix = meta.get("appendix")
    section = meta.get("section")

    clause_display = clause_span or clause

    parts = [
        f"Источник {idx}: файл={source_file}",
        f"пункт={clause_display}",
    ]
    if appendix:
        parts.append(f"приложение={appendix}")
    if section:
        parts.append(f"раздел={section}")
    return "; ".join(parts)


async def build_retrieved_contexts(
    retriever: Retriever,
    question: str,
    *,
    top_k: int = 5,
    max_context_chars: int = 14000,
) -> List[str]:
    chunks: List[RetrievedChunk] = await retriever.aretrieve(question, top_k=top_k)

    blocks: List[str] = []
    total = 0
    for i, c in enumerate(chunks, 1):
        loc = format_locator(i, c.meta or {})
        block = f"{loc}\n{c.text}".strip()
        if total + len(block) > max_context_chars:
            break
        blocks.append(block)
        total += len(block)
    return blocks


async def warmup_embedder(embedder: Embedder) -> None:
    # Avoid first-call overhead during evaluation
    await asyncio.to_thread(embedder.embed, ["warmup"], input_type="query")
