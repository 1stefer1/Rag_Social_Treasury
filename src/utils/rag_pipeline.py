from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.utils.generator import LLM
from src.utils.retriever import RetrievedChunk, Retriever

logger = logging.getLogger(__name__)


@dataclass
class SourceRef:
    id: str
    score: float
    source_file: str
    doc_id: str
    clause: Optional[str] = None
    appendix: Optional[str] = None
    section: Optional[str] = None


@dataclass
class RAGResult:
    question: str
    answer: str
    sources: List[SourceRef]
    used_top_k: int


class VanillaRAG:
    """
    Vanilla RAG:
      - retrieve: top-k чанков
      - prompt: собираем контекст
      - generate: вызываем LLM
      - return: answer + sources

    Важно:
    - LLM отвечает строго по контексту.
    - Если ответа нет в контексте — возвращает явное сообщение.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        *,
        default_top_k: int = 5,
        max_context_chars: int = 14000,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.default_top_k = default_top_k
        self.max_context_chars = max_context_chars

    async def aask(
        self,
        question: str,
        *,
        top_k: Optional[int] = None,
        temperature: float = 0.0,
        max_tokens: int = 700,
    ) -> RAGResult:
        k = top_k or self.default_top_k

        chunks = await self.retriever.aretrieve(question, top_k=k)

        prompt, used_chunks = self._build_prompt(question, chunks)

        answer = await self.llm.arun(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        sources = [self._to_source_ref(c) for c in used_chunks]

        return RAGResult(
            question=question,
            answer=answer.strip(),
            sources=sources,
            used_top_k=len(used_chunks),
        )

    @staticmethod
    def format_sources(sources: List[SourceRef]) -> str:
        """Render sources into user-facing bullet list."""
        lines: List[str] = []
        for s in sources:
            clause = s.clause or "no_clause"
            appendix = s.appendix
            parts = [f"{s.source_file}", f"пункт={clause}"]
            if appendix:
                parts.append(f"приложение={appendix}")
            lines.append("- " + "; ".join(parts))
        return "\n".join(lines).strip()

    # -------------------------
    # Prompt building
    # -------------------------

    def _build_prompt(
        self, question: str, chunks: List[RetrievedChunk]
    ) -> tuple[str, List[RetrievedChunk]]:
        """
        Собираем контекст с ограничением по размеру.
        Возвращаем (prompt, used_chunks).
        """
        used: List[RetrievedChunk] = []
        blocks: List[str] = []
        total = 0

        for i, c in enumerate(chunks, 1):
            locator = self._format_locator(i, c)
            block = f"{locator}\n{c.text}".strip()

            # контроль размера контекста
            if total + len(block) > self.max_context_chars:
                break

            used.append(c)
            blocks.append(block)
            total += len(block)

        context = "\n\n---\n\n".join(blocks).strip()

        # prompt = (
        #     "Ты юридический ассистент. Ответь на вопрос строго на основе КОНТЕКСТА.\n"
        #     "Если в контексте нет информации для ответа, напиши ровно:\n"
        #     "\"В предоставленном контексте нет информации для ответа.\"\n\n"
        #     f"ВОПРОС:\n{question}\n\n"
        #     f"КОНТЕКСТ:\n{context}\n\n"
        #     "ТРЕБОВАНИЯ К ОТВЕТУ:\n"
        #     "1) Дай краткий, прямой ответ (2–6 предложений).\n"
        #     "2) Затем перечисли источники строками в формате:\n"
        #     "   - <файл>; пункт=<пункт или no_clause>; приложение=<если есть>\n"
        #     "3) Не выдумывай. Не используй знания вне контекста.\n"
        # )

        prompt = (
            "Ты юридический ассистент. Используй предоставленный КОНТЕКСТ для ответа на вопрос.\n"
            "Если информация явно есть — дай прямой, конкретный ответ.\n"
            "Если ответ можно логически вывести из контекста (в том числе через перефразирование) — дай его.\n"
            "Только если в контексте действительно нет релевантной информации — верни строго:\n"
            '"В предоставленном контексте нет информации для ответа."\n\n'
            f"ВОПРОС:\n{question}\n\n"
            f"КОНТЕКСТ:\n{context}\n\n"
            "ТРЕБОВАНИЯ К ОТВЕТУ:\n"
            "1) Дай краткий, прямой ответ (2–6 предложений).\n"
            "2) Не перечисляй источники и не упоминай названия файлов/пунктов.\n"
            "3) Не выдумывай ничего вне контекста.\n"
        )

        if not used:
            # если вообще ничего не влезло/не пришло, всё равно отправим LLM корректный запрос
            prompt = (
                "Ты юридический ассистент.\n"
                "Контекст пуст.\n"
                "Ответь:\n"
                '"В предоставленном контексте нет информации для ответа."\n\n'
                f"ВОПРОС:\n{question}\n"
            )

        return prompt, used

    def _format_locator(self, idx: int, c: RetrievedChunk) -> str:
        meta = c.meta or {}
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

    # -------------------------
    # Sources
    # -------------------------

    def _to_source_ref(self, c: RetrievedChunk) -> SourceRef:
        meta: Dict[str, Any] = c.meta or {}

        return SourceRef(
            id=c.id,
            score=float(c.score),
            source_file=str(meta.get("source_file", "unknown_file")),
            doc_id=str(meta.get("doc_id", "unknown_doc")),
            clause=meta.get("clause"),
            appendix=meta.get("appendix"),
            section=meta.get("section"),
        )
