from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.utils.embedder import Embedder
from src.utils.vector_store import SearchResult, StoredChunk, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """
    Результат ретривала, который удобно отдавать в генератор.
    """
    id: str
    score: float
    text: str
    meta: Dict[str, Any]


class Retriever:
    """
    Ретривер для vanilla RAG:
    - загружает чанки из JSON
    - строит FAISS индекс
    - по запросу возвращает top_k релевантных чанков

    Важно:
    - Embedder должен выдавать нормированные эмбеддинги (normalize=True),
      чтобы FAISS IndexFlatIP давал cosine similarity.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: Optional[VectorStore] = None,
        *,
        top_k: int = 5,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k

    def save(self, dir_path: Path, name: str = "kb") -> None:
        """
        Сохраняет готовый VectorStore на диск (FAISS индекс + метаданные).
        """
        if self.vector_store is None:
            raise RuntimeError(
                "VectorStore не инициализирован. Сначала вызови build_from_*()."
            )

        dir_path = dir_path.resolve()
        dir_path.mkdir(parents=True, exist_ok=True)

        self.vector_store.save(dir_path, name=name)
        logger.info("Retriever: индекс сохранён (%s, name=%s)", dir_path, name)

    async def asave(self, dir_path: Path, name: str = "kb") -> None:
        await asyncio.to_thread(self.save, dir_path, name=name)

    def load(self, dir_path: Path, name: str = "kb") -> None:
        """
        Загружает VectorStore с диска (без пересчёта эмбеддингов).
        """
        dir_path = dir_path.resolve()
        self.vector_store = VectorStore.load(dir_path, name=name)
        logger.info("Retriever: индекс загружен (%s, name=%s)", dir_path, name)

    async def aload(self, dir_path: Path, name: str = "kb") -> None:
        await asyncio.to_thread(self.load, dir_path, name=name)

    # -------------------------
    # Загрузка чанков и построение индекса
    # -------------------------

    def build_from_chunks(self, chunks: Sequence[Dict[str, Any]]) -> None:
        """
        Синхронно строит индекс из списка чанков (dict).
        """
        stored = self._chunks_to_store(chunks)
        if not stored:
            raise ValueError("Нет валидных чанков для индексации (пустые тексты).")

        texts = [c.text for c in stored]
        vectors = self.embedder.embed(texts, input_type="document")

        vs = VectorStore(dim=vectors.shape[1])
        vs.add(vectors, stored)

        self.vector_store = vs
        logger.info(
            "Retriever: индекс построен (чанков=%d, dim=%d)",
            len(stored),
            vectors.shape[1],
        )

    async def abuild_from_chunks(self, chunks: Sequence[Dict[str, Any]]) -> None:
        """
        Асинхронно строит индекс из списка чанков.
        """
        await asyncio.to_thread(self.build_from_chunks, chunks)

    def build_from_json_files(self, chunks_dir: Path, pattern: str = "*.json") -> None:
        """
        Синхронно загружает все JSON из директории и строит индекс.
        """
        chunks_dir = chunks_dir.resolve()
        paths = sorted(chunks_dir.glob(pattern))
        if not paths:
            raise FileNotFoundError(
                f"Не найдены файлы чанков по шаблону {pattern} в {chunks_dir}"
            )

        all_chunks: List[Dict[str, Any]] = []
        for p in paths:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                logger.warning(
                    "Пропускаю %s: ожидался list, получили %s", p.name, type(data)
                )
                continue
            all_chunks.extend(data)

        logger.info("Загружено чанков: %d из %d файлов", len(all_chunks), len(paths))
        self.build_from_chunks(all_chunks)

    async def abuild_from_json_files(
        self, chunks_dir: Path, pattern: str = "*.json"
    ) -> None:
        await asyncio.to_thread(self.build_from_json_files, chunks_dir, pattern)

    # -------------------------
    # Retrieval
    # -------------------------

    async def aretrieve(
        self, query: str, *, top_k: Optional[int] = None
    ) -> List[RetrievedChunk]:
        """
        Асинхронно возвращает релевантные чанки из векторного стора.
        """
        if self.vector_store is None:
            raise RuntimeError(
                "VectorStore не инициализирован. Сначала вызови build_from_*()."
            )

        q_vec = await self.embedder.aembed(query, input_type="query")
        results = await asyncio.to_thread(
            self.vector_store.search, q_vec, top_k or self.top_k
        )

        return [
            RetrievedChunk(id=r.id, score=r.score, text=r.text, meta=r.meta)
            for r in results
        ]

    # -------------------------
    # Helpers
    # -------------------------

    def _chunks_to_store(self, chunks: Sequence[Dict[str, Any]]) -> List[StoredChunk]:
        """
        Преобразует чанки из твоего JSON в StoredChunk для VectorStore.
        """
        stored: List[StoredChunk] = []

        for i, ch in enumerate(chunks):
            text = (ch.get("text") or "").strip()
            if not text:
                continue

            doc_id = ch.get("doc_id") or "unknown_doc"
            source_file = ch.get("source_file") or "unknown_file"
            clause = ch.get("clause") or "no_clause"
            clause_span = ch.get("clause_span")

            # Стабильный id (достаточно хороший для MVP)
            clause_for_id = clause_span or clause
            chunk_id = f"{doc_id}|{source_file}|{clause_for_id}|{i}"

            meta = {
                "doc_id": doc_id,
                "source_file": source_file,
                "doc_type": ch.get("doc_type"),
                "appendix": ch.get("appendix"),
                "section": ch.get("section"),
                "clause": clause,
                "clause_span": clause_span,
                "language": ch.get("language", "ru"),
            }

            stored.append(StoredChunk(id=chunk_id, text=text, meta=meta))

        return stored
