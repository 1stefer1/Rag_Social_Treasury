from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.utils.embedder import Embedder
from src.utils.reranker import CrossEncoderReranker
from src.utils.vector_store import SearchResult, StoredChunk, VectorStore

logger = logging.getLogger(__name__)


try:
    from rank_bm25 import BM25Okapi  # type: ignore

    _HAS_BM25 = True
except Exception:
    BM25Okapi = None  # type: ignore
    _HAS_BM25 = False


_TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


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
        use_bm25: bool = False,
        bm25_weight: float = 0.25,
        bm25_candidates_k: int = 50,
        use_reranker: bool = False,
        reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        reranker_candidates_k: int = 50,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k

        self.use_bm25 = use_bm25
        self.bm25_weight = bm25_weight
        self.bm25_candidates_k = bm25_candidates_k

        self.use_reranker = use_reranker
        self.reranker_model = reranker_model
        self.reranker_candidates_k = reranker_candidates_k

        self._reranker: Optional[CrossEncoderReranker] = None

        self._id_to_pos: Dict[str, int] = {}
        self._bm25 = None

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
        self._post_load_build_aux_indexes()
        logger.info("Retriever: индекс загружен (%s, name=%s)", dir_path, name)

    def _post_load_build_aux_indexes(self) -> None:
        """Build lightweight helper indexes (id->pos, optional BM25)."""
        self._id_to_pos = {}
        self._bm25 = None
        if self.vector_store is None:
            return

        for i, ch in enumerate(self.vector_store.store):
            self._id_to_pos[ch.id] = i

        if not self.use_bm25:
            return
        if not _HAS_BM25 or BM25Okapi is None:
            logger.warning(
                "BM25 requested but rank-bm25 is not installed; falling back to vector search"
            )
            return

        corpus_tokens = [_tokenize(ch.text) for ch in self.vector_store.store]
        self._bm25 = BM25Okapi(corpus_tokens)

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
        self._post_load_build_aux_indexes()
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

        req_k = top_k or self.top_k
        req_k = int(req_k)

        q_vec = await self.embedder.aembed(query, input_type="query")

        # Candidate pool size
        cand_k = req_k
        if self.use_bm25:
            cand_k = max(cand_k, 50)
        if self.use_reranker:
            cand_k = max(cand_k, int(self.reranker_candidates_k))

        sem_results = await asyncio.to_thread(self.vector_store.search, q_vec, cand_k)

        # FAISS-only path
        if not self.use_bm25 or self._bm25 is None:
            candidates = [
                RetrievedChunk(id=r.id, score=r.score, text=r.text, meta=r.meta)
                for r in sem_results
            ]
            return await self._maybe_rerank(query, candidates, req_k)

        # BM25 candidates
        q_tokens = _tokenize(query)
        bm_scores = self._bm25.get_scores(q_tokens)
        if bm_scores is None:
            bm_scores = []

        # Take top bm25 candidates
        k_bm = min(int(self.bm25_candidates_k), len(bm_scores))
        bm_top_pos: List[int] = []
        if k_bm > 0:
            bm_top_pos = sorted(
                range(len(bm_scores)), key=lambda i: bm_scores[i], reverse=True
            )[:k_bm]

        # Map semantic results to positions
        sem_pos_score: Dict[int, float] = {}
        for r in sem_results:
            pos = self._id_to_pos.get(r.id)
            if pos is None:
                continue
            sem_pos_score[pos] = float(r.score)

        cand_pos = set(sem_pos_score.keys()) | set(bm_top_pos)
        if not cand_pos:
            return []

        best_sem = max(sem_pos_score.values()) if sem_pos_score else 0.0
        best_bm = max((float(bm_scores[i]) for i in bm_top_pos), default=0.0)
        w_bm = float(self.bm25_weight)
        w_bm = 0.0 if w_bm < 0 else 1.0 if w_bm > 1 else w_bm

        scored: List[tuple[float, int]] = []
        for pos in cand_pos:
            sem = sem_pos_score.get(pos, 0.0)
            sem_n = (sem / best_sem) if best_sem > 0 else 0.0
            bm = float(bm_scores[pos]) if pos < len(bm_scores) else 0.0
            bm_n = (bm / best_bm) if best_bm > 0 else 0.0
            score = (1.0 - w_bm) * sem_n + w_bm * bm_n
            scored.append((score, pos))

        scored.sort(key=lambda x: x[0], reverse=True)
        candidates: List[RetrievedChunk] = []
        for score, pos in scored:
            item = self.vector_store.store[pos]
            candidates.append(
                RetrievedChunk(
                    id=item.id, score=float(score), text=item.text, meta=item.meta
                )
            )
        return await self._maybe_rerank(query, candidates, req_k)

    async def _maybe_rerank(
        self, query: str, candidates: List[RetrievedChunk], top_k: int
    ) -> List[RetrievedChunk]:
        if not candidates:
            return []

        if not self.use_reranker:
            return candidates[:top_k]

        n = min(int(self.reranker_candidates_k), len(candidates))
        pool = candidates[:n]

        if self._reranker is None:
            self._reranker = CrossEncoderReranker(model_name=self.reranker_model)

        scores = await self._reranker.ascore(query, [c.text for c in pool])
        if len(scores) != len(pool):
            logger.warning(
                "Reranker returned %d scores for %d passages", len(scores), len(pool)
            )
            return candidates[:top_k]

        reranked = []
        for c, s in zip(pool, scores):
            reranked.append(
                RetrievedChunk(id=c.id, score=float(s), text=c.text, meta=c.meta)
            )
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]

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
