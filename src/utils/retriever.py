from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from src.utils.embedder import Embedder
from src.utils.reranker import CrossEncoderReranker
from src.utils.sparse_store import ElasticsearchSparseStore
from src.utils.vector_store import QdrantVectorStore, SearchResult, StoredChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    score: float
    text: str
    meta: dict[str, Any]


class Retriever:
    """Hybrid retrieval: Qdrant dense + Elasticsearch BM25 + RRF + reranking."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: QdrantVectorStore,
        sparse_store: ElasticsearchSparseStore,
        *,
        top_k: int = 5,
        dense_candidates_k: int = 50,
        sparse_candidates_k: int = 50,
        rrf_k: int = 60,
        use_reranker: bool = False,
        reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        reranker_candidates_k: int = 50,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.sparse_store = sparse_store
        self.top_k = top_k
        self.dense_candidates_k = dense_candidates_k
        self.sparse_candidates_k = sparse_candidates_k
        self.rrf_k = rrf_k
        self.use_reranker = use_reranker
        self.reranker_model = reranker_model
        self.reranker_candidates_k = reranker_candidates_k
        self._reranker: CrossEncoderReranker | None = None

    def build_from_chunks(self, chunks: Sequence[dict[str, Any]]) -> None:
        """Rebuild both retrieval indexes from the same normalized chunk set."""
        stored = self._chunks_to_store(chunks)
        if not stored:
            raise ValueError("No non-empty chunks were provided for indexing")

        vectors = self.embedder.embed([chunk.text for chunk in stored], input_type="document")
        self.vector_store.recreate_collection(vector_size=vectors.shape[1])
        self.vector_store.add(vectors, stored)
        self.sparse_store.recreate_index()
        self.sparse_store.add(stored)
        logger.info("Built hybrid indexes from %d chunks", len(stored))

    async def abuild_from_chunks(self, chunks: Sequence[dict[str, Any]]) -> None:
        await asyncio.to_thread(self.build_from_chunks, chunks)

    def build_from_json_files(self, chunks_dir: Path, pattern: str = "*.json") -> None:
        paths = sorted(chunks_dir.resolve().glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No chunk files matching {pattern} in {chunks_dir}")

        chunks: list[dict[str, Any]] = []
        for path in paths:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            if not isinstance(data, list):
                logger.warning("Skipping %s: expected a JSON list", path.name)
                continue
            chunks.extend(item for item in data if isinstance(item, dict))
        self.build_from_chunks(chunks)

    async def abuild_from_json_files(self, chunks_dir: Path, pattern: str = "*.json") -> None:
        await asyncio.to_thread(self.build_from_json_files, chunks_dir, pattern)

    async def aretrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        requested_k = int(top_k or self.top_k)
        if requested_k < 1:
            raise ValueError("top_k must be positive")

        query_vector = await self.embedder.aembed(query, input_type="query")
        dense_task = asyncio.to_thread(
            self.vector_store.search, query_vector, self.dense_candidates_k
        )
        sparse_task = asyncio.to_thread(self.sparse_store.search, query, self.sparse_candidates_k)
        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)

        candidates = reciprocal_rank_fusion(dense_results, sparse_results, rrf_k=self.rrf_k)
        return await self._maybe_rerank(query, candidates, requested_k)

    async def _maybe_rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates or not self.use_reranker:
            return candidates[:top_k]

        pool = candidates[: self.reranker_candidates_k]
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(model_name=self.reranker_model)
        scores = await self._reranker.ascore(query, [candidate.text for candidate in pool])
        if len(scores) != len(pool):
            raise RuntimeError(f"Reranker returned {len(scores)} scores for {len(pool)} candidates")
        reranked = [
            RetrievedChunk(id=item.id, score=float(score), text=item.text, meta=item.meta)
            for item, score in zip(pool, scores, strict=True)
        ]
        return sorted(reranked, key=lambda item: item.score, reverse=True)[:top_k]

    @staticmethod
    def _chunks_to_store(chunks: Sequence[dict[str, Any]]) -> list[StoredChunk]:
        stored: list[StoredChunk] = []
        for index, chunk in enumerate(chunks):
            text = str(chunk.get("text") or "").strip()
            if not text:
                continue
            doc_id = str(chunk.get("doc_id") or "unknown_doc")
            source_file = str(chunk.get("source_file") or "unknown_file")
            clause = str(chunk.get("clause") or "no_clause")
            clause_span = str(chunk.get("clause_span") or "")
            chunk_id = f"{doc_id}|{source_file}|{clause_span or clause}|{index}"
            stored.append(
                StoredChunk(
                    id=chunk_id,
                    text=text,
                    meta={
                        "doc_id": doc_id,
                        "source_file": source_file,
                        "doc_type": str(chunk.get("doc_type") or ""),
                        "appendix": str(chunk.get("appendix") or ""),
                        "section": str(chunk.get("section") or ""),
                        "clause": clause,
                        "clause_span": clause_span,
                        "language": str(chunk.get("language") or "ru"),
                    },
                )
            )
        return stored


def reciprocal_rank_fusion(
    dense_results: Sequence[SearchResult],
    sparse_results: Sequence[SearchResult],
    *,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """Fuse ranked lists without assuming comparable backend score scales."""
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")

    scores: dict[str, float] = {}
    chunks: dict[str, SearchResult] = {}
    for result_list in (dense_results, sparse_results):
        for rank, result in enumerate(result_list, start=1):
            scores[result.id] = scores.get(result.id, 0.0) + 1.0 / (rrf_k + rank)
            chunks.setdefault(result.id, result)

    return [
        RetrievedChunk(
            id=chunk_id, score=score, text=chunks[chunk_id].text, meta=chunks[chunk_id].meta
        )
        for chunk_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ]
