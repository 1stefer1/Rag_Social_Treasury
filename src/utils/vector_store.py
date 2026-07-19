from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoredChunk:
    id: str
    text: str
    meta: dict[str, Any]


@dataclass(frozen=True)
class SearchResult:
    id: str
    score: float
    text: str
    meta: dict[str, Any]


class QdrantVectorStore:
    """Dense-vector storage backed by a Qdrant collection."""

    def __init__(self, client: QdrantClient, collection_name: str) -> None:
        self.client = client
        self.collection_name = collection_name

    def recreate_collection(self, vector_size: int) -> None:
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            on_disk_payload=True,
        )
        logger.info(
            "Created Qdrant collection %s (vector_size=%d)",
            self.collection_name,
            vector_size,
        )

    def add(self, vectors: np.ndarray, chunks: Sequence[StoredChunk]) -> None:
        vectors = self._validate_vectors(vectors)
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"Vector/chunk count mismatch: vectors={vectors.shape[0]}, chunks={len(chunks)}"
            )

        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.id)),
                vector=vector.tolist(),
                payload={"chunk_id": chunk.id, "text": chunk.text, **chunk.meta},
            )
            for vector, chunk in zip(vectors, chunks, strict=True)
        ]
        self.client.upload_points(
            collection_name=self.collection_name,
            points=points,
            batch_size=128,
            max_retries=3,
            wait=True,
        )
        logger.info("Indexed %d dense vectors in Qdrant", len(points))

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        vector = self._validate_query(query_vector)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector.tolist(),
            limit=top_k,
            with_payload=True,
        )
        results: list[SearchResult] = []
        for point in response.points:
            payload = dict(point.payload or {})
            chunk_id = str(payload.pop("chunk_id", point.id))
            text = str(payload.pop("text", ""))
            results.append(
                SearchResult(id=chunk_id, score=float(point.score), text=text, meta=payload)
            )
        return results

    def is_ready(self) -> bool:
        return self.client.collection_exists(self.collection_name)

    @staticmethod
    def _validate_vectors(vectors: np.ndarray) -> np.ndarray:
        if not isinstance(vectors, np.ndarray):
            raise TypeError("vectors must be a numpy.ndarray")
        if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
            raise ValueError("vectors must have non-empty shape (N, D)")
        return vectors.astype(np.float32, copy=False)

    @staticmethod
    def _validate_query(query_vector: np.ndarray) -> np.ndarray:
        if not isinstance(query_vector, np.ndarray):
            raise TypeError("query_vector must be a numpy.ndarray")
        if query_vector.ndim == 2 and query_vector.shape[0] == 1:
            query_vector = query_vector[0]
        if query_vector.ndim != 1 or query_vector.size == 0:
            raise ValueError("query_vector must have shape (D,) or (1, D)")
        return query_vector.astype(np.float32, copy=False)
