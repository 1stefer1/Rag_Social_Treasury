from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError as e:
    raise ImportError(
        "faiss не установлен. Добавь зависимость 'faiss-cpu' в pyproject.toml и сделай uv sync."
    ) from e


@dataclass
class StoredChunk:
    """
    То, что мы храним параллельно с FAISS (потому что FAISS хранит только вектора).
    """

    id: str
    text: str
    meta: Dict[str, Any]


@dataclass
class SearchResult:
    id: str
    score: float
    text: str
    meta: Dict[str, Any]


class VectorStore:
    """
    Векторное хранилище на базе FAISS (IndexFlatIP) + JSON store.

    Требование:
    - вектора должны быть float32 shape (N, D)
    - для cosine similarity вектора должны быть L2-нормированы заранее
      (Embedder должен выдавать normalize_embeddings=True).
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.store: List[StoredChunk] = []

    # -------------------------
    # Public API
    # -------------------------
    def add(self, vectors: np.ndarray, chunks: Sequence[StoredChunk]) -> None:
        """
        Добавляет пачку векторов и соответствующие чанки в store.
        Args:
            vectors: np.ndarray (N, D) float32
            chunks: список StoredChunk длины N
        """
        vectors = self._validate_vectors(vectors)

        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"Несовпадение размеров: vectors N={vectors.shape[0]}, chunks={len(chunks)}"
            )
        start_size = len(self.store)
        self.index.add(vectors)
        self.store.extend(chunks)
        logger.info(
            "Добавлено %d векторов. Store: %d -> %d", vectors.shape[0], start_size, len(self.store)
        )

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        """
        Ищет top_k ближайших.

        Args:
            query_vector: np.ndarray (D,) или (1, D), float32, L2-нормированный
            top_k: количество результатов

        Returns:
            список SearchResult (id, score, text, meta)
        """
        q = self._validate_query(query_vector)

        if self.index.ntotal == 0:
            return []

        k = min(top_k, self.index.ntotal)
        scores, idxs = self.index.search(q, k)  # scores: (1,k), idxs: (1,k)

        results: List[SearchResult] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0:
                continue
            item = self.store[idx]
            results.append(
                SearchResult(
                    id=item.id,
                    score=float(score),
                    text=item.text,
                    meta=item.meta,
                )
            )
        return results

    def save(self, dir_path: Path, name: str = "kb") -> Tuple[Path, Path]:
        """
        Сохраняет индекс и store в директорию.
        """
        dir_path.mkdir(parents=True, exist_ok=True)
        index_path = dir_path / f"{name}.faiss"
        store_path = dir_path / f"{name}.store.json"

        faiss.write_index(self.index, str(index_path))
        with store_path.open("w", encoding="utf-8") as f:
            json.dump(
                [{"id": c.id, "text": c.text, "meta": c.meta} for c in self.store],
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info("Сохранено: %s и %s", index_path, store_path)
        return index_path, store_path

    @classmethod
    def load(cls, dir_path: Path, name: str = "kb") -> "VectorStore":
        """
        Загружает индекс и store из директории.
        """
        index_path = dir_path / f"{name}.faiss"
        store_path = dir_path / f"{name}.store.json"

        if not index_path.exists():
            raise FileNotFoundError(f"Не найден FAISS индекс: {index_path}")
        if not store_path.exists():
            raise FileNotFoundError(f"Не найден store: {store_path}")

        index = faiss.read_index(str(index_path))
        dim = index.d

        vs = cls(dim=dim)
        vs.index = index

        with store_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        store: List[StoredChunk] = []
        for item in data:
            store.append(
                StoredChunk(
                    id=item["id"],
                    text=item["text"],
                    meta=item.get("meta", {}),
                )
            )

        # Важно: порядок store должен соответствовать порядку векторов в индексе.
        if len(store) != vs.index.ntotal:
            logger.warning(
                "Несоответствие store (%d) и индекса (%d). Это может привести к неверным соответствиям.",
                len(store),
                vs.index.ntotal,
            )

        vs.store = store
        logger.info("Загружено: %d векторов", vs.index.ntotal)
        return vs

    # -------------------------
    # Validation
    # -------------------------

    def _validate_vectors(self, vectors: np.ndarray) -> np.ndarray:
        if not isinstance(vectors, np.ndarray):
            raise TypeError("vectors должны быть numpy.ndarray")
        if vectors.ndim != 2:
            raise ValueError("vectors должны иметь форму (N, D)")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Ожидалась размерность D={self.dim}, получено {vectors.shape[1]}")
        if vectors.dtype != np.float32:
            vectors = vectors.astype("float32")
        return vectors

    def _validate_query(self, query_vector: np.ndarray) -> np.ndarray:
        if not isinstance(query_vector, np.ndarray):
            raise TypeError("query_vector должен быть numpy.ndarray")

        if query_vector.ndim == 1:
            if query_vector.shape[0] != self.dim:
                raise ValueError(
                    f"Ожидалась форма (D,), D={self.dim}, получено {query_vector.shape}"
                )
            q = query_vector.reshape(1, -1)
        elif query_vector.ndim == 2:
            if query_vector.shape != (1, self.dim):
                raise ValueError(f"Ожидалась форма (1, D), получено {query_vector.shape}")
            q = query_vector
        else:
            raise ValueError("query_vector должен иметь форму (D,) или (1, D)")

        if q.dtype != np.float32:
            q = q.astype("float32")

        return q
