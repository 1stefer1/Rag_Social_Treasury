from __future__ import annotations

import numpy as np
import pytest

from src.utils.retriever import Retriever, reciprocal_rank_fusion
from src.utils.vector_store import SearchResult


def _result(chunk_id: str, score: float) -> SearchResult:
    return SearchResult(id=chunk_id, score=score, text=f"text-{chunk_id}", meta={})


@pytest.mark.unit
def test_rrf_rewards_chunks_returned_by_both_backends() -> None:
    dense = [_result("dense-only", 0.99), _result("shared", 0.80)]
    sparse = [_result("shared", 12.0), _result("sparse-only", 8.0)]

    fused = reciprocal_rank_fusion(dense, sparse, rrf_k=60)

    assert [item.id for item in fused] == ["shared", "dense-only", "sparse-only"]
    assert fused[0].score > fused[1].score


@pytest.mark.unit
def test_rrf_does_not_compare_backend_specific_raw_scores() -> None:
    dense = [_result("dense", 0.1)]
    sparse = [_result("sparse", 10_000.0)]

    fused = reciprocal_rank_fusion(dense, sparse, rrf_k=60)

    assert fused[0].score == fused[1].score


@pytest.mark.unit
def test_rrf_rejects_invalid_rank_constant() -> None:
    with pytest.raises(ValueError, match="rrf_k must be positive"):
        reciprocal_rank_fusion([], [], rrf_k=0)


class FakeEmbedder:
    async def aembed(self, text: str, *, input_type: str) -> np.ndarray:
        assert text == "query"
        assert input_type == "query"
        return np.array([1.0, 0.0], dtype=np.float32)


class FakeDenseStore:
    def search(self, vector: np.ndarray, top_k: int) -> list[SearchResult]:
        assert vector.tolist() == [1.0, 0.0]
        assert top_k == 20
        return [_result("shared", 0.8), _result("dense", 0.7)]


class FakeSparseStore:
    def search(self, query: str, top_k: int) -> list[SearchResult]:
        assert query == "query"
        assert top_k == 30
        return [_result("sparse", 9.0), _result("shared", 8.0)]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retriever_fuses_qdrant_and_elasticsearch_results() -> None:
    retriever = Retriever(
        FakeEmbedder(),  # type: ignore[arg-type]
        FakeDenseStore(),  # type: ignore[arg-type]
        FakeSparseStore(),  # type: ignore[arg-type]
        dense_candidates_k=20,
        sparse_candidates_k=30,
        use_reranker=False,
    )

    results = await retriever.aretrieve("query", top_k=3)

    assert [result.id for result in results] == ["shared", "sparse", "dense"]
