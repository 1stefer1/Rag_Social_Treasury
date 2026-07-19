from __future__ import annotations

import numpy as np
import pytest
from qdrant_client import QdrantClient

from src.utils.vector_store import QdrantVectorStore, StoredChunk


@pytest.mark.integration
def test_qdrant_store_indexes_payload_and_searches_dense_vectors() -> None:
    store = QdrantVectorStore(QdrantClient(":memory:"), "test_chunks")
    chunks = [
        StoredChunk(id="benefit", text="Назначение выплаты", meta={"doc_id": "law"}),
        StoredChunk(id="transport", text="Оплата проезда", meta={"doc_id": "notice"}),
    ]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    store.recreate_collection(vector_size=2)
    store.add(vectors, chunks)
    results = store.search(np.array([0.9, 0.1], dtype=np.float32), top_k=1)

    assert store.is_ready()
    assert results[0].id == "benefit"
    assert results[0].text == "Назначение выплаты"
    assert results[0].meta["doc_id"] == "law"
