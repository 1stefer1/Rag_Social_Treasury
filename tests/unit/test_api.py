from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

import src.api.routers.rag as rag_router
from main import app


class FakeRetriever:
    async def aretrieve(self, query: str, *, top_k: int) -> list[Any]:
        del query, top_k
        return [
            SimpleNamespace(
                id="chunk-1",
                score=0.91,
                text="Право на выплату установлено законом.",
                meta={"source_file": "law.docx", "doc_type": "law"},
            ),
            SimpleNamespace(
                id="chunk-2",
                score=0.75,
                text="Нерелевантный документ.",
                meta={"source_file": "notice.docx", "doc_type": "notice"},
            ),
        ]


@pytest.mark.unit
def test_health_endpoint_does_not_initialize_ml_runtime() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.unit
def test_search_applies_metadata_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_runtime = SimpleNamespace(retriever=FakeRetriever())
    monkeypatch.setattr(rag_router, "_get_runtime", lambda: fake_runtime)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/search",
            json={
                "query": "выплата",
                "top_k": 5,
                "filters": {"doc_type": "law"},
            },
        )

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["results"]] == ["chunk-1"]
