from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient

from src.settings.config import Settings, get_settings
from src.utils.embedder import Embedder
from src.utils.generator import LLM
from src.utils.rag_pipeline import VanillaRAG
from src.utils.retriever import Retriever
from src.utils.sparse_store import ElasticsearchSparseStore
from src.utils.vector_store import QdrantVectorStore


@dataclass
class RAGRuntime:
    embedder: Embedder
    retriever: Retriever
    llm: LLM
    rag: VanillaRAG
    top_k: int
    qdrant_collection: str
    es_index: str


def create_rag_runtime(config: Settings | None = None) -> RAGRuntime:
    config = config or get_settings()
    embedder = Embedder()
    qdrant_client = QdrantClient(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key_value,
        timeout=config.qdrant_timeout,
    )
    elastic_client = Elasticsearch(config.es_url, request_timeout=30)
    retriever = Retriever(
        embedder,
        QdrantVectorStore(qdrant_client, config.qdrant_collection),
        ElasticsearchSparseStore(elastic_client, config.es_index),
        top_k=config.top_k,
        dense_candidates_k=config.dense_candidates_k,
        sparse_candidates_k=config.sparse_candidates_k,
        rrf_k=config.rrf_k,
        use_reranker=config.use_reranker,
        reranker_model=config.reranker_model,
        reranker_candidates_k=config.reranker_candidates_k,
    )
    llm = LLM(config=config)
    rag = VanillaRAG(
        retriever,
        llm,
        default_top_k=config.top_k,
        max_context_chars=config.max_context_chars,
    )

    return RAGRuntime(
        embedder=embedder,
        retriever=retriever,
        llm=llm,
        rag=rag,
        top_k=config.top_k,
        qdrant_collection=config.qdrant_collection,
        es_index=config.es_index,
    )


@lru_cache(maxsize=1)
def get_rag_runtime() -> RAGRuntime:
    return create_rag_runtime()
