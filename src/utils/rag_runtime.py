from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.settings.config import Settings, get_settings
from src.utils.embedder import Embedder
from src.utils.generator import LLM
from src.utils.rag_pipeline import VanillaRAG
from src.utils.retriever import Retriever


@dataclass
class RAGRuntime:
    embedder: Embedder
    retriever: Retriever
    llm: LLM
    rag: VanillaRAG
    top_k: int
    index_dir: Path
    index_name: str


def create_rag_runtime(config: Settings | None = None) -> RAGRuntime:
    config = config or get_settings()
    index_dir = config.index_dir.resolve()
    embedder = Embedder()
    retriever = Retriever(
        embedder,
        top_k=config.top_k,
        use_bm25=config.use_bm25,
        bm25_weight=config.bm25_weight,
        bm25_candidates_k=config.bm25_candidates_k,
        use_reranker=config.use_reranker,
        reranker_model=config.reranker_model,
        reranker_candidates_k=config.reranker_candidates_k,
    )
    retriever.load(index_dir, name=config.index_name)

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
        index_dir=index_dir,
        index_name=config.index_name,
    )


@lru_cache(maxsize=1)
def get_rag_runtime() -> RAGRuntime:
    return create_rag_runtime()
