from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.utils.embedder import Embedder
from src.utils.generator import LLM
from src.utils.rag_pipeline import VanillaRAG
from src.utils.retriever import Retriever

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_DIR = BASE_DIR / "data" / "faiss_index"
DEFAULT_INDEX_NAME = "moscow_kb"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no")


@dataclass
class RAGRuntime:
    embedder: Embedder
    retriever: Retriever
    llm: LLM
    rag: VanillaRAG
    top_k: int
    index_dir: Path
    index_name: str


def create_rag_runtime() -> RAGRuntime:
    top_k = int(os.environ.get("TOP_K", "5"))
    use_bm25 = _env_flag("USE_BM25", False)
    use_reranker = _env_flag("USE_RERANKER", True)
    reranker_model = os.environ.get(
        "RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    )
    reranker_candidates_k = int(os.environ.get("RERANKER_CANDIDATES_K", "50"))
    bm25_weight = float(os.environ.get("BM25_WEIGHT", "0.25"))
    bm25_candidates_k = int(os.environ.get("BM25_CANDIDATES_K", "50"))
    max_context_chars = int(os.environ.get("MAX_CONTEXT_CHARS", "14000"))
    index_dir = Path(os.environ.get("INDEX_DIR", str(DEFAULT_INDEX_DIR))).resolve()
    index_name = os.environ.get("INDEX_NAME", DEFAULT_INDEX_NAME)

    embedder = Embedder()
    retriever = Retriever(
        embedder,
        top_k=top_k,
        use_bm25=use_bm25,
        bm25_weight=bm25_weight,
        bm25_candidates_k=bm25_candidates_k,
        use_reranker=use_reranker,
        reranker_model=reranker_model,
        reranker_candidates_k=reranker_candidates_k,
    )
    retriever.load(index_dir, name=index_name)

    llm = LLM()
    rag = VanillaRAG(retriever, llm, default_top_k=top_k, max_context_chars=max_context_chars)

    return RAGRuntime(
        embedder=embedder,
        retriever=retriever,
        llm=llm,
        rag=rag,
        top_k=top_k,
        index_dir=index_dir,
        index_name=index_name,
    )
