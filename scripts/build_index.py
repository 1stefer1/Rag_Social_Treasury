import asyncio
import logging
import sys
from pathlib import Path

# allow running as: python scripts/build_index.py
sys.path.append(str(Path(__file__).resolve().parents[1]))

from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient

from src.settings.config import get_settings
from src.utils.embedder import Embedder
from src.utils.retriever import Retriever
from src.utils.sparse_store import ElasticsearchSparseStore
from src.utils.vector_store import QdrantVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)


async def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    chunks_dir = base_dir / "data" / "chunks_json"
    settings = get_settings()
    embedder = Embedder()
    qdrant_client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key_value,
        timeout=settings.qdrant_timeout,
    )
    retriever = Retriever(
        embedder,
        QdrantVectorStore(qdrant_client, settings.qdrant_collection),
        ElasticsearchSparseStore(
            Elasticsearch(settings.es_url, request_timeout=30), settings.es_index
        ),
        top_k=settings.top_k,
        dense_candidates_k=settings.dense_candidates_k,
        sparse_candidates_k=settings.sparse_candidates_k,
        rrf_k=settings.rrf_k,
    )

    await retriever.abuild_from_json_files(chunks_dir)


if __name__ == "__main__":
    asyncio.run(main())
