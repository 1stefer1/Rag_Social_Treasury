from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Sequence

from elasticsearch import Elasticsearch, helpers

from src.utils.vector_store import SearchResult, StoredChunk

logger = logging.getLogger(__name__)


class ElasticsearchSparseStore:
    """BM25 retrieval and chunk payload storage backed by Elasticsearch."""

    def __init__(self, client: Elasticsearch, index_name: str) -> None:
        self.client = client
        self.index_name = index_name

    def recreate_index(self) -> None:
        self.client.indices.delete(index=self.index_name, ignore_unavailable=True)
        self.client.indices.create(index=self.index_name, body=self._index_mapping())
        logger.info("Created Elasticsearch index %s", self.index_name)

    def add(self, chunks: Sequence[StoredChunk]) -> None:
        updated_at = datetime.now(timezone.utc).isoformat()
        actions = [
            {
                "_op_type": "index",
                "_index": self.index_name,
                "_id": chunk.id,
                "_source": {
                    "chunk_id": chunk.id,
                    "text": chunk.text,
                    **chunk.meta,
                    "updated_at": updated_at,
                },
            }
            for chunk in chunks
        ]
        indexed, errors = helpers.bulk(self.client, actions, refresh=True, raise_on_error=False)
        failed = len(errors) if isinstance(errors, list) else errors
        if failed:
            raise RuntimeError(f"Elasticsearch rejected {failed} chunks")
        logger.info("Indexed %d sparse documents in Elasticsearch", indexed)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        response = self.client.search(
            index=self.index_name,
            query={
                "multi_match": {
                    "query": query,
                    "fields": ["text^3", "source_file^2", "section", "clause"],
                    "type": "best_fields",
                    "operator": "or",
                }
            },
            size=top_k,
        )
        results: list[SearchResult] = []
        for hit in response.get("hits", {}).get("hits", []):
            source = dict(hit.get("_source", {}))
            chunk_id = str(source.pop("chunk_id", hit.get("_id", "")))
            text = str(source.pop("text", ""))
            source.pop("updated_at", None)
            results.append(
                SearchResult(
                    id=chunk_id,
                    score=float(hit.get("_score") or 0.0),
                    text=text,
                    meta=source,
                )
            )
        return results

    def is_ready(self) -> bool:
        return bool(self.client.ping() and self.client.indices.exists(index=self.index_name))

    @staticmethod
    def _index_mapping() -> dict[str, Any]:
        return {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "ru_text": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase"],
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "source_file": {
                        "type": "text",
                        "analyzer": "ru_text",
                        "fields": {"keyword": {"type": "keyword", "ignore_above": 1024}},
                    },
                    "doc_type": {"type": "keyword"},
                    "clause": {"type": "keyword"},
                    "clause_span": {"type": "keyword"},
                    "appendix": {"type": "keyword"},
                    "section": {"type": "keyword"},
                    "language": {"type": "keyword"},
                    "text": {"type": "text", "analyzer": "ru_text"},
                    "updated_at": {"type": "date"},
                }
            },
        }
