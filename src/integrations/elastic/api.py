from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional

import uvicorn
from elasticsearch import Elasticsearch, helpers
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MetadataFilters(BaseModel):
    source_file: Optional[str] = None
    clause: Optional[str] = None
    appendix: Optional[str] = None
    section: Optional[str] = None
    doc_type: Optional[str] = None
    doc_id: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)
    filters: Optional[MetadataFilters] = None


class SearchHit(BaseModel):
    chunk_id: str
    doc_id: str
    source_file: str
    doc_type: str
    clause: str
    appendix: str
    section: str
    score: float
    snippet: str
    text: str


class SearchResponse(BaseModel):
    query: str
    took_ms: float
    total_hits: int
    results: List[SearchHit]


class UpsertChunk(BaseModel):
    chunk_id: str
    doc_id: str
    source_file: str
    text: str
    doc_type: Optional[str] = None
    clause: Optional[str] = None
    appendix: Optional[str] = None
    section: Optional[str] = None
    language: Optional[str] = "ru"


class UpsertRequest(BaseModel):
    chunks: List[UpsertChunk]


class RebuildResponse(BaseModel):
    indexed_chunks: int
    files_scanned: int
    index_name: str


class DeleteResponse(BaseModel):
    deleted: int


class HealthResponse(BaseModel):
    status: str
    elasticsearch: str
    index_name: str


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


ES_URL = _env("ES_URL", "http://elasticsearch:9200")
ES_INDEX = _env("ES_INDEX", "kb_chunks")
CHUNKS_DIR = Path(_env("CHUNKS_DIR", "/app/data/chunks_json"))


@dataclass
class ESRuntime:
    client: Elasticsearch


runtime: Optional[ESRuntime] = None


def _make_client() -> Elasticsearch:
    return Elasticsearch(ES_URL, request_timeout=60)


def _index_mapping() -> Dict[str, Any]:
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
                "appendix": {"type": "keyword"},
                "section": {"type": "keyword"},
                "language": {"type": "keyword"},
                "text": {"type": "text", "analyzer": "ru_text"},
                "updated_at": {"type": "date"},
            }
        },
    }


def _ensure_index(client: Elasticsearch) -> None:
    if client.indices.exists(index=ES_INDEX):
        return
    client.indices.create(index=ES_INDEX, body=_index_mapping())
    logger.info("Created Elasticsearch index: %s", ES_INDEX)


def _coerce_chunk(ch: Dict[str, Any], i: int) -> Dict[str, Any]:
    text = (ch.get("text") or "").strip()
    if not text:
        return {}
    doc_id = str(ch.get("doc_id") or "unknown_doc")
    source_file = str(ch.get("source_file") or "unknown_file")
    clause = str(ch.get("clause") or "no_clause")
    clause_span = str(ch.get("clause_span") or "")
    chunk_id = f"{doc_id}|{source_file}|{clause_span or clause}|{i}"
    ts = datetime.now(timezone.utc).isoformat()
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "source_file": source_file,
        "doc_type": str(ch.get("doc_type") or ""),
        "clause": clause,
        "appendix": str(ch.get("appendix") or ""),
        "section": str(ch.get("section") or ""),
        "language": str(ch.get("language") or "ru"),
        "text": text,
        "updated_at": ts,
    }


def _load_chunks_from_dir(chunks_dir: Path) -> tuple[list[dict[str, Any]], int]:
    paths = sorted(glob.glob(str(chunks_dir / "*.json")))
    out: List[Dict[str, Any]] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for i, ch in enumerate(data):
            doc = _coerce_chunk(ch, i)
            if doc:
                out.append(doc)
    return out, len(paths)


def _bulk_upsert(client: Elasticsearch, docs: List[Dict[str, Any]]) -> int:
    if not docs:
        return 0
    actions = [
        {
            "_op_type": "index",
            "_index": ES_INDEX,
            "_id": d["chunk_id"],
            "_source": d,
        }
        for d in docs
    ]
    ok, _ = helpers.bulk(client, actions, refresh=True)
    return int(ok)


def _to_term_filters(filters: MetadataFilters | None) -> list[dict[str, Any]]:
    if filters is None:
        return []
    values = filters.model_dump(exclude_none=True)
    clauses: List[Dict[str, Any]] = []
    for k, v in values.items():
        val = str(v).strip()
        if not val:
            continue
        if k == "source_file":
            clauses.append({"match_phrase": {"source_file": val}})
        else:
            clauses.append({"term": {k: val}})
    return clauses


app = FastAPI(title="Elasticsearch Search Service", version="0.1.0")


@app.on_event("startup")
def startup() -> None:
    global runtime
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )
    client = _make_client()
    if not client.ping():
        raise RuntimeError(f"Elasticsearch is not reachable: {ES_URL}")
    _ensure_index(client)
    runtime = ESRuntime(client=client)
    logger.info("Elastic service started. ES=%s index=%s", ES_URL, ES_INDEX)


@app.get("/es/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if runtime is None or not runtime.client.ping():
        raise HTTPException(status_code=503, detail="Elasticsearch unavailable")
    return HealthResponse(status="ok", elasticsearch=ES_URL, index_name=ES_INDEX)


@app.post("/es/index/rebuild", response_model=RebuildResponse)
def rebuild_index() -> RebuildResponse:
    if runtime is None:
        raise HTTPException(status_code=503, detail="Service runtime is not ready")
    if not CHUNKS_DIR.exists():
        raise HTTPException(
            status_code=400, detail=f"Chunks directory does not exist: {CHUNKS_DIR}"
        )

    started = perf_counter()
    docs, files_count = _load_chunks_from_dir(CHUNKS_DIR)
    runtime.client.indices.delete(index=ES_INDEX, ignore_unavailable=True)
    _ensure_index(runtime.client)
    indexed = _bulk_upsert(runtime.client, docs)
    logger.info(
        "Rebuilt index %s: files=%d chunks=%d took=%.2fs",
        ES_INDEX,
        files_count,
        indexed,
        perf_counter() - started,
    )
    return RebuildResponse(
        indexed_chunks=indexed, files_scanned=files_count, index_name=ES_INDEX
    )


@app.post("/es/index/upsert")
def upsert_chunks(request: UpsertRequest) -> Dict[str, Any]:
    if runtime is None:
        raise HTTPException(status_code=503, detail="Service runtime is not ready")
    ts = datetime.now(timezone.utc).isoformat()
    docs = [
        {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "source_file": c.source_file,
            "doc_type": c.doc_type or "",
            "clause": c.clause or "",
            "appendix": c.appendix or "",
            "section": c.section or "",
            "language": c.language or "ru",
            "text": c.text,
            "updated_at": ts,
        }
        for c in request.chunks
        if c.text.strip()
    ]
    indexed = _bulk_upsert(runtime.client, docs)
    return {"indexed_chunks": indexed}


@app.delete("/es/index/doc/{doc_id}", response_model=DeleteResponse)
def delete_document(doc_id: str) -> DeleteResponse:
    if runtime is None:
        raise HTTPException(status_code=503, detail="Service runtime is not ready")
    resp = runtime.client.delete_by_query(
        index=ES_INDEX,
        query={"term": {"doc_id": doc_id}},
        refresh=True,
        conflicts="proceed",
    )
    return DeleteResponse(deleted=int(resp.get("deleted", 0)))


@app.post("/es/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    if runtime is None:
        raise HTTPException(status_code=503, detail="Service runtime is not ready")

    started = perf_counter()
    query: Dict[str, Any] = {
        "bool": {
            "must": [
                {
                    "multi_match": {
                        "query": request.query,
                        "fields": ["text^3", "source_file^2", "section", "clause"],
                        "type": "best_fields",
                        "operator": "or",
                    }
                }
            ],
            "filter": _to_term_filters(request.filters),
        }
    }

    resp = runtime.client.search(
        index=ES_INDEX,
        query=query,
        size=request.top_k,
        highlight={
            "pre_tags": ["<mark>"],
            "post_tags": ["</mark>"],
            "fields": {"text": {"fragment_size": 220, "number_of_fragments": 1}},
        },
    )
    hits = resp.get("hits", {}).get("hits", [])
    total = resp.get("hits", {}).get("total", {})
    total_hits = (
        int(total.get("value", len(hits))) if isinstance(total, dict) else len(hits)
    )

    results: List[SearchHit] = []
    for h in hits:
        src = h.get("_source", {})
        highlight = h.get("highlight", {}).get("text", [])
        snippet = highlight[0] if highlight else (src.get("text") or "")[:260]
        results.append(
            SearchHit(
                chunk_id=str(src.get("chunk_id", "")),
                doc_id=str(src.get("doc_id", "")),
                source_file=str(src.get("source_file", "")),
                doc_type=str(src.get("doc_type", "")),
                clause=str(src.get("clause", "")),
                appendix=str(src.get("appendix", "")),
                section=str(src.get("section", "")),
                score=float(h.get("_score") or 0.0),
                snippet=snippet,
                text=str(src.get("text", "")),
            )
        )

    return SearchResponse(
        query=request.query,
        took_ms=(perf_counter() - started) * 1000.0,
        total_hits=total_hits,
        results=results,
    )


def main() -> None:
    uvicorn.run(
        "src.integrations.elastic.api:app", host="0.0.0.0", port=8010, reload=False
    )


if __name__ == "__main__":
    main()
