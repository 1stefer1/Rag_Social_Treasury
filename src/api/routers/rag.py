from __future__ import annotations

import logging
import time
from typing import Any, Dict

import httpx
from fastapi import APIRouter, HTTPException

from src.schemas.rag_schema import (
    AskRequest,
    AskResponse,
    ConfigResponse,
    HealthResponse,
    MetadataFilters,
    RAGSource,
    SearchRequest,
    SearchResponse,
)
from src.utils.output_guard import build_russian_rewrite_prompt, looks_non_russian

router = APIRouter()
logger = logging.getLogger(__name__)

NO_ANSWER_TEXT = "В предоставленном контексте нет информации для ответа."
IGNORED_FILTER_VALUES = {"", "string", "null", "none", "undefined"}


def _get_runtime() -> Any:
    """Import the ML runtime lazily so health checks do not load models."""
    from src.utils.rag_runtime import get_rag_runtime

    return get_rag_runtime()


def _is_effective_filter_value(value: Any) -> bool:
    return str(value or "").strip().lower() not in IGNORED_FILTER_VALUES


def _has_effective_filters(filters: MetadataFilters | None) -> bool:
    if filters is None:
        return False
    return any(
        _is_effective_filter_value(value)
        for value in filters.model_dump(exclude_none=True).values()
    )


def _matches_metadata(meta: Dict[str, Any], filters: MetadataFilters | None) -> bool:
    if filters is None:
        return True
    for key, expected in filters.model_dump(exclude_none=True).items():
        expected_text = str(expected).strip().lower()
        if not _is_effective_filter_value(expected_text):
            continue
        actual = str(meta.get(key) or "").lower()
        if expected_text not in actual:
            return False
    return True


def _is_no_answer(answer: str) -> bool:
    return NO_ANSWER_TEXT in (answer or "").strip().strip('"').strip()


def _to_source(chunk) -> RAGSource:
    return RAGSource(
        id=chunk.id,
        score=float(chunk.score),
        text=chunk.text,
        meta=chunk.meta or {},
    )


async def _retrieve_filtered(query: str, top_k: int, filters: MetadataFilters | None):
    runtime = _get_runtime()
    candidate_k = top_k
    if _has_effective_filters(filters):
        candidate_k = max(top_k * 6, 20)
    chunks = await runtime.retriever.aretrieve(query, top_k=candidate_k)
    chunks = [c for c in chunks if _matches_metadata(c.meta or {}, filters)]
    return chunks[:top_k]


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/ready", response_model=HealthResponse)
async def ready() -> HealthResponse:
    try:
        runtime = _get_runtime()
        if not runtime.retriever.vector_store.is_ready():
            raise RuntimeError("Qdrant collection is unavailable")
        if not runtime.retriever.sparse_store.is_ready():
            raise RuntimeError("Elasticsearch index is unavailable")
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return HealthResponse(status="ready")


@router.get("/config", response_model=ConfigResponse)
async def config() -> ConfigResponse:
    runtime = _get_runtime()
    return ConfigResponse(
        qdrant_collection=runtime.qdrant_collection,
        es_index=runtime.es_index,
        top_k=runtime.top_k,
        llm_provider=runtime.llm.provider,
        llm_model=runtime.llm.model,
        llm_base_url=runtime.llm.base_url,
    )


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    started_at = time.perf_counter()
    chunks = await _retrieve_filtered(request.query, request.top_k, request.filters)
    return SearchResponse(
        query=request.query,
        results=[_to_source(c) for c in chunks],
        latency_ms=(time.perf_counter() - started_at) * 1000.0,
    )


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    started_at = time.perf_counter()
    runtime = _get_runtime()
    chunks = await _retrieve_filtered(request.question, request.top_k, request.filters)

    if not chunks:
        return AskResponse(
            question=request.question,
            answer=f"{NO_ANSWER_TEXT}\n\nПереформулируйте запрос или обратитесь напрямую в техподдержку.",
            no_answer=True,
            sources=[],
            used_top_k=0,
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
        )

    prompt, used_chunks = runtime.rag._build_prompt(request.question, chunks)
    try:
        answer = await runtime.llm.arun(
            prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    except httpx.HTTPError as e:
        logger.exception("LLM generation failed")
        raise HTTPException(
            status_code=503,
            detail="LLM backend is temporarily unavailable. Retry in a few moments.",
        ) from e
    answer = (answer or "").strip()

    if looks_non_russian(answer):
        rewrite_prompt = build_russian_rewrite_prompt(answer)
        try:
            answer_ru = await runtime.llm.arun(
                rewrite_prompt,
                temperature=0.0,
                max_tokens=min(request.max_tokens, 500),
            )
            answer = (answer_ru or "").strip()
        except httpx.HTTPError:
            logger.warning("Russian rewrite step failed, keeping original answer")

    if _is_no_answer(answer):
        return AskResponse(
            question=request.question,
            answer=f"{NO_ANSWER_TEXT}\n\nПереформулируйте запрос или обратитесь напрямую в техподдержку.",
            no_answer=True,
            sources=[],
            used_top_k=0,
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
        )

    return AskResponse(
        question=request.question,
        answer=answer,
        no_answer=False,
        sources=[_to_source(c) for c in used_chunks],
        used_top_k=len(used_chunks),
        latency_ms=(time.perf_counter() - started_at) * 1000.0,
    )
