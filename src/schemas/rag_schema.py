from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class MetadataFilters(BaseModel):
    source_file: Optional[str] = None
    clause: Optional[str] = None
    appendix: Optional[str] = None
    section: Optional[str] = None
    doc_type: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"source_file": "Закон г. Москвы"},
                {"clause": "9"},
            ]
        }
    )


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    filters: Optional[MetadataFilters] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "единовременная денежная выплата",
                    "top_k": 5,
                    "filters": None,
                },
                {
                    "query": "единовременная денежная выплата",
                    "top_k": 5,
                    "filters": {"source_file": "Закон г. Москвы"},
                },
            ]
        }
    )


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=30)
    filters: Optional[MetadataFilters] = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=700, ge=1, le=4096)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "question": "От чего зависит размер выплаты за конкретный учебный год?",
                    "top_k": 5,
                    "filters": None,
                    "temperature": 0,
                    "max_tokens": 700,
                },
                {
                    "question": "От чего зависит размер выплаты за конкретный учебный год?",
                    "top_k": 5,
                    "filters": {"source_file": "Закон г. Москвы"},
                    "temperature": 0,
                    "max_tokens": 700,
                },
            ]
        }
    )


class RAGSource(BaseModel):
    id: str
    score: float
    text: str
    meta: Dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    results: List[RAGSource]
    latency_ms: float


class AskResponse(BaseModel):
    question: str
    answer: str
    no_answer: bool
    sources: List[RAGSource]
    used_top_k: int
    latency_ms: float


class ConfigResponse(BaseModel):
    index_dir: str
    index_name: str
    top_k: int
    llm_provider: str
    llm_model: str
    llm_base_url: str


class HealthResponse(BaseModel):
    status: str
