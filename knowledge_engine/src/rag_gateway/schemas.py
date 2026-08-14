"""Схемы Directional RAG Gateway (Модуль 3)."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from knowledge_engine.config import RAG_DEFAULT_MAX_FACTS, RAG_DEFAULT_MIN_RELEVANCE
from knowledge_engine.src.rag_gateway.fact_text import (
    FACT_MAX_CHARS,
    truncate_fact_at_word_boundary,
)


class SearchDirection(BaseModel):
    direction_label: str = Field(min_length=2, max_length=200)
    vector_query: str = Field(min_length=4, max_length=2000)
    weight: float = Field(default=1.0, ge=0.1, le=1.5)


class DirectionalRAGQuery(BaseModel):
    """Запрос от Модуля 2 (Node Deep-Dive)."""

    target_node: str = Field(min_length=2, max_length=80)
    search_directions: list[SearchDirection] = Field(min_length=1, max_length=8)
    relevance_criteria: str = Field(min_length=10, max_length=2000)
    max_facts: int = Field(default=RAG_DEFAULT_MAX_FACTS, ge=1, le=12)
    min_relevance_threshold: float = Field(
        default=RAG_DEFAULT_MIN_RELEVANCE, ge=0.1, le=0.99
    )


class RankedMemoryFact(BaseModel):
    direction: str = Field(max_length=200)
    fact: str = Field(min_length=8, max_length=FACT_MAX_CHARS)
    relevance_score: float = Field(ge=0.0, le=1.0)

    @field_validator("fact", mode="before")
    @classmethod
    def _clamp_fact(cls, v: object) -> str:
        """Аварийный guardrail: не должен срабатывать после compress_fact_if_needed."""
        s = str(v or "").strip()
        if len(s) > FACT_MAX_CHARS:
            return truncate_fact_at_word_boundary(s, FACT_MAX_CHARS)
        return s


class DirectionalRAGResponse(BaseModel):
    target_node: str
    total_found: int = Field(ge=0)
    facts: list[RankedMemoryFact] = Field(default_factory=list)
    latency_ms: float = 0.0


class SaveUserFactRequest(BaseModel):
    fact_text: str = Field(min_length=12, max_length=4000)
    category: str = Field(default="learning_gap", max_length=200)
    node_id: str = Field(min_length=2, max_length=80)

    @field_validator("category")
    @classmethod
    def _strip_cat(cls, v: str) -> str:
        return (v or "learning_gap").strip() or "learning_gap"
