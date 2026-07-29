"""Модуль 3 — Directional RAG Gateway (Background Memory Engine)."""

from knowledge_engine.src.rag_gateway.gateway import (
    query_directional_rag,
    query_rag_gateway,
    save_user_fact,
    save_user_fact_request,
)
from knowledge_engine.src.rag_gateway.schemas import (
    DirectionalRAGQuery,
    DirectionalRAGResponse,
    RankedMemoryFact,
    SaveUserFactRequest,
    SearchDirection,
)

__all__ = [
    "DirectionalRAGQuery",
    "DirectionalRAGResponse",
    "RankedMemoryFact",
    "SaveUserFactRequest",
    "SearchDirection",
    "query_directional_rag",
    "query_rag_gateway",
    "save_user_fact",
    "save_user_fact_request",
]
