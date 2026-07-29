"""Модуль 3 — API Directional RAG Gateway (отладка / прямой вызов)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from knowledge_engine.src.rag_gateway.gateway import (
    query_directional_rag,
    save_user_fact_request,
)
from knowledge_engine.src.rag_gateway.schemas import (
    DirectionalRAGQuery,
    DirectionalRAGResponse,
    SaveUserFactRequest,
)
from knowledge_engine.ui.run_log import trace

router = APIRouter(prefix="/rag-gateway", tags=["rag-gateway-module-3"])


@router.post("/query", response_model=DirectionalRAGResponse)
async def post_rag_query(body: DirectionalRAGQuery) -> dict[str, Any]:
    """Детерминированный поиск фактов (без LLM)."""
    trace(f"API ▶ POST /rag-gateway/query | {body.target_node}")
    result = await query_directional_rag(body)
    return result.model_dump()


@router.post("/facts")
async def post_save_fact(body: SaveUserFactRequest) -> dict[str, Any]:
    """Индексация нового личного факта / пробела."""
    trace(f"API ▶ POST /rag-gateway/facts | node={body.node_id}")
    n = await save_user_fact_request(body)
    return {"indexed": n, "node_id": body.node_id}


@router.get("/memory-status")
async def get_rag_memory_status() -> dict[str, Any]:
    from knowledge_engine.src.memory.light_rag import LightRAG

    rag = LightRAG()
    rows = await rag.count_indexed_rows()
    connected = rows > 0
    label = (
        "Персональный профиль подключен"
        if connected
        else "RAG-память пуста — загрузите профиль или пройдите ноды"
    )
    return {
        "connected": connected,
        "indexed_rows": rows,
        "label": label,
    }
