"""Модуль 3 — API Directional RAG Gateway (очередь на worker)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from knowledge_engine.api.helpers.work_enqueue import (
    enqueue_rag_gateway,
    wait_job_result,
)
from knowledge_engine.config import KE_RAG_TIMEOUT_SEC
from knowledge_engine.src.rag_gateway.schemas import (
    DirectionalRAGQuery,
    DirectionalRAGResponse,
    SaveUserFactRequest,
)
from knowledge_engine.ui.run_log import trace

router = APIRouter(prefix="/rag-gateway", tags=["rag-gateway-module-3"])


@router.post("/query", response_model=DirectionalRAGResponse)
def post_rag_query(body: DirectionalRAGQuery) -> dict[str, Any]:
    """Детерминированный поиск фактов (без LLM) — выполняется worker."""
    trace(f"API ▶ POST /rag-gateway/query (queue) | {body.target_node}")
    job_id = enqueue_rag_gateway({"op": "query", "body": body.model_dump()})
    return wait_job_result(job_id, timeout_sec=KE_RAG_TIMEOUT_SEC + 30.0)


@router.post("/facts")
def post_save_fact(body: SaveUserFactRequest) -> dict[str, Any]:
    """Индексация нового личного факта / пробела — выполняется worker."""
    trace(f"API ▶ POST /rag-gateway/facts (queue) | node={body.node_id}")
    job_id = enqueue_rag_gateway({"op": "facts", "body": body.model_dump()})
    return wait_job_result(job_id, timeout_sec=KE_RAG_TIMEOUT_SEC + 30.0)


@router.get("/memory-status")
def get_rag_memory_status() -> dict[str, Any]:
    from knowledge_engine.src.memory.light_rag import count_light_rag_rows_sync

    rows = count_light_rag_rows_sync()
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
