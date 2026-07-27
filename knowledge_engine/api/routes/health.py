"""Health и конфигурация."""

from __future__ import annotations

from fastapi import APIRouter

import knowledge_engine.config as cfg
from knowledge_engine.api.schemas.responses import ConfigResponse, HealthResponse
from knowledge_engine.services.gemini_stateless import is_gemini_available
from knowledge_engine.services.search_service import searxng_health

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ok, msg = searxng_health()
    return HealthResponse(status="ok", searxng_ok=ok, searxng_message=msg)


@router.get("/config", response_model=ConfigResponse)
def runtime_config() -> ConfigResponse:
    return ConfigResponse(
        graph_version=(cfg.GRAPH_VERSION or "0.4").strip(),
        gemini_model=cfg.GEMINI_MODEL,
        gemini_configured=is_gemini_available(),
        ollama_base_url=cfg.OLLAMA_BASE_URL,
        searxng_base_url=cfg.SEARXNG_BASE_URL,
        local_heavy_model=cfg.LOCAL_HEAVY_MODEL,
        local_router_model=cfg.LOCAL_ROUTER_MODEL,
    )
