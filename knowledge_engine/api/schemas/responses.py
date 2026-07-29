"""Pydantic модели ответов API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from knowledge_engine.schemas import AnalysisReport


class HealthResponse(BaseModel):
    status: str = "ok"
    searxng_ok: bool
    searxng_message: str
    worker_ok: bool = False
    redis_ok: bool = False


class ConfigResponse(BaseModel):
    graph_version: str
    gemini_model: str
    gemini_configured: bool
    ollama_base_url: str
    searxng_base_url: str
    local_heavy_model: str
    local_router_model: str


class AnalysisJobResponse(BaseModel):
    id: str
    status: str
    problem: str
    constraints: str
    matrix_only: bool
    created_at: datetime
    updated_at: datetime
    report: Optional[dict[str, Any]] = None
    unraveled_details: Optional[str] = None
    selected_option_id: Optional[int] = None
    error: Optional[str] = None
    log_path: Optional[str] = None
    clarify_question: Optional[str] = None

    @staticmethod
    def from_job(job) -> "AnalysisJobResponse":
        return AnalysisJobResponse(
            id=job.id,
            status=(
                job.status.value if hasattr(job.status, "value") else str(job.status)
            ),
            problem=job.problem,
            constraints=job.constraints,
            matrix_only=job.matrix_only,
            created_at=job.created_at,
            updated_at=job.updated_at,
            report=job.report,
            unraveled_details=job.unraveled_details,
            selected_option_id=job.selected_option_id,
            error=job.error,
            log_path=job.log_path,
            clarify_question=job.clarify_question,
        )


class AnalyzeCreatedResponse(BaseModel):
    job: AnalysisJobResponse
    message: str = ""


class AnalysisJobWaitResponse(BaseModel):
    """Long polling: текущий job + мета ожидания."""

    job: AnalysisJobResponse
    done: bool = Field(description="Достигнут target или failed/clarify")
    timed_out: bool = Field(description="timeout_sec истёк, job ещё в работе")
    waited_sec: float = 0.0


class SearchHit(BaseModel):
    source: str = ""
    title: str = ""
    url: str = ""
    horizon: Optional[str] = None


class SearchTestResponse(BaseModel):
    mode: str
    hits: list[SearchHit] = Field(default_factory=list)
    meta: Optional[dict[str, Any]] = None


class ReportOptionsResponse(BaseModel):
    """Сериализованный AnalysisReport для клиентов."""

    report: AnalysisReport
