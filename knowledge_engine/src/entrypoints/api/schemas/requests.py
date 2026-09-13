"""Pydantic модели запросов API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyzeCreate(BaseModel):
    problem: str = Field(..., min_length=3, description="Инженерная задача")
    constraints: str = Field(default="", description="Ограничения (стек, железо)")
    matrix_only: bool = Field(default=False, description="Без unraveling")
    async_mode: bool = Field(
        default=True,
        description="Запуск в фоне (рекомендуется для длинных прогонов)",
    )
    reuse_cached_sources: bool = Field(
        default=False,
        description="Сначала использовать ссылки из архива (cache-first), затем SearXNG",
    )


class ClarifySubmit(BaseModel):
    answer: str = Field(
        ..., min_length=1, description="Ответ на уточняющий вопрос графа"
    )


class UnravelRequest(BaseModel):
    option_id: int = Field(..., ge=1, le=3, description="ID варианта матрицы")
    async_mode: bool = Field(default=True)
    force_rerun: bool = Field(
        default=False,
        description="Перезапустить unravel даже если job уже completed с этим option_id",
    )


class SearchTestRequest(BaseModel):
    query: str = Field(default="cache invalidation RAG")
    constraints: str = Field(default="")
    flat: bool = Field(default=False, description="Один запрос на все провайдеры")
    limit_per_provider: int = Field(default=3, ge=1, le=10)
