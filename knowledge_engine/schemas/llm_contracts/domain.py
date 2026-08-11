"""Domain profiler — Gemini batch contract."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class DomainProfilerBatchItemContract(BaseModel):
    domain: str = Field(..., description="Домен без схемы")
    trust_score: float = Field(default=0.0, ge=0.0, le=1.0)
    category: str = Field(default="")
    is_valid_for_research: bool = Field(default=True)
    reason: str = Field(default="")


class DomainProfilerBatchContract(BaseModel):
    """Пачка доменов в одном Gemini запросе."""

    domains: List[DomainProfilerBatchItemContract] = Field(default_factory=list)


__all__ = ["DomainProfilerBatchContract", "DomainProfilerBatchItemContract"]
