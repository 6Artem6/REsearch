"""Lite curriculum search — Gemini contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LiteQueryPlanContract(BaseModel):
    selected_domains: list[str] = Field(
        default_factory=list,
        description="3–5 whitelist доменов",
    )
    queries: list[str] = Field(
        default_factory=list,
        description="SearXNG queries с site: dorks",
    )


class ArxivQueryParamsContract(BaseModel):
    """Structured arXiv Atom search fields (one-pass with academic_query_en)."""

    title_keywords: list[str] = Field(
        default_factory=list,
        description="Phrases/terms for ti: field (precision title match)",
    )
    abstract_keywords: list[str] = Field(
        default_factory=list,
        description="Phrases/terms for abs: field",
    )
    categories: list[str] = Field(
        default_factory=list,
        description="arXiv cats e.g. cs.AI, cs.CL, cs.LG, stat.ML",
    )
    exclude_terms: list[str] = Field(
        default_factory=list,
        description="Terms to exclude via ANDNOT (surveys, homework, …)",
    )
    start_year: int | None = Field(
        default=None,
        description="Inclusive submittedDate start year (YYYY) or null",
    )
    end_year: int | None = Field(
        default=None,
        description="Inclusive submittedDate end year (YYYY) or null",
    )


class LiteAcademicQueryContract(BaseModel):
    academic_query_en: str = Field(
        ...,
        description="English literature search query 1–2 sentences",
    )
    notes: str = Field(default="", description="Кратко на русском")
    arxiv_params: ArxivQueryParamsContract = Field(
        default_factory=ArxivQueryParamsContract,
        description="Precision arXiv query fields in the same LLM pass",
    )


class LiteHitEvaluationContract(BaseModel):
    id: int = Field(..., description="hit id из batch")
    is_sufficient: bool = Field(default=False)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = Field(default="")


class LiteBatchEvalContract(BaseModel):
    evaluations: list[LiteHitEvaluationContract] = Field(default_factory=list)


class LiteSourceEvalItemContract(BaseModel):
    id: int = Field(...)
    status: Literal["APPROVED", "REJECTED"] = Field(default="REJECTED")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = Field(default="")
    suggested_action: Literal["RETRY_WITH_NEW_SOURCE", "REMOVE_LINK", "KEEP"] = "KEEP"


class LiteSourceBatchContract(BaseModel):
    evaluations: list[LiteSourceEvalItemContract] = Field(default_factory=list)


class LiteSiteSuggestionsContract(BaseModel):
    sites: list[str] = Field(
        default_factory=list,
        max_length=6,
        description="Дополнительные домены для поиска",
    )
    rationale: str = Field(default="")
