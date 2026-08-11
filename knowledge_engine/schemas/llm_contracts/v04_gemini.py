"""v0.4 LangGraph — Gemini structured contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field

from knowledge_engine.schemas import CSAbstraction, TradeOffOption


class L1PatternStubContract(BaseModel):
    title: str = Field(..., description="Название L1 паттерна")
    description: str = Field(default="", description="Краткое описание")


class GeminiL0DecompositionContract(BaseModel):
    l0_summary: str = Field(..., description="Мета-карта задачи L0")
    l1_patterns: list[L1PatternStubContract] = Field(
        default_factory=list,
        description="Список L1 паттернов",
    )
    search_queries: list[str] = Field(
        default_factory=list,
        description="Запросы для discovery",
    )


class L2EvidenceItemContract(BaseModel):
    fact: str = Field(..., description="Инженерный факт из источника")
    failure_mode: str = Field(default="", description="Failure mode если есть")
    metric: str = Field(default="", description="Метрика / benchmark")


class L2EvidenceExtractionContract(BaseModel):
    evidences: list[L2EvidenceItemContract] = Field(default_factory=list)
    l1_title_hint: str = Field(
        default="",
        description="Ближайший L1-паттерн для фактов",
    )


class ResearchEvaluationContract(BaseModel):
    is_sufficient: bool = Field(
        ...,
        description="Достаточно L2 для trade-off анализа",
    )
    missing_gaps: list[str] = Field(
        default_factory=list,
        description="Непокрытые gaps",
    )
    new_search_queries: list[str] = Field(
        default_factory=list,
        description="Точечные запросы если insufficient",
    )


class AnalysisReportContract(BaseModel):
    abstractions: list[CSAbstraction] = Field(
        ...,
        description="CS абстракции задачи",
    )
    options: list[TradeOffOption] = Field(
        ...,
        description="Trade-off варианты (3 колонки)",
    )
