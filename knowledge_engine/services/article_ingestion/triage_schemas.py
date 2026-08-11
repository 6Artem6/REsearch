"""Контракты Document Triage & TOC."""

from __future__ import annotations

from typing import List, Tuple

from pydantic import BaseModel, Field


class TOCNode(BaseModel):
    title: str = Field(..., description="Название секции/главы")
    level: int = Field(
        default=1, description="Уровень вложенности (1 for H1, 2 for H2)"
    )
    start_p_id: str = Field(
        ..., description="ID первого абзаца этой секции, например 'P_12'"
    )
    end_p_id: str | None = Field(None, description="ID последнего абзаца этой секции")
    page_number: int | None = Field(
        None, description="Номер страницы в PDF (если есть)"
    )


class DocumentStructureTree(BaseModel):
    has_explicit_toc: bool = Field(
        ..., description="Было ли найдено явное оглавление в файле"
    )
    nodes: List[TOCNode] = Field(
        ..., description="Иерархическое дерево всех секций документа"
    )


class TriageDecisionResponse(BaseModel):
    keep_paragraph_ranges: List[Tuple[str, str]] = Field(
        ...,
        description="Пары [start_P_id, end_P_id] для секций с основным техническим контентом",
    )
    pruned_sections_reason: List[str] = Field(
        ...,
        description="Удалённые секции и причины",
    )


class TriageOutcome(BaseModel):
    """Результат triage для логов и downstream."""

    structure: DocumentStructureTree
    decision: TriageDecisionResponse
    kept_p_ids: List[str] = Field(default_factory=list)
