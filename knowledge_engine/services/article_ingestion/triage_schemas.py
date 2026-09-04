"""Контракты Document Triage & TOC."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, field_validator


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
    keep_paragraph_ranges: List[List[str]] = Field(
        default_factory=list,
        description=(
            "Inclusive ranges as [start_P_id, end_P_id] pairs for MAIN technical "
            "content. Example: [[\"P_2\", \"P_123\"]]. Use list-of-lists, not objects."
        ),
    )
    # RU: пары абзацев, которые оставляем в теле статьи.
    pruned_sections_reason: List[str] = Field(
        default_factory=list,
        description=(
            "Dropped sections and why (bibliography, appendix, legal). "
            "Empty list if nothing was pruned."
        ),
    )
    # RU: причины выкинутых секций; пустой список, если подрезки не было.

    @field_validator("keep_paragraph_ranges", mode="before")
    @classmethod
    def _coerce_keep_ranges(cls, value: object) -> list[list[str]]:
        if value is None:
            return []
        if not isinstance(value, list):
            return []
        out: list[list[str]] = []
        for item in value:
            if isinstance(item, dict):
                start = item.get("start_p_id") or item.get("start")
                end = item.get("end_p_id") or item.get("end")
                if start and end:
                    out.append([str(start).strip(), str(end).strip()])
                continue
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                start, end = str(item[0]).strip(), str(item[1]).strip()
                if start and end:
                    out.append([start, end])
        return out

    @field_validator("pruned_sections_reason", mode="before")
    @classmethod
    def _coerce_reasons(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            line = value.strip()
            return [line] if line else []
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        return []


class TriageOutcome(BaseModel):
    """Результат triage для логов и downstream."""

    structure: DocumentStructureTree
    decision: TriageDecisionResponse
    kept_p_ids: List[str] = Field(default_factory=list)
