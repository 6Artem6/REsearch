"""Pydantic v2 — spatial Map-Reduce + legacy single-pass schemas."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field, field_validator

from knowledge_engine.schemas.extraction import (
    KnowledgeAtom,
    ParagraphInspectionResult,
    attach_source_chunk_id,
    coerce_scope_type,
    normalize_knowledge_atoms,
)

_LOG = logging.getLogger(__name__)


def _coerce_window_diagram_list(value: object) -> list[object]:
    """Heal LLM list items: bare strings / partial dicts → WindowDiagramCheck payloads."""
    if value is None:
        return []
    raw_items: list[object]
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, (list, tuple)):
        raw_items = list(value)
    else:
        raw_items = [value]
    out: list[object] = []
    for item in raw_items:
        if item is None:
            continue
        if isinstance(item, WindowDiagramCheck):
            out.append(item)
            continue
        if isinstance(item, str):
            name = item.strip()
            if not name:
                continue
            _LOG.warning(
                "required_diagrams item is a bare string %r; "
                "coercing to WindowDiagramCheck",
                name,
            )
            out.append(
                {
                    "figure_id": name,
                    "referenced_paragraphs": [],
                    "reason": "",
                }
            )
            continue
        if isinstance(item, dict):
            fid = str(
                item.get("figure_id")
                or item.get("diagram_id")
                or item.get("id")
                or item.get("name")
                or ""
            ).strip()
            paras = (
                item.get("referenced_paragraphs")
                or item.get("relevant_paragraphs")
                or []
            )
            if isinstance(paras, str):
                paras = [paras] if paras.strip() else []
            elif not isinstance(paras, list):
                paras = []
            reason = item.get("reason") or item.get("semantic_reason") or ""
            out.append(
                {
                    "figure_id": fid or "unknown",
                    "referenced_paragraphs": paras,
                    "reason": str(reason or ""),
                }
            )
            continue
        _LOG.warning(
            "required_diagrams item has unsupported type %s; dropping",
            type(item).__name__,
        )
    return out


class TargetDiagramLocation(BaseModel):
    figure_id: str = Field(
        ...,
        description="Exact figure id from the text, e.g. 'FIG_1' or 'FIG_3'",
    )
    relevant_paragraphs: list[str] = Field(
        ...,
        description="Paragraph ids P_X that reference or rely on the figure",
    )
    semantic_reason: str = Field(
        ...,
        description="Architectural reason why the figure is critical for these paragraphs",
    )


class BlogArticleSummaryResponse(BaseModel):
    summary: str = Field(..., description="Deep engineering article digest in Markdown")
    key_takeaways: list[str] = Field(
        ...,
        description=("Key insights prefixed with [SCOPE: PRINCIPLE|MECHANIC|INSTANCE]"),
    )
    knowledge_atoms: list[KnowledgeAtom] = Field(
        default_factory=list,
        description="Structured Knowledge Triangulation atoms (mirror of takeaways)",
    )
    critical_diagram_locations: list[TargetDiagramLocation] = Field(
        default_factory=list,
        description="Critical FIG_X diagrams linked to P_Y; empty if none",
    )


class WindowDiagramCheck(BaseModel):
    figure_id: str = Field(
        default="unknown",
        min_length=1,
        description=(
            "Exact figure id from the window text, e.g. FIG_1. "
            "FORBIDDEN: conceptual camelCase names as the list item itself; "
            "each required_diagrams entry is this object, never a bare string."
        ),
    )
    # RU: id фигуры окна (FIG_n); не имя придуманной схемы.
    referenced_paragraphs: list[str] = Field(
        default_factory=list,
        description="Paragraphs in the current window that critically need the figure",
    )
    reason: str = Field(
        default="",
        description="Architectural reason why the figure is needed here",
    )


class MapWindowResponse(BaseModel):
    """Lite/Gemma MAP: window inspection + ParagraphInspectionResult.atoms."""

    window_role: str = Field(
        default="",
        description=(
            "Short window role tag (2–6 words): «Benchmarks», «System architecture»"
        ),
    )
    window_summary: str = Field(..., description="Dense summary of the current window")
    knowledge_atoms: list[KnowledgeAtom] = Field(
        default_factory=list,
        max_length=24,
        description=(
            "Required atoms with scope PRINCIPLE / MECHANIC / INSTANCE "
            "(ParagraphInspectionResult.atoms)"
        ),
    )
    required_diagrams: list[WindowDiagramCheck] = Field(
        default_factory=list,
        description=(
            "WindowDiagramCheck objects only: "
            "{figure_id, referenced_paragraphs, reason}. "
            "Prefer []. Never emit bare strings. "
            "figure_id must be FIG_n from the window, not an invented name."
        ),
    )
    # RU: объекты диаграмм окна; строки от LLM приводятся к объекту, окно не падает.

    @field_validator("required_diagrams", mode="before")
    @classmethod
    def _coerce_required_diagrams(cls, v: object) -> object:
        return _coerce_window_diagram_list(v)

    def as_inspection(self) -> ParagraphInspectionResult:
        return ParagraphInspectionResult(atoms=list(self.knowledge_atoms or []))


class DeduplicatedAtomsResponse(BaseModel):
    """REDUCE phase 1: merged knowledge atoms only (no executive prose)."""

    knowledge_atoms: list[KnowledgeAtom] = Field(
        default_factory=list,
        max_length=32,
        description="Deduplicated full KnowledgeAtom objects (no unique fact lost)",
    )

    @field_validator("knowledge_atoms", mode="before")
    @classmethod
    def _coerce_atoms(cls, v: object) -> object:
        return v if v is not None else []


class FinalArticleSummaryResponse(BaseModel):
    executive_summary: str = Field(
        ...,
        description="Single coherent Markdown digest from all window summaries",
    )
    key_takeaways: list[str] = Field(
        ...,
        description=(
            "Compressed synthesis takeaways (3–7 lines) prefixed with "
            "[SCOPE: PRINCIPLE|MECHANIC|INSTANCE]; not the full knowledge_atoms catalog."
        ),
    )
    """ RU: сжатые выводы фазы синтеза; полный каталог фактов — в knowledge_atoms. """
    knowledge_atoms: list[KnowledgeAtom] = Field(
        default_factory=list,
        max_length=32,
        description="Aggregated atoms from all windows (scope tags preserved)",
    )
    target_diagrams_for_vlm: list[WindowDiagramCheck] = Field(
        default_factory=list,
        description="Deduplicated figure list across windows",
    )

    @field_validator("executive_summary", mode="before")
    @classmethod
    def _coerce_executive(cls, v: object) -> object:
        if v is None:
            return ""
        if isinstance(v, str):
            return v.strip()
        return str(v)

    @field_validator("key_takeaways", mode="before")
    @classmethod
    def _coerce_takeaways(cls, v: object) -> object:
        if v is None:
            return []
        if isinstance(v, str):
            line = v.strip()
            return [line] if line else []
        if isinstance(v, list):
            out: list[str] = []
            for item in v:
                if item is None:
                    continue
                if isinstance(item, dict):
                    scope = coerce_scope_type(item.get("scope"))
                    stmt = str(item.get("statement") or item.get("text") or "").strip()
                    if stmt:
                        out.append(f"[SCOPE: {scope.value}] {stmt}")
                    continue
                s = str(item).strip()
                if s:
                    out.append(s)
            return out
        return v

    @field_validator("knowledge_atoms", mode="before")
    @classmethod
    def _coerce_atoms(cls, v: object) -> object:
        return v if v is not None else []

    @field_validator("target_diagrams_for_vlm", mode="before")
    @classmethod
    def _coerce_diagrams(cls, v: object) -> object:
        return _coerce_window_diagram_list(v)


def normalize_final_knowledge(
    final: FinalArticleSummaryResponse,
) -> FinalArticleSummaryResponse:
    """Нормализация knowledge_atoms после Reduce; takeaways синтеза не затираются."""
    final.knowledge_atoms = normalize_knowledge_atoms(
        final.knowledge_atoms or [],
        fallback_lines=[],
    )
    return final


def normalize_map_knowledge(
    mapped: MapWindowResponse,
    *,
    source_chunk_id: str | None = None,
) -> MapWindowResponse:
    atoms = normalize_knowledge_atoms(
        mapped.knowledge_atoms or [],
        fallback_lines=[],
    )
    # если LLM положила теги только в summary — вытянем маркированные строки
    if not atoms and mapped.window_summary:
        from knowledge_engine.schemas.extraction import extract_tagged_lines

        atoms = normalize_knowledge_atoms(
            [],
            fallback_lines=extract_tagged_lines(mapped.window_summary),
        )
    if source_chunk_id:
        atoms = attach_source_chunk_id(atoms, source_chunk_id)
    mapped.knowledge_atoms = atoms
    return mapped


def final_to_legacy_summary(
    final: FinalArticleSummaryResponse,
) -> BlogArticleSummaryResponse:
    final = normalize_final_knowledge(final)
    return BlogArticleSummaryResponse(
        summary=final.executive_summary,
        key_takeaways=list(final.key_takeaways),
        knowledge_atoms=list(final.knowledge_atoms or []),
        critical_diagram_locations=[
            TargetDiagramLocation(
                figure_id=d.figure_id,
                relevant_paragraphs=list(d.referenced_paragraphs),
                semantic_reason=d.reason,
            )
            for d in final.target_diagrams_for_vlm
        ],
    )
