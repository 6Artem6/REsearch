"""Pydantic v2 — spatial Map-Reduce + legacy single-pass schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from knowledge_engine.schemas.extraction import (
    KnowledgeAtom,
    ParagraphInspectionResult,
    attach_source_chunk_id,
    coerce_scope_type,
    normalize_knowledge_atoms,
    tagged_takeaways_from_atoms,
)


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
    figure_id: str = Field(..., description="Exact figure id, e.g. 'FIG_1'")
    referenced_paragraphs: list[str] = Field(
        ...,
        description="Paragraphs in the current window that critically need the figure",
    )
    reason: str = Field(
        ...,
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
        description="Figures needed to close gaps in this window",
    )

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
            "8–12 takeaways prefixed with [SCOPE: PRINCIPLE|MECHANIC|INSTANCE]"
        ),
    )
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
        return v if v is not None else []


def normalize_final_knowledge(
    final: FinalArticleSummaryResponse,
) -> FinalArticleSummaryResponse:
    """Pydantic-нормализация: atoms ↔ tagged takeaways после Reduce."""
    atoms = normalize_knowledge_atoms(
        final.knowledge_atoms or [],
        fallback_lines=final.key_takeaways or [],
    )
    final.knowledge_atoms = atoms
    tagged = tagged_takeaways_from_atoms(atoms, max_items=12)
    if tagged:
        final.key_takeaways = tagged
    elif final.key_takeaways:
        # сохранить сырые строки, пометив как PRINCIPLE при отсутствии тега
        final.key_takeaways = tagged_takeaways_from_atoms(
            normalize_knowledge_atoms([], fallback_lines=final.key_takeaways),
            max_items=12,
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
