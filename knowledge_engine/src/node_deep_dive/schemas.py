"""Схемы Node Deep-Dive (Модуль 2)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from knowledge_engine.config import CURRICULUM_DEEP_NODE_MAX_HITS
from knowledge_engine.src.curriculum.schemas import (
    LayerKind,
    LearningMaterials,
    NodeCurriculumBreakdown,
    NodeSourceRef,
    _normalize_node_id,
)
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    CoreConceptRecord,
    LectureExtractedConcept,
    NodeStatus,
)

UserAction = Literal["init", "chat", "verify"]


class NodeDataInput(BaseModel):
    """Объект ноды из Модуля 1."""

    node_id: str = Field(min_length=2, max_length=80)
    title: str = Field(min_length=2, max_length=300)
    layer: LayerKind
    core_concepts: list[str] = Field(min_length=1, max_length=8)
    prerequisites: list[str] = Field(default_factory=list, max_length=24)
    brief_summary: str = Field(default="", max_length=1200)
    category: str = Field(default="", max_length=200)
    learning_materials: LearningMaterials | None = None
    mapped_source_ids: list[str] = Field(
        default_factory=list,
        max_length=CURRICULUM_DEEP_NODE_MAX_HITS,
    )
    learning_goal: str = Field(default="", max_length=600)
    primary_source_id: str = Field(default="", max_length=16)
    source_ref: NodeSourceRef | None = None
    node_curriculum_breakdown: NodeCurriculumBreakdown | None = None
    resource_urls: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="URL маршрута (как в CurriculumNode)",
    )
    learning_resources: list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=8,
        description="Карточки ресурсов из skill-tree graph",
    )

    @field_validator("learning_resources", mode="before")
    @classmethod
    def _norm_learning_resources(cls, v: Any) -> list[dict[str, Any]]:
        if not v:
            return []
        out: list[dict[str, Any]] = []
        for item in v:
            if isinstance(item, dict):
                out.append(item)
            elif hasattr(item, "model_dump"):
                out.append(item.model_dump())
        return out[:8]

    @field_validator("source_ref", "node_curriculum_breakdown", mode="before")
    @classmethod
    def _norm_plan_fields(cls, v: Any) -> Any:
        if v is None or isinstance(v, (NodeSourceRef, NodeCurriculumBreakdown)):
            return v
        if isinstance(v, dict) and v:
            return v
        return None

    @field_validator("learning_materials", mode="before")
    @classmethod
    def _norm_lm(cls, v: LearningMaterials | dict | None) -> LearningMaterials | None:
        if v is None:
            return None
        if isinstance(v, LearningMaterials):
            return v
        if isinstance(v, dict) and v:
            return LearningMaterials.model_validate(v)
        return None

    @field_validator("node_id", mode="before")
    @classmethod
    def _norm_id(cls, v: str) -> str:
        return _normalize_node_id(str(v))


class ReferenceItem(BaseModel):
    source_name: str = Field(min_length=2, max_length=300)
    url: str = Field(min_length=8, max_length=2000)


class RichReferenceItem(BaseModel):
    """Обогащённая карточка ресурса (не сухий URL)."""

    asset_id: str = Field(default="", max_length=32)
    source_name: str = Field(min_length=2, max_length=300)
    url: str = Field(min_length=8, max_length=2000)
    title: str = Field(default="", max_length=400)
    why_read: str = Field(default="", max_length=1200)
    key_focus: str = Field(default="", max_length=800)
    read_time_minutes: int = Field(default=0, ge=0, le=180)


CoverageItemState = Literal["verified", "in_progress", "unchecked", "gap"]
CoverageLayerStatus = Literal["passed", "in_progress", "locked", "failed", "gloss"]
ActiveDepthLayer = Literal["WHY", "HOW", "MECHANIC"]
FactEvalStatus = Literal["matched", "omitted", "pending"]


class FactEvaluation(BaseModel):
    """Optional atom-level credit (future AtomicFactMatcher → UI)."""

    fact_id: str = Field(
        ...,
        min_length=1,
        max_length=80,
        description="ID атома из knowledge_atoms",
    )
    statement: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Текст концепта/факта",
    )
    layer: ActiveDepthLayer = "WHY"
    status: FactEvalStatus = "pending"
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    sub_concept_id: str = Field(default="", max_length=64)


class CoverageItem(BaseModel):
    id: str = Field(min_length=2, max_length=64)
    label: str = Field(min_length=1, max_length=200)
    state: CoverageItemState = "unchecked"
    why_passed: bool = False
    how_passed: bool = False
    mechanic_passed: bool = False
    status_hint: str = Field(
        default="",
        max_length=240,
        description=(
            "Short UI status without repeating the label "
            "(e.g. «Не хватает механик реализации»)"
        ),
    )
    facts: list[FactEvaluation] = Field(
        default_factory=list,
        max_length=24,
        description="Per-subtopic fact credits (empty until matcher exists)",
    )


class CoverageLayerProgress(BaseModel):
    status: CoverageLayerStatus = "locked"
    score: float = Field(default=0.0, ge=0.0, le=1.0)


class CoverageLayers(BaseModel):
    why: CoverageLayerProgress = Field(default_factory=CoverageLayerProgress)
    how: CoverageLayerProgress = Field(default_factory=CoverageLayerProgress)
    mechanic: CoverageLayerProgress = Field(default_factory=CoverageLayerProgress)


class CoverageSummary(BaseModel):
    total: int = Field(default=0, ge=0, le=12)
    verified: int = Field(default=0, ge=0, le=12)
    items: list[CoverageItem] = Field(default_factory=list, max_length=8)
    layers: CoverageLayers | None = None
    overall_score: int = Field(default=0, ge=0, le=100)
    active_layer: ActiveDepthLayer | None = None
    gloss_hint: str = Field(default="", max_length=400)
    facts_breakdown: list[FactEvaluation] = Field(
        default_factory=list,
        max_length=64,
        description="Flat fact evaluations for UI (empty until matcher exists)",
    )


class MasteryDashboard(BaseModel):
    topic_mastery_score: int = Field(default=0, ge=0, le=100)
    node_status: NodeStatus = "in_progress"
    strengths: list[str] = Field(default_factory=list, max_length=12)
    polish_zones: list[str] = Field(default_factory=list, max_length=12)
    critical_gaps: list[str] = Field(default_factory=list, max_length=8)
    learning_phase: str = "intro_assessment"
    learning_mode: str = "lecture"
    pathway_bridge: str = ""
    coverage_summary: CoverageSummary | None = None


class DiagramAsset(BaseModel):
    id: str = Field(min_length=3, max_length=32)
    title: str = Field(default="", max_length=200)
    mermaid: str = Field(default="", max_length=8000)


class CodeAsset(BaseModel):
    id: str = Field(min_length=3, max_length=32)
    title: str = Field(default="", max_length=200)
    language: str = Field(default="", max_length=32)
    code: str = Field(default="", max_length=12_000)


class NodeContentBlock(BaseModel):
    summary: str = Field(default="", max_length=12_000)
    summary_html: str = Field(default="", max_length=200_000)
    diagram: str = Field(default="", max_length=8000)
    diagrams: list[DiagramAsset] = Field(default_factory=list, max_length=12)
    references: list[RichReferenceItem] = Field(default_factory=list, max_length=16)
    code_snippets: list[str] = Field(default_factory=list, max_length=4)
    code_assets: list[CodeAsset] = Field(default_factory=list, max_length=16)


class NodeDeepDiveRequest(BaseModel):
    curriculum_id: str = Field(min_length=3, max_length=80)
    node_data: NodeDataInput
    user_action: UserAction
    user_message: str = Field(default="", max_length=8000)


class NodeDeepDiveResponse(BaseModel):
    node_id: str
    node_status: NodeStatus
    content: NodeContentBlock
    tutor_message: str = Field(max_length=12_000)
    tutor_message_html: str = Field(default="", max_length=200_000)
    tutor_dialogue_feedback: str = Field(
        default="",
        max_length=4000,
        description="Семантическое поле feedback_on_answer (для UI склейки)",
    )
    tutor_dialogue_technical: str = Field(
        default="",
        max_length=10_000,
        description="Семантическое поле technical_explanation",
    )
    tutor_dialogue_follow_up: str = Field(
        default="",
        max_length=2000,
        description="Семантическое поле follow_up_question",
    )
    quick_replies: list[str] = Field(
        default_factory=list,
        max_length=4,
        description="Chip labels for UI Quick Replies (e.g. Gloss / MECH / next)",
    )
    ready_for_transition: bool = Field(
        default=False,
        description="True when topic/phase closed — UI may show transition Quick Reply chips",
    )
    last_eval_directive: str = Field(
        default="",
        max_length=64,
        description="Last Threshold Engine directive (e.g. PASSED_WITH_GLOSS)",
    )
    history: list[dict[str, str]] = Field(default_factory=list, max_length=40)
    new_gap_to_record: str | None = None
    session_key: str = ""
    rag_facts_count: int = 0
    rag_fact_labels: list[str] = Field(default_factory=list, max_length=8)
    topic_mastery_score: int = Field(default=0, ge=0, le=100)
    concepts_matrix: list[CoreConceptRecord] = Field(
        default_factory=list, max_length=12
    )
    mastery_dashboard: MasteryDashboard | None = None
    coverage_summary: CoverageSummary | None = None
    learning_phase: str = "intro_assessment"
    learning_mode: str = "lecture"
    source_registry: list[dict[str, Any]] = Field(default_factory=list)
    lecture_rag_inspector: list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=16,
        description="RAG-чанки, переданные в Reduce (сверка [R1]…[Rn])",
    )


class IntroAssessmentOutput(BaseModel):
    """Flash: один экспресс-вопрос при init."""

    tutor_message: str = Field(max_length=2000)
    node_status: NodeStatus = "in_progress"


class DenseMaterialOutput(BaseModel):
    """Heavy: плотный блок материала (внутренний UI/session; генерация через StructuredLectureResponse)."""

    lecture_body: str = Field(
        default="",
        max_length=24_000,
        description=(
            "Полная лекция для чата (Markdown). См. также StructuredLectureResponse.lecture_body"
        ),
    )
    summary: str = Field(default="", max_length=12_000)
    referenced_diagram_id: str | None = Field(
        default=None,
        max_length=64,
        description="Catalog diagram id for panel; server resolves Mermaid",
    )
    references: list[RichReferenceItem] = Field(default_factory=list, max_length=6)
    code_snippets: list[str] = Field(default_factory=list, max_length=4)
    bridge_to_next: str = Field(default="", max_length=2000)
    checkpoint_prompt: str = Field(default="", max_length=2000)
    extracted_concepts: list[LectureExtractedConcept] = Field(
        default_factory=list,
        max_length=5,
        description="Микро-темы лекции для Coverage Registry",
    )
    introduced_terms: list[str] = Field(
        default_factory=list,
        max_length=24,
        description="Термины, впервые расшифрованные в этой лекции",
    )


class DeepDiveLLMOutput(BaseModel):
    """Структурированный ответ тьютора (Flash)."""

    node_status: NodeStatus = "in_progress"
    summary: str = ""
    referenced_diagram_id: str | None = Field(
        default=None,
        max_length=64,
        description="Catalog diagram id for panel; server resolves Mermaid",
    )
    references: list[RichReferenceItem] = Field(default_factory=list)
    feedback_on_answer: str = Field(default="", max_length=4000)
    technical_explanation: str = Field(default="", max_length=10_000)
    follow_up_question: str = Field(default="", max_length=2000)
    question_sub_concept_id: str | None = None
    new_gap_to_record: str | None = None
    introduced_terms: list[str] = Field(default_factory=list, max_length=16)
    verified_sub_concept_ids: list[str] = Field(default_factory=list, max_length=8)
    ready_for_transition: bool = False
    suggested_next_step: str | None = None
    quick_replies: list[str] = Field(default_factory=list, max_length=4)

    def compose_tutor_message(self) -> str:
        from knowledge_engine.src.node_deep_dive.tutor_dialogue import (
            compose_tutor_dialogue_from_output,
        )

        return compose_tutor_dialogue_from_output(self)
