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

    source_name: str = Field(min_length=2, max_length=300)
    url: str = Field(min_length=8, max_length=2000)
    title: str = Field(default="", max_length=400)
    why_read: str = Field(default="", max_length=1200)
    key_focus: str = Field(default="", max_length=800)
    read_time_minutes: int = Field(default=0, ge=0, le=180)


class MasteryDashboard(BaseModel):
    topic_mastery_score: int = Field(default=0, ge=0, le=100)
    node_status: NodeStatus = "in_progress"
    strengths: list[str] = Field(default_factory=list, max_length=12)
    polish_zones: list[str] = Field(default_factory=list, max_length=12)
    critical_gaps: list[str] = Field(default_factory=list, max_length=8)
    learning_phase: str = "intro_assessment"
    learning_mode: str = "lecture"
    pathway_bridge: str = ""


class NodeContentBlock(BaseModel):
    summary: str = Field(default="", max_length=12_000)
    summary_html: str = Field(default="", max_length=200_000)
    diagram: str = Field(default="", max_length=8000)
    references: list[RichReferenceItem] = Field(default_factory=list, max_length=6)
    code_snippets: list[str] = Field(default_factory=list, max_length=4)


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
    history: list[dict[str, str]] = Field(default_factory=list, max_length=40)
    new_gap_to_record: str | None = None
    session_key: str = ""
    rag_facts_count: int = 0
    rag_fact_labels: list[str] = Field(default_factory=list, max_length=8)
    topic_mastery_score: int = Field(default=0, ge=0, le=100)
    concepts_matrix: list[CoreConceptRecord] = Field(default_factory=list, max_length=12)
    mastery_dashboard: MasteryDashboard | None = None
    learning_phase: str = "intro_assessment"
    learning_mode: str = "lecture"
    source_registry: list[dict[str, Any]] = Field(default_factory=list)


class IntroAssessmentOutput(BaseModel):
    """Flash: один экспресс-вопрос при init."""

    tutor_message: str = Field(max_length=2000)
    node_status: NodeStatus = "in_progress"


class DenseMaterialOutput(BaseModel):
    """Heavy: плотный блок материала."""

    lecture_body: str = Field(
        default="",
        max_length=10_000,
        description="Полная лекция для чата (300–600 слов)",
    )
    summary: str = Field(default="", max_length=12_000)
    diagram: str = Field(default="", max_length=8000)
    references: list[RichReferenceItem] = Field(default_factory=list, max_length=6)
    code_snippets: list[str] = Field(default_factory=list, max_length=4)
    bridge_to_next: str = Field(default="", max_length=2000)
    checkpoint_prompt: str = Field(default="", max_length=2000)


class DeepDiveLLMOutput(BaseModel):
    """Структурированный ответ тьютора (Flash)."""

    node_status: NodeStatus = "in_progress"
    summary: str = ""
    diagram: str = ""
    references: list[RichReferenceItem] = Field(default_factory=list)
    tutor_message: str = Field(default="", max_length=12_000)
    new_gap_to_record: str | None = None
