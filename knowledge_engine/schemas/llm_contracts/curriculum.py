"""Curriculum generation — Gemini Flash / Lite / Reasoner contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field

from knowledge_engine.src.curriculum.schemas import CurriculumNode


class FlashSourceRefContract(BaseModel):
    source_id: str = Field(default="", description="src_N из registry")
    url: str = Field(default="", description="URL источника")
    relevant_extracts: list[str] = Field(
        default_factory=list,
        description="Выдержки релевантные ноде",
    )


class FlashBreakdownContract(BaseModel):
    key_concepts: list[str] = Field(
        default_factory=list, description="Ключевые концепты"
    )
    architectural_focus: str = Field(
        default="",
        description="Архитектурный фокус ноды",
    )


class FlashNodeContract(BaseModel):
    node_id: str = Field(..., description="snake_case id")
    title: str = Field(..., description="Название ноды")
    layer: str = Field(default="foundation", description="foundation | advanced | sota")
    category: str = Field(default="", description="Категория DAG")
    brief_summary: str = Field(default="", description="Краткое описание")
    prerequisites: list[str] = Field(
        default_factory=list, description="node_id предков"
    )
    source_ref: FlashSourceRefContract = Field(default_factory=FlashSourceRefContract)
    node_curriculum_breakdown: FlashBreakdownContract = Field(
        default_factory=FlashBreakdownContract
    )


class FlashCurriculumPayloadContract(BaseModel):
    curriculum_id: str = Field(default="", description="slug curriculum")
    title: str = Field(default="", description="Название маршрута")
    description: str = Field(default="", description="Описание цели")
    nodes: list[FlashNodeContract] = Field(
        default_factory=list,
        description="Узлы DAG с prerequisites",
    )


class ModelFirstNodeContract(BaseModel):
    node_id: str = Field(..., description="snake_case id")
    title: str = Field(...)
    layer: str = Field(default="foundation")
    category: str = Field(default="")
    brief_summary: str = Field(default="")
    core_concepts: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)


class ModelFirstPayloadContract(BaseModel):
    curriculum_id: str = Field(default="")
    title: str = Field(default="")
    description: str = Field(default="")
    nodes: list[ModelFirstNodeContract] = Field(default_factory=list)


class CurriculumReasonerContract(BaseModel):
    curriculum_id: str = Field(default="", description="Уникальный slug")
    title: str = Field(default="")
    description: str = Field(default="")
    total_nodes: int = Field(default=0, ge=0)
    nodes: list[CurriculumNode] = Field(
        default_factory=list,
        description="Полные CurriculumNode для DAG",
    )


class GeminiRegistryEntryContract(BaseModel):
    source_id: str = Field(default="", description="src_1 …")
    title: str = Field(default="")
    whitelist_domain: str = Field(default="")
    source_type: str = Field(default="")
    url: str = Field(default="")
    why_read: str = Field(default="")


class GeminiPrimarySourceContract(BaseModel):
    source_name: str = Field(default="")
    chapter_or_article: str = Field(default="")
    core_concepts: list[str] = Field(default_factory=list)


class GeminiLearningResourceContract(BaseModel):
    title: str = Field(default="")
    url: str = Field(default="")
    why_read: str = Field(default="")


class GeminiNodePatchContract(BaseModel):
    node_id: str = Field(default="")
    mapped_source_ids: list[str] = Field(default_factory=list)
    learning_goal: str = Field(default="")
    primary_source_id: str = Field(default="")
    primary_whitelist_source: GeminiPrimarySourceContract = Field(
        default_factory=GeminiPrimarySourceContract
    )
    learning_resources: list[GeminiLearningResourceContract] = Field(
        default_factory=list
    )
    resource_urls: list[str] = Field(default_factory=list)


class GeminiSourcesEnrichmentContract(BaseModel):
    curriculum_sources_registry: list[GeminiRegistryEntryContract] = Field(
        default_factory=list,
        description="8–15 ресурсов whitelist",
    )
    nodes: list[GeminiNodePatchContract] = Field(
        default_factory=list,
        description="Патч mapped_source_ids per node",
    )


class ExpansionVectorContract(BaseModel):
    expansion_vector: str = Field(
        ...,
        min_length=20,
        max_length=4000,
        description="Текстовый вектор углубления без URL и списка нод",
    )


class FlashExpansionEdgeContract(BaseModel):
    from_node_id: str = Field(default="", description="Prerequisite node")
    to_node_id: str = Field(default="", description="Dependent node")


class FlashExpansionPatchContract(BaseModel):
    new_nodes: list[FlashNodeContract] = Field(
        default_factory=list,
        description="2–3 атомарные new_nodes",
    )
    new_edges: list[FlashExpansionEdgeContract] = Field(
        default_factory=list,
        description="DAG edges among new + anchor nodes",
    )
