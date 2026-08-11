"""Схемы входа/выхода Curriculum Generator (roadmap.sh / React Flow)."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from knowledge_engine.config import CURRICULUM_DEEP_NODE_MAX_HITS

DepthLevel = Literal["Overview", "Standard", "Deep Mechanics"]
GenerationMode = Literal["fast", "consensus"]
LayerKind = Literal["foundation", "advanced", "sota"]
NodeRiskKind = Literal["BASE", "DEEP"]
GroundingStatus = Literal[
    "model_only",
    "grounded",
    "unverified_deep",
    "pending_grounding",
]

_NODE_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _normalize_node_id(raw: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (raw or "").strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    if not s or not s[0].isalpha():
        s = f"n_{s or 'topic'}"
    return s[:80]


class CurriculumGenerateInput(BaseModel):
    """Входной протокол приложения."""

    target_goal: str = Field(
        min_length=8,
        max_length=4000,
        description="Глобальная инженерная цель",
    )
    user_level: str = Field(
        default="Intermediate/Advanced",
        max_length=120,
        description="Базовый уровень подготовки",
    )
    depth_level: DepthLevel = Field(
        default="Standard",
        description="Глубина: Overview | Standard | Deep Mechanics",
    )
    generation_mode: GenerationMode = Field(
        default="fast",
        description="legacy: fast | consensus (используйте source_policy)",
    )
    source_policy: str = Field(
        default="practical_only",
        description="hybrid | practical_only | academic_only",
    )

    @field_validator("source_policy", mode="before")
    @classmethod
    def _norm_source_policy(cls, v: str) -> str:
        from knowledge_engine.src.curriculum.source_policy import (
            normalize_source_policy,
        )

        return normalize_source_policy(v, default="practical_only")

    @field_validator("generation_mode", mode="before")
    @classmethod
    def _norm_generation_mode(cls, v: str) -> str:
        m = str(v or "fast").strip().lower()
        if m in ("deep", "consensus"):
            return "consensus"
        return "fast"


class CurriculumResourceRef(BaseModel):
    """Ссылка из генерации маршрута (Flash/Reasoner) для RAG лекций."""

    title: str = Field(default="", max_length=400)
    url: str = Field(min_length=8, max_length=2000)
    why_read: str = Field(default="", max_length=800)


class RouteSourceEntry(BaseModel):
    """Legacy зеркало реестра для UI [S1] / [src_1]."""

    source_id: str = Field(min_length=2, max_length=16)
    source_name: str = Field(min_length=2, max_length=400)
    url: str = Field(min_length=8, max_length=2000)
    whitelist_category: str = Field(default="", max_length=120)
    why_read: str = Field(default="", max_length=800)


class CurriculumSourceRegistryEntry(BaseModel):
    """Глобальная библиотека курса (curriculum_sources_registry)."""

    source_id: str = Field(min_length=3, max_length=16)
    title: str = Field(min_length=2, max_length=400)
    whitelist_domain: str = Field(default="", max_length=200)
    source_type: str = Field(default="", max_length=120)
    url: str = Field(default="", max_length=2000)
    why_read: str = Field(default="", max_length=800)
    snippet: str = Field(default="", max_length=1200)
    key_extracts: list[str] = Field(default_factory=list, max_length=12)
    source_tier: str = Field(
        default="",
        max_length=24,
        description="consensus | gemini_grounding | whitelist_blog",
    )


class NodeSourceRef(BaseModel):
    """Привязка ноды к источнику и выдержкам (Search-First / Flash)."""

    source_id: str = Field(default="", max_length=16)
    url: str = Field(default="", max_length=2000)
    relevant_extracts: list[str] = Field(default_factory=list, max_length=12)


class NodeCurriculumBreakdown(BaseModel):
    key_concepts: list[str] = Field(default_factory=list, max_length=24)
    architectural_focus: str = Field(default="", max_length=800)


class PrimaryWhitelistSource(BaseModel):
    """Главный источник ноды — строго из APPROVED_SOURCES_WHITELIST."""

    source_name: str = Field(min_length=2, max_length=400)
    chapter_or_article: str = Field(min_length=2, max_length=800)
    core_concepts: list[str] = Field(min_length=1, max_length=12)


class CurriculumSearchHit(BaseModel):
    """Результат предпоиска перед Flash (Search-First)."""

    source_id: str = Field(default="", max_length=16)
    url: str = Field(min_length=8, max_length=2000)
    title: str = Field(default="", max_length=400)
    snippet: str = Field(default="", max_length=1200)
    published_date: str = Field(default="", max_length=32)
    key_extracts: list[str] = Field(default_factory=list, max_length=12)
    source_tier: str = Field(default="", max_length=24)
    skip_ollama_summary: bool = Field(
        default=False,
        description="Exa highlights уже достаточны — не гонять 7B summarizer",
    )
    exa_relevance_score: float | None = Field(
        default=None,
        description="Exa neural relevance score (если API вернул score)",
    )


class LearningMaterials(BaseModel):
    primary_whitelist_source: PrimaryWhitelistSource | None = None


class CurriculumNode(BaseModel):
    node_id: str = Field(
        min_length=2,
        max_length=80,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Уникальный snake_case ID",
    )
    title: str = Field(min_length=2, max_length=300)
    layer: LayerKind
    category: str = Field(min_length=2, max_length=200)
    brief_summary: str = Field(min_length=10, max_length=1200)
    core_concepts: list[str] = Field(min_length=1, max_length=8)
    prerequisites: list[str] = Field(default_factory=list, max_length=24)
    resource_urls: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="Прямые URL из генерации маршрута",
    )
    learning_resources: list[CurriculumResourceRef] = Field(
        default_factory=list,
        max_length=8,
    )
    learning_materials: LearningMaterials = Field(default_factory=LearningMaterials)
    primary_source_id: str = Field(
        default="",
        max_length=16,
        description="Главный source_id из реестра (первый из mapped_source_ids)",
    )
    mapped_source_ids: list[str] = Field(
        default_factory=list,
        max_length=CURRICULUM_DEEP_NODE_MAX_HITS,
        description="source_id из curriculum_sources_registry (до CURRICULUM_DEEP_NODE_MAX_HITS)",
    )
    learning_goal: str = Field(default="", max_length=600)
    source_ref: NodeSourceRef | None = None
    node_curriculum_breakdown: NodeCurriculumBreakdown | None = None
    node_risk_kind: NodeRiskKind = Field(
        default="BASE",
        description="BASE — без веб-поиска; DEEP — требуется RAG",
    )
    grounding_status: GroundingStatus = Field(
        default="model_only",
        description="model_only | grounded | unverified_deep | pending_grounding",
    )

    @field_validator("source_ref", "node_curriculum_breakdown", mode="before")
    @classmethod
    def _norm_nested(cls, v: Any) -> Any:
        if v is None or isinstance(v, BaseModel):
            return v
        if isinstance(v, dict) and v:
            return v
        return None

    @field_validator("mapped_source_ids")
    @classmethod
    def _strip_mapped_ids(cls, v: list[str]) -> list[str]:
        out: list[str] = []
        for raw in v or []:
            s = str(raw).strip()
            if s and s not in out:
                out.append(s[:16])
        return out[:CURRICULUM_DEEP_NODE_MAX_HITS]

    @field_validator("learning_materials", mode="before")
    @classmethod
    def _norm_learning_materials(
        cls, v: LearningMaterials | dict | None
    ) -> LearningMaterials:
        if v is None:
            return LearningMaterials()
        if isinstance(v, LearningMaterials):
            return v
        if isinstance(v, dict):
            return LearningMaterials.model_validate(v)
        return LearningMaterials()

    @field_validator("node_id", mode="before")
    @classmethod
    def _norm_node_id(cls, v: str) -> str:
        return _normalize_node_id(str(v))

    @field_validator("node_id")
    @classmethod
    def _check_node_id(cls, v: str) -> str:
        if not _NODE_ID_RE.match(v):
            raise ValueError(f"некорректный node_id: {v}")
        return v

    @field_validator("core_concepts")
    @classmethod
    def _strip_concepts(cls, v: list[str]) -> list[str]:
        out = [c.strip() for c in v if c and c.strip()]
        if not out:
            raise ValueError("core_concepts не может быть пустым")
        return out[:8]

    @field_validator("prerequisites", mode="before")
    @classmethod
    def _norm_prereqs(cls, v: list[str] | None) -> list[str]:
        if not v:
            return []
        return [_normalize_node_id(str(p)) for p in v if str(p).strip()]

    @field_validator("prerequisites")
    @classmethod
    def _strip_prereqs(cls, v: list[str]) -> list[str]:
        return [p.strip() for p in v if p and p.strip()]


class CurriculumGraph(BaseModel):
    curriculum_id: str = Field(min_length=3, max_length=80)
    title: str = Field(min_length=3, max_length=300)
    description: str = Field(min_length=10, max_length=4000)
    total_nodes: int = Field(ge=1, le=40)
    curriculum_sources_registry: list[CurriculumSourceRegistryEntry] = Field(
        default_factory=list,
        max_length=32,
    )
    route_sources: list[RouteSourceEntry] = Field(default_factory=list, max_length=24)
    nodes: list[CurriculumNode] = Field(min_length=3, max_length=40)

    @field_validator("nodes")
    @classmethod
    def _unique_node_ids(cls, v: list[CurriculumNode]) -> list[CurriculumNode]:
        seen: set[str] = set()
        for n in v:
            if n.node_id in seen:
                raise ValueError(f"дублирующий node_id: {n.node_id}")
            seen.add(n.node_id)
        return v


class CurriculumReasonerPayload(BaseModel):
    """Структурированный ответ Reasoner перед нормализацией."""

    curriculum_id: str = ""
    title: str = ""
    description: str = ""
    total_nodes: int = 0
    nodes: list[CurriculumNode] = Field(default_factory=list)


class ExpansionVectorOutput(BaseModel):
    """Lite: вектор направления расширения (без нод и ссылок)."""

    expansion_vector: str = Field(min_length=20, max_length=4000)


class CurriculumExpansionEdge(BaseModel):
    """Новое DAG-ребро: prerequisite → dependent."""

    from_node_id: str = Field(min_length=2, max_length=80)
    to_node_id: str = Field(min_length=2, max_length=80)


class CurriculumExpansionPatch(BaseModel):
    """Flash: JSON Patch для expand_curriculum."""

    new_nodes: list[CurriculumNode] = Field(default_factory=list, max_length=20)
    new_edges: list[CurriculumExpansionEdge] = Field(
        default_factory=list, max_length=40
    )
