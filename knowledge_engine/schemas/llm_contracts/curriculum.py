"""Curriculum generation — Gemini Flash / Lite / Reasoner contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

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


class NodeListNodeContract(BaseModel):
    """Pass 1 (Two-Pass Model-First): содержание ноды без топологии —
    prerequisites сюда намеренно не входит, генерируется отдельным Pass 2."""

    node_id: str = Field(..., description="snake_case id, unique within this list.")
    title: str = Field(..., description="Node title.")
    layer: str = Field(default="foundation", description="foundation | advanced | sota")
    category: str = Field(default="", description="DAG category label.")
    brief_summary: str = Field(default="", description="Short node summary.")
    core_concepts: list[str] = Field(
        default_factory=list, description="Key concepts covered by this node."
    )


class NodeListContract(BaseModel):
    """Pass 1 payload: декомпозиция темы на ноды, без рёбер."""

    curriculum_id: str = Field(default="")
    title: str = Field(default="")
    description: str = Field(default="")
    nodes: list[NodeListNodeContract] = Field(
        default_factory=list,
        description="8-12 decomposed topic nodes, no prerequisites yet.",
    )


class CurriculumDAGNodeContract(BaseModel):
    """Pass 2 (Two-Pass Model-First): рёбра одной ноды (её prerequisites) —
    node_id должен совпадать с одним из уже зафиксированных в Pass 1."""

    node_id: str = Field(..., description="Must match a node_id from Pass 1's node list.")
    prerequisites: list[str] = Field(
        default_factory=list,
        max_length=24,
        description=(
            "node_id of direct prerequisites (parents) for this node. Every node "
            "in the graph MUST have in_degree + out_degree >= 1 — a node with an "
            "empty prerequisites list here MUST be referenced as a prerequisite by "
            "at least one OTHER node in this same list, or it becomes an isolated "
            "orphan and the whole payload is rejected."
        ),
    )
    # RU: связность по всему графу (referential integrity / orphan /
    # weak connectivity) проверяет model_validator ниже, не это поле само по
    # себе — Pydantic не может выразить межобъектный инвариант в Field(...).


class CurriculumDAGContract(BaseModel):
    """Pass 2 payload: полный набор рёбер для зафиксированного в Pass 1
    списка нод. model_validator кидает ValueError на любое нарушение
    связности — тот же путь, что и обычная Pydantic ValidationError, уходит
    в существующий Repair Feedback Loop (см. _parse_structured в
    gemini_stateless.py: ValidationError оборачивается в RuntimeError,
    вызывающий код ловит его и повторяет запрос с текстом ошибки)."""

    nodes: list[CurriculumDAGNodeContract] = Field(
        default_factory=list,
        description="One entry per Pass-1 node_id, each carrying its prerequisites.",
    )

    @model_validator(mode="after")
    def _validate_topology(self) -> "CurriculumDAGContract":
        ids: list[str] = [n.node_id for n in self.nodes]
        id_set = set(ids)
        if len(id_set) != len(ids):
            seen: set[str] = set()
            for nid in ids:
                if nid in seen:
                    raise ValueError(f"Duplicate node_id in Pass 2 payload: '{nid}'.")
                seen.add(nid)

        for n in self.nodes:
            for p in n.prerequisites:
                if p == n.node_id:
                    raise ValueError(
                        f"Node '{n.node_id}': self-reference in prerequisites."
                    )
                if p not in id_set:
                    raise ValueError(
                        f"Node '{n.node_id}': prerequisite '{p}' does not match "
                        "any node_id from the Pass 1 node list."
                    )

        out_degree: dict[str, int] = {nid: 0 for nid in id_set}
        for n in self.nodes:
            for p in n.prerequisites:
                out_degree[p] = out_degree.get(p, 0) + 1

        for n in self.nodes:
            if not n.prerequisites and out_degree.get(n.node_id, 0) == 0:
                raise ValueError(
                    f"Node '{n.node_id}' is isolated (orphan node with 0 edges). "
                    "Connect it as a prerequisite to advanced nodes or assign a "
                    "parent."
                )

        # RU: слабая связность — Union-Find по неориентированным рёбрам.
        parent: dict[str, str] = {nid: nid for nid in id_set}

        def _find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: str, b: str) -> None:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[ra] = rb

        for n in self.nodes:
            for p in n.prerequisites:
                _union(n.node_id, p)

        roots = {_find(nid) for nid in id_set}
        if len(roots) > 1:
            raise ValueError(
                "Graph is not weakly connected: found "
                f"{len(roots)} disconnected component(s) across {len(id_set)} "
                "nodes. Every node must belong to a single connected graph."
            )
        return self


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
