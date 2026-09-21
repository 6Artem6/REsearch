"""Pydantic schemas for v0.7 analytics stages L2a–L2c and chunking."""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


class ChunkExtractionItem(BaseModel):
    text: str = Field(description="Text fragment (Russian, or EN for terms)")
    # RU: фрагмент текста (русский, или EN для терминов).
    concepts: List[str] = Field(default_factory=list)
    code_snippets: List[str] = Field(default_factory=list)
    p99_relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Relevance to tail latency/p99",
        # RU: релевантность tail latency/p99.
    )


class ChunkExtractionResult(BaseModel):
    chunks: List[ChunkExtractionItem] = Field(default_factory=list)


class ConceptNode(BaseModel):
    id: str
    label: str
    kind: str = Field(
        description="concept | invariant | constraint | mechanism | algorithm"
    )
    source_doc_ids: List[str] = Field(
        default_factory=list,
        description="doc_id of chunks where the concept is explicitly discussed",
        # RU: doc_id чанков, где концепт явно обсуждается.
    )
    detail: str = Field(
        default="",
        description="In-depth technical description of the mechanic/algorithm from sources",
        # RU: развёрнутое техническое описание механики/алгоритма из источников.
    )


class ConceptEdge(BaseModel):
    source: str
    target: str
    relation: str
    nuance: str = Field(
        default="",
        description="Non-obvious nuance or constraint of the relation, from sources",
        # RU: неочевидный нюанс связи или ограничение из источников.
    )


class SourceContrast(BaseModel):
    topic: str
    approach_a: str = Field(
        description="Approach/mechanic A with details from the source"
        # RU: подход/механика A с деталями из источника.
    )
    approach_b: str = Field(
        description="Approach/mechanic B with details from the source"
        # RU: подход/механика B с деталями из источника.
    )
    principal_difference: str = Field(
        description="Principal difference: algorithm, data structure, assumptions"
        # RU: принципиальное различие: алгоритм, структура данных, допущения.
    )
    pitfall: str = Field(
        default="",
        description="Pitfall noted in the sources",
        # RU: подводный камень, отмеченный в источниках.
    )


class ConceptGraph(BaseModel):
    task_summary: str = Field(
        description=(
            "In-depth research synthesis of the task from sources "
            "(not a single line)"
        ),
        # RU: развёрнутый исследовательский синтез задачи по источникам
        # (не одна строка).
    )
    research_synthesis: str = Field(
        default="",
        description=(
            "Deep synthesis: how different sources describe the problem "
            "and solutions"
        ),
        # RU: глубокий синтез: как разные источники описывают проблему и решения.
    )
    nodes: List[ConceptNode] = Field(default_factory=list)
    edges: List[ConceptEdge] = Field(default_factory=list)
    invariants: List[str] = Field(
        default_factory=list,
        description="Invariants and hard constraints from literature/practice",
        # RU: инварианты и жёсткие ограничения из литературы/практики.
    )
    contrasts: List[str] = Field(
        default_factory=list,
        description="Brief contrasts (complements cross_source_contrasts)",
        # RU: краткие противопоставления (дополнение к cross_source_contrasts).
    )
    cross_source_contrasts: List[SourceContrast] = Field(default_factory=list)
    engineering_pitfalls: List[str] = Field(
        default_factory=list,
        description="Non-obvious engineering nuances and failure modes from the authors",
        # RU: неочевидные инженерные нюансы и failure modes от авторов.
    )
    theory_practice_bridges: List[str] = Field(
        default_factory=list,
        description="Mapping theory (papers) to the task's practical implications",
        # RU: сопоставление теории (papers) с практическими импликациями задачи.
    )


class ProfileGap(BaseModel):
    area: str
    risk: str
    severity: str = Field(description="low | medium | high | critical")
    mitigation_hint: str = ""
    source_basis: str = Field(
        default="",
        description="Which ideas from ConceptGraph/sources this is based on",
        # RU: на каких идеях из ConceptGraph/источников основано.
    )


class ProfileGapMap(BaseModel):
    context_synthesis: str = Field(
        default="",
        description=(
            "Breakdown of the task's conditions, assumptions, and the "
            "theories' applicability boundaries"
        ),
        # RU: разбор условий, допущений и границ применимости теорий к задаче.
    )
    assumption_clashes: List[str] = Field(
        default_factory=list,
        description=(
            "Where article/approach assumptions conflict with the task's "
            "real conditions"
        ),
        # RU: где допущения статей/подходов конфликтуют с реальными
        # условиями задачи.
    )
    context_flags: List[str] = Field(
        default_factory=list,
        description=(
            "Context flags (hardware, SLA, stack) — markers, not a "
            "solution filter"
        ),
        # RU: контекстные флаги (железо, SLA, стек) — маркеры, не фильтр решений.
    )
    gaps: List[ProfileGap] = Field(default_factory=list)
    uma_risks: List[str] = Field(
        default_factory=list,
        description="Optional: UMA/memory risks, if relevant to the task",
        # RU: опционально: риски UMA/памяти, если релевантны задаче.
    )
    latency_risks: List[str] = Field(default_factory=list)
    sla_risks: List[str] = Field(default_factory=list)
    stack_incompatibilities: List[str] = Field(
        default_factory=list,
        description="Stack incompatibilities — only if clearly relevant",
        # RU: несовместимости стека — только если явно релевантны.
    )


TradeoffColumn = Literal["classical", "sota", "minimalist"]


class TradeoffMatrixOption(BaseModel):
    column: TradeoffColumn
    pattern_name: str
    category: str = Field(description="Classical | SOTA (state of the art) | Minimalist")
    # RU: Классика | SOTA (современное) | Минимализм.
    fundamental_idea: str = Field(
        description=(
            "In-depth description of the idea with technical detail, not "
            "an abstract-level summary"
        ),
        # RU: развёрнутое описание идеи с техническими деталями, не
        # abstract-сводка.
    )
    mechanics_detail: str = Field(
        default="",
        description=(
            "Detailed mechanics: algorithms, order of operations, data structures"
        ),
        # RU: детальная механика: алгоритмы, порядок операций, структуры данных.
    )
    implementation_details: List[str] = Field(
        default_factory=list,
        description="Concrete implementation steps/components",
        # RU: конкретные шаги/компоненты реализации.
    )
    data_structure_notes: str = Field(
        default="",
        description="Indexes, hashes, graph, vector store — how it is built",
        # RU: индексы, хеши, граф, векторное хранилище — как устроено.
    )
    pros: List[str] = Field(default_factory=list)
    cons_and_risks: List[str] = Field(
        default_factory=list,
        description="Downsides, failure modes, operational risks",
        # RU: минусы, failure modes, операционные риски.
    )
    fundamental_limits: List[str] = Field(
        default_factory=list,
        description="Fundamental limits of the approach (not fixable by tuning)",
        # RU: фундаментальные ограничения подхода (не исправить настройкой).
    )
    applicability: str = Field(
        default="",
        description="When the approach fits / when it doesn't",
        # RU: когда подход уместен / когда не подходит.
    )
    operational_cost: str = Field(
        default="",
        description="Operational complexity: CPU, IO, memory, engineer-hours",
        # RU: операционная сложность: CPU, IO, память, человекочасы.
    )
    aligning_sources: List[str] = Field(
        default_factory=list,
        description="Which ideas from ConceptGraph/sources support this option",
        # RU: какие идеи из ConceptGraph/источников поддерживают этот вариант.
    )


class TradeoffMatrixResult(BaseModel):
    options: List[TradeoffMatrixOption] = Field(
        default_factory=list,
        description="Three columns: classical, sota, minimalist",
        # RU: три колонки: classical, sota, minimalist.
    )
