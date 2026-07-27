"""Pydantic schemas for v0.7 analytics stages L2a–L2c and chunking."""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


class ChunkExtractionItem(BaseModel):
    text: str = Field(description="Фрагмент текста (русский или термины EN)")
    concepts: List[str] = Field(default_factory=list)
    code_snippets: List[str] = Field(default_factory=list)
    p99_relevance_score: float = Field(
        ge=0.0, le=1.0, description="Релевантность tail latency/p99"
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
        description="doc_id чанков, где концепт явно обсуждается",
    )
    detail: str = Field(
        default="",
        description="Развернутое техническое описание механики/алгоритма из источников",
    )


class ConceptEdge(BaseModel):
    source: str
    target: str
    relation: str
    nuance: str = Field(
        default="",
        description="Неочевидный нюанс связи или ограничение из источников",
    )


class SourceContrast(BaseModel):
    topic: str
    approach_a: str = Field(description="Подход/механика A с деталями из источника")
    approach_b: str = Field(description="Подход/механика B с деталями из источника")
    principal_difference: str = Field(
        description="Принципиальное различие: алгоритм, структура данных, допущения"
    )
    pitfall: str = Field(
        default="", description="Подводный камень, отмеченный в источниках"
    )


class ConceptGraph(BaseModel):
    task_summary: str = Field(
        description="Развернутый исследовательский синтез задачи по источникам (не одна строка)",
    )
    research_synthesis: str = Field(
        default="",
        description="Глубокий синтез: как разные источники описывают проблему и решения",
    )
    nodes: List[ConceptNode] = Field(default_factory=list)
    edges: List[ConceptEdge] = Field(default_factory=list)
    invariants: List[str] = Field(
        default_factory=list,
        description="Инварианты и жёсткие ограничения из литературы/практики",
    )
    contrasts: List[str] = Field(
        default_factory=list,
        description="Краткие противопоставления (дополнение к cross_source_contrasts)",
    )
    cross_source_contrasts: List[SourceContrast] = Field(default_factory=list)
    engineering_pitfalls: List[str] = Field(
        default_factory=list,
        description="Неочевидные инженерные нюансы и failure modes из авторов",
    )
    theory_practice_bridges: List[str] = Field(
        default_factory=list,
        description="Сопоставление теории (papers) с практическими импликациями задачи",
    )


class ProfileGap(BaseModel):
    area: str
    risk: str
    severity: str = Field(description="low | medium | high | critical")
    mitigation_hint: str = ""
    source_basis: str = Field(
        default="",
        description="На каких идеях из ConceptGraph/источников основано",
    )


class ProfileGapMap(BaseModel):
    context_synthesis: str = Field(
        default="",
        description="Разбор условий, допущений и границ применимости теорий к задаче",
    )
    assumption_clashes: List[str] = Field(
        default_factory=list,
        description="Где допущения статей/подходов конфликтуют с реальными условиями задачи",
    )
    context_flags: List[str] = Field(
        default_factory=list,
        description="Контекстные флаги (железо, SLA, стек) — маркеры, не фильтр решений",
    )
    gaps: List[ProfileGap] = Field(default_factory=list)
    uma_risks: List[str] = Field(
        default_factory=list,
        description="Опционально: риски UMA/памяти, если релевантны задаче",
    )
    latency_risks: List[str] = Field(default_factory=list)
    sla_risks: List[str] = Field(default_factory=list)
    stack_incompatibilities: List[str] = Field(
        default_factory=list,
        description="Несовместимости стека — только если явно релевантны",
    )


TradeoffColumn = Literal["classical", "sota", "minimalist"]


class TradeoffMatrixOption(BaseModel):
    column: TradeoffColumn
    pattern_name: str
    category: str = Field(description="Классика | SOTA (Современное) | Минимализм")
    fundamental_idea: str = Field(
        description="Развернутое описание идеи с техническими деталями, не abstract-сводка",
    )
    mechanics_detail: str = Field(
        default="",
        description="Детальная механика: алгоритмы, порядок операций, структуры данных",
    )
    implementation_details: List[str] = Field(
        default_factory=list,
        description="Конкретные шаги/компоненты реализации",
    )
    data_structure_notes: str = Field(
        default="",
        description="Индексы, хеши, граф, векторное хранилище — как устроено",
    )
    pros: List[str] = Field(default_factory=list)
    cons_and_risks: List[str] = Field(
        default_factory=list,
        description="Минусы, failure modes, operational риски",
    )
    fundamental_limits: List[str] = Field(
        default_factory=list,
        description="Фундаментальные ограничения подхода (не исправить настройкой)",
    )
    applicability: str = Field(
        default="",
        description="Когда подход уместен / когда не подходит",
    )
    operational_cost: str = Field(
        default="",
        description="Операционная сложность: CPU, IO, память, человекочасы",
    )
    aligning_sources: List[str] = Field(
        default_factory=list,
        description="Какие идеи из ConceptGraph/источников поддерживают этот вариант",
    )


class TradeoffMatrixResult(BaseModel):
    options: List[TradeoffMatrixOption] = Field(
        default_factory=list,
        description="Три колонки: classical, sota, minimalist",
    )
