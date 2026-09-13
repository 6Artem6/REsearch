"""Pydantic schemas for LLM outputs and LangGraph state."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict


class CSAbstraction(BaseModel):
    title: str = Field(description="Название на русском")
    cs_concept: str = Field(
        description="Термин CS на русском или общепринятый (Cache Invalidation, …)"
    )
    description: str = Field(description="Объяснение на русском")


class TradeOffOption(BaseModel):
    id: int = Field(description="Уникальный ID варианта (1, 2, 3)")
    pattern_name: str = Field(description="Название паттерна на русском")
    category: str = Field(description="Классика / SOTA (Современное) / Минимализм")
    fundamental_idea: str = Field(description="Суть паттерна на русском")
    pros: List[str] = Field(description="Плюсы на русском")
    cons_and_risks: List[str] = Field(description="Риски и failure modes на русском")
    operational_cost: str = Field(
        description="Нагрузка на инфраструктуру, RAM и сложность поддержки"
    )


class DocumentSummary(BaseModel):
    title: str = Field(description="Краткий заголовок на русском")
    url: str
    executive_summary: str = Field(
        default="",
        description=(
            "Phase-2 Reduce passport prose (1–2 paragraphs); empty on legacy rows."
        ),
    )
    """ RU: итоговая синтезированная проза паспорта документа. """
    cs_concepts: List[str] = Field(
        default_factory=list, description="Концепты на русском"
    )
    key_takeaways: List[str] = Field(
        default_factory=list,
        description=(
            "Compressed synthesis takeaways (3–7), preferably tagged "
            "[SCOPE: PRINCIPLE|MECHANIC|INSTANCE]; not the atom catalog."
        ),
    )
    """ RU: сжатые выводы синтеза; полный каталог фактов — в knowledge_atoms. """
    failure_modes: List[str] = Field(
        default_factory=list, description="Failure modes на русском"
    )
    diagram_descriptions: List[str] = Field(default_factory=list)


class KnowledgeNode(BaseModel):
    """Узел иерархического графа знаний (LanceDB v0.3)."""

    id: str
    level: str = Field(description="L0_META | L1_PATTERN | L2_EVIDENCE")
    parent_id: Optional[str] = None
    content: str = ""
    source_url: Optional[str] = None


class L1PatternStub(BaseModel):
    title: str
    description: str = ""


class L0DecompositionResult(BaseModel):
    l0_summary: str = Field(description="Мета-карта задачи (L0)")
    l1_patterns: List[L1PatternStub] = Field(default_factory=list)
    search_queries: List[str] = Field(
        default_factory=list, description="Запросы для discovery"
    )


class L2EvidenceItem(BaseModel):
    fact: str
    failure_mode: str = ""
    metric: str = ""


class L2EvidenceExtraction(BaseModel):
    evidences: List[L2EvidenceItem] = Field(default_factory=list)
    l1_title_hint: str = Field(
        default="",
        description="К какому L1-паттерну ближе всего эти факты",
    )


class ResearchEvaluation(BaseModel):
    """Re-Act: достаточно ли L2 для матрицы."""

    is_sufficient: bool = Field(
        description="Достаточно для глубокого Trade-off анализа"
    )
    missing_gaps: List[str] = Field(
        default_factory=list,
        description="Что не покрыто (failure modes, metrics, LanceDB invalidation…)",
    )
    new_search_queries: List[str] = Field(
        default_factory=list,
        description="Точечные запросы при is_sufficient=false",
    )


class DocumentMetaSummary(BaseModel):
    title: str = ""
    description: str = ""
    keywords: List[str] = Field(default_factory=list)


class TocEntry(BaseModel):
    level: int = Field(ge=1, le=6)
    title: str = ""


class CodeArtifact(BaseModel):
    language: str = ""
    context: str = Field(default="", description="Заголовок секции или окружение")
    code: str = ""


class MediaArtifact(BaseModel):
    url: str = ""
    alt: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    reason: str = Field(default="", description="Почему важно для анализа")


class DocumentStructure(BaseModel):
    source_url: str = ""
    meta_summary: DocumentMetaSummary = Field(default_factory=DocumentMetaSummary)
    abstract: str = ""
    toc: List[TocEntry] = Field(default_factory=list)
    sections: Dict[str, str] = Field(default_factory=dict)
    code_artifacts: List[CodeArtifact] = Field(default_factory=list)
    media_artifacts: List[MediaArtifact] = Field(default_factory=list)


class QueryExpansionResult(BaseModel):
    expanded_queries: List[str] = Field(
        description="10–15 точных поисковых векторов для SearchRegistry"
    )


class StructureFilterResult(BaseModel):
    is_relevant: bool = True
    reject_reason: str = ""
    selected_section_keys: List[str] = Field(default_factory=list)
    selected_code_indices: List[int] = Field(default_factory=list)
    selected_media_indices: List[int] = Field(default_factory=list)


class PreSynthesisDraft(BaseModel):
    deduplicated_summary: str = ""
    matrix_input: str = Field(
        default="",
        description="Структурированный черновик для matrix_node (Gemini)",
    )
    tags: List[str] = Field(default_factory=list)


class RouterDecision(BaseModel):
    next_node: str = Field(
        description="document_fetch_node | discovery_node | pre_synthesis_clusterizer_node"
    )
    rationale: str = ""


class DomainProfilerGeminiResult(BaseModel):
    """Ответ Gemini Profiler (v0.5) — один домен (legacy)."""

    trust_score: float = Field(ge=0.0, le=1.0)
    category: str = Field(
        description="tech_blog | academic | official_docs | documentation | e_commerce | seo_spam | seo_farm | general_news | media"
    )
    is_valid_for_research: bool = True
    reason: str = ""


class DomainProfilerBatchItem(BaseModel):
    domain: str
    trust_score: float = Field(ge=0.0, le=1.0)
    category: str = ""
    is_valid_for_research: bool = True
    reason: str = ""


class DomainProfilerBatchResult(BaseModel):
    """Пачка доменов в одном Gemini запросе."""

    domains: List[DomainProfilerBatchItem] = Field(default_factory=list)


class DomainTrustResult(BaseModel):
    domain: str
    trust_score: float = Field(ge=0.0, le=1.0)
    category: str
    reason: str = ""
    created_at: Optional[str] = None
    from_cache: bool = False
    is_valid_for_research: bool = True


class AnalysisReport(BaseModel):
    abstractions: List[CSAbstraction]
    options: List[TradeOffOption]


class EngineState(BaseModel):
    user_problem: str = Field(description="Исходная проблема от пользователя")
    context_constraints: str = Field(
        default="",
        description="Ограничения (например: Mac M1, Python, local, low latency)",
    )
    abstractions: List[CSAbstraction] = Field(default_factory=list)
    search_queries: List[str] = Field(default_factory=list)
    search_horizon_queries: Dict[str, str] = Field(
        default_factory=dict,
        description="Запросы по горизонтам: sota, infra, prod",
    )
    found_facts: List[str] = Field(default_factory=list)
    search_iterations: int = Field(default=0)
    is_facts_sufficient: bool = Field(default=False)
    found_summaries: List[DocumentSummary] = Field(default_factory=list)
    external_ai_dialogue_history: List[Dict[str, str]] = Field(default_factory=list)
    is_rag_sufficient: bool = Field(default=False)
    collected_urls: List[str] = Field(default_factory=list)
    report: Optional[AnalysisReport] = None
    selected_option_id: Optional[int] = None
    unraveled_details: Optional[str] = None
    dialogue_rolling_summary: str = Field(
        default="",
        description="Сжатая 1.5B история уточнений (Rolling Context)",
    )
    gemini_payload: str = Field(default="", description="Sandwich payload для Gemini")
    gemini_raw_response: str = Field(
        default="", description="Сырой ответ Gemini (heavy)"
    )
    pending_clarification: bool = Field(default=False)
    research_source_urls: List[str] = Field(default_factory=list)
    research_source_index: int = Field(default=0)
    research_find_rounds: int = Field(default=0)
    validated_source_count: int = Field(default=0)
    last_validator_signal: str = Field(default="")
    last_extraction: Optional[Dict[str, Any]] = None
    last_validation: Optional[Dict[str, Any]] = None
    context_corrected_once: bool = Field(
        default=False,
        description="Предохранитель: одна итерация очистки контекста перед Gemini",
    )
    is_context_optimal: bool = Field(
        default=False,
        description="1.5B: sandwich payload без шума в профиле/источниках",
    )
    is_ready_for_gemini: bool = Field(
        default=False,
        description="Payload готов для gemini_heavy_reasoning",
    )
    context_blocks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Стабильные блоки контекста (профиль MD, источники, задача)",
    )
    context_block_selections: Dict[str, bool] = Field(
        default_factory=dict,
        description="block_id → включить в финальный payload (галочки 1.5B)",
    )
    # v0.3 hierarchical research
    l0_node_id: str = Field(default="")
    l1_node_ids: List[str] = Field(default_factory=list)
    pending_urls: List[str] = Field(default_factory=list)
    explored_urls: List[str] = Field(default_factory=list)
    depth: int = Field(default=0)
    l0_summary: str = Field(default="", description="L0 мета-карта (v0.3)")
    knowledge_node_ids: List[str] = Field(default_factory=list)
    research_sufficient: bool = Field(default=False)
    last_research_gaps: List[str] = Field(default_factory=list)
    expanded_search_queries: List[str] = Field(default_factory=list)
    current_page_url: str = Field(default="")
    document_structure: Optional[Dict[str, Any]] = None
    structure_filter: Optional[Dict[str, Any]] = None
    pre_synthesis_draft: str = Field(default="")
    router_target: str = Field(default="")
    material_source_urls: List[str] = Field(
        default_factory=list,
        description="Все ссылки из discovery (архив + поиск) для повторного анализа",
    )
    discovery_cache_first: bool = Field(
        default=False,
        description="Приоритет ссылок из архива перед новым поиском",
    )


class ContextBlock(BaseModel):
    """Один фрагмент контекста для Gemini — контент не перегенерируется, только вкл/выкл."""

    block_id: str = Field(
        description="Стабильный id, например profile:hardware-ecosystem"
    )
    kind: str = Field(
        description="system | profile | source | fact | abstraction | dialogue | user_task"
    )
    title: str = Field(description="Заголовок секции или источника")
    content: str = Field(description="Готовый текст блока (markdown / выжимка)")
    always_include: bool = Field(
        default=False,
        description="user_task — всегда в payload; system оценяется, но обычно нужен",
    )
    default_include: bool = Field(
        default=True,
        description="Стартовая галочка до 1.5B (эвристики: пустой source → false)",
    )
    hints: List[str] = Field(
        default_factory=list,
        description="Подсказки для 1.5B: empty_source, artifact, likely_junk",
    )


class ContextBlockIncludeDecision(BaseModel):
    block_id: str = Field(description="id из каталога блоков")
    include: bool = Field(description="Включить блок в финальный контекст для Gemini")
    reason: str = Field(default="", description="Кратко на русском")


class ContextBlocksEvaluation(BaseModel):
    """Галочки 1.5B по каждому блоку — без переписывания текста."""

    decisions: List[ContextBlockIncludeDecision] = Field(
        description="Ровно одно решение include для каждого block_id из каталога"
    )
    is_context_optimal: bool = Field(
        description="Все нужные блоки включены, шум и мусор исключены"
    )
    rationale: str = Field(default="", description="Кратко на русском")


class ContextQualityAssessment(BaseModel):
    """Чек-лист качества Sandwich payload (1.5B)."""

    is_context_optimal: bool = Field(
        description="Контекст чистый: профиль, источники и фокус задачи достаточны"
    )
    profile_has_noise: bool = Field(
        description="Шум в профиле: периферия, часы, непрофильные проекты"
    )
    sources_have_junk: bool = Field(
        description="Мусор в источниках: пустые блоки, нерелевантные темы"
    )
    task_focus_weak: bool = Field(
        description="Не выделены целевые ограничения (RAM, tail latency, failure modes)"
    )
    rationale: str = Field(default="", description="Кратко на русском")


class RefinedGeminiPayload(BaseModel):
    """Очищенный Sandwich payload после one-step correction."""

    gemini_payload: str = Field(
        description="Полный текст: SYSTEM ROLE, PROFILE, SOURCES, TASK — без мусора"
    )


class EngineGraphState(TypedDict):
    """LangGraph state schema (required for correct merge with MemorySaver)."""

    user_problem: str
    context_constraints: str
    abstractions: NotRequired[List[Any]]
    search_queries: NotRequired[List[str]]
    search_horizon_queries: NotRequired[Dict[str, str]]
    found_facts: NotRequired[List[str]]
    search_iterations: NotRequired[int]
    is_facts_sufficient: NotRequired[bool]
    found_summaries: NotRequired[List[Any]]
    external_ai_dialogue_history: NotRequired[List[Dict[str, str]]]
    is_rag_sufficient: NotRequired[bool]
    collected_urls: NotRequired[List[str]]
    report: NotRequired[Any]
    selected_option_id: NotRequired[int]
    unraveled_details: NotRequired[str]
    dialogue_rolling_summary: NotRequired[str]
    gemini_payload: NotRequired[str]
    gemini_raw_response: NotRequired[str]
    pending_clarification: NotRequired[bool]
    research_source_urls: NotRequired[List[str]]
    research_source_index: NotRequired[int]
    research_find_rounds: NotRequired[int]
    validated_source_count: NotRequired[int]
    last_validator_signal: NotRequired[str]
    last_extraction: NotRequired[Any]
    last_validation: NotRequired[Any]
    context_corrected_once: NotRequired[bool]
    is_context_optimal: NotRequired[bool]
    is_ready_for_gemini: NotRequired[bool]
    context_blocks: NotRequired[List[Any]]
    context_block_selections: NotRequired[Dict[str, bool]]
    l0_node_id: NotRequired[str]
    l1_node_ids: NotRequired[List[str]]
    pending_urls: NotRequired[List[str]]
    explored_urls: NotRequired[List[str]]
    depth: NotRequired[int]
    knowledge_node_ids: NotRequired[List[str]]
    original_query: NotRequired[str]
    constraints: NotRequired[str]
    l0_summary: NotRequired[str]
    research_sufficient: NotRequired[bool]
    last_research_gaps: NotRequired[List[str]]
    expanded_search_queries: NotRequired[List[str]]
    current_page_url: NotRequired[str]
    document_structure: NotRequired[Any]
    structure_filter: NotRequired[Any]
    pre_synthesis_draft: NotRequired[str]
    router_target: NotRequired[str]
    material_source_urls: NotRequired[List[str]]
    discovery_cache_first: NotRequired[bool]


class GeminiSourceExtraction(BaseModel):
    """Шаг B: структурированный разбор одного источника (из ответа Gemini)."""

    source_url: str = Field(description="URL источника")
    key_engineering_findings: str = Field(description="Инженерные тезисы на русском")
    extracted_failure_modes: str = Field(description="Failure modes на русском")
    proposed_next_steps: str = Field(description="Что изучить дальше, на русском")


class GeminiSourceList(BaseModel):
    """Шаг A: целевые URL от Gemini."""

    urls: List[str] = Field(description="Прямые URL (блоги, postmortem, RFC, papers)")
    search_notes: str = Field(default="", description="Кратко: почему эти источники")


class ProfileValidationResult(BaseModel):
    """1.5B validator vs user_profile.md."""

    is_valuable: bool = Field(description="Практическая ценность для профиля")
    reason: str = Field(description="Обоснование на русском")
    actions: List[str] = Field(
        default_factory=list,
        description="save_to_lancedb | continue_deep_dive",
    )
    validator_signal: str = Field(
        description="Короткий сигнал для Gemini: VALIDATED/REJECTED + ключевые аспекты"
    )


class ClarificationAssessment(BaseModel):
    """SLM routing: нужны ли уточнения перед discovery."""

    needs_clarification: bool = Field(
        description="Критически не хватает параметров для архитектурного анализа"
    )
    clarification_question: Optional[str] = Field(
        default=None,
        description="Один конкретный вопрос к пользователю (ТТЖ/ТЗ)",
    )


class DialogueRollingSummary(BaseModel):
    summary: str = Field(description="Краткая сжатая история диалога и уточнений")


class CSAbstractionList(BaseModel):
    items: List[CSAbstraction] = Field(description="Список CS-абстракций задачи")


class ReactSearchAssessment(BaseModel):
    new_queries: List[str] = Field(
        description="Новые целевые поисковые запросы (1–3 штуки)"
    )
    simulated_facts: List[str] = Field(description="Краткие проверенные факты")
    is_facts_sufficient: bool = Field(description="Достаточно ли фактов для матрицы")


class AIDialogueEvaluation(BaseModel):
    """Оценка ответа внешнего ИИ роутером 1.5B."""

    has_sufficient_links: bool = Field(description="Достаточно прямых ссылок и фактов")
    extracted_urls: List[str] = Field(default_factory=list, description="URL из ответа")
    extracted_facts: List[str] = Field(
        default_factory=list, description="Ключевые факты"
    )
    follow_up_question: Optional[str] = Field(
        default=None,
        description="Уточняющий вопрос, если данных недостаточно",
    )


class AIDialoguePrompt(BaseModel):
    """Системный/пользовательский промпт для внешнего ИИ."""

    system_prompt: str
    user_message: str


class HorizonQuerySet(BaseModel):
    """Короткие поисковые запросы по горизонтам (без списков ключевых слов)."""

    sota: str = Field(description="SOTA: papers/survey/benchmark, до ~12 слов")
    infra: str = Field(description="Infra: deploy/observability/stack, до ~12 слов")
    prod: str = Field(description="Prod: incidents/failure modes, до ~12 слов")
