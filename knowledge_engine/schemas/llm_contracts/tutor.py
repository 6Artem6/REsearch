"""Node Deep-Dive / Tutor — Gemini structured contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from knowledge_engine.src.node_deep_dive.memory_schemas import (
    ConceptMasteryStatus,
    LectureExtractedConcept,
    UserIntent,
)
from knowledge_engine.src.node_deep_dive.schemas import (
    DenseMaterialOutput,
    NodeStatus,
    RichReferenceItem,
)

SuggestedNextStep = Literal["next_node", "deep_dive_optional"]

STRUCTURED_LECTURE_FIELD_RULES = (
    "Generate technical lectures matching JSON Schema contract `StructuredLectureResponse`.\n\n"
    "=== FIELD-BY-FIELD GENERATION RULES ===\n\n"
    "1. `lecture_body`:\n"
    "   - NO http/https and NO Markdown `[label](url)` in prose.\n"
    "   - RAG (RAG MATERIAL / RAG CHUNK SOURCE INDEX): cite `[R1]`, `[R2]`, … — same N as chunk line; "
    "multi-source statements: `[R1][R3]` adjacently; never replace `[R2]` with `[R1]` only; "
    "closed-world — no parametric facts/APIs absent from [RN] chunks.\n"
    "   - Course whitelist (SOURCE REGISTRY): cite `[S1]`, `[S2]` — separate from `[R*]` RAG index.\n"
    "   - MANDATORY: claims from `[RN]` end with `[RN]`; cite only R/S ids actually used.\n"
    "   - NODE MATERIALS: [diagram-N], [code-N] from [AVAILABLE NODE MATERIALS] — "
    "cite ≥1–2 times with element-level analysis (not abstract textbook).\n"
    "   - CONCEPT INTRODUCTION: first mention of each acronym/algorithm — Russian gloss in output + "
    "plain-language intuition before deep dive (no bare RRF/HNSW/BM25).\n"
    "   - DEPTH: Big-O, memory, LaTeX, code, [Diagram N] analyses.\n"
    "   - CODE: every Python snippet in lecture_body MUST use fenced blocks "
    "(```python … ```) with real line breaks and PEP8 indentation (4 spaces); "
    "in JSON, preserve newline characters as \\n inside lecture_body — do not collapse code "
    "into one line; never glue imports/defs (e.g. import hmacimport hashlib, osdef verify).\n"
    "   - DIAGRAMS: cite [Diagram N] only if N exists in DIAGRAM_CATALOG / PINNED_DIAGRAMS; "
    "never reference missing external charts/SVG.\n"
    "   - FORBIDDEN in lecture_body: meta verdicts / self-check scoreboards "
    "(«Вердикт самопроверки», «Пользователь корректно…» as assessment headers).\n\n"
    "2. `diagrams_referenced`: tags discussed in body.\n"
    "3. `referenced_diagram_id`: exact asset id from DIAGRAM_CATALOG for the panel "
    "(or null). NEVER write raw Mermaid code in any JSON field.\n"
    "4. `used_sources`: cited URLs with titles — ONLY copies from VERIFIED_EXTERNAL_SOURCES "
    "(for panel JSON; do not paste URLs into lecture_body).\n"
    "5. `next_recommended_subtopics`: exactly 3 Deep Dive topics.\n"
    "6. `extracted_concepts`: 3–5 micro-topics actually covered in lecture_body "
    "(snake_case key + summary ≤600 chars in Russian); do not duplicate next_recommended_subtopics.\n"
    "7. `introduced_terms`: terms/acronyms you FIRST introduced or glossed in this lecture "
    "(e.g. RRF, BM25, HNSW); exclude terms from ALREADY_EXPLAINED_TERMS in payload.\n"
    "OUTPUT MUST BE VALID JSON CONFORMING TO THE PYDANTIC SCHEMA.\n"
)
"""
RU (пояснение): правила полей StructuredLectureResponse для system prompt лекции.
"""


class VerifiedSourceReference(BaseModel):
    title: str = Field(
        ..., min_length=2, max_length=400, description="Title из VERIFIED block"
    )
    url: str = Field(
        ..., min_length=8, max_length=2000, description="URL только из verified"
    )


class StructuredLectureResponse(BaseModel):
    lecture_body: str = Field(
        ...,
        max_length=24_000,
        description=(
            "Markdown лекция: LaTeX, код в ```python fences с переносами строк, "
            "[Diagram N], сноски [S1] без URL в тексте; "
            "обязательная экскурсия по [AVAILABLE NODE MATERIALS] при наличии кода/схем"
        ),
    )
    diagrams_referenced: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="Diagram tags разобранные в lecture_body",
    )
    used_sources: list[VerifiedSourceReference] = Field(
        default_factory=list,
        max_length=8,
        description="Процитированные verified URL",
    )
    next_recommended_subtopics: list[str] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="3 узкие темы для Deep Dive",
    )
    extracted_concepts: list[LectureExtractedConcept] = Field(
        default_factory=list,
        max_length=5,
        description="3–5 ключевых микро-тем, фактически разобранных в lecture_body",
    )
    introduced_terms: list[str] = Field(
        default_factory=list,
        max_length=24,
        description=(
            "Термины/аббревиатуры, впервые расшифрованные в lecture_body в этой реплике"
        ),
    )
    summary: str = Field(
        default="", max_length=12_000, description="Выжимка для панели"
    )
    referenced_diagram_id: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "ID of the diagram chosen from the provided node diagram catalog. "
            "Do NOT write raw Mermaid code here."
        ),
    )
    code_snippets: list[str] = Field(
        default_factory=list,
        max_length=4,
        description="Code blocks",
    )
    checkpoint_prompt: str = Field(
        default="",
        max_length=2000,
        description="Один вопрос самопроверки (контент для пользователя, не meta-verdict)",
    )
    bridge_to_next: str = Field(
        default="",
        max_length=2000,
        description="Следующий шаг без риторических вопросов",
    )


class IntroAssessmentContract(BaseModel):
    tutor_message: str = Field(
        ...,
        max_length=2000,
        description=(
            "Вводный контекст + один практический вопрос или мини-кейс (≤400 символов вопроса). "
            "Без «голых» аббревиатур; вопрос на интуицию/проблему, не на настройку "
            "незнакомого алгоритма (Concept Introduction для intro)."
        ),
    )
    node_status: NodeStatus = Field(
        default="in_progress",
        description="Статус ноды после intro",
    )


class DeepDiveTutorContract(BaseModel):
    node_status: NodeStatus = Field(
        default="in_progress",
        description="Прогресс по ноде",
    )
    summary: str = Field(
        default="",
        max_length=12_000,
        description="Выжимка для панели Materials",
    )
    referenced_diagram_id: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "ID of the diagram chosen from the provided node diagram catalog. "
            "Do NOT write raw Mermaid code here."
        ),
    )
    references: list[RichReferenceItem] = Field(
        default_factory=list,
        max_length=6,
        description=(
            "Карточки источников для панели: только строки из SOURCE REGISTRY в payload "
            "(asset_id S1…, url/title копировать дословно). Пустой список, если не цитировал."
        ),
    )
    feedback_on_answer: str = Field(
        default="",
        max_length=4000,
        description="Короткая реакция и разбор предыдущего ответа пользователя",
    )
    technical_explanation: str = Field(
        default="",
        max_length=10_000,
        description=(
            "Сухой инженерный разбор темы: без «?», без follow-up и без анонса следующих подтем"
        ),
    )
    follow_up_question: str = Field(
        default="",
        max_length=2000,
        description="Подводка и вопрос по следующей подтеме (с «?»)",
    )
    question_sub_concept_id: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Точный id подконцепта из concept_map, по которому задан follow_up_question. "
            "null если вопрос не задаётся."
        ),
    )
    new_gap_to_record: str | None = Field(
        default=None,
        max_length=2000,
        description="Пробел для LightRAG если выявлен",
    )
    introduced_terms: list[str] = Field(
        default_factory=list,
        max_length=16,
        description=(
            "Термины/аббревиатуры, впервые расшифрованные в реплике (dialogue поля) в этой реплике"
        ),
    )
    verified_sub_concept_ids: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="ID sub_concepts, подтверждённых в этом ходе (для реестра)",
    )
    ready_for_transition: bool = Field(
        default=False,
        description=(
            "True если все ключевые sub_concepts/фаза закрыты: без нового тех. вопроса, "
            "только итог и выбор пользователя (next_node / deep_dive_optional)."
        ),
    )
    suggested_next_step: SuggestedNextStep | None = Field(
        default=None,
        description=(
            "При ready_for_transition=true: next_node — следующая нода/лекция; "
            "deep_dive_optional — предложена опциональная углублённая подтема."
        ),
    )
    quick_replies: list[str] = Field(
        default_factory=list,
        max_length=4,
        description=(
            "Короткие chip-labels для UI Quick Replies. "
            "На PASSED_WITH_GLOSS fork при закрытии ноды: ровно "
            "«Хочу Gloss», «Дожать MECH», «Идем дальше». Иначе []."
        ),
    )


class ConceptUpdateContract(BaseModel):
    concept: str = Field(..., min_length=1, max_length=400)
    status: ConceptMasteryStatus | None = None
    evidence: str = Field(default="", max_length=2000)
    mastery_score: int | None = Field(default=None, ge=0, le=100)


class StepAnalysisContract(BaseModel):
    intent: UserIntent = Field(
        default="ANSWER",
        description="INTENT_EXPLAIN только при явной лекции; ANSWER для диалога",
    )
    concept_updates: list[ConceptUpdateContract] = Field(
        default_factory=list,
        max_length=12,
        description="Обновления mastery по концептам",
    )
    critical_gap: str | None = Field(
        default=None,
        max_length=2000,
        description="Критический пробел если есть",
    )


class SubConceptStatusUpdate(BaseModel):
    """
    Layer fact-extract from one user answer (no mastery verdict).

    Threshold Engine (Python) OR-merges flags and sets VERIFIED/PARTIAL.
    """

    id: str = Field(..., min_length=2, max_length=64)
    why_passed: bool = Field(
        default=False,
        description="WHY: concept / problem / motivation present in THIS answer",
    )
    how_passed: bool = Field(
        default=False,
        description="HOW: architecture / invariants / role split in THIS answer",
    )
    mechanic_passed: bool = Field(
        default=False,
        description="MECHANIC: math / algorithm / code detail in THIS answer",
    )
    evidence: str = Field(
        default="",
        max_length=2000,
        description="Short digest of what the user demonstrated this turn",
    )
    focus_hint: str = Field(
        default="",
        max_length=500,
        description=(
            "Optional note on the weakest missing layer in THIS answer "
            "(Python may override from threshold)"
        ),
    )
    # Legacy optional — ignored by Threshold Engine (kept for schema soft-compat).
    status: Literal["VERIFIED", "PARTIAL", "GAP", "UNCHECKED"] | None = Field(
        default=None,
        description="Deprecated: Python Threshold Engine owns status",
    )


class SubConceptGapEvalContract(BaseModel):
    updates: list[SubConceptStatusUpdate] = Field(
        default_factory=list,
        max_length=1,
        description=(
            "Exactly 0–1 layer update for active_question_sub_concept_id "
            "(id must match evaluation_target); booleans only — no mastery verdict"
        ),
    )


class NodeExplainContract(BaseModel):
    explanation: str = Field(
        ...,
        min_length=1,
        max_length=12000,
        description="Markdown: деталь из источника, коротко для инженера",
    )
    cited_source_ids: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="ID из выделения: R6, R7 (RAG chunks) и при необходимости S1 (registry)",
    )


class DialogueFactManifestContract(BaseModel):
    agreed_concepts: list[str] = Field(
        default_factory=list,
        max_length=24,
        description="Принятые концепты/механики",
    )
    rejected_options: list[str] = Field(
        default_factory=list,
        max_length=16,
        description="Отвергнутые варианты",
    )
    open_bottlenecks: list[str] = Field(
        default_factory=list,
        max_length=16,
        description="Открытые bottlenecks (latency, RAM, index)",
    )
    stack_mentions: list[str] = Field(
        default_factory=list,
        max_length=24,
        description="Конкретные технологии из реплики",
    )
    current_subtopic: str = Field(
        default="",
        max_length=400,
        description="Активная подтема одной строкой",
    )


def structured_lecture_to_dense(
    out: StructuredLectureResponse,
    *,
    allowed_urls: set[str] | None = None,
) -> DenseMaterialOutput:
    from knowledge_engine.utils.link_sanitizer import normalize_lecture_url

    allowed = {
        normalize_lecture_url(u) for u in (allowed_urls or set()) if (u or "").strip()
    }
    refs: list[RichReferenceItem] = []
    for i, src in enumerate(out.used_sources or []):
        url = (src.url or "").strip()
        if len(url) < 8:
            continue
        key = normalize_lecture_url(url)
        if allowed and key not in allowed:
            continue
        from knowledge_engine.services.node_source_registry import (
            is_disallowed_source_url,
        )

        if is_disallowed_source_url(url):
            continue
        title = (src.title or url).strip()
        refs.append(
            RichReferenceItem(
                asset_id=f"ref-{i + 1}",
                source_name=title[:300],
                url=url,
                title=title[:400],
            )
        )
    snippets: list[str] = []
    from knowledge_engine.src.node_deep_dive.code_snippet_heuristic import (
        filter_code_snippets,
    )

    snippets = filter_code_snippets(out.code_snippets or [])
    ref_id = (out.referenced_diagram_id or "").strip() or None
    return DenseMaterialOutput(
        lecture_body=(out.lecture_body or "").strip(),
        summary=(out.summary or "").strip(),
        referenced_diagram_id=ref_id,
        references=refs[:6],
        code_snippets=snippets[:4],
        bridge_to_next=(out.bridge_to_next or "").strip(),
        checkpoint_prompt=(out.checkpoint_prompt or "").strip(),
        extracted_concepts=list(out.extracted_concepts or [])[:5],
        introduced_terms=list(out.introduced_terms or [])[:24],
    )
