"""Node Deep-Dive / Tutor — Gemini structured contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

from knowledge_engine.src.domains.grounding.drill_schemas import (
    AnswerAccuracyGrade,
    TechnicalConceptAudit,
    audit_feedback_text,
    validate_grade_matches_errors,
)
from knowledge_engine.src.domains.grounding.memory_schemas import (
    ConceptMasteryStatus,
    LectureExtractedConcept,
)
from knowledge_engine.src.domains.grounding.schemas import (
    DenseMaterialOutput,
    NodeStatus,
    RichReferenceItem,
)
from knowledge_engine.src.domains.grounding.tutor_field_limits import (
    PROMPT_FOLLOW_UP_MAX_CHARS,
    SCHEMA_BRIDGE_TO_NEXT_MAX,
    SCHEMA_BULLET_SUMMARY_MAX_ITEMS,
    SCHEMA_EXECUTION_PLAN_MAX,
    SCHEMA_FOLLOW_UP_QUESTION_MAX,
    SCHEMA_LECTURE_BODY_MAX,
    SCHEMA_NEW_GAP_MAX,
    SCHEMA_SUMMARY_MAX,
    SCHEMA_TECHNICAL_EXPLANATION_MAX,
    SCHEMA_TUTOR_MESSAGE_MAX,
)
from knowledge_engine.src.domains.grounding.tutor_field_limits import (
    truncate_bullet_summary as _truncate_bullet_summary,
)


SuggestedNextStep = Literal["next_node", "deep_dive_optional"]

STRUCTURED_LECTURE_FIELD_RULES = (
    "Generate a dense technical lecture.\n\n"
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
    "(«Вердикт самопроверки», «Пользователь корректно…» as assessment headers); "
    "closing quiz questions; «?» in the last paragraph; "
    "'Самопроверка' / 'Вопрос:' self-check headers. "
    "lecture_body is theory, code, and architecture only.\n"
    "CRITICAL NEGATIVE CONSTRAINT: Do NOT include any closing questions, self-check "
    "queries, or 'Самопроверка:' headers inside lecture_body. The lecture_body MUST "
    "contain pure educational content only. The checkpoint question belongs EXCLUSIVELY "
    "in follow_up_question.\n\n"
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
    "8. `follow_up_question`: the ONLY field for ONE technical self-check question "
    "(must contain «?»; PART 1 lecture_body must end without «?» / without a quiz); "
    "target ≤400 characters. Every scored criterion MUST be named here and "
    "introduced in lecture_body first.\n"
)
"""
RU (пояснение): правила полей StructuredLectureResponse для system prompt лекции.
"""


class VerifiedSourceReference(BaseModel):
    title: str = Field(
        ..., min_length=2, max_length=400, description="Title from the VERIFIED block"
        # RU: title из блока VERIFIED.
    )
    url: str = Field(
        ..., min_length=8, max_length=2000, description="URL from verified sources only"
        # RU: URL только из verified.
    )


class _DenseLectureFieldsBase(BaseModel):
    """Поля, общие для lecture_self_check и Topic Q&A dense-ответов.

    follow_up_question сознательно НЕ здесь: он существует только в
    StructuredLectureResponse (lecture_self_check). Topic Q&A использует
    TopicQnaLectureResponse — без этого поля в JSON Schema, которую видит
    Gemini structured output, так что self-check вопрос физически
    невозможно записать, а не просто "не рекомендуется текстом". Раньше
    follow_up_question был общим полем с условным Field(description=...)
    ("оставь пустым для Topic Q&A") — Gemini всё равно заполнял его вопреки
    инструкции (подтверждено live-тестом на ноде sql_cte):
    описание поля — не менее сильный канал для модели, чем system prompt,
    и условность в нём так же ненадёжна. Известный инвариант (это поле
    вообще не существует в этом режиме) должен быть закодирован в схеме,
    а не в тексте инструкции."""

    # --- 1. Message (User-Facing Text, in reading order) ---
    lecture_body: str = Field(
        ...,
        max_length=SCHEMA_LECTURE_BODY_MAX,
        description=(
            "Markdown theory only: LaTeX, ```python fences with real newlines, "
            "[Diagram N], [S1]/[R1] citations without URLs; node-materials tour when "
            "code/diagrams exist. FORBIDDEN: closing self-check questions, a final «?» "
            "paragraph, or 'Самопроверка' / 'Вопрос:' headers. Checkpoint belongs "
            "exclusively in follow_up_question. Explains the mechanic/physics of HOW — "
            "never states the consequence/trade-off/cost that follow_up_question will "
            "ask about; the user must derive that conclusion themselves. May use 1-2 "
            "short rhetorical micro-questions answered in the SAME paragraph to build "
            "the causal chain (e.g. 'Can it just write directly? No, because...') — "
            "these are not the checkpoint and must not be the final paragraph."
        ),
        # RU: теоретический блок без контрольного вопроса и без заголовков
        # самопроверки; описывает механику/физику HOW, никогда не
        # формулирует следствие/цену, которую спросит follow_up_question —
        # вывод пользователь делает сам. Можно 1-2 коротких риторических
        # микро-вопроса, отвеченных в том же абзаце.
    )
    message_bullet_summary: list[str] = Field(
        default_factory=list,
        max_length=SCHEMA_BULLET_SUMMARY_MAX_ITEMS,
        description=(
            "Up to 5 short key theses (claims) of NEW content, SOURCED "
            "EXCLUSIVELY from lecture_body, with [S*]/[R*] citations, for "
            "compressed history. IGNORE `summary` and `bridge_to_next` "
            "entirely as sources — never paraphrase or reword their "
            "evaluation/transition framing into a thesis here. Do NOT "
            "include meta-statements about answer evaluation or subtopic "
            "credit status (e.g. 'subtopic passed', 'moving to the next "
            "topic') even if such a phrase opens lecture_body itself — "
            "that status is already conveyed separately via "
            "verified_sub_concept_ids and the progress dashboard. Do NOT "
            "write recap/frame phrases like 'discussed X' or 'covered Y' — "
            "state the fact ITSELF, not that it was discussed (bad: "
            "'Обсудили MergeTree'; good: 'MergeTree хранит части "
            "независимо [R1]'). Content claims only. The last thesis, if "
            "present, records the question just asked."
        ),
        # RU: до 5 кратких тезисов НОВОГО материала — ИСКЛЮЧИТЕЛЬНО из
        # lecture_body; поля summary и bridge_to_next как источник
        # ИГНОРИРОВАТЬ полностью (их формулировки про зачёт/переход не
        # пересказывать сюда, даже если похожая фраза открывает сам
        # lecture_body). Без рекап-фреймов вида «обсудили X».
    )
    used_sources: list[VerifiedSourceReference] = Field(
        default_factory=list,
        max_length=8,
        description="Cited verified URLs",
        # RU: процитированные verified URL.
    )

    # --- 2. Additional materials & context ---
    summary: str = Field(
        default="",
        max_length=SCHEMA_SUMMARY_MAX,
        description="Panel digest",
        # RU: выжимка для панели.
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
    introduced_terms: list[str] = Field(
        default_factory=list,
        max_length=24,
        description=(
            "Terms/acronyms first spelled out in lecture_body in this reply"
        ),
        # RU: термины/аббревиатуры, впервые расшифрованные в lecture_body
        # в этой реплике.
    )
    bridge_to_next: str = Field(
        default="",
        max_length=SCHEMA_BRIDGE_TO_NEXT_MAX,
        description="Next step, without rhetorical questions",
        # RU: следующий шаг без риторических вопросов.
    )

    # --- 3. System & RAG fields (LightRAG, FSM, knowledge graph) ---
    diagrams_referenced: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="Diagram tags discussed in lecture_body",
        # RU: теги диаграмм, разобранные в lecture_body.
    )
    next_recommended_subtopics: list[str] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="3 narrow topics for Deep Dive",
        # RU: 3 узкие темы для Deep Dive.
    )
    extracted_concepts: list[LectureExtractedConcept] = Field(
        default_factory=list,
        max_length=5,
        description="3-5 key micro-topics actually covered in lecture_body",
        # RU: 3-5 ключевых микро-тем, фактически разобранных в lecture_body.
    )

    @field_validator("message_bullet_summary", mode="before")
    @classmethod
    def _clip_message_bullet_summary(cls, v: list[str]) -> list[str]:
        return _truncate_bullet_summary(v)


class StructuredLectureResponse(BaseModel):
    """dense_material for lecture_self_check — includes follow_up_question.

    Deliberately NOT ``_DenseLectureFieldsBase`` + appended field: Pydantic
    v2 always appends subclass-added fields AFTER every inherited field,
    which would put `follow_up_question` (displayed) after the base's
    system-only fields, breaking the displayed-before-system invariant. All
    fields below are therefore re-declared in full — kept in sync with
    `_DenseLectureFieldsBase` by
    test_tutor_execution_plan.py::test_lecture_plan_field_set_matches_structured_lecture_response.
    """

    # --- 1. Message (User-Facing Text, in reading order) ---
    lecture_body: str = Field(
        ...,
        max_length=SCHEMA_LECTURE_BODY_MAX,
        description=(
            "Markdown theory only: LaTeX, ```python fences with real newlines, "
            "[Diagram N], [S1]/[R1] citations without URLs; node-materials tour when "
            "code/diagrams exist. FORBIDDEN: closing self-check questions, a final «?» "
            "paragraph, or 'Самопроверка' / 'Вопрос:' headers. Checkpoint belongs "
            "exclusively in follow_up_question. Explains the mechanic/physics of HOW — "
            "never states the consequence/trade-off/cost that follow_up_question will "
            "ask about; the user must derive that conclusion themselves. May use 1-2 "
            "short rhetorical micro-questions answered in the SAME paragraph to build "
            "the causal chain (e.g. 'Can it just write directly? No, because...') — "
            "these are not the checkpoint and must not be the final paragraph."
        ),
        # RU: теоретический блок без контрольного вопроса и без заголовков
        # самопроверки; описывает механику/физику HOW, никогда не
        # формулирует следствие/цену, которую спросит follow_up_question —
        # вывод пользователь делает сам. Можно 1-2 коротких риторических
        # микро-вопроса, отвеченных в том же абзаце.
    )
    follow_up_question: str = Field(
        default="",
        max_length=SCHEMA_FOLLOW_UP_QUESTION_MAX,
        description=(
            "The ONLY field for the single self-check question "
            "(must contain «?»). Do not duplicate it in lecture_body. "
            "Asks about a CONSEQUENCE/TRADE-OFF/COST the user must derive "
            "from lecture_body — never a fact already stated as a "
            "conclusion there. Every scored criterion MUST be named here "
            "and introduced in lecture_body first (never only here). "
            "FORBIDDEN: a surface question whose hidden rubric is a "
            "deeper unasked layer."
        ),
        # RU: спрашивает про следствие/компромисс/цену, которую пользователь
        # выводит сам из lecture_body — не факт, уже сформулированный там
        # как вывод; каждый критерий должен быть назван И введён в
        # lecture_body, не только здесь.
    )
    message_bullet_summary: list[str] = Field(
        default_factory=list,
        max_length=SCHEMA_BULLET_SUMMARY_MAX_ITEMS,
        description=(
            "Up to 5 short key theses (claims) of NEW content, SOURCED "
            "EXCLUSIVELY from lecture_body, with [S*]/[R*] citations, for "
            "compressed history. IGNORE `summary` and `bridge_to_next` "
            "entirely as sources — never paraphrase or reword their "
            "evaluation/transition framing into a thesis here. Do NOT "
            "include meta-statements about answer evaluation or subtopic "
            "credit status (e.g. 'subtopic passed', 'moving to the next "
            "topic') even if such a phrase opens lecture_body itself — "
            "that status is already conveyed separately via "
            "verified_sub_concept_ids and the progress dashboard. Do NOT "
            "write recap/frame phrases like 'discussed X' or 'covered Y' — "
            "state the fact ITSELF, not that it was discussed (bad: "
            "'Обсудили MergeTree'; good: 'MergeTree хранит части "
            "независимо [R1]'). Content claims only. The last thesis, if "
            "present, records the question just asked."
        ),
        # RU: до 5 кратких тезисов НОВОГО материала — ИСКЛЮЧИТЕЛЬНО из
        # lecture_body; поля summary и bridge_to_next как источник
        # ИГНОРИРОВАТЬ полностью (их формулировки про зачёт/переход не
        # пересказывать сюда, даже если похожая фраза открывает сам
        # lecture_body). Без рекап-фреймов вида «обсудили X».
    )
    used_sources: list[VerifiedSourceReference] = Field(
        default_factory=list,
        max_length=8,
        description="Cited verified URLs",
        # RU: процитированные verified URL.
    )

    # --- 2. Additional materials & context ---
    summary: str = Field(
        default="",
        max_length=SCHEMA_SUMMARY_MAX,
        description="Panel digest",
        # RU: выжимка для панели.
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
    introduced_terms: list[str] = Field(
        default_factory=list,
        max_length=24,
        description=(
            "Terms/acronyms first spelled out in lecture_body in this reply"
        ),
        # RU: термины/аббревиатуры, впервые расшифрованные в lecture_body
        # в этой реплике.
    )
    bridge_to_next: str = Field(
        default="",
        max_length=SCHEMA_BRIDGE_TO_NEXT_MAX,
        description="Next step, without rhetorical questions",
        # RU: следующий шаг без риторических вопросов.
    )

    # --- 3. System & RAG fields (LightRAG, FSM, knowledge graph) ---
    diagrams_referenced: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="Diagram tags discussed in lecture_body",
        # RU: теги диаграмм, разобранные в lecture_body.
    )
    next_recommended_subtopics: list[str] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="3 narrow topics for Deep Dive",
        # RU: 3 узкие темы для Deep Dive.
    )
    extracted_concepts: list[LectureExtractedConcept] = Field(
        default_factory=list,
        max_length=5,
        description="3-5 key micro-topics actually covered in lecture_body",
        # RU: 3-5 ключевых микро-тем, фактически разобранных в lecture_body.
    )

    @field_validator("message_bullet_summary", mode="before")
    @classmethod
    def _clip_message_bullet_summary(cls, v: list[str]) -> list[str]:
        return _truncate_bullet_summary(v)


class TopicQnaLectureResponse(_DenseLectureFieldsBase):
    """dense_material для interaction_axis="topic_qna" (expert-consultant).

    Идентична StructuredLectureResponse минус follow_up_question — само
    поле отсутствует в схеме, так что self-check/follow-up вопрос
    структурно невозможен, а не просто запрещён текстом промпта."""


class StructuredLectureResponseWithPlanContract(BaseModel):
    """CoT Execution Plan variant of ``StructuredLectureResponse`` — gated by
    ``ENABLE_TUTOR_EXECUTION_PLAN`` (default off), see
    node_content_generator.py::generate_dense_material.

    Same reasoning as DeepDiveTutorWithPlanContract above (not a subclass of
    StructuredLectureResponse — Pydantic v2 always appends subclass-added
    fields last, which would put execution_plan after lecture_body, not
    before it). Fields re-declared in full, kept in sync with
    _DenseLectureFieldsBase/StructuredLectureResponse by
    test_tutor_execution_plan.py::test_lecture_plan_field_set_matches_structured_lecture_response.
    """

    execution_plan: str = Field(
        ...,
        max_length=SCHEMA_EXECUTION_PLAN_MAX,
        description=(
            "DYNAMIC CAUSAL GRAPH (graph TD) FOR LIGHTWEIGHT MODEL. Generated "
            "FIRST, before any other field.\n"
            "Build a logical chain adapting to the specific topic (3 to 6 "
            "nodes max).\n\n"
            "GRAPH CONSTRUCTION INVARIANTS:\n"
            "1. GROUNDING: Move from high-level context down to concrete "
            "implementation mechanics (RAM, disk, data structures, locks, "
            "protocols).\n"
            "2. SOCRATIC HOOK: Include at least 1 node with a short "
            "micro-question/tension ('Can X handle Y? No/Why').\n"
            "3. SHADOW TARGET: Mark the ultimate unrevealed architectural "
            "consequence/trade-off as `SHADOW: <target>`.\n"
            "4. LINKING: Connect main nodes with solid arrows `-->`. Connect "
            "the approach to `SHADOW` with a dotted arrow "
            "`-. BANNED_IN_TEXT .->`.\n\n"
            "FLEXIBLE EXAMPLES:\n"
            "- Architecture topic: Concept --> TradeOff --> CodePattern -. "
            "BANNED_IN_TEXT .-> [SHADOW: Scale Bottleneck]\n"
            "- Database topic: AppendOnly --> PartsAccumulation --> "
            "SocraticPause --> MergeTree -. BANNED_IN_TEXT .-> [SHADOW: Read "
            "IO Impact]"
        ),
        # RU: короткий Mermaid-граф причинно-следственных связей (3-6 узлов),
        # генерируется ПЕРВЫМ полем — план для лёгкой модели до того, как
        # писать lecture_body. SHADOW = скрытое следствие, которое запрещено
        # раскрывать в lecture_body/follow_up_question.
    )
    lecture_body: str = Field(
        ...,
        max_length=SCHEMA_LECTURE_BODY_MAX,
        description=(
            "Markdown theory only: LaTeX, ```python fences with real newlines, "
            "[Diagram N], [S1]/[R1] citations without URLs; node-materials tour when "
            "code/diagrams exist. FORBIDDEN: closing self-check questions, a final «?» "
            "paragraph, or 'Самопроверка' / 'Вопрос:' headers. Checkpoint belongs "
            "exclusively in follow_up_question. Write a natural, engaging "
            "explanation FOLLOWING THE FLOW of execution_plan (same node "
            "order) — stop strictly before the SHADOW target; never state "
            "it, even paraphrased. May use 1-2 short rhetorical "
            "micro-questions answered in the SAME paragraph to build the "
            "causal chain (e.g. 'Can it just write directly? No, "
            "because...') — these are not the checkpoint and must not be "
            "the final paragraph."
        ),
        # RU: теоретический блок, следующий порядку узлов execution_plan;
        # останавливается строго ДО SHADOW-узла — его содержание нельзя
        # раскрывать даже перефразированно. Без контрольного вопроса и без
        # заголовков самопроверки.
    )
    follow_up_question: str = Field(
        default="",
        max_length=SCHEMA_FOLLOW_UP_QUESTION_MAX,
        description=(
            "The ONLY field for the single self-check question "
            "(must contain «?»). Do not duplicate it in lecture_body. "
            "Asks about the SHADOW target from execution_plan — the "
            "CONSEQUENCE/TRADE-OFF/COST the user must derive from "
            "lecture_body — never a fact already stated as a conclusion "
            "there. Every scored criterion MUST be named here and "
            "introduced in lecture_body first (never only here). "
            "FORBIDDEN: a surface question whose hidden rubric is a "
            "deeper unasked layer."
        ),
        # RU: спрашивает про SHADOW-цель из execution_plan — следствие/
        # компромисс/цену, которую пользователь выводит сам из lecture_body,
        # не факт, уже сформулированный там как вывод.
    )
    message_bullet_summary: list[str] = Field(
        default_factory=list,
        max_length=SCHEMA_BULLET_SUMMARY_MAX_ITEMS,
        description=(
            "Up to 5 short key theses (claims) of NEW content, SOURCED "
            "EXCLUSIVELY from lecture_body, with [S*]/[R*] citations, for "
            "compressed history. IGNORE `summary` and `bridge_to_next` "
            "entirely as sources — never paraphrase or reword their "
            "evaluation/transition framing into a thesis here. Do NOT "
            "include meta-statements about answer evaluation or subtopic "
            "credit status (e.g. 'subtopic passed', 'moving to the next "
            "topic') even if such a phrase opens lecture_body itself — "
            "that status is already conveyed separately via "
            "verified_sub_concept_ids and the progress dashboard. Do NOT "
            "write recap/frame phrases like 'discussed X' or 'covered Y' — "
            "state the fact ITSELF, not that it was discussed (bad: "
            "'Обсудили MergeTree'; good: 'MergeTree хранит части "
            "независимо [R1]'). Content claims only. The last thesis, if "
            "present, records the question just asked."
        ),
        # RU: до 5 кратких тезисов НОВОГО материала — ИСКЛЮЧИТЕЛЬНО из
        # lecture_body; поля summary и bridge_to_next как источник
        # ИГНОРИРОВАТЬ полностью. Без рекап-фреймов вида «обсудили X».
    )
    used_sources: list[VerifiedSourceReference] = Field(
        default_factory=list,
        max_length=8,
        description="Cited verified URLs",
        # RU: процитированные verified URL.
    )

    # --- 2. Additional materials & context ---
    summary: str = Field(
        default="",
        max_length=SCHEMA_SUMMARY_MAX,
        description="Panel digest",
        # RU: выжимка для панели.
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
    introduced_terms: list[str] = Field(
        default_factory=list,
        max_length=24,
        description=(
            "Terms/acronyms first spelled out in lecture_body in this reply"
        ),
        # RU: термины/аббревиатуры, впервые расшифрованные в lecture_body
        # в этой реплике.
    )
    bridge_to_next: str = Field(
        default="",
        max_length=SCHEMA_BRIDGE_TO_NEXT_MAX,
        description="Next step, without rhetorical questions",
        # RU: следующий шаг без риторических вопросов.
    )

    # --- 3. System & RAG fields (LightRAG, FSM, knowledge graph) ---
    diagrams_referenced: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="Diagram tags discussed in lecture_body",
        # RU: теги диаграмм, разобранные в lecture_body.
    )
    next_recommended_subtopics: list[str] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="3 narrow topics for Deep Dive",
        # RU: 3 узкие темы для Deep Dive.
    )
    extracted_concepts: list[LectureExtractedConcept] = Field(
        default_factory=list,
        max_length=5,
        description="3-5 key micro-topics actually covered in lecture_body",
        # RU: 3-5 ключевых микро-тем, фактически разобранных в lecture_body.
    )

    @field_validator("message_bullet_summary", mode="before")
    @classmethod
    def _clip_message_bullet_summary(cls, v: list[str]) -> list[str]:
        return _truncate_bullet_summary(v)


class IntroAssessmentContract(BaseModel):
    tutor_message: str = Field(
        ...,
        max_length=SCHEMA_TUTOR_MESSAGE_MAX,
        description=(
            "Intro context plus ONE practical question or mini-case "
            "(question ≤400 chars). No bare unexplained jargon. Stay at the "
            "asked layer. Every criterion later scored MUST appear in this "
            "text. FORBIDDEN: a surface question whose hidden rubric is a "
            "deeper unasked layer."
        ),
    )
    node_status: NodeStatus = Field(
        default="in_progress",
        description="Node status after intro",
        # RU: статус ноды после intro.
    )


class _TutorFieldsBase(BaseModel):
    """Общие поля диалогового контракта тьютора (audit + Host-owned поля).

    follow_up_question / question_sub_concept_id сознательно НЕ здесь — тот
    же принцип, что у _DenseLectureFieldsBase/_ExplainFieldsBase: в Topic
    Q&A вопрос после оценки self-check ответа запрещён категорически (пока
    пользователь сам не нажмёт «Самопроверка» снова), а условная текстовая
    инструкция для этого поля уже дважды подряд не сработала на живых
    тестах. TopicQnaTutorContract просто не добавляет поле — Gemini не
    может в него написать."""

    # --- 1. Message (User-Facing Text, in reading order) ---
    audit: TechnicalConceptAudit = Field(
        ...,
        description=(
            "Strict technical audit of the learner's previous answer, filled "
            "BEFORE any other learner-facing text. Discriminated on "
            "feedback_kind: EXACT → confirmation; NEEDS_CORRECTION → "
            "correction_breakdown only."
        ),
    )
    technical_explanation: str = Field(
        default="",
        max_length=SCHEMA_TECHNICAL_EXPLANATION_MAX,
        description=(
            "Dry engineering breakdown of the topic: no closing/follow-up "
            "question, no announcing next subtopics. Explains the "
            "mechanic/physics of HOW — never states the consequence/"
            "trade-off/cost that follow_up_question will ask about; the "
            "user must derive that conclusion themselves. May use 1-2 "
            "short rhetorical micro-questions answered in the SAME "
            "paragraph to build the causal chain (e.g. 'Can it just write "
            "directly? No, because...') — these are not the closing "
            "question and must not end the field. In [mode:deep_analysis] "
            "— a long multi-section Deep Material Analysis."
        ),
        # RU: сухой инженерный разбор темы: без закрывающего/follow-up
        # вопроса, без анонса следующих подтем; описывает механику/физику
        # HOW, никогда не формулирует следствие/цену, которую спросит
        # follow_up_question — вывод пользователь делает сам. Можно 1-2
        # коротких риторических микро-вопроса, отвеченных в том же
        # абзаце. В [mode:deep_analysis] — длинный многосекционный Deep
        # Material Analysis.
    )
    message_bullet_summary: list[str] = Field(
        default_factory=list,
        max_length=SCHEMA_BULLET_SUMMARY_MAX_ITEMS,
        description=(
            "Up to 5 short key theses (claims) of NEW content, SOURCED "
            "EXCLUSIVELY from technical_explanation, with [S*]/[R*] "
            "citations, for compressed history. IGNORE the `audit` object "
            "entirely as a source — never read, paraphrase, or reword "
            "`confirmation` / `correction_breakdown` / `praise_points` "
            "into a thesis, even loosely (e.g. `confirmation`'s 'подтема "
            "разобрана/оценена верно' must NOT resurface here reworded as "
            "'разбор завершён' or similar) — that evaluation status is "
            "already conveyed separately via audit/verified_sub_concept_ids "
            "and the progress dashboard. Do NOT write recap/frame phrases "
            "like 'discussed X' or 'covered Y' — state the fact ITSELF, "
            "not that it was discussed (bad: 'Обсудили MergeTree'; good: "
            "'MergeTree хранит части независимо [R1]'). Content claims "
            "only. The last thesis, if present, records the question "
            "just asked."
        ),
        # RU: до 5 кратких тезисов НОВОГО материала — ИСКЛЮЧИТЕЛЬНО из
        # technical_explanation; поле audit (confirmation/correction_
        # breakdown/praise_points) как источник ИГНОРИРОВАТЬ полностью —
        # запрещено даже переформулировать «подтема разобрана верно» из
        # confirmation как «разбор завершён» здесь. Без рекап-фреймов вида
        # «обсудили X» — нужен сам факт, а не пересказ того, что говорили.
    )
    references: list[RichReferenceItem] = Field(
        default_factory=list,
        max_length=6,
        description=(
            "Source cards for the panel: only rows from SOURCE REGISTRY in "
            "the payload (asset_id S1…, copy url/title verbatim). Empty "
            "list if nothing was cited."
        ),
        # RU: карточки источников для панели: только строки из SOURCE
        # REGISTRY в payload; пустой список, если не цитировал.
    )

    # --- 2. Additional materials & context ---
    node_status: NodeStatus = Field(
        default="in_progress",
        description="Node progress",
        # RU: прогресс по ноде.
    )
    summary: str = Field(
        default="",
        max_length=SCHEMA_SUMMARY_MAX,
        description="Digest for the Materials panel",
        # RU: выжимка для панели Materials.
    )
    referenced_diagram_id: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "ID of the diagram chosen from the provided node diagram catalog. "
            "Do NOT write raw Mermaid code here."
        ),
    )
    introduced_terms: list[str] = Field(
        default_factory=list,
        max_length=16,
        description=(
            "Terms/acronyms first spelled out in the dialogue fields of "
            "this reply"
        ),
        # RU: термины/аббревиатуры, впервые расшифрованные в реплике
        # (dialogue поля) в этой реплике.
    )

    # --- 3. System & RAG fields (LightRAG, FSM, knowledge graph) ---
    new_gap_to_record: str | None = Field(
        default=None,
        max_length=SCHEMA_NEW_GAP_MAX,
        description="Gap for LightRAG, if detected",
        # RU: пробел для LightRAG, если выявлен.
    )
    verified_sub_concept_ids: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="sub_concept ids credited in this turn (for the registry)",
        # RU: ID sub_concepts, подтверждённых в этом ходе (для реестра).
    )
    ready_for_transition: bool = Field(
        default=False,
        description=(
            "Host-owned (Python). Leave false/inert — the host overwrites from "
            "BGE/FSM after generation. Do not invent topic-close logic."
        ),
    )
    suggested_next_step: SuggestedNextStep | None = Field(
        default=None,
        description=(
            "Host-owned (Python). Leave null — the host overwrites after generation."
        ),
    )
    quick_replies: list[str] = Field(
        default_factory=list,
        max_length=4,
        description=(
            "Host-owned (Python). Leave empty — the host sets UI chips from "
            "open optional layers / FSM. Do not invent chip labels."
        ),
    )

    @field_validator("message_bullet_summary", mode="before")
    @classmethod
    def _clip_message_bullet_summary(cls, v: list[str]) -> list[str]:
        return _truncate_bullet_summary(v)

    @computed_field
    @property
    def feedback_on_answer(self) -> str:
        return audit_feedback_text(self.audit)

    @model_validator(mode="after")
    def validate_audit_branch_consistency(self) -> _TutorFieldsBase:
        validate_grade_matches_errors(self.audit)
        return self


class DeepDiveTutorContract(BaseModel):
    """Dialogue turn with a follow-up question — the common Evaluator-scored path.

    Deliberately NOT ``_TutorFieldsBase`` + appended fields: Pydantic v2
    always appends subclass-added fields AFTER every inherited field, which
    would put `follow_up_question` (displayed) after the base's system-only
    fields (host-owned bookkeeping), breaking the displayed-before-system
    invariant. All fields below are therefore re-declared in full — kept in
    sync with `_TutorFieldsBase` by
    test_tutor_execution_plan.py::test_field_set_matches_deep_dive_tutor_contract.
    """

    # --- 1. Message (User-Facing Text, in reading order) ---
    audit: TechnicalConceptAudit = Field(
        ...,
        description=(
            "Strict technical audit of the learner's previous answer, filled "
            "BEFORE any other learner-facing text. Discriminated on "
            "feedback_kind: EXACT → confirmation; NEEDS_CORRECTION → "
            "correction_breakdown only."
        ),
    )
    technical_explanation: str = Field(
        default="",
        max_length=SCHEMA_TECHNICAL_EXPLANATION_MAX,
        description=(
            "Dry engineering breakdown of the topic: no closing/follow-up "
            "question, no announcing next subtopics. Explains the "
            "mechanic/physics of HOW — never states the consequence/"
            "trade-off/cost that follow_up_question will ask about; the "
            "user must derive that conclusion themselves. May use 1-2 "
            "short rhetorical micro-questions answered in the SAME "
            "paragraph to build the causal chain (e.g. 'Can it just write "
            "directly? No, because...') — these are not the closing "
            "question and must not end the field. In [mode:deep_analysis] "
            "— a long multi-section Deep Material Analysis."
        ),
        # RU: сухой инженерный разбор темы: без закрывающего/follow-up
        # вопроса, без анонса следующих подтем; описывает механику/физику
        # HOW, никогда не формулирует следствие/цену, которую спросит
        # follow_up_question — вывод пользователь делает сам. Можно 1-2
        # коротких риторических микро-вопроса, отвеченных в том же
        # абзаце. В [mode:deep_analysis] — длинный многосекционный Deep
        # Material Analysis.
    )
    follow_up_question: str = Field(
        default="",
        max_length=SCHEMA_FOLLOW_UP_QUESTION_MAX,
        description=(
            "Lead-in plus ONE question on the next sub-topic (must contain "
            "«?») ONLY when the system instructions for this session call "
            "for one — leave this field EMPTY when they say the session has "
            "no follow-up/self-check question. "
            f"Target ≤{PROMPT_FOLLOW_UP_MAX_CHARS} characters. Asks about a "
            "CONSEQUENCE/TRADE-OFF/COST the user must derive from "
            "technical_explanation — never a fact already stated as a "
            "conclusion there. Every criterion the Evaluator may require "
            "MUST be named or scope-locked here and introduced in "
            "technical_explanation on first mention (never only in this "
            "question)."
        ),
        # RU: спрашивает про следствие/компромисс/цену, которую пользователь
        # выводит сам из technical_explanation — не факт, уже
        # сформулированный там как вывод. Каждый критерий, который проверит
        # Evaluator, должен быть назван здесь И введён в technical_explanation
        # при первом упоминании — не только в вопросе.
    )
    message_bullet_summary: list[str] = Field(
        default_factory=list,
        max_length=SCHEMA_BULLET_SUMMARY_MAX_ITEMS,
        description=(
            "Up to 5 short key theses (claims) of NEW content, SOURCED "
            "EXCLUSIVELY from technical_explanation, with [S*]/[R*] "
            "citations, for compressed history. IGNORE the `audit` object "
            "entirely as a source — never read, paraphrase, or reword "
            "`confirmation` / `correction_breakdown` / `praise_points` "
            "into a thesis, even loosely (e.g. `confirmation`'s 'подтема "
            "разобрана/оценена верно' must NOT resurface here reworded as "
            "'разбор завершён' or similar) — that evaluation status is "
            "already conveyed separately via audit/verified_sub_concept_ids "
            "and the progress dashboard. Do NOT write recap/frame phrases "
            "like 'discussed X' or 'covered Y' — state the fact ITSELF, "
            "not that it was discussed (bad: 'Обсудили MergeTree'; good: "
            "'MergeTree хранит части независимо [R1]'). Content claims "
            "only. The last thesis, if present, records the question "
            "just asked."
        ),
        # RU: до 5 кратких тезисов НОВОГО материала — ИСКЛЮЧИТЕЛЬНО из
        # technical_explanation; поле audit (confirmation/correction_
        # breakdown/praise_points) как источник ИГНОРИРОВАТЬ полностью —
        # запрещено даже переформулировать «подтема разобрана верно» из
        # confirmation как «разбор завершён» здесь. Без рекап-фреймов вида
        # «обсудили X» — нужен сам факт, а не пересказ того, что говорили.
    )
    references: list[RichReferenceItem] = Field(
        default_factory=list,
        max_length=6,
        description=(
            "Source cards for the panel: only rows from SOURCE REGISTRY in "
            "the payload (asset_id S1…, copy url/title verbatim). Empty "
            "list if nothing was cited."
        ),
        # RU: карточки источников для панели: только строки из SOURCE
        # REGISTRY в payload; пустой список, если не цитировал.
    )

    # --- 2. Additional materials & context ---
    node_status: NodeStatus = Field(
        default="in_progress",
        description="Node progress",
        # RU: прогресс по ноде.
    )
    summary: str = Field(
        default="",
        max_length=SCHEMA_SUMMARY_MAX,
        description="Digest for the Materials panel",
        # RU: выжимка для панели Materials.
    )
    referenced_diagram_id: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "ID of the diagram chosen from the provided node diagram catalog. "
            "Do NOT write raw Mermaid code here."
        ),
    )
    introduced_terms: list[str] = Field(
        default_factory=list,
        max_length=16,
        description=(
            "Terms/acronyms first spelled out in the dialogue fields of "
            "this reply"
        ),
        # RU: термины/аббревиатуры, впервые расшифрованные в реплике
        # (dialogue поля) в этой реплике.
    )

    # --- 3. System & RAG fields (LightRAG, FSM, knowledge graph) ---
    new_gap_to_record: str | None = Field(
        default=None,
        max_length=SCHEMA_NEW_GAP_MAX,
        description="Gap for LightRAG, if detected",
        # RU: пробел для LightRAG, если выявлен.
    )
    verified_sub_concept_ids: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="sub_concept ids credited in this turn (for the registry)",
        # RU: ID sub_concepts, подтверждённых в этом ходе (для реестра).
    )
    question_sub_concept_id: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Exact sub-concept id from concept_map that follow_up_question "
            "targets. null if no question is asked."
        ),
        # RU: точный id подконцепта из concept_map, по которому задан
        # follow_up_question; null, если вопрос не задаётся.
    )
    ready_for_transition: bool = Field(
        default=False,
        description=(
            "Host-owned (Python). Leave false/inert — the host overwrites from "
            "BGE/FSM after generation. Do not invent topic-close logic."
        ),
    )
    suggested_next_step: SuggestedNextStep | None = Field(
        default=None,
        description=(
            "Host-owned (Python). Leave null — the host overwrites after generation."
        ),
    )
    quick_replies: list[str] = Field(
        default_factory=list,
        max_length=4,
        description=(
            "Host-owned (Python). Leave empty — the host sets UI chips from "
            "open optional layers / FSM. Do not invent chip labels."
        ),
    )

    @field_validator("message_bullet_summary", mode="before")
    @classmethod
    def _clip_message_bullet_summary(cls, v: list[str]) -> list[str]:
        return _truncate_bullet_summary(v)

    @computed_field
    @property
    def feedback_on_answer(self) -> str:
        return audit_feedback_text(self.audit)

    @model_validator(mode="after")
    def validate_audit_branch_consistency(self) -> "DeepDiveTutorContract":
        validate_grade_matches_errors(self.audit)
        return self


class DeepDiveTutorWithPlanContract(BaseModel):
    """CoT Execution Plan variant of ``DeepDiveTutorContract`` — gated by
    ``ENABLE_TUTOR_EXECUTION_PLAN`` (default off), see
    engine.py::_resolve_tutor_response_schema.

    Deliberately NOT ``class DeepDiveTutorWithPlanContract(DeepDiveTutorContract)``:
    Pydantic v2 always appends subclass-added fields AFTER every inherited
    field (verified — MRO order of the bases does not change this), so a
    plain subclass would put `execution_plan` LAST, not first, which defeats
    the entire point (the plan must be written before the prose it steers).
    All fields below are therefore re-declared in full, execution_plan
    genuinely first — kept in sync with _TutorFieldsBase/DeepDiveTutorContract
    by test_tutor_execution_plan.py::test_field_set_matches_deep_dive_tutor_contract.
    """

    execution_plan: str = Field(
        ...,
        max_length=SCHEMA_EXECUTION_PLAN_MAX,
        description=(
            "DYNAMIC CAUSAL GRAPH (graph TD) FOR LIGHTWEIGHT MODEL. Generated "
            "FIRST, before any other field.\n"
            "Build a logical chain adapting to the specific topic (3 to 6 "
            "nodes max).\n\n"
            "GRAPH CONSTRUCTION INVARIANTS:\n"
            "1. GROUNDING: Move from high-level context down to concrete "
            "implementation mechanics (RAM, disk, data structures, locks, "
            "protocols).\n"
            "2. SOCRATIC HOOK: Include at least 1 node with a short "
            "micro-question/tension ('Can X handle Y? No/Why').\n"
            "3. SHADOW TARGET: Mark the ultimate unrevealed architectural "
            "consequence/trade-off as `SHADOW: <target>`.\n"
            "4. LINKING: Connect main nodes with solid arrows `-->`. Connect "
            "the approach to `SHADOW` with a dotted arrow "
            "`-. BANNED_IN_TEXT .->`.\n\n"
            "FLEXIBLE EXAMPLES:\n"
            "- Architecture topic: Concept --> TradeOff --> CodePattern -. "
            "BANNED_IN_TEXT .-> [SHADOW: Scale Bottleneck]\n"
            "- Database topic: AppendOnly --> PartsAccumulation --> "
            "SocraticPause --> MergeTree -. BANNED_IN_TEXT .-> [SHADOW: Read "
            "IO Impact]"
        ),
        # RU: короткий Mermaid-граф причинно-следственных связей (3-6 узлов),
        # генерируется ПЕРВЫМ полем — план для лёгкой модели до того, как
        # писать текст. SHADOW = скрытое следствие, которое запрещено
        # раскрывать в technical_explanation/follow_up_question.
    )
    # --- 1. Message (User-Facing Text, in reading order) ---
    audit: TechnicalConceptAudit = Field(
        ...,
        description=(
            "Strict technical audit of the learner's previous answer, filled "
            "BEFORE any other learner-facing text. Discriminated on "
            "feedback_kind: EXACT → confirmation; NEEDS_CORRECTION → "
            "correction_breakdown only."
        ),
    )
    technical_explanation: str = Field(
        default="",
        max_length=SCHEMA_TECHNICAL_EXPLANATION_MAX,
        description=(
            "Dry engineering breakdown of the topic: no closing/follow-up "
            "question, no announcing next subtopics. Write a natural, "
            "engaging explanation FOLLOWING THE FLOW of execution_plan "
            "(same node order) — stop strictly before the SHADOW target; "
            "never state it, even paraphrased. May use 1-2 short rhetorical "
            "micro-questions answered in the SAME paragraph to build the "
            "causal chain (e.g. 'Can it just write directly? No, "
            "because...') — these are not the closing question and must "
            "not end the field. In [mode:deep_analysis] — a long "
            "multi-section Deep Material Analysis."
        ),
        # RU: сухой инженерный разбор темы, следующий порядку узлов
        # execution_plan; останавливается строго ДО SHADOW-узла — его
        # содержание нельзя раскрывать даже перефразированно. Можно 1-2
        # коротких риторических микро-вопроса в том же абзаце.
    )
    follow_up_question: str = Field(
        default="",
        max_length=SCHEMA_FOLLOW_UP_QUESTION_MAX,
        description=(
            "Lead-in plus ONE question on the next sub-topic (must contain "
            "«?») ONLY when the system instructions for this session call "
            "for one — leave this field EMPTY when they say the session has "
            "no follow-up/self-check question. "
            f"Target ≤{PROMPT_FOLLOW_UP_MAX_CHARS} characters. Asks about "
            "the SHADOW target from execution_plan — the "
            "CONSEQUENCE/TRADE-OFF/COST the user must derive from "
            "technical_explanation — never a fact already stated as a "
            "conclusion there. Every criterion the Evaluator may require "
            "MUST be named or scope-locked here and introduced in "
            "technical_explanation on first mention (never only in this "
            "question)."
        ),
        # RU: спрашивает про SHADOW-цель из execution_plan — следствие/
        # компромисс/цену, которую пользователь выводит сам из
        # technical_explanation, не факт, уже сформулированный там как
        # вывод.
    )
    message_bullet_summary: list[str] = Field(
        default_factory=list,
        max_length=SCHEMA_BULLET_SUMMARY_MAX_ITEMS,
        description=(
            "Up to 5 short key theses (claims) of NEW content, SOURCED "
            "EXCLUSIVELY from technical_explanation, with [S*]/[R*] "
            "citations, for compressed history. IGNORE the `audit` object "
            "entirely as a source — never read, paraphrase, or reword "
            "`confirmation` / `correction_breakdown` / `praise_points` "
            "into a thesis, even loosely (e.g. `confirmation`'s 'подтема "
            "разобрана/оценена верно' must NOT resurface here reworded as "
            "'разбор завершён' or similar) — that evaluation status is "
            "already conveyed separately via audit/verified_sub_concept_ids "
            "and the progress dashboard. Do NOT write recap/frame phrases "
            "like 'discussed X' or 'covered Y' — state the fact ITSELF, "
            "not that it was discussed (bad: 'Обсудили MergeTree'; good: "
            "'MergeTree хранит части независимо [R1]'). Content claims "
            "only. The last thesis, if present, records the question "
            "just asked."
        ),
        # RU: до 5 кратких тезисов НОВОГО материала — ИСКЛЮЧИТЕЛЬНО из
        # technical_explanation; поле audit как источник ИГНОРИРОВАТЬ
        # полностью. Без рекап-фреймов вида «обсудили X».
    )
    references: list[RichReferenceItem] = Field(
        default_factory=list,
        max_length=6,
        description=(
            "Source cards for the panel: only rows from SOURCE REGISTRY in "
            "the payload (asset_id S1…, copy url/title verbatim). Empty "
            "list if nothing was cited."
        ),
        # RU: карточки источников для панели: только строки из SOURCE
        # REGISTRY в payload; пустой список, если не цитировал.
    )

    # --- 2. Additional materials & context ---
    node_status: NodeStatus = Field(
        default="in_progress",
        description="Node progress",
        # RU: прогресс по ноде.
    )
    summary: str = Field(
        default="",
        max_length=SCHEMA_SUMMARY_MAX,
        description="Digest for the Materials panel",
        # RU: выжимка для панели Materials.
    )
    referenced_diagram_id: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "ID of the diagram chosen from the provided node diagram catalog. "
            "Do NOT write raw Mermaid code here."
        ),
    )
    introduced_terms: list[str] = Field(
        default_factory=list,
        max_length=16,
        description=(
            "Terms/acronyms first spelled out in the dialogue fields of "
            "this reply"
        ),
        # RU: термины/аббревиатуры, впервые расшифрованные в реплике
        # (dialogue поля) в этой реплике.
    )

    # --- 3. System & RAG fields (LightRAG, FSM, knowledge graph) ---
    new_gap_to_record: str | None = Field(
        default=None,
        max_length=SCHEMA_NEW_GAP_MAX,
        description="Gap for LightRAG, if detected",
        # RU: пробел для LightRAG, если выявлен.
    )
    verified_sub_concept_ids: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="sub_concept ids credited in this turn (for the registry)",
        # RU: ID sub_concepts, подтверждённых в этом ходе (для реестра).
    )
    question_sub_concept_id: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Exact sub-concept id from concept_map that follow_up_question "
            "targets. null if no question is asked."
        ),
        # RU: точный id подконцепта из concept_map, по которому задан
        # follow_up_question; null, если вопрос не задаётся.
    )
    ready_for_transition: bool = Field(
        default=False,
        description=(
            "Host-owned (Python). Leave false/inert — the host overwrites from "
            "BGE/FSM after generation. Do not invent topic-close logic."
        ),
    )
    suggested_next_step: SuggestedNextStep | None = Field(
        default=None,
        description=(
            "Host-owned (Python). Leave null — the host overwrites after generation."
        ),
    )
    quick_replies: list[str] = Field(
        default_factory=list,
        max_length=4,
        description=(
            "Host-owned (Python). Leave empty — the host sets UI chips from "
            "open optional layers / FSM. Do not invent chip labels."
        ),
    )

    @field_validator("message_bullet_summary", mode="before")
    @classmethod
    def _clip_message_bullet_summary(cls, v: list[str]) -> list[str]:
        return _truncate_bullet_summary(v)

    @computed_field
    @property
    def feedback_on_answer(self) -> str:
        return audit_feedback_text(self.audit)

    @model_validator(mode="after")
    def validate_audit_branch_consistency(self) -> "DeepDiveTutorWithPlanContract":
        validate_grade_matches_errors(self.audit)
        return self


class TopicQnaTutorContract(_TutorFieldsBase):
    """Audited dialogue turn (Evaluator ran, grading a Self-Check answer)

    for interaction_axis="topic_qna". No follow_up_question /
    question_sub_concept_id field: after grading, the expert-consultant
    gives feedback and stops — no new question until the learner explicitly
    presses "Самопроверка" again. Auto-advance to the next module stays
    driven by the Host-owned ready_for_transition / suggested_next_step /
    quick_replies fields, unaffected by this split."""


class _ExplainFieldsBase(BaseModel):
    """Общие поля Explain-контракта (Evaluator пропущен) для

    lecture_self_check/прочих skip-причин и Topic Q&A. follow_up_question /
    question_sub_concept_id сознательно НЕ здесь: DeepDiveExplainContract
    (обычный skip: пустое сообщение / явный lecture request / нет pending)
    добавляет их как ОПЦИОНАЛЬНОЕ поле — там follow-up на суждение модели
    приемлем. TopicQnaExplainContract НЕ добавляет их вовсе: в Topic Q&A
    follow-up/self-check вопрос запрещён категорически (эксперт-консультант
    отвечает, не спрашивает), а "Optional... if set" без axis-условия — тот
    же класс бага, что уже был у follow_up_question (см.
    StructuredLectureResponse/TopicQnaLectureResponse): текстовая
    инструкция ненадёжна, известный инвариант должен быть в схеме."""

    # --- 1. Message (User-Facing Text, in reading order) ---
    technical_explanation: str = Field(
        default="",
        max_length=SCHEMA_TECHNICAL_EXPLANATION_MAX,
        description=(
            "Engineering explanation only. No learner-answer verdict. "
            "No questions (no '?') in this field."
        ),
    )
    references: list[RichReferenceItem] = Field(
        default_factory=list,
        max_length=6,
        description="Source cards for the panel: only rows from SOURCE REGISTRY.",
        # RU: карточки источников для панели: только строки из SOURCE REGISTRY.
    )

    # --- 2. Additional materials & context ---
    node_status: NodeStatus = Field(
        default="in_progress",
        description="Node progress",
        # RU: прогресс по ноде.
    )
    summary: str = Field(
        default="",
        max_length=SCHEMA_SUMMARY_MAX,
        description="Digest for the Materials panel",
        # RU: выжимка для панели Materials.
    )
    referenced_diagram_id: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "ID of the diagram chosen from the provided node diagram catalog. "
            "Do NOT write raw Mermaid code here."
        ),
    )
    introduced_terms: list[str] = Field(
        default_factory=list,
        max_length=16,
        description="Terms first glossed in this turn.",
    )

    # --- 3. System & RAG fields (LightRAG, FSM, knowledge graph) ---
    verified_sub_concept_ids: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="Host-owned; leave empty.",
    )
    ready_for_transition: bool = Field(
        default=False,
        description="Host-owned. Leave false.",
    )
    suggested_next_step: SuggestedNextStep | None = Field(
        default=None,
        description="Host-owned. Leave null.",
    )
    quick_replies: list[str] = Field(
        default_factory=list,
        max_length=4,
        description="Host-owned. Leave empty.",
    )

    @computed_field
    @property
    def feedback_on_answer(self) -> str:
        return ""

    # RU: оценка пропущена — вердикта нет.


class DeepDiveExplainContract(BaseModel):
    """Tutor turn when Host skipped Evaluator for a non-Topic-Q&A reason

    (empty message / explicit lecture request / no pending target) — see
    TopicQnaExplainContract for the Topic Q&A variant.

    Deliberately NOT ``_ExplainFieldsBase`` + appended fields: Pydantic v2
    always appends subclass-added fields AFTER every inherited field, which
    would put `follow_up_question` (displayed) after the base's system-only
    fields, breaking the message-before-materials-before-system invariant.
    All fields below are therefore re-declared in full — kept in sync with
    `_ExplainFieldsBase` by
    test_pydantic_contract_field_order.py::test_explain_field_set_matches_base.
    """

    # --- 1. Message (User-Facing Text, in reading order) ---
    technical_explanation: str = Field(
        default="",
        max_length=SCHEMA_TECHNICAL_EXPLANATION_MAX,
        description=(
            "Engineering explanation only. No learner-answer verdict. "
            "No questions (no '?') in this field."
        ),
    )
    follow_up_question: str = Field(
        default="",
        max_length=SCHEMA_FOLLOW_UP_QUESTION_MAX,
        description=(
            f"Optional one follow-up question (must contain «?» if set); "
            f"target ≤{PROMPT_FOLLOW_UP_MAX_CHARS} characters"
        ),
    )
    references: list[RichReferenceItem] = Field(
        default_factory=list,
        max_length=6,
        description="Source cards for the panel: only rows from SOURCE REGISTRY.",
        # RU: карточки источников для панели: только строки из SOURCE REGISTRY.
    )

    # --- 2. Additional materials & context ---
    node_status: NodeStatus = Field(
        default="in_progress",
        description="Node progress",
        # RU: прогресс по ноде.
    )
    summary: str = Field(
        default="",
        max_length=SCHEMA_SUMMARY_MAX,
        description="Digest for the Materials panel",
        # RU: выжимка для панели Materials.
    )
    referenced_diagram_id: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "ID of the diagram chosen from the provided node diagram catalog. "
            "Do NOT write raw Mermaid code here."
        ),
    )
    introduced_terms: list[str] = Field(
        default_factory=list,
        max_length=16,
        description="Terms first glossed in this turn.",
    )

    # --- 3. System & RAG fields (LightRAG, FSM, knowledge graph) ---
    verified_sub_concept_ids: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="Host-owned; leave empty.",
    )
    question_sub_concept_id: str | None = Field(
        default=None,
        max_length=64,
        description="Map id for follow_up_question, or null.",
    )
    ready_for_transition: bool = Field(
        default=False,
        description="Host-owned. Leave false.",
    )
    suggested_next_step: SuggestedNextStep | None = Field(
        default=None,
        description="Host-owned. Leave null.",
    )
    quick_replies: list[str] = Field(
        default_factory=list,
        max_length=4,
        description="Host-owned. Leave empty.",
    )

    @computed_field
    @property
    def feedback_on_answer(self) -> str:
        return ""

    # RU: оценка пропущена — вердикта нет.


class TopicQnaExplainContract(_ExplainFieldsBase):
    """Tutor chat turn for interaction_axis="topic_qna" (expert-consultant).

    No follow_up_question / question_sub_concept_id field — Gemini
    structured output cannot write a self-check/follow-up question, since
    there is no slot for it in this schema."""


class DeepDiveDeepAnalysisContract(DeepDiveTutorContract):
    """
    Structural contract for [mode:deep_analysis] / open Star Task turns.

    Only validates that follow_up_question is present and non-empty.
    Orchestration flags (ready_for_transition, quick_replies) are forced in
    Python after the LLM call — not via phrase / boolean validators.
    """

    follow_up_question: str = Field(
        ...,
        min_length=1,
        max_length=SCHEMA_FOLLOW_UP_QUESTION_MAX,
        description=(
            "REQUIRED non-empty: exactly ONE engineering design / evaluation "
            f"question with «?»; target ≤{PROMPT_FOLLOW_UP_MAX_CHARS} characters. "
            "Question derived from the Problem / Edge / Trade-off analysis "
            "(FACT_ATTRACTION). When SOURCE REGISTRY is empty, references=[]."
        ),
    )

    @field_validator("follow_up_question")
    @classmethod
    def _follow_up_nonempty(cls, v: str) -> str:
        text = (v or "").strip()
        if not text:
            raise ValueError(
                "follow_up_question is REQUIRED and must be non-empty for deep_analysis"
            )
        return text


class ConceptUpdateContract(BaseModel):
    concept: str = Field(..., min_length=1, max_length=400)
    status: ConceptMasteryStatus | None = None
    evidence: str = Field(default="", max_length=2000)
    mastery_score: int | None = Field(default=None, ge=0, le=100)


class StepAnalysisContract(BaseModel):
    """
    RU (пояснение): intent сюда больше не входит — тип сообщения (lecture /
    finalize / shift_focus / control chips) резолвится детерминированно через
    VectorIntentRouter выше по пайплайну (см. step_analysis_node), LLM отвечает
    только за concept_updates/critical_gap.
    """

    concept_updates: list[ConceptUpdateContract] = Field(
        default_factory=list,
        max_length=12,
        description="Mastery updates per concept",
        # RU: обновления mastery по концептам.
    )
    critical_gap: str | None = Field(
        default=None,
        max_length=2000,
        description="Critical gap, if any",
        # RU: критический пробел, если есть.
    )


class SubConceptStatusUpdate(BaseModel):
    """
    Layer fact-extract from one user answer (no mastery verdict).

    Threshold Engine (Python) credits layer flags and VERIFIED only on
    ``accuracy_grade=EXACT_AND_CORRECT`` with an empty error list.
    """

    id: str = Field(..., min_length=2, max_length=64)
    why_passed: bool = Field(
        default=False,
        description="WHY: concept / problem / motivation present in THIS answer",
    )
    how_passed: bool = Field(
        default=False,
        description="HOW: components / roles / invariants present in THIS answer",
    )
    mechanic_passed: bool = Field(
        default=False,
        description="MECHANIC: named execution/detail layer present in THIS answer",
    )
    accuracy_grade: AnswerAccuracyGrade = Field(
        ...,
        description=(
            "Accuracy strictly within the explicitly requested scope of the "
            "question. Unasked deeper layers MUST NOT reduce this grade. "
            "Host credits layers / VERIFIED only on EXACT_AND_CORRECT with "
            "empty detected_errors_or_misconceptions. PARTIAL does not close "
            "a layer."
        ),
    )
    detected_errors_or_misconceptions: list[str] = Field(
        default_factory=list,
        max_length=16,
        description=(
            "Only explicit factually false statements in THIS answer. "
            "Must be empty on EXACT_AND_CORRECT. Silence or omission of "
            "unasked topics is NOT an error. PARTIAL may be empty when the "
            "answer is incomplete but not factually wrong."
        ),
    )
    correct_claims: list[str] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "Theses from THIS answer that are correct (facts only, "
            "no praise). Required non-empty when accuracy_grade is PARTIAL."
        ),
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
            "On PARTIAL / non-exact: the single missing element from the "
            "explicitly asked scope. NEVER demand unasked deeper layers. "
            "Python may override from the threshold probe layer."
        ),
    )
    # Legacy optional — ignored by Threshold Engine (kept for schema soft-compat).
    status: Literal["VERIFIED", "PARTIAL", "GAP", "UNCHECKED"] | None = Field(
        default=None,
        description="Deprecated: Python Threshold Engine owns status",
    )

    @model_validator(mode="after")
    def grade_must_match_errors(self) -> SubConceptStatusUpdate:
        errors = [
            e.strip()
            for e in (self.detected_errors_or_misconceptions or [])
            if (e or "").strip()
        ]
        claims = [c.strip() for c in (self.correct_claims or []) if (c or "").strip()]
        if self.accuracy_grade == AnswerAccuracyGrade.EXACT_AND_CORRECT:
            if errors:
                raise ValueError(
                    "EXACT_AND_CORRECT forbids non-empty "
                    "detected_errors_or_misconceptions."
                )
            return self
        if self.accuracy_grade == AnswerAccuracyGrade.PARTIAL and not claims:
            raise ValueError(
                "PARTIAL requires non-empty correct_claims "
                "(theses that were already right)."
            )
        return self


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
        description="Markdown: detail from the source, brief, for an engineer",
        # RU: Markdown: деталь из источника, коротко для инженера.
    )
    cited_source_ids: list[str] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "IDs from the selection: R6, R7 (RAG chunks) and, if needed, "
            "S1 (registry)"
        ),
        # RU: ID из выделения: R6, R7 (RAG chunks) и при необходимости S1 (registry).
    )


class DialogueFactManifestContract(BaseModel):
    agreed_concepts: list[str] = Field(
        default_factory=list,
        max_length=24,
        description="Accepted concepts/mechanics",
        # RU: принятые концепты/механики.
    )
    rejected_options: list[str] = Field(
        default_factory=list,
        max_length=16,
        description="Rejected options",
        # RU: отвергнутые варианты.
    )
    open_bottlenecks: list[str] = Field(
        default_factory=list,
        max_length=16,
        description="Open bottlenecks (latency, RAM, index)",
        # RU: открытые bottlenecks (latency, RAM, index).
    )
    stack_mentions: list[str] = Field(
        default_factory=list,
        max_length=24,
        description="Specific technologies mentioned in the reply",
        # RU: конкретные технологии из реплики.
    )
    current_subtopic: str = Field(
        default="",
        max_length=400,
        description="Active subtopic, one line",
        # RU: активная подтема одной строкой.
    )


def structured_lecture_to_dense(
    out: (
        StructuredLectureResponse
        | TopicQnaLectureResponse
        | StructuredLectureResponseWithPlanContract
    ),
    *,
    allowed_urls: set[str] | None = None,
) -> DenseMaterialOutput:
    from knowledge_engine.src.core.link_sanitizer import normalize_lecture_url

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
        from knowledge_engine.src.domains.grounding.node_source_registry import (
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
    from knowledge_engine.src.domains.grounding.code_snippet_heuristic import (
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
        follow_up_question=(getattr(out, "follow_up_question", "") or "").strip(),
        extracted_concepts=list(out.extracted_concepts or [])[:5],
        introduced_terms=list(out.introduced_terms or [])[:24],
        message_bullet_summary=list(out.message_bullet_summary or []),
    )
