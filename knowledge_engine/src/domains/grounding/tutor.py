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
    SCHEMA_CHECKPOINT_PROMPT_MAX,
    SCHEMA_FEEDBACK_ON_ANSWER_MAX,
    SCHEMA_FOLLOW_UP_QUESTION_MAX,
    SCHEMA_LECTURE_BODY_MAX,
    SCHEMA_NEW_GAP_MAX,
    SCHEMA_SUMMARY_MAX,
    SCHEMA_TECHNICAL_EXPLANATION_MAX,
    SCHEMA_TUTOR_MESSAGE_MAX,
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
    "(«Вердикт самопроверки», «Пользователь корректно…» as assessment headers); "
    "closing quiz questions; «?» in the last paragraph; "
    "'Самопроверка' / 'Вопрос:' self-check headers. "
    "lecture_body is theory, code, and architecture only.\n"
    "CRITICAL NEGATIVE CONSTRAINT: Do NOT include any closing questions, self-check "
    "queries, or 'Самопроверка:' headers inside lecture_body. The lecture_body MUST "
    "contain pure educational content only. The checkpoint question belongs EXCLUSIVELY "
    "in checkpoint_prompt.\n\n"
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
    "8. `checkpoint_prompt`: the ONLY field for ONE technical self-check question "
    "(must contain «?»; PART 1 lecture_body must end without «?» / without a quiz); "
    "target ≤400 characters. Every scored criterion MUST be named here and "
    "introduced in lecture_body first.\n"
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


class _DenseLectureFieldsBase(BaseModel):
    """Поля, общие для lecture_self_check и Topic Q&A dense-ответов.

    checkpoint_prompt сознательно НЕ здесь: он существует только в
    StructuredLectureResponse (lecture_self_check). Topic Q&A использует
    TopicQnaLectureResponse — без этого поля в JSON Schema, которую видит
    Gemini structured output, так что self-check вопрос физически
    невозможно записать, а не просто "не рекомендуется текстом". Раньше
    checkpoint_prompt был общим полем с условным Field(description=...)
    ("оставь пустым для Topic Q&A") — Gemini всё равно заполнял его вопреки
    инструкции (подтверждено live-тестом на ноде sql_cte, prompt.log):
    описание поля — не менее сильный канал для модели, чем system prompt,
    и условность в нём так же ненадёжна. Известный инвариант (это поле
    вообще не существует в этом режиме) должен быть закодирован в схеме,
    а не в тексте инструкции."""

    lecture_body: str = Field(
        ...,
        max_length=SCHEMA_LECTURE_BODY_MAX,
        description=(
            "Markdown theory only: LaTeX, ```python fences with real newlines, "
            "[Diagram N], [S1]/[R1] citations without URLs; node-materials tour when "
            "code/diagrams exist. FORBIDDEN: closing self-check questions, a final «?» "
            "paragraph, or 'Самопроверка' / 'Вопрос:' headers. Checkpoint belongs "
            "exclusively in checkpoint_prompt."
        ),
        # RU: теоретический блок без контрольного вопроса и без заголовков самопроверки.
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
        default="", max_length=SCHEMA_SUMMARY_MAX, description="Выжимка для панели"
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
    bridge_to_next: str = Field(
        default="",
        max_length=SCHEMA_BRIDGE_TO_NEXT_MAX,
        description="Следующий шаг без риторических вопросов",
    )


class StructuredLectureResponse(_DenseLectureFieldsBase):
    checkpoint_prompt: str = Field(
        default="",
        max_length=SCHEMA_CHECKPOINT_PROMPT_MAX,
        description=(
            "The ONLY field for the single self-check question "
            "(must contain «?»). Do not duplicate it in lecture_body. "
            "Every scored criterion MUST be named here and introduced in "
            "lecture_body first. FORBIDDEN: a surface question whose hidden "
            "rubric is a deeper unasked layer."
        ),
    )


class TopicQnaLectureResponse(_DenseLectureFieldsBase):
    """dense_material для interaction_axis="topic_qna" (expert-consultant).

    Идентична StructuredLectureResponse минус checkpoint_prompt — само
    поле отсутствует в схеме, так что self-check/follow-up вопрос
    структурно невозможен, а не просто запрещён текстом промпта."""


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
        description="Статус ноды после intro",
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

    audit: TechnicalConceptAudit = Field(
        ...,
        description=(
            "Strict technical audit of the learner's previous answer, filled "
            "BEFORE any other learner-facing text. Discriminated on "
            "feedback_kind: EXACT → confirmation; NEEDS_CORRECTION → "
            "correction_breakdown only."
        ),
    )
    node_status: NodeStatus = Field(
        default="in_progress",
        description="Прогресс по ноде",
    )
    summary: str = Field(
        default="",
        max_length=SCHEMA_SUMMARY_MAX,
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
    technical_explanation: str = Field(
        default="",
        max_length=SCHEMA_TECHNICAL_EXPLANATION_MAX,
        description=(
            "Сухой инженерный разбор темы: без «?», без follow-up и без анонса следующих подтем. "
            "В [mode:deep_analysis] — длинный многосекционный Deep Material Analysis."
        ),
    )
    new_gap_to_record: str | None = Field(
        default=None,
        max_length=SCHEMA_NEW_GAP_MAX,
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

    @computed_field
    @property
    def feedback_on_answer(self) -> str:
        return audit_feedback_text(self.audit)

    @model_validator(mode="after")
    def validate_audit_branch_consistency(self) -> _TutorFieldsBase:
        validate_grade_matches_errors(self.audit)
        return self


class DeepDiveTutorContract(_TutorFieldsBase):
    follow_up_question: str = Field(
        default="",
        max_length=SCHEMA_FOLLOW_UP_QUESTION_MAX,
        description=(
            "Lead-in plus ONE question on the next sub-topic (must contain "
            "«?») ONLY when the system instructions for this session call "
            "for one — leave this field EMPTY when they say the session has "
            "no follow-up/self-check question. "
            f"Target ≤{PROMPT_FOLLOW_UP_MAX_CHARS} characters. When used, "
            "every criterion the Evaluator may require MUST be named or "
            "scope-locked here and introduced in technical_explanation on "
            "first mention."
        ),
    )
    question_sub_concept_id: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Точный id подконцепта из concept_map, по которому задан follow_up_question. "
            "null если вопрос не задаётся."
        ),
    )


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
    же класс бага, что уже был у checkpoint_prompt (см.
    StructuredLectureResponse/TopicQnaLectureResponse): текстовая
    инструкция ненадёжна, известный инвариант должен быть в схеме."""

    node_status: NodeStatus = Field(
        default="in_progress",
        description="Прогресс по ноде",
    )
    summary: str = Field(
        default="",
        max_length=SCHEMA_SUMMARY_MAX,
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
            "Карточки источников для панели: только строки из SOURCE REGISTRY."
        ),
    )
    technical_explanation: str = Field(
        default="",
        max_length=SCHEMA_TECHNICAL_EXPLANATION_MAX,
        description=(
            "Engineering explanation only. No learner-answer verdict. "
            "No questions (no '?') in this field."
        ),
    )
    introduced_terms: list[str] = Field(
        default_factory=list,
        max_length=16,
        description="Terms first glossed in this turn.",
    )
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


class DeepDiveExplainContract(_ExplainFieldsBase):
    """Tutor turn when Host skipped Evaluator for a non-Topic-Q&A reason

    (empty message / explicit lecture request / no pending target) — see
    TopicQnaExplainContract for the Topic Q&A variant."""

    follow_up_question: str = Field(
        default="",
        max_length=SCHEMA_FOLLOW_UP_QUESTION_MAX,
        description=(
            f"Optional one follow-up question (must contain «?» if set); "
            f"target ≤{PROMPT_FOLLOW_UP_MAX_CHARS} characters"
        ),
    )
    question_sub_concept_id: str | None = Field(
        default=None,
        max_length=64,
        description="Map id for follow_up_question, or null.",
    )


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
    out: StructuredLectureResponse | TopicQnaLectureResponse,
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
        checkpoint_prompt=(getattr(out, "checkpoint_prompt", "") or "").strip(),
        extracted_concepts=list(out.extracted_concepts or [])[:5],
        introduced_terms=list(out.introduced_terms or [])[:24],
    )
