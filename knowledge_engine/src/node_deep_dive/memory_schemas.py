"""Tiered memory и прогресс Сократовского диалога (Модуль 2)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ConceptMasteryStatus = Literal["unverified", "in_progress", "verified"]
SubConceptStatus = Literal["unchecked", "partial", "verified", "gap"]
UserIntent = Literal[
    "ANSWER",
    "INTENT_EXPLAIN",
    "INTENT_SHIFT_FOCUS",
    "INTENT_FINALIZE",
]
NodeStatus = Literal[
    "unexplored",
    "in_progress",
    "deep_understanding",
    "mastered",
    "gap",
    "passed_by_equivalence",
]
LearningPhase = Literal[
    "intro_assessment",
    "dense_material",
    "checkpoint",
    "pathway_decision",
    "socratic_focus",
]
LearningMode = Literal["lecture", "express_blitz", "socratic_point"]


class CoreConceptRecord(BaseModel):
    concept: str = Field(min_length=1, max_length=400)
    status: ConceptMasteryStatus = "unverified"
    evidence: str = Field(default="", max_length=2000)
    mastery_score: int = Field(default=0, ge=0, le=100)


class SubConceptRecord(BaseModel):
    """Атомарная подтема ноды (Topic Concept Map)."""

    id: str = Field(min_length=2, max_length=64)
    label: str = Field(min_length=2, max_length=200)
    success_criterion: str = Field(default="", max_length=600)
    status: SubConceptStatus = "unchecked"
    evidence: str = Field(default="", max_length=2000)
    focus_hint: str = Field(default="", max_length=500)
    why_passed: bool = Field(
        default=False,
        description="WHY: концепция, проблематика, мотивация (накопительный)",
    )
    how_passed: bool = Field(
        default=False,
        description="HOW: архитектура, инварианты, роли (накопительный)",
    )
    mechanic_passed: bool = Field(
        default=False,
        description="MECHANIC: формулы, алгоритмы, код (накопительный)",
    )
    updated_at: str = Field(
        default="",
        max_length=40,
        description="ISO-8601 UTC последнего изменения статуса",
    )

    @property
    def is_verified(self) -> bool:
        return self.status == "verified"


class LectureExtractedConcept(BaseModel):
    """Микро-тема из плотной лекции (контракт Gemini + реестр покрытия)."""

    key: str = Field(
        min_length=2, max_length=64, description="snake_case id микро-темы"
    )
    summary: str = Field(
        min_length=4,
        max_length=600,
        description="Кратко: что именно разобрали в лекции",
    )


class DialogueFactManifest(BaseModel):
    """Структурированная память диалога (вместо прозаического rolling_compress)."""

    agreed_concepts: list[str] = Field(default_factory=list, max_length=24)
    rejected_options: list[str] = Field(default_factory=list, max_length=16)
    open_bottlenecks: list[str] = Field(default_factory=list, max_length=16)
    stack_mentions: list[str] = Field(default_factory=list, max_length=24)
    current_subtopic: str = Field(default="", max_length=400)


class SessionMemory(BaseModel):
    """Двухуровневая память сессии для контекста LLM."""

    rag_profile_compressed: str = Field(default="", max_length=4000)
    concepts_matrix: list[CoreConceptRecord] = Field(
        default_factory=list, max_length=12
    )
    rolling_dialogue_summary: str = Field(
        default="",
        max_length=8000,
        description="Legacy; не в hot path тьютора — используй fact_manifest",
    )
    fact_manifest: DialogueFactManifest = Field(default_factory=DialogueFactManifest)
    anchor_turn: dict[str, str] = Field(default_factory=dict)
    active_window: list[dict[str, str]] = Field(default_factory=list, max_length=8)
    dialog_seq: int = Field(
        default=0,
        ge=0,
        description="Монотонный счётчик msg_id для истории диалога",
    )
    topic_mastery_score: int = Field(default=0, ge=0, le=100)
    node_goal: str = Field(
        default="",
        max_length=800,
        description="High-level goal ноды для concept map",
    )
    sub_concepts: list[SubConceptRecord] = Field(
        default_factory=list,
        max_length=8,
        description="Topic Concept Map / coverage chips",
    )
    learning_phase: LearningPhase = "intro_assessment"
    learning_mode: LearningMode = "lecture"
    pathway_bridge: str = Field(default="", max_length=2000)
    intro_question_pending: bool = Field(
        default=False,
        description="Ждём первый ответ на intro; до него node_status=unexplored",
    )
    last_tutor_sub_concept_id: str = Field(
        default="",
        max_length=64,
        description="Legacy alias for pending_evaluation_concept_id",
    )
    last_evaluator_feedback: str = Field(
        default="",
        max_length=1200,
        description="Явная обратная связь gap evaluator для тьютора (последний ход)",
    )
    last_eval_directive: str = Field(
        default="",
        max_length=64,
        description=(
            "Директива Threshold Engine для тьютора: "
            "PROBE_NEXT_LAYER:WHY|HOW | PASSED_WITH_GLOSS | PASSED_CLEAN"
        ),
    )
    pending_evaluation_concept_id: str = Field(
        default="",
        max_length=64,
        description=(
            "ID подтемы, по которой предыдущая реплика тьютора задала вопрос; "
            "только этот концепт может быть оценён следующим сообщением пользователя"
        ),
    )
    asked_question_sub_concept_id: str = Field(
        default="",
        max_length=64,
        description=(
            "ID подтемы последнего ЗАДАННОГО вопроса тьютора; "
            "эвалюатор оценивает user_message строго относительно этого id "
            "(не относительно next_question / active generation focus)"
        ),
    )
    next_question_concept_id: str = Field(
        default="",
        max_length=64,
        description=(
            "ID подтемы, выбранной для формируемого следующего вопроса/лекции "
            "(active generation focus); становится asked/pending только после "
            "отправки реплики тьютора"
        ),
    )
    chat_sessions: dict[str, dict] = Field(default_factory=dict)
    covered_subtopics: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Персистентный реестр покрытых микро-тем лекций: id → выдержка "
            "(не сбрасывается при compact_dialog_session)"
        ),
    )
    introduced_terms: list[str] = Field(
        default_factory=list,
        max_length=64,
        description=(
            "Термины/аббревиатуры, уже расшифрованные в сессии ноды "
            "(не сбрасывается при compact_dialog_session)"
        ),
    )
    memory_updated_at: str = Field(
        default="",
        max_length=40,
        description="ISO-8601 UTC последнего коммита SessionMemory",
    )
    last_tutor_question_angle: str = Field(
        default="",
        max_length=32,
        description="Эвристический угол последнего вопроса тьютора (ротация)",
    )
    last_tutor_display_message: str = Field(
        default="",
        max_length=12_000,
        description="Полный текст последней реплики тьютора для UI (с follow_up_question)",
    )
    last_tutor_follow_up_question: str = Field(
        default="",
        max_length=2000,
        description="Последний follow_up_question (для восстановления history при reload)",
    )
    lecture_rag_inspector: list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=16,
        description="Последний набор RAG-чанков для dense_lecture (UI Inspector)",
    )


class ConceptUpdate(BaseModel):
    concept: str = Field(min_length=1, max_length=400)
    status: ConceptMasteryStatus | None = None
    evidence: str = Field(default="", max_length=2000)
    mastery_score: int | None = Field(default=None, ge=0, le=100)


class StepAnalysisOutput(BaseModel):
    intent: UserIntent = "ANSWER"
    concept_updates: list[ConceptUpdate] = Field(default_factory=list, max_length=12)
    critical_gap: str | None = Field(default=None, max_length=2000)


class RollingCompressOutput(BaseModel):
    """Структурированное сжатие диалога (собирается в markdown для layer_3)."""

    current_state: str = Field(
        default="",
        max_length=1500,
        description="[CURRENT_STATE]: режим, фаза, активная тема",
    )
    covered_points: str = Field(
        default="",
        max_length=4000,
        description="[COVERED_POINTS]: только что реально выдал ассистент и что понял пользователь",
    )
    pending_deliverables: str = Field(
        default="",
        max_length=2500,
        description="[PENDING_ACTION]: невыполненные обязательства (лекция, материал)",
    )
    next_action_for_tutor: str = Field(
        default="",
        max_length=1500,
        description="Прямая инструкция для следующего хода тьютора",
    )
