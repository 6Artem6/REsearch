"""Tiered memory и прогресс Сократовского диалога (Модуль 2)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ConceptMasteryStatus = Literal["unverified", "in_progress", "verified"]
UserIntent = Literal[
    "ANSWER",
    "INTENT_EXPLAIN",
    "INTENT_SHIFT_FOCUS",
    "INTENT_FINALIZE",
]
NodeStatus = Literal[
    "in_progress",
    "deep_understanding",
    "mastered",
    "gap",
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
    concepts_matrix: list[CoreConceptRecord] = Field(default_factory=list, max_length=12)
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
    learning_phase: LearningPhase = "intro_assessment"
    learning_mode: LearningMode = "lecture"
    pathway_bridge: str = Field(default="", max_length=2000)
    chat_sessions: dict[str, dict] = Field(default_factory=dict)


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
