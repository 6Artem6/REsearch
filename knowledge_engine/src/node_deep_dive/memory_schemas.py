"""Tiered memory и прогресс Сократовского диалога (Модуль 2)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from knowledge_engine.src.node_deep_dive.tutor_field_limits import (
    SCHEMA_FOLLOW_UP_QUESTION_MAX,
    SCHEMA_TUTOR_MESSAGE_MAX,
)

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
# Overlay asterisk-question tracks (parallel to core WHY/HOW/MECH).
OverlayKind = Literal["advanced_analysis", "deep_design"]
OverlayType = Literal["ADVANCED_ASTERISK", "DEEP_ASTERISK"]
LayerDrillLayer = Literal[
    "WHY",
    "HOW",
    "MECH",
    "ADVANCED_ASTERISK",
    "DEEP_ASTERISK",
]
PendingEvalKind = Literal[
    "",
    "gap",
    "deep_analysis",  # legacy alias of deep_design (Bloom L5/L6)
    "advanced_analysis",
    "deep_design",
]
PendingControlSlot = Literal["", "mode_selection"]


SUBCONCEPT_EVIDENCE_MAX = 2000
_DEGRADED_EVIDENCE_PREFIX = "evaluator_degraded"


def accumulate_evidence_text(
    existing: str,
    incoming: str,
    *,
    max_len: int = SUBCONCEPT_EVIDENCE_MAX,
) -> str:
    """Append credited theses without wiping prior evidence (dedup + cap).

    Service markers such as ``evaluator_degraded:…`` never replace real credit
    and are dropped once a real thesis arrives.
    """
    existing = (existing or "").strip()
    incoming = (incoming or "").strip()
    if not incoming:
        return existing[:max_len]
    incoming_degraded = incoming.casefold().startswith(_DEGRADED_EVIDENCE_PREFIX)
    existing_degraded = existing.casefold().startswith(_DEGRADED_EVIDENCE_PREFIX)
    if incoming_degraded:
        if existing and not existing_degraded:
            return existing[:max_len]
        return incoming[:max_len]
    if existing_degraded:
        existing = ""
    if not existing:
        return incoming[:max_len]
    existing_cf = existing.casefold()
    incoming_cf = incoming.casefold()
    if incoming_cf in existing_cf:
        return existing[:max_len]
    if existing_cf in incoming_cf:
        return incoming[:max_len]

    def _fragments(text: str) -> list[str]:
        parts: list[str] = []
        for chunk in text.replace("\n", ";").split(";"):
            item = chunk.strip()
            if item:
                parts.append(item)
        return parts

    seen: set[str] = set()
    ordered: list[str] = []
    for frag in _fragments(existing) + _fragments(incoming):
        key = frag.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(frag)
    joined = "; ".join(ordered)
    if len(joined) <= max_len:
        return joined
    newest = _fragments(incoming)
    newest_keys = {f.casefold() for f in newest}
    kept: list[str] = []
    size = 0

    def _fits(frag: str) -> bool:
        nonlocal size
        extra = len(frag) + (2 if kept else 0)
        if size + extra > max_len:
            return False
        size += extra
        kept.append(frag)
        return True

    for frag in ordered:
        if frag.casefold() in newest_keys:
            continue
        if not _fits(frag):
            break
    for frag in newest:
        key = frag.casefold()
        if any(item.casefold() == key for item in kept):
            continue
        if not _fits(frag):
            overflow = "; ".join(kept + [frag]) if kept else frag
            return overflow[:max_len]
    return "; ".join(kept) if kept else incoming[:max_len]


class CoreConceptRecord(BaseModel):
    concept: str = Field(min_length=1, max_length=400)
    status: ConceptMasteryStatus = "unverified"
    evidence: str = Field(default="", max_length=SUBCONCEPT_EVIDENCE_MAX)
    mastery_score: int = Field(default=0, ge=0, le=100)


class SubConceptRecord(BaseModel):
    """Атомарная подтема ноды (Topic Concept Map)."""

    id: str = Field(min_length=2, max_length=64)
    label: str = Field(min_length=2, max_length=200)
    success_criterion: str = Field(default="", max_length=600)
    status: SubConceptStatus = "unchecked"
    evidence: str = Field(default="", max_length=SUBCONCEPT_EVIDENCE_MAX)
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
    failed_attempts: int = Field(
        default=0,
        ge=0,
        le=99,
        description="Consecutive non-exact answers on this sub-topic (Host counter)",
    )
    last_accuracy_grade: str = Field(
        default="",
        max_length=32,
        description="Last Evaluator AnswerAccuracyGrade value (empty if never scored)",
    )
    is_extension: bool = Field(
        default=False,
        description=(
            "True for overlay/extension subtopics (ADVANCED / DEEP asterisk-question). "
            "Excluded from core topic_mastery_score denominator."
        ),
    )
    updated_at: str = Field(
        default="",
        max_length=40,
        description="ISO-8601 UTC последнего изменения статуса",
    )

    @property
    def is_verified(self) -> bool:
        return self.status == "verified"

    def merge_evidence(self, incoming: str) -> None:
        """Accumulate credited theses; never wipe prior credit with a last-turn digest."""
        self.evidence = accumulate_evidence_text(self.evidence, incoming)


class OverlayMasteryRecord(BaseModel):
    """Asterisk-question overlay award for one sub-concept (L4 or L5/L6)."""

    concept_id: str = Field(min_length=1, max_length=64)
    overlay_type: OverlayType = "DEEP_ASTERISK"


class LayerDrillSession(BaseModel):
    """Host-owned Layer Drill Session — persists until every queued subtopic passes."""

    is_active: bool = False
    target_layer: LayerDrillLayer | None = None
    target_sub_concept_ids: list[str] = Field(default_factory=list, max_length=8)
    current_index: int = Field(default=0, ge=0)
    status: Literal["", "DRILL_IN_PROGRESS", "DRILL_COMPLETE"] = ""

    def get_current_sub_concept_id(self) -> str | None:
        if self.is_active and 0 <= self.current_index < len(
            self.target_sub_concept_ids
        ):
            return self.target_sub_concept_ids[self.current_index]
        return None

    def has_more_questions(self) -> bool:
        """True while a queued sub-topic still needs a checkpoint question."""
        if not self.is_active:
            return False
        ids = self.target_sub_concept_ids or []
        return 0 <= self.current_index < len(ids)

    def advance_or_complete(self) -> bool:
        """Move to the next queued subtopic. True when the layer drill is finished."""
        self.current_index += 1
        if self.current_index >= len(self.target_sub_concept_ids):
            self.is_active = False
            self.status = "DRILL_COMPLETE"
            return True
        self.status = "DRILL_IN_PROGRESS"
        return False


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
    manifest_version: int = Field(
        default=0,
        ge=0,
        description=(
            "CAS-версия fact_manifest — растёт на каждый успешный merge из "
            "фонового context_compressor_worker; используется для отказа от "
            "записи устаревшего summary (см. apply_fact_manifest_patch)"
        ),
    )
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
    pending_control_slot: PendingControlSlot = Field(
        default="",
        description=(
            "Активный FSM-слот Host. mode_selection — ждать "
            "практика / проверка / пропустить после fast-track."
        ),
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
    last_evaluator_critique: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Phase 1+: structured EvaluatorCritiqueContract dump for overlay/"
            "deep_analysis turns (Tutor Phase 2 consumes this directly)."
        ),
    )
    last_eval_directive: str = Field(
        default="",
        max_length=64,
        description=(
            "Директива Threshold Engine для тьютора: "
            "PROBE_NEXT_LAYER:WHY|HOW | PASSED_WITH_GLOSS | PASSED_CLEAN"
        ),
    )
    evaluator_skipped: bool = Field(
        default=False,
        description=(
            "True when this turn did not run the Evaluator (no pending, "
            "lecture, control chip, or answer shorter than 5 characters). "
            "Tutor must not emit TechnicalConceptAudit."
        ),
    )
    is_layer_just_completed: bool = Field(
        default=False,
        description=(
            "One-shot Evaluator latch: this scored turn closed the current "
            "layer. Host routes Tutor to LayerCompletionTutorOutput (no "
            "next_question) then clears the flag after commit."
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
    pending_eval_kind: PendingEvalKind = Field(
        default="",
        description=(
            "Какой evaluator использовать для pending: "
            "gap (default) | advanced_analysis (Bloom L4 asterisk-question) | "
            "deep_design (Bloom L5/L6 asterisk-question) | "
            "deep_analysis (legacy alias of deep_design). "
            "Сбрасывается вместе с pending после оценки "
            "(кроме star_task needs_refinement — kind сохраняется)."
        ),
    )
    active_optional_layer: Literal["", "HOW", "MECHANIC"] = Field(
        default="",
        description=(
            "Compat mirror of layer_drill when target_layer is HOW/MECH. "
            "Prefer layer_drill.is_active as the session SSOT."
        ),
    )
    layer_drill: LayerDrillSession = Field(
        default_factory=LayerDrillSession,
        description=(
            "Layer Drill Session (HOW / MECH / overlay asterisk). "
            "While is_active, Host must not declare the node/layer complete "
            "and must walk every queued subtopic with that layer still open."
        ),
    )
    star_task_status: Literal[
        "not_started",
        "in_progress",
        "needs_refinement",
        "resolved",
    ] = Field(
        default="not_started",
        description=(
            "FSM задачки со звёздочкой (asterisk-question overlay): "
            "not_started | in_progress | needs_refinement | resolved. "
            "Пока in_progress/needs_refinement — ready_for_transition запрещён."
        ),
    )
    asterisk_fsm_hops: int = Field(
        default=0,
        ge=0,
        description="Asterisk-question FSM transitions this host turn (cap 5).",
    )
    deep_mastery_concepts: list[OverlayMasteryRecord] = Field(
        default_factory=list,
        max_length=16,
        description=(
            "Реестр Overlay-наград (concept_id + overlay_type ADVANCED_ASTERISK | "
            "DEEP_ASTERISK) после успешной задачки со звёздочкой. Параллелен "
            "WHY/HOW/MECH — не входит в базовый 100% depth score. "
            "Legacy sessions may store bare concept-id strings (coerced to "
            "DEEP_ASTERISK)."
        ),
    )

    @field_validator("deep_mastery_concepts", mode="before")
    @classmethod
    def _coerce_overlay_mastery_records(cls, v: Any) -> list[Any]:
        if not v:
            return []
        out: list[Any] = []
        for item in v:
            if isinstance(item, str):
                cid = item.strip()
                if cid:
                    out.append({"concept_id": cid, "overlay_type": "DEEP_ASTERISK"})
            else:
                out.append(item)
        return out

    deep_analysis_used_source_ids: list[str] = Field(
        default_factory=list,
        max_length=32,
        description=(
            "Asterisk-question coverage: cited [Sx] / asset_id from prior deep_analysis turns "
            "(separate from lecture covered_subtopics)."
        ),
    )
    deep_analysis_used_atom_keys: list[str] = Field(
        default_factory=list,
        max_length=32,
        description=(
            "Asterisk-question coverage: stable keys of dialog atoms cited/shown as [Rx] "
            "(exclude on next retrieve)."
        ),
    )
    deep_analysis_used_chunk_ids: list[str] = Field(
        default_factory=list,
        max_length=64,
        description=(
            "Asterisk-question coverage: vector DB chunk/atom ids already fed into deep_analysis "
            "(hard exclude on next LanceDB search)."
        ),
    )
    deep_analysis_prior_digests: list[str] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "Asterisk-question coverage: short digests of prior technical_explanation "
            "(section headings / themes) for novelty instructions."
        ),
    )
    last_deep_analysis_atom_keys: list[str] = Field(
        default_factory=list,
        max_length=24,
        description=(
            "Turn stash: ordered atom keys for the current Asterisk-question retrieval "
            "([R1]=index 0). Used by commit to map citations → used_atom_keys."
        ),
    )
    last_deep_analysis_chunk_ids: list[str] = Field(
        default_factory=list,
        max_length=48,
        description=(
            "Turn stash: vector chunk/atom ids shown this Asterisk-question turn "
            "(merged into used_chunk_ids on commit)."
        ),
    )
    last_deep_analysis_atom_ids: list[str] = Field(
        default_factory=list,
        max_length=24,
        description=(
            "Turn stash: knowledge_atoms row ids for [R1]… this Asterisk-question turn."
        ),
    )
    socratic_poles_snapshot: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Host cache of last Asterisk-question Socratic Poles payload "
            "(repulsion/attraction FACT_* rows) for commit upsert "
            "into the mental-map vector store."
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
        max_length=SCHEMA_TUTOR_MESSAGE_MAX,
        description="Полный текст последней реплики тьютора для UI (с follow_up_question)",
    )
    last_tutor_follow_up_question: str = Field(
        default="",
        max_length=SCHEMA_FOLLOW_UP_QUESTION_MAX,
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
