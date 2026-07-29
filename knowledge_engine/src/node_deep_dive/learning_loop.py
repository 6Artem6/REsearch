"""Ритмичный учебный цикл и панель mastery (Модуль 2)."""

from __future__ import annotations

from knowledge_engine.src.node_deep_dive.memory_schemas import (
    CoreConceptRecord,
    LearningMode,
    LearningPhase,
    NodeStatus,
    SessionMemory,
)
from knowledge_engine.src.node_deep_dive.schemas import MasteryDashboard


def build_mastery_dashboard(
    memory: SessionMemory | None,
    node_status: NodeStatus,
) -> MasteryDashboard:
    score = memory.topic_mastery_score if memory else 0
    phase: LearningPhase = (
        memory.learning_phase if memory else "intro_assessment"
    )
    mode: LearningMode = memory.learning_mode if memory else "lecture"
    strengths: list[str] = []
    polish: list[str] = []
    critical: list[str] = []
    for row in (memory.concepts_matrix if memory else []):
        ev = (row.evidence or "").strip()
        label = row.concept
        if row.status == "verified" and row.mastery_score >= 70:
            strengths.append(f"{label}" + (f": {ev[:120]}" if ev else ""))
        elif row.status == "in_progress" or 20 <= row.mastery_score < 70:
            polish.append(label)
        elif row.status == "unverified" and row.mastery_score < 20:
            polish.append(label)
        if row.mastery_score < 25 and row.status != "verified":
            critical.append(label)
    return MasteryDashboard(
        topic_mastery_score=score,
        node_status=node_status,
        strengths=strengths[:8],
        polish_zones=polish[:8],
        critical_gaps=critical[:6],
        learning_phase=phase,
        learning_mode=mode,
        pathway_bridge=(memory.pathway_bridge or "") if memory else "",
    )


def set_learning_mode(memory: SessionMemory, mode: LearningMode) -> None:
    memory.learning_mode = mode
    if mode == "socratic_point":
        memory.learning_phase = "socratic_focus"
    elif mode == "express_blitz":
        memory.learning_phase = "intro_assessment"
    elif mode == "lecture":
        if memory.learning_phase == "socratic_focus":
            memory.learning_phase = "dense_material"
        elif memory.learning_phase in ("checkpoint", "pathway_decision"):
            memory.learning_phase = "dense_material"


def advance_phase_after_chat(
    memory: SessionMemory,
    intent: str,
    action: str,
) -> None:
    if action == "verify":
        memory.learning_mode = "socratic_point"
        memory.learning_phase = "socratic_focus"
        return
    if intent == "INTENT_EXPLAIN":
        memory.learning_mode = "lecture"
        memory.learning_phase = "dense_material"
        return
    if intent == "INTENT_FINALIZE":
        memory.learning_phase = "pathway_decision"
        return
    if memory.learning_phase == "intro_assessment":
        memory.learning_phase = "checkpoint"
        return
    if memory.learning_phase == "dense_material":
        memory.learning_phase = "checkpoint"
    elif memory.learning_phase == "checkpoint":
        memory.learning_phase = "pathway_decision"
