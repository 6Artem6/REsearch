"""Шаг пайплайна: интент, обновление матрицы, fact manifest (без rolling_compress в hot path)."""

from __future__ import annotations

import json

from knowledge_engine.config import GEMINI_LITE_MODEL, GEMINI_RPM_PAUSE_SEC
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.services.chat_session_manager import ChatSessionManager
from knowledge_engine.services.gemini_stateless import run_gemini_structured_with_chain
from knowledge_engine.ui.run_log import trace
from knowledge_engine.src.node_deep_dive.fact_manifest import (
    format_fact_manifest_block,
    update_manifest_from_evicted,
)
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    SessionMemory,
    StepAnalysisOutput,
    UserIntent,
)
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput
from knowledge_engine.src.node_deep_dive.tutor_behavior_state import (
    build_tutor_behavior_state,
    format_tutor_behavior_state_block,
)
from knowledge_engine.src.node_deep_dive.tiered_memory import (
    ACTIVE_WINDOW_MAX,
    apply_concept_updates,
    append_to_active_window,
    build_handoff_summary,
    format_matrix_for_llm,
    format_window_for_llm,
    pop_evicted_message,
)

_STEP_ANALYSIS_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "Ты — аналитик шага Сократовского диалога по одной учебной ноде.\n"
    "На вход: матрица концептов, sliding window, fact_manifest (JSON), сообщение пользователя.\n\n"
    "1) intent — тип сообщения:\n"
    "  ANSWER — ответ на вопрос тьютора / аргумент по теме (режим dialogue_feedback, НЕ лекция).\n"
    "  INTENT_EXPLAIN — только явный запрос теории, схемы, плотной лекции, [mode:lecture].\n"
    "  INTENT_SHIFT_FOCUS — смена ракурса внутри ноды.\n"
    "  INTENT_FINALIZE — подведение итогов, «закрой тему».\n\n"
    "Не классифицируй развёрнутый ответ пользователя на вопрос тьютора как INTENT_EXPLAIN.\n"
    "Если learning_phase=intro_assessment — intent почти всегда ANSWER, "
    "кроме [mode:lecture] / явного запроса плотной лекции.\n\n"
    "2) concept_updates — сопоставь аргументы user_message и окна с core_concepts.\n"
    "   Для доказанного понимания: status=verified, evidence=короткая цитата пользователя, "
    "mastery_score 70-100. Частичное: in_progress, 20-60. Не выдумывай концепты вне списка.\n\n"
    "3) critical_gap — только если базовый пробел в ключевом концепте (блокер), иначе null.\n"
)


def build_tutor_behavior_state_block(
    intent: UserIntent,
    action: str,
    learning_mode: str,
    learning_phase: str,
    user_message: str,
    *,
    has_user_focus: bool = False,
) -> str:
    state = build_tutor_behavior_state(
        intent,
        action,
        learning_mode,
        learning_phase,
        user_message,
        has_user_focus=has_user_focus,
    )
    return format_tutor_behavior_state_block(state)


def heuristic_step_analysis(
    user_message: str,
    learning_phase: str = "",
) -> StepAnalysisOutput:
    """Без LLM, если step_analysis недоступен (503/квота)."""
    text = (user_message or "").lower()
    phase = (learning_phase or "").strip()
    if phase == "intro_assessment":
        if not any(
            k in text
            for k in (
                "[mode:lecture]",
                "mode:lecture",
                "дай лекцию",
                "плотный материал",
                "dense material",
            )
        ):
            return StepAnalysisOutput(intent="ANSWER", concept_updates=[], critical_gap=None)
    intent: UserIntent = "ANSWER"
    if any(
        k in text
        for k in (
            "[mode:lecture]",
            "mode:lecture",
            "дай лекцию",
            "плотный материал",
            "dense material",
            "объясни подроб",
            "разжуй",
            "покажи пример",
            "пример кода",
            "дай схем",
            "нарисуй",
            "mermaid",
            "как работает",
        )
    ):
        intent = "INTENT_EXPLAIN"
    elif any(
        k in text
        for k in ("итог", "закрой тему", "заверш", "подведи", "резюме", "finalize")
    ):
        intent = "INTENT_FINALIZE"
    elif any(
        k in text
        for k in ("другой ракурс", "смени тему", "другую тему", "переключ")
    ):
        intent = "INTENT_SHIFT_FOCUS"
    return StepAnalysisOutput(intent=intent, concept_updates=[], critical_gap=None)


def run_step_analysis(
    user_message: str,
    memory: SessionMemory,
    node: NodeDataInput,
    anchor: str,
) -> StepAnalysisOutput:
    payload = (
        f"### learning_phase\n{memory.learning_phase}\n"
        f"### learning_mode\n{memory.learning_mode}\n"
        f"### node_title\n{node.title}\n"
        f"### core_concepts\n"
        + "\n".join(f"- {c}" for c in node.core_concepts)
        + f"\n\n### concepts_matrix\n{format_matrix_for_llm(memory.concepts_matrix)}\n"
        f"{format_fact_manifest_block(memory)}\n"
        f"### sliding_window\n{format_window_for_llm(memory.active_window)}\n"
    )
    chat_mgr = ChatSessionManager.from_memory_blob(anchor, memory.chat_sessions)
    handoff = build_handoff_summary(memory)
    out = run_gemini_structured_with_chain(
        GEMINI_LITE_MODEL,
        _STEP_ANALYSIS_SYSTEM,
        payload,
        anchor,
        StepAnalysisOutput,
        "node_deep_dive / step_analysis",
        rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
        chat_manager=chat_mgr,
        chat_label="node_deep_dive/step_analysis",
        delta_user_message=(user_message or "").strip(),
        handoff_summary=handoff,
    )
    memory.chat_sessions = chat_mgr.to_memory_blob()
    return out


def rotate_window_after_message(
    memory: SessionMemory,
    anchor: str,
) -> None:
    while len(memory.active_window) > ACTIVE_WINDOW_MAX:
        evicted = pop_evicted_message(memory)
        if not evicted:
            break
        update_manifest_from_evicted(memory, evicted, anchor)


def process_user_message_pipeline(
    user_message: str,
    memory: SessionMemory,
    node: NodeDataInput,
    anchor: str,
    action: str,
) -> tuple[UserIntent, str | None]:
    """Интент, обновление матрицы; user ещё не в active_window."""
    try:
        analysis = run_step_analysis(user_message, memory, node, anchor)
    except Exception as exc:
        trace(
            f"NODE_DIVE step_analysis fallback (heuristic) | {type(exc).__name__}: {exc}"
        )
        analysis = heuristic_step_analysis(
            user_message, memory.learning_phase
        )
    apply_concept_updates(memory, analysis.concept_updates)
    intent = analysis.intent
    gap = (analysis.critical_gap or "").strip() or None
    append_to_active_window(memory, "user", user_message)
    rotate_window_after_message(memory, anchor)
    return intent, gap
