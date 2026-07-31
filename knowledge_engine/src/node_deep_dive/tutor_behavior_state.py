"""State Vector для tutor_behavior — только динамика, без дублирования system rules."""

from __future__ import annotations

import json
from typing import Any

from knowledge_engine.src.node_deep_dive.memory_schemas import UserIntent

TutorMode = str  # dialogue_feedback | lecture_dense | verify | socratic | finalize


def _wants_lecture(
    intent: UserIntent,
    learning_mode: str,
    user_message: str,
) -> bool:
    msg = (user_message or "").strip().lower()
    if intent == "INTENT_EXPLAIN":
        return True
    if learning_mode != "lecture":
        return False
    return any(
        k in msg
        for k in (
            "дай лекцию",
            "плотный материал",
            "dense material",
            "[mode:lecture]",
            "mode:lecture",
            "объясни подроб",
            "дай плотн",
        )
    )


def resolve_tutor_mode(
    intent: UserIntent,
    action: str,
    learning_mode: str,
    learning_phase: str,
    user_message: str,
) -> TutorMode:
    if action == "verify":
        return "verify"
    if learning_mode == "socratic_point" or learning_phase == "socratic_focus":
        return "socratic"
    if intent == "INTENT_FINALIZE":
        return "finalize"
    if _wants_lecture(intent, learning_mode, user_message):
        return "lecture_dense"
    if intent == "INTENT_SHIFT_FOCUS":
        return "dialogue_feedback"
    return "dialogue_feedback"


def build_tutor_behavior_state(
    intent: UserIntent,
    action: str,
    learning_mode: str,
    learning_phase: str,
    user_message: str,
    *,
    has_user_focus: bool = False,
) -> dict[str, Any]:
    mode = resolve_tutor_mode(
        intent, action, learning_mode, learning_phase, user_message
    )
    focus_restriction = (
        "Узкий фокус текущей подтемы; не разворачивать всю ноду."
        if has_user_focus or mode == "dialogue_feedback"
        else ""
    )
    next_action = _next_action_for_mode(mode, intent, action)
    return {
        "current_mode": mode,
        "step_intent": intent,
        "learning_phase": learning_phase,
        "learning_mode": learning_mode,
        "focus_restriction": focus_restriction,
        "next_action": next_action,
    }


def _next_action_for_mode(
    mode: TutorMode,
    intent: UserIntent,
    action: str,
) -> str:
    if mode == "lecture_dense":
        return "Выдать полную лекцию в tutor_message; заземлить на реальный стек."
    if mode == "verify":
        return "Финальная проверка по матрице концептов; без бесконечного допроса."
    if mode == "socratic":
        return "Один контрвопрос или edge-case; без лекции."
    if mode == "finalize":
        return "Итог по матрице и mastery; честно назвать пробелы."
    if intent == "INTENT_SHIFT_FOCUS":
        return "Сменить ракурс внутри ноды; заземлить на стек; один вопрос."
    return (
        "Валидировать ответ пользователя, заземлить на стек, задать один уточняющий вопрос."
    )


def format_tutor_behavior_state_block(state: dict[str, Any]) -> str:
    return (
        "### tutor_behavior_state\n"
        + json.dumps(state, ensure_ascii=False, indent=0)
    )
