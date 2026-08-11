"""State Vector для tutor_behavior — только динамика, без дублирования system rules."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from knowledge_engine.src.node_deep_dive.memory_schemas import UserIntent

if TYPE_CHECKING:
    from knowledge_engine.src.node_deep_dive.memory_schemas import SessionMemory

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
    memory: SessionMemory | None = None,
    node_layer: str = "",
) -> dict[str, Any]:
    mode = resolve_tutor_mode(
        intent, action, learning_mode, learning_phase, user_message
    )
    focus_restriction = (
        "Узкий фокус текущей подтемы; не разворачивать всю ноду."
        if has_user_focus or mode == "dialogue_feedback"
        else ""
    )
    next_action = _next_action_for_mode(
        mode,
        intent,
        action,
        learning_phase,
        memory=memory,
        user_message=user_message,
        node_layer=node_layer,
    )
    return {
        "current_mode": mode,
        "step_intent": intent,
        "learning_phase": learning_phase,
        "learning_mode": learning_mode,
        "node_layer": (node_layer or "").strip() or "foundation",
        "focus_restriction": focus_restriction,
        "next_action": next_action,
        "last_eval_directive": (
            (memory.last_eval_directive or "").strip() if memory is not None else ""
        ),
    }


def _next_action_for_mode(
    mode: TutorMode,
    intent: UserIntent,
    action: str,
    learning_phase: str = "",
    *,
    memory: SessionMemory | None = None,
    user_message: str = "",
    node_layer: str = "",
) -> str:
    from knowledge_engine.src.node_deep_dive.concept_map import (
        classify_gloss_fork_choice,
        first_optional_layer_concept_id,
        gloss_fork_quick_replies,
        is_full_depth_closure,
        open_optional_layers,
        sub_concept_coverage_complete,
        user_accepted_optional_deep_dive,
    )
    from knowledge_engine.src.node_deep_dive.sub_concept_evaluator import (
        normalize_node_layer,
    )

    phase = (learning_phase or "").strip()
    ly = normalize_node_layer(node_layer)
    if memory is not None and sub_concept_coverage_complete(memory):
        choice = classify_gloss_fork_choice(user_message)
        open_layers = open_optional_layers(memory, ly)
        if choice == "gloss":
            named = "/".join(open_layers) or "optional"
            return (
                f"GLOSS_FORK_CHOICE=gloss: give a short Glossary digest of open optional "
                f"layer(s) [{named}] with key formulas/patterns (2–8 sentences); NO quiz; "
                "system will auto-credit those optional layers; "
                "ready_for_transition=true; suggested_next_step=next_node; "
                "question_sub_concept_id=null; quick_replies=[]; "
                "invite UI next-node choice; do not invent next node titles."
            )
        if choice in ("how", "mech"):
            layer_name = "HOW" if choice == "how" else "MECHANIC"
            cid = (
                first_optional_layer_concept_id(memory, layer_name) or "open sub-topic"
            )
            if choice == "mech":
                return (
                    f"GLOSS_FORK_CHOICE=mech DEEP_DIVE: Active Teaching on `{cid}`. "
                    "ready_for_transition=false; suggested_next_step=null; "
                    f"question_sub_concept_id={cid}. "
                    "HARD MECH RULES: in technical_explanation you MUST include at least one "
                    "hands-on artifact — Python/Pydantic code, asyncio architecture snippet, "
                    "OR explicit mathematical consensus/weighting formula in $LaTeX$. "
                    "FORBIDDEN: abstract summary without code/math. "
                    "Then follow_up_question = ONE short edge-case question about THAT "
                    "code/formula (verify mastery). Do NOT close the node; wait for the "
                    "user answer (Evaluator scores the next turn)."
                )
            return (
                f"GLOSS_FORK_CHOICE=how DEEP_DIVE: Active Teaching on `{cid}`. "
                "ready_for_transition=false; suggested_next_step=null; "
                f"question_sub_concept_id={cid}. "
                "Deliver concrete architecture/invariants (diagrams/pipelines OK); "
                "follow_up_question = ONE targeted HOW question. "
                "Do NOT set ready_for_transition=true; wait for user answer."
            )
        if choice == "next":
            return (
                "GLOSS_FORK_CHOICE=next: confirm readiness; no quiz; "
                "ready_for_transition=true; suggested_next_step=next_node; "
                "question_sub_concept_id=null; quick_replies=[]; "
                "do not invent next node titles — UI picks the next node."
            )
        if open_layers and not is_full_depth_closure(memory, ly) and ly != "sota":
            labels = gloss_fork_quick_replies(open_layers)
            named = " и ".join(open_layers)
            return (
                f"TOPIC_COMPLETE + OPTIONAL_LAYER FORK (node_layer={ly}): "
                f"threshold met; optional [{named}] still open. "
                "Russian: «Концептуальный минимум ноды освоен! Но у нас остался "
                f"опциональный слой {named}. Выбери одно из действий ниже…». "
                f"JSON: ready_for_transition=true; quick_replies={labels!r}; "
                "NON-EMPTY follow_up_question; FORBIDDEN: inventing next node titles; "
                "FORBIDDEN on SotA (not this branch)."
            )
        return (
            f"TOPIC_COMPLETE FULL DEPTH (node_layer={ly}): "
            "Russian: «Нода полностью освоена на 100%! Мы готовы двигаться дальше "
            "по графу знаний. Выбери следующее действие.» "
            "JSON: ready_for_transition=true; suggested_next_step=next_node; "
            "question_sub_concept_id=null; quick_replies=[]; "
            "FORBIDDEN: technical quiz; inventing next node titles."
        )
    if phase == "pathway_decision":
        return (
            "PATHWAY: node wrap-up; ready_for_transition=true; "
            "suggested_next_step=next_node or deep_dive_optional; "
            "NON-EMPTY follow_up_question CTA; no automatic technical quiz; "
            "do not invent next node titles."
        )
    if user_accepted_optional_deep_dive(user_message):
        return (
            "Пользователь согласился на углубление — один edge-case вопрос по выбранной "
            "подтеме; ready_for_transition=false."
        )
    if mode == "lecture_dense":
        return "Выдать полную лекцию в tutor_message; заземлить на реальный стек."
    if mode == "verify":
        return "Финальная проверка по матрице концептов; без бесконечного допроса."
    if mode == "socratic":
        return "Один контрвопрос или edge-case; без лекции."
    if mode == "finalize":
        return "Итог по матрице и mastery; честно назвать пробелы."
    if intent == "INTENT_SHIFT_FOCUS":
        return (
            "Сменить ракурс в рамках ответа пользователя; разбор его паттернов; "
            "без сброса на базовую методичку."
        )
    if memory is not None:
        from knowledge_engine.src.node_deep_dive.concept_map import find_sub_concept

        directive = (memory.last_eval_directive or "").strip()
        if directive == "PASSED_WITH_GLOSS":
            return (
                "THRESHOLD PASSED_WITH_GLOSS (mid-map): credit required layers, gloss "
                "open optional depth yourself, then advance to next sub_concept; "
                "FORBIDDEN to quiz optional depth unless user asks; "
                "ready_for_transition stays false unless map is fully VERIFIED "
                "(then use TOPIC_COMPLETE fork by node_layer)."
            )
        if directive == "PASSED_CLEAN":
            return (
                "THRESHOLD PASSED_CLEAN: brief credit and move to next UNCHECKED "
                "sub_concept; no extra grilling on the closed id."
            )
        if directive.startswith("PROBE_NEXT_LAYER:"):
            layer = directive.split(":", 1)[-1].strip().upper() or "WHY"
            return (
                f"THRESHOLD {directive}: stay on the SAME sub_concept; "
                f"follow_up_question MUST probe ONLY layer {layer} "
                "(do not demand deeper layers). "
                "feedback_on_answer MUST open with CRITICAL TRANSPARENCY block "
                "(📋 credited / 🎯 missing = verbatim focus_hint); "
                "ready_for_transition=false."
            )
        nid = (memory.next_question_concept_id or "").strip()
        row = find_sub_concept(memory, nid) if nid else None
        if row is not None and row.status in ("partial", "gap"):
            hint = (row.focus_hint or "").strip()
            return (
                "PARTIAL/GAP on current sub-topic: do NOT switch to a new sub_concept. "
                "feedback_on_answer MUST open with CRITICAL TRANSPARENCY block "
                "(📋 credited / 🎯 missing = verbatim focus_hint)"
                + (f": «{hint[:200]}»" if hint else "")
                + "; follow_up_question — probe same id per last_eval_directive "
                "(WHY/HOW only as directed); ready_for_transition=false."
            )
    if phase in ("checkpoint", "pathway_decision"):
        return (
            "Peer-разбор ответа на «ты»; затем следующая подтема concept_map; "
            "без отчётных заголовков и без третьего вопроса по той же теме."
        )
    return (
        "Peer-to-peer на «ты» (без «пользователь указал»); развернутый инженерный разбор "
        "(~180+ слов минимум, без искусственного потолка длины); "
        "разбери паттерны из user_message; obey last_eval_directive from Threshold Engine; "
        "после VERIFIED — переход к следующей sub_concept; при PARTIAL — probe only the "
        "directed layer; один новый вопрос максимум."
    )


def format_tutor_behavior_state_block(state: dict[str, Any]) -> str:
    return "### tutor_behavior_state\n" + json.dumps(
        state, ensure_ascii=False, indent=0
    )
