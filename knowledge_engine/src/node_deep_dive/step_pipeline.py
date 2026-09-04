"""Шаг пайплайна: интент, обновление матрицы, fact manifest (без rolling_compress в hot path)."""

from __future__ import annotations

from knowledge_engine.config import (
    GEMINI_LITE_MAX_OUTPUT_TOKENS,
    GEMINI_LITE_MODEL,
    GEMINI_RPM_PAUSE_SEC,
)
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.schemas.llm_contracts.tutor import StepAnalysisContract
from knowledge_engine.services.chat_session_manager import ChatSessionManager
from knowledge_engine.services.gemini_stateless import run_gemini_structured_with_chain
from knowledge_engine.src.node_deep_dive.concept_map import (
    process_sub_concept_user_answer,
    stored_pending_evaluation_id,
)
from knowledge_engine.src.node_deep_dive.fact_manifest import (
    format_fact_manifest_block,
    prepare_evicted_for_manifest_extraction,
)
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    SessionMemory,
    StepAnalysisOutput,
    UserIntent,
)
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput
from knowledge_engine.src.node_deep_dive.tiered_memory import (
    ACTIVE_WINDOW_MAX,
    append_to_active_window,
    apply_concept_updates,
    build_handoff_summary,
    format_matrix_for_llm,
    format_window_for_llm,
    pop_evicted_message,
)
from knowledge_engine.src.node_deep_dive.tutor_behavior_state import (
    build_tutor_behavior_state,
    format_tutor_behavior_state_block,
)
from knowledge_engine.ui.run_log import trace

_STEP_ANALYSIS_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "You analyze one Socratic dialogue step for a single curriculum node.\n"
    "Input: concept matrix, sliding window, fact_manifest (JSON), user message.\n"
    "Message-type routing (lecture / finalize / shift-focus / control chips) is "
    "already resolved deterministically upstream by VectorIntentRouter — do not "
    "classify intent here, there is no intent field to fill.\n\n"
    "1) concept_updates — map user_message and window to core_concepts.\n"
    "   For proven understanding: status=verified, evidence=short user quote, mastery_score 70-100.\n"
    "   Partial: in_progress, 20-60. Do not invent concepts outside the list.\n\n"
    "2) critical_gap — only if fundamental gap in a key concept (blocker), else null.\n"
    "   A senior architecture answer with several patterns is not critical_gap.\n"
    "3) If user_message is system design (queues, active-passive, delta index, isolation): "
    "extract concept_updates for the touched core_concepts.\n"
)
"""
RU (пояснение): step analysis теперь отвечает только за concept_updates и
critical_gap — intent резолвится детерминированно через VectorIntentRouter
в step_analysis_node (см. resolve_user_intent_from_chip ниже), LLM больше не
классифицирует тип сообщения (устраняет задержку и галлюцинации на hot path).
"""


def build_tutor_behavior_state_block(
    intent: UserIntent,
    action: str,
    learning_mode: str,
    learning_phase: str,
    user_message: str,
    *,
    has_user_focus: bool = False,
    memory: object | None = None,
    node_layer: str = "",
) -> str:
    state = build_tutor_behavior_state(
        intent,
        action,
        learning_mode,
        learning_phase,
        user_message,
        has_user_focus=has_user_focus,
        memory=memory,  # type: ignore[arg-type]
        node_layer=node_layer,
    )
    return format_tutor_behavior_state_block(state)


def heuristic_step_analysis(
    user_message: str,
    learning_phase: str = "",
) -> StepAnalysisOutput:
    """
    Без LLM, если step_analysis недоступен (503/квота) или пропущен по гейтингу.

    RU (пояснение): intent сюда не входит (см. resolve_user_intent_from_chip) —
    эта функция отвечает только за concept_updates/critical_gap, которых при
    фолбэке просто нет; никакого substring-разбора текста здесь больше нет.
    """
    _ = (user_message, learning_phase)
    return StepAnalysisOutput(concept_updates=[], critical_gap=None)


_CHIP_TO_LEGACY_INTENT: dict[str, UserIntent] = {
    "lecture": "INTENT_EXPLAIN",
    "finalize": "INTENT_FINALIZE",
    "shift_focus": "INTENT_SHIFT_FOCUS",
}


def resolve_user_intent_from_chip(chip: str) -> UserIntent:
    """
    Deterministic UserIntent from VectorIntentRouter chip — no LLM classification.

    RU (пояснение): только lecture/finalize/shift_focus транслируются в старые
    значения UserIntent (совместимость с engine.py/tutor_behavior_state.py,
    которые сравнивают именно эти строки). Любой другой распознанный чип
    (gloss/how/mech/blitz/socratic/next/practice/check/skip/begin/accept_deep/
    *_analysis/deep_design/clarify) или отсутствие совпадения — ANSWER; эти
    чипы уже отдельно маршрутизируются в coverage_router_node.
    """
    return _CHIP_TO_LEGACY_INTENT.get((chip or "").strip(), "ANSWER")


def should_run_step_analysis_llm(
    user_message: str,
    memory: SessionMemory,
    action: str,
) -> bool:
    """LLM step_analysis только при смене режима / finalize / verify."""
    from knowledge_engine.src.node_deep_dive.control_intent import (
        classify_control_chip,
        has_explicit_control_tag,
    )

    act = (action or "").strip().lower()
    if act == "verify":
        return True
    raw = (user_message or "").strip()
    from knowledge_engine.src.node_deep_dive.intent_definitions import (
        MODE_SELECTION_SLOT_INTENTS,
    )

    chip = classify_control_chip(raw, memory=memory)
    # Слот практика/проверка/пропустить уже классифицирован — Lite LLM не нужен.
    if chip in MODE_SELECTION_SLOT_INTENTS:
        return False
    if has_explicit_control_tag(raw) or "[mode:" in (raw or "").lower():
        return True
    return bool(chip)


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
    raw = run_gemini_structured_with_chain(
        GEMINI_LITE_MODEL,
        _STEP_ANALYSIS_SYSTEM,
        payload,
        anchor,
        StepAnalysisContract,
        "node_deep_dive / step_analysis",
        rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
        chat_manager=chat_mgr,
        chat_label="node_deep_dive/step_analysis",
        delta_user_message=(user_message or "").strip(),
        handoff_summary=handoff,
        max_output_tokens=GEMINI_LITE_MAX_OUTPUT_TOKENS,
    )
    memory.chat_sessions = chat_mgr.to_memory_blob()
    return StepAnalysisOutput.model_validate(raw.model_dump())


def rotate_window_after_message(
    memory: SessionMemory,
    anchor: str,
    curriculum_id: str = "",
    node_id: str = "",
) -> None:
    """Эвикция окна остаётся синхронной (дёшево), но LLM-экстракция
    fact_manifest ушла в фон (context_compressor_worker) — раньше
    update_manifest_from_evicted блокировал ответ пользователю доп.
    вызовом Gemini прямо на hot path.
    """
    from knowledge_engine.services.context_compressor_worker import (
        enqueue_dialog_summarize,
    )

    while len(memory.active_window) > ACTIVE_WINDOW_MAX:
        evicted = pop_evicted_message(memory)
        if not evicted:
            break
        payload = prepare_evicted_for_manifest_extraction(memory, evicted, anchor)
        if payload is None:
            continue
        enqueue_dialog_summarize(curriculum_id, node_id, payload)


def process_user_message_pipeline(
    user_message: str,
    memory: SessionMemory,
    node: NodeDataInput,
    anchor: str,
    action: str,
) -> tuple[UserIntent, str | None]:
    """Интент, обновление матрицы; user ещё не в active_window."""
    if should_run_step_analysis_llm(user_message, memory, action):
        try:
            analysis = run_step_analysis(user_message, memory, node, anchor)
        except Exception as exc:
            trace(
                f"NODE_DIVE step_analysis fallback (heuristic) | {type(exc).__name__}: {exc}"
            )
            analysis = heuristic_step_analysis(user_message, memory.learning_phase)
    else:
        trace("NODE_DIVE step_analysis skip | heuristic (latency)")
        analysis = heuristic_step_analysis(user_message, memory.learning_phase)
    apply_concept_updates(memory, analysis.concept_updates)
    intent = analysis.intent
    gap = (analysis.critical_gap or "").strip() or None
    act = (action or "").strip().lower()
    if act in ("chat", "verify") and (user_message or "").strip():
        try:
            process_sub_concept_user_answer(
                user_message,
                memory,
                node,
                anchor,
            )
        except Exception as exc:
            trace(f"EVALUATOR_ERROR | pipeline | {type(exc).__name__}: {exc}")
            pending = stored_pending_evaluation_id(memory)
            if pending:
                from knowledge_engine.src.node_deep_dive.concept_map import (
                    find_sub_concept,
                )

                row = find_sub_concept(memory, pending)
                if row is not None and row.status == "unchecked":
                    row.status = "partial"
                    row.focus_hint = "Оценка ответа не завершилась; уточните детали по критерию подтемы."
                    memory.last_evaluator_feedback = (
                        f"Подтема «{row.label}»: автоматическая оценка не завершилась "
                        f"({type(exc).__name__})."
                    )
            trace(
                f"NODE_DIVE sub_concept evaluation FAILED | "
                f"{type(exc).__name__}: {exc}"
            )
    append_to_active_window(memory, "user", user_message)
    rotate_window_after_message(memory, anchor)
    return intent, gap
