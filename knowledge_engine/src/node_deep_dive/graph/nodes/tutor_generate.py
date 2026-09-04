"""Tutor LLM generation node (dialogue_feedback / lecture_chat)."""

from __future__ import annotations

from typing import Any

from knowledge_engine.schemas.fsm import TutorStage
from knowledge_engine.services.chat_session_manager import ChatSessionManager
from knowledge_engine.src.node_deep_dive.graph.stage_events import stage_scope
from knowledge_engine.src.node_deep_dive.graph.state import TutorGraphState
from knowledge_engine.src.node_deep_dive.tiered_memory import build_handoff_summary
from knowledge_engine.src.node_deep_dive.tutor_dialogue import (
    resolve_tutor_display_message,
)
from knowledge_engine.ui.run_log import trace


def _stream_from_config(config: dict[str, Any] | None) -> Any:
    if not config:
        return None
    return (config.get("configurable") or {}).get("stream_callback")


def tutor_generate_node(
    state: TutorGraphState,
    config: dict[str, Any] | None = None,
) -> TutorGraphState:
    """Compose system prompt and call Gemini (``_invoke_tutor``) — FSM stage
    wrapper, см. graph/stage_events.py."""
    with stage_scope(
        state,
        config,
        TutorStage.LLM_GENERATE,
        running_message="Генерация нового сообщения…",
    ):
        return _tutor_generate_node_impl(state, config)


def _tutor_generate_node_impl(
    state: TutorGraphState,
    config: dict[str, Any] | None = None,
) -> TutorGraphState:
    from knowledge_engine.src.node_deep_dive.engine import _invoke_tutor, _merge_content

    req = state["request"]
    memory = state["memory"]
    node = req.node_data
    anchor = state["anchor"]
    action = (req.user_action or "").strip().lower()
    # Keep [mode:…] prefixes — Prompt Factory in _invoke_tutor selects
    # the isolated system prompt and strips the tag for the LLM body.
    user_msg = (req.user_message or "").strip()

    intent = state.get("intent") or "ANSWER"
    chat_mgr = ChatSessionManager.from_memory_blob(anchor, memory.chat_sessions)
    handoff = build_handoff_summary(memory)
    stream_callback = _stream_from_config(config)
    content = state.get("content")
    try:
        llm_out = _invoke_tutor(
            memory,
            node,
            intent,
            action,
            user_msg,
            anchor,
            f"node_deep_dive / {action}",
            chat_mgr,
            handoff,
            req.curriculum_id,
            content,
            stream_callback,
            f"{req.curriculum_id}/{node.node_id}",
        )
        from knowledge_engine.src.node_deep_dive.subconcept_invariants import (
            enforce_question_sub_concept_invariant,
        )

        llm_out, drifted = enforce_question_sub_concept_invariant(memory, llm_out)
        if drifted:
            trace(
                "STATE_DRIFT retry | regenerating tutor turn with stripped chat_history"
            )
            llm_out = _invoke_tutor(
                memory,
                node,
                intent,
                action,
                user_msg,
                anchor,
                f"node_deep_dive / {action} / drift_retry",
                chat_mgr,
                handoff,
                req.curriculum_id,
                content,
                stream_callback,
                f"{req.curriculum_id}/{node.node_id}",
                strip_chat_history=True,
                emit_stream_plaque=False,
            )
            llm_out, drifted2 = enforce_question_sub_concept_invariant(memory, llm_out)
            if drifted2:
                trace(
                    "STATE_DRIFT persist | force question_sub_concept_id="
                    f"{llm_out.question_sub_concept_id}"
                )
    except Exception as exc:
        from knowledge_engine.src.node_deep_dive.schemas import DeepDiveLLMOutput
        from knowledge_engine.src.resilience_manager import (
            degraded_student_message,
            is_llm_resilience_error,
            is_tutor_contract_validation_error,
        )

        if not (
            is_llm_resilience_error(exc) or is_tutor_contract_validation_error(exc)
        ):
            raise
        trace(f"TUTOR_GENERATE degrade | {type(exc).__name__}: {exc} | FSM preserved")
        qid = (
            (memory.next_question_concept_id or "").strip()
            or (memory.pending_evaluation_concept_id or "").strip()
            or None
        )
        msg = degraded_student_message()
        llm_out = DeepDiveLLMOutput(
            technical_explanation=msg,
            feedback_on_answer=msg,
            follow_up_question="Какой следующий шаг вы хотите разобрать?",
            question_sub_concept_id=qid,
            ready_for_transition=False,
        )

    memory.chat_sessions = chat_mgr.to_memory_blob()
    if content is not None:
        content = _merge_content(
            content,
            llm_out,
            False,
            curriculum_id=req.curriculum_id,
            node=node,
        )
    from knowledge_engine.src.node_deep_dive.prompt_factory import (
        parse_tutor_mode_prefix,
        requires_deep_analysis_guard,
    )
    from knowledge_engine.src.node_deep_dive.star_task_fsm import (
        get_star_task_status,
        star_task_blocks_transition,
    )

    _, factory_mode = parse_tutor_mode_prefix(user_msg)
    star_guard = requires_deep_analysis_guard(
        factory_mode,
        star_task_status=get_star_task_status(memory),
    )
    if star_guard or star_task_blocks_transition(memory):
        llm_out = llm_out.model_copy(
            update={
                "ready_for_transition": False,
                "suggested_next_step": None,
                "quick_replies": [],
            }
        )
    tutor = resolve_tutor_display_message(llm_out)
    if not tutor:
        tutor = "Продолжим по теме — один конкретный вопрос или уточнение."
        # Never wipe a required deep_analysis follow_up on empty-display fallback.
        patch: dict[str, Any] = {"technical_explanation": tutor}
        if not star_guard:
            patch["follow_up_question"] = ""
        llm_out = llm_out.model_copy(update=patch)
    return {
        **state,
        "memory": memory,
        "content": content,
        "llm_out": llm_out,
        "tutor_message": tutor,
    }
