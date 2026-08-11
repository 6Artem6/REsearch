"""Tutor LLM generation node (dialogue_feedback / lecture_chat)."""

from __future__ import annotations

from typing import Any

from knowledge_engine.services.chat_session_manager import ChatSessionManager
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
    """Compose system prompt and call Gemini (``_invoke_tutor``)."""
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
        trace("STATE_DRIFT retry | regenerating tutor turn with stripped chat_history")
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
        )
        llm_out, drifted2 = enforce_question_sub_concept_invariant(memory, llm_out)
        if drifted2:
            trace(
                "STATE_DRIFT persist | force question_sub_concept_id="
                f"{llm_out.question_sub_concept_id}"
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
    tutor = resolve_tutor_display_message(llm_out)
    if not tutor:
        tutor = "Продолжим по теме — один конкретный вопрос или уточнение."
        llm_out = llm_out.model_copy(
            update={
                "technical_explanation": tutor,
                "follow_up_question": "",
            }
        )
    return {
        **state,
        "memory": memory,
        "content": content,
        "llm_out": llm_out,
        "tutor_message": tutor,
    }
