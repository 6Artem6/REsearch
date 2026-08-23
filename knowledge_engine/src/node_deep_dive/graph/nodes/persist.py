"""Persist session memory and sync UI history."""

from __future__ import annotations

from knowledge_engine.src.node_deep_dive.dialog_ids import (
    patch_last_tutor_history_content,
    sync_session_history_turns,
)
from knowledge_engine.src.node_deep_dive.graph.state import TutorGraphState
from knowledge_engine.src.node_deep_dive.session_store import (
    get_session,
    persist_session_memory,
    repair_history_with_memory,
)
from knowledge_engine.src.node_deep_dive.tutor_dialogue import (
    coerce_deep_dive_llm_output,
    resolve_tutor_display_message,
)
from knowledge_engine.src.node_deep_dive.tutor_field_limits import (
    SCHEMA_FOLLOW_UP_QUESTION_MAX,
    SCHEMA_TUTOR_MESSAGE_MAX,
)


def persist_node(state: TutorGraphState) -> TutorGraphState:
    """``persist_session_memory`` + ``sync_session_history_turns`` once per invoke."""
    req = state["request"]
    memory = state["memory"]
    node = req.node_data
    action = (req.user_action or "").strip().lower()
    llm_out = coerce_deep_dive_llm_output(state.get("llm_out"))
    tutor = resolve_tutor_display_message(
        llm_out,
        (state.get("tutor_message") or "").strip(),
    )

    persist_session_memory(req.curriculum_id, node.node_id, memory)
    session = get_session(req.curriculum_id, node.node_id)
    if action in ("chat", "verify"):
        history = sync_session_history_turns(
            session.history,
            memory,
            user_message=(req.user_message or "").strip(),
            tutor_message=tutor,
        )
    else:
        history = sync_session_history_turns(
            session.history,
            memory,
            tutor_message=tutor,
        )
    history = patch_last_tutor_history_content(history, tutor)
    history = repair_history_with_memory(history, memory)
    history = patch_last_tutor_history_content(history, tutor)
    if llm_out is not None and tutor:
        memory.last_tutor_display_message = tutor[:SCHEMA_TUTOR_MESSAGE_MAX]
        memory.last_tutor_follow_up_question = (
            llm_out.follow_up_question or ""
        ).strip()[:SCHEMA_FOLLOW_UP_QUESTION_MAX]
    return {
        **state,
        "memory": memory,
        "session_history": history,
        "tutor_message": tutor,
    }
