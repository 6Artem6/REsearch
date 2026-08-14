"""Sub-concept evaluator node (single-writer for sub_concepts coverage status)."""

from __future__ import annotations

import logging

from knowledge_engine.src.node_deep_dive.concept_map import (
    process_sub_concept_user_answer,
    stored_pending_evaluation_id,
)
from knowledge_engine.src.node_deep_dive.graph.state import TutorGraphState
from knowledge_engine.ui.run_log import trace

logger = logging.getLogger(__name__)


def sub_concept_eval_node(state: TutorGraphState) -> TutorGraphState:
    """Gap eval for ``pending_evaluation_concept_id`` only; skip if no pending."""
    req = state["request"]
    memory = state["memory"]
    user_message = (req.user_message or "").strip()
    if not user_message:
        logger.info("sub_concept_eval_node skip | empty user_message")
        return state

    pending = stored_pending_evaluation_id(memory)
    if not pending:
        logger.warning(
            "sub_concept_eval_node skip | no pending target "
            "(asked=%r pending_field=%r) — tutor may run without credit",
            getattr(memory, "asked_question_sub_concept_id", ""),
            getattr(memory, "pending_evaluation_concept_id", ""),
        )
        trace(
            "NODE_DIVE sub_concept evaluation skip | no pending "
            "(silent credit loss risk)"
        )
        return state

    from knowledge_engine.src.node_deep_dive.lecture_scope import (
        is_lecture_request_message,
    )

    if is_lecture_request_message(user_message):
        logger.info("sub_concept_eval_node skip | lecture request")
        trace(
            "NODE_DIVE sub_concept evaluation skip | lecture request "
            "(not a user answer)"
        )
        return state

    from knowledge_engine.src.node_deep_dive.concept_map import (
        is_quick_reply_control_message,
    )

    if is_quick_reply_control_message(user_message):
        logger.info(
            "sub_concept_eval_node skip | quick-reply control chip "
            "(not a scored answer)"
        )
        trace(
            "NODE_DIVE sub_concept evaluation skip | quick-reply control "
            "(Gloss / Дожать / next — evaluator off)"
        )
        return state

    try:
        process_sub_concept_user_answer(
            user_message,
            memory,
            req.node_data,
            state["anchor"],
        )
    except Exception as exc:
        logger.exception("sub_concept_eval_node FAILED pending=%s", pending)
        trace(f"EVALUATOR_ERROR | pipeline | {type(exc).__name__}: {exc}")
        trace(f"NODE_DIVE sub_concept evaluation FAILED | {type(exc).__name__}: {exc}")
        # Last-resort: do not leave unchecked without a mark.
        try:
            from knowledge_engine.src.node_deep_dive.concept_map import find_sub_concept
            from knowledge_engine.src.node_deep_dive.sub_concept_evaluator import (
                apply_degraded_threshold,
            )

            row = find_sub_concept(memory, pending)
            if row is not None and row.status == "unchecked":
                layer = str(getattr(req.node_data, "layer", "") or "foundation")
                directive = apply_degraded_threshold(
                    row, layer=layer, reason=f"node_exc:{type(exc).__name__}"
                )
                memory.last_eval_directive = directive
                from knowledge_engine.src.node_deep_dive.concept_map_state import (
                    build_evaluator_feedback,
                )

                memory.last_evaluator_feedback = build_evaluator_feedback(row)
        except Exception as inner:
            logger.exception("degraded threshold also failed: %s", inner)
    return {**state, "memory": memory}
