"""Commit turn: windows, pending question, orchestrate transition (no tutor verified ids)."""

from __future__ import annotations

from knowledge_engine.src.curriculum.global_tracker import infer_question_angle
from knowledge_engine.src.node_deep_dive.concept_map import (
    list_verified_sub_concept_ids,
    orchestrate_tutor_llm_output,
    set_pending_evaluation_for_tutor_turn,
)
from knowledge_engine.src.node_deep_dive.graph.state import TutorGraphState
from knowledge_engine.src.node_deep_dive.step_pipeline import (
    rotate_window_after_message,
)
from knowledge_engine.src.node_deep_dive.term_registry import merge_introduced_terms
from knowledge_engine.src.node_deep_dive.tiered_memory import append_to_active_window
from knowledge_engine.src.node_deep_dive.tutor_dialogue import (
    compose_tutor_dialogue_from_output,
    deep_dive_llm_output_from_chat_text,
    resolve_tutor_display_message,
)
from knowledge_engine.src.node_deep_dive.tutor_memory_content import (
    tutor_content_for_active_window,
)
from knowledge_engine.ui.run_log import trace


def commit_turn_node(state: TutorGraphState) -> TutorGraphState:
    """Append user/tutor windows; pending from tutor contract; orchestrate output."""
    req = state["request"]
    memory = state["memory"]
    anchor = state["anchor"]
    action = (req.user_action or "").strip().lower()
    raw_user = (req.user_message or "").strip()

    tutor = (state.get("tutor_message") or "").strip()
    llm_out = state.get("llm_out")
    if llm_out is None:
        llm_out = deep_dive_llm_output_from_chat_text(tutor)

    if action in ("chat", "verify") and raw_user:
        from knowledge_engine.src.node_deep_dive.prompt_factory import (
            parse_tutor_mode_prefix,
        )

        display_user, _mode = parse_tutor_mode_prefix(raw_user)
        append_to_active_window(memory, "user", display_user or raw_user)
        rotate_window_after_message(memory, anchor)

    if tutor:
        merge_introduced_terms(memory, list(llm_out.introduced_terms or []))
        window_tutor = tutor_content_for_active_window(
            llm_out, fallback_compose_text=tutor
        )
        append_to_active_window(memory, "tutor", window_tutor or tutor)
        rotate_window_after_message(memory, anchor)
        memory.last_tutor_question_angle = infer_question_angle(tutor)

    if not llm_out.ready_for_transition:
        follow = (llm_out.follow_up_question or "").strip()
        from knowledge_engine.src.node_deep_dive.subconcept_invariants import (
            enforce_question_sub_concept_invariant,
            resolve_active_subconcept_id,
        )

        llm_out, drifted = enforce_question_sub_concept_invariant(memory, llm_out)
        if drifted:
            trace(
                "STATE_DRIFT commit_turn | repaired question_sub_concept_id → "
                f"{resolve_active_subconcept_id(memory)}"
            )
        qid = (llm_out.question_sub_concept_id or "").strip()
        if follow and qid:
            cid = set_pending_evaluation_for_tutor_turn(memory, qid)
            if cid:
                trace(
                    f"NODE_DIVE pending question set | concept={cid} "
                    "(question_sub_concept_id / asked_question_sub_concept_id)"
                )
        elif follow:
            # No id from model — bind outstanding question to active focus
            active = resolve_active_subconcept_id(memory)
            if active:
                cid = set_pending_evaluation_for_tutor_turn(memory, active)
                llm_out = llm_out.model_copy(update={"question_sub_concept_id": active})
                if cid:
                    trace(
                        f"NODE_DIVE pending question set | concept={cid} "
                        "(fallback active_subconcept_id)"
                    )
            else:
                trace(
                    "WARN commit_turn | follow_up_question без question_sub_concept_id; "
                    "pending_evaluation_concept_id не установлен"
                )

    llm_out = orchestrate_tutor_llm_output(
        memory,
        llm_out,
        user_message=raw_user,
        node_layer=str(getattr(req.node_data, "layer", "") or ""),
    )
    if llm_out.ready_for_transition:
        tutor = compose_tutor_dialogue_from_output(llm_out) or tutor
    else:
        # Deep-dive push («Дожать»): ensure pending is bound to the practice question.
        from knowledge_engine.src.node_deep_dive.concept_map import (
            classify_gloss_fork_choice,
            first_optional_layer_concept_id,
        )

        choice = classify_gloss_fork_choice(raw_user)
        if choice in ("how", "mech"):
            layer_name = "HOW" if choice == "how" else "MECHANIC"
            qid = (llm_out.question_sub_concept_id or "").strip()
            if not qid:
                qid = first_optional_layer_concept_id(memory, layer_name)
            if qid:
                cid = set_pending_evaluation_for_tutor_turn(memory, qid)
                llm_out = llm_out.model_copy(update={"question_sub_concept_id": qid})
                if cid:
                    trace(
                        f"NODE_DIVE deep_dive pending set | concept={cid} "
                        f"layer={layer_name} (Дожать chip)"
                    )
            # MECH push → await edge-case answer (Evaluator scores the next turn).
            if choice == "mech":
                memory.last_eval_directive = "AWAITING_EDGE_CASE_ANSWER"

    verified_ids = list_verified_sub_concept_ids(memory)[:8]
    llm_out = llm_out.model_copy(update={"verified_sub_concept_ids": verified_ids})
    tutor = resolve_tutor_display_message(llm_out, tutor)
    if tutor:
        memory.last_tutor_display_message = tutor[:12_000]
        memory.last_tutor_follow_up_question = (
            llm_out.follow_up_question or ""
        ).strip()[:2000]

    return {
        **state,
        "memory": memory,
        "tutor_message": tutor,
        "llm_out": llm_out,
        "response_verified_sub_concept_ids": verified_ids,
    }
