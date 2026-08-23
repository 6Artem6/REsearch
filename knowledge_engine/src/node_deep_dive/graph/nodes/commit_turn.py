"""Commit turn: windows, pending question, orchestrate transition (no tutor verified ids)."""

from __future__ import annotations

import logging

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
    extract_follow_up_from_chat_text,
    resolve_tutor_display_message,
)
from knowledge_engine.src.node_deep_dive.tutor_memory_content import (
    tutor_content_for_active_window,
)
from knowledge_engine.src.node_deep_dive.tutor_field_limits import (
    SCHEMA_FOLLOW_UP_QUESTION_MAX,
    SCHEMA_TUTOR_MESSAGE_MAX,
)
from knowledge_engine.ui.run_log import trace

logger = logging.getLogger(__name__)


def _emit_commit_telemetry(req, memory, factory_mode: str, raw_user: str, t0: float) -> None:
    """Cheap host telemetry: exact chip only (never re-embed on the P99 path)."""
    import time

    from knowledge_engine.context_drift_manager import ContextDriftManager
    from knowledge_engine.src.node_deep_dive.control_intent import (
        classify_exact_control_chip,
    )
    from knowledge_engine.src.telemetry_auditor import (
        HostTurnTelemetry,
        emit_host_telemetry,
    )

    hit = classify_exact_control_chip(raw_user) if raw_user else ""
    mode = (factory_mode or "").strip().lower()
    if hit:
        intent, source = str(hit), "exact"
    elif mode and mode not in ("default", ""):
        intent, source = mode, "exact"
    else:
        intent, source = "", "fallback"
    cid = str(getattr(req, "curriculum_id", "") or "").strip()
    tags: list[str] = []
    if cid:
        try:
            tags = ContextDriftManager(cid, persist=False).open_weakness_tags()
        except Exception:
            tags = []
    overlay = str(getattr(memory, "pending_eval_kind", "") or "").strip()
    node_id = str(getattr(getattr(req, "node_data", None), "node_id", "") or "")
    emit_host_telemetry(
        HostTurnTelemetry(
            session_id=cid,
            node_id=node_id,
            intent_detected=intent,
            intent_source=source,  # type: ignore[arg-type]
            active_overlay=overlay,
            weakness_tags=tags,
            latency_host_ms=round((time.perf_counter() - t0) * 1000.0, 3),
        )
    )


def _ensure_follow_up_on_llm_out(llm_out, tutor: str):
    """Fill follow_up_question from Самопроверка / trailing ``?`` when missing."""
    follow = (llm_out.follow_up_question or "").strip()
    if follow:
        return llm_out
    raw = (tutor or "").strip()
    if not raw:
        return llm_out
    _tech, extracted = extract_follow_up_from_chat_text(raw)
    if not extracted:
        return llm_out
    return llm_out.model_copy(update={"follow_up_question": extracted})


def _bind_pending_for_follow_up(
    memory,
    llm_out,
    *,
    focus_sub_concept_id: str = "",
) -> object:
    """Unconditionally bind pending when a follow-up question exists + focus id."""
    from knowledge_engine.src.node_deep_dive.subconcept_invariants import (
        enforce_question_sub_concept_invariant,
        resolve_active_subconcept_id,
    )

    follow = (llm_out.follow_up_question or "").strip()
    if not follow or llm_out.ready_for_transition:
        return llm_out

    llm_out, drifted = enforce_question_sub_concept_invariant(memory, llm_out)
    if drifted:
        trace(
            "STATE_DRIFT commit_turn | repaired question_sub_concept_id → "
            f"{resolve_active_subconcept_id(memory)}"
        )

    qid = (llm_out.question_sub_concept_id or "").strip()
    focus = (focus_sub_concept_id or "").strip()
    active = resolve_active_subconcept_id(memory)
    bind_id = qid or focus or active
    if not bind_id:
        logger.warning(
            "commit_turn | follow_up_question present but no focus_sub_concept_id "
            "(pending_evaluation_concept_id not set) follow=%r",
            follow[:120],
        )
        trace(
            "WARN commit_turn | follow_up_question без focus_sub_concept_id; "
            "pending_evaluation_concept_id не установлен"
        )
        return llm_out

    cid = set_pending_evaluation_for_tutor_turn(memory, bind_id)
    llm_out = llm_out.model_copy(update={"question_sub_concept_id": bind_id})
    if cid:
        source = (
            "question_sub_concept_id"
            if qid == bind_id
            else (
                "focus_sub_concept_id" if focus == bind_id else "active_subconcept_id"
            )
        )
        trace(f"NODE_DIVE pending question set | concept={cid} " f"(bind via {source})")
    else:
        logger.warning(
            "commit_turn | set_pending failed for bind_id=%s "
            "(verified/unknown?) — silent credit loss risk",
            bind_id,
        )
        trace(
            f"WARN commit_turn | pending bind failed id={bind_id} "
            "(silent credit loss risk)"
        )
    return llm_out


def _warn_if_question_without_pending(memory, llm_out, tutor: str) -> None:
    """Safety net: lecture/tutor ended with a question but pending stayed empty."""
    if llm_out.ready_for_transition:
        return
    pending = (memory.pending_evaluation_concept_id or "").strip()
    if pending:
        return
    follow = (llm_out.follow_up_question or "").strip()
    raw = (tutor or "").strip()
    has_q = bool(follow)
    if not has_q and raw:
        _t, extracted = extract_follow_up_from_chat_text(raw)
        has_q = bool(extracted)
    if not has_q and ("?" in raw[-600:] or "？" in raw[-600:]):
        has_q = True
    if has_q:
        logger.warning(
            "commit_turn | lecture/tutor has question but "
            "pending_evaluation_concept_id is empty — silent credit loss risk"
        )
        trace(
            "WARN commit_turn | question without pending_evaluation_concept_id "
            "(silent credit loss risk)"
        )


def commit_turn_node(state: TutorGraphState) -> TutorGraphState:
    """Append user/tutor windows; pending from tutor contract; orchestrate output."""
    import time

    t0 = time.perf_counter()
    req = state["request"]
    memory = state["memory"]
    anchor = state["anchor"]
    action = (req.user_action or "").strip().lower()
    raw_user = (req.user_message or "").strip()
    focus_id = (state.get("focus_sub_concept_id") or "").strip()
    if focus_id:
        # Keep generation focus pinned across commit even if asked/pending empty.
        memory.next_question_concept_id = focus_id

    tutor = (state.get("tutor_message") or "").strip()
    llm_out = state.get("llm_out")
    if llm_out is None:
        llm_out = deep_dive_llm_output_from_chat_text(
            tutor,
            question_sub_concept_id=focus_id or None,
        )
    else:
        from knowledge_engine.src.node_deep_dive.drill_orchestrator import (
            is_layer_just_completed,
        )

        if not is_layer_just_completed(memory):
            llm_out = _ensure_follow_up_on_llm_out(llm_out, tutor)
        if focus_id and not (llm_out.question_sub_concept_id or "").strip():
            llm_out = llm_out.model_copy(update={"question_sub_concept_id": focus_id})

    if action in ("chat", "verify") and raw_user:
        from knowledge_engine.src.node_deep_dive.prompt_factory import (
            display_user_after_mode_prefix,
        )

        display_user, factory_mode = display_user_after_mode_prefix(raw_user)
        append_to_active_window(memory, "user", display_user or raw_user)
        rotate_window_after_message(memory, anchor)
    else:
        factory_mode = "default"

    if tutor:
        merge_introduced_terms(memory, list(llm_out.introduced_terms or []))
        window_tutor = tutor_content_for_active_window(
            llm_out, fallback_compose_text=tutor
        )
        append_to_active_window(memory, "tutor", window_tutor or tutor)
        rotate_window_after_message(memory, anchor)
        memory.last_tutor_question_angle = infer_question_angle(tutor)

    llm_out = orchestrate_tutor_llm_output(
        memory,
        llm_out,
        user_message=raw_user,
        node_layer=str(getattr(req.node_data, "layer", "") or ""),
    )
    from knowledge_engine.src.node_deep_dive.star_task_fsm import (
        is_overlay_eval_kind,
        star_task_blocks_transition,
    )

    if star_task_blocks_transition(memory):
        llm_out = llm_out.model_copy(
            update={
                "ready_for_transition": False,
                "suggested_next_step": None,
                "quick_replies": [],
            }
        )

    if not llm_out.ready_for_transition:
        llm_out = _bind_pending_for_follow_up(
            memory, llm_out, focus_sub_concept_id=focus_id
        )
        # Mark overlay asterisk-question pending for specialized evaluator + FSM.
        if is_overlay_eval_kind(factory_mode) and (
            memory.pending_evaluation_concept_id or ""
        ).strip():
            from knowledge_engine.src.node_deep_dive.star_task_fsm import (
                mark_star_task_in_progress,
            )

            kind = (factory_mode or "").strip().lower()
            mark_star_task_in_progress(
                memory,
                concept_id=memory.pending_evaluation_concept_id,
                overlay_kind=kind,
            )
            trace(
                f"NODE_DIVE pending_eval_kind={kind} | "
                f"concept={memory.pending_evaluation_concept_id} | "
                f"star_task={memory.star_task_status}"
            )

    # Asterisk-question coverage: record citations after overlay generation.
    factory_l = (factory_mode or "").strip().lower()
    star_st = (memory.star_task_status or "").strip()
    should_record_star_coverage = is_overlay_eval_kind(factory_l) or (
        is_overlay_eval_kind(memory.pending_eval_kind)
        and star_st in ("in_progress", "needs_refinement")
    )
    tech = (llm_out.technical_explanation or "").strip()
    # Require a real analysis body (avoid polluting digests on tiny review lines).
    if should_record_star_coverage and (
        is_overlay_eval_kind(factory_l) or len(tech) >= 400
    ):
        from knowledge_engine.src.node_deep_dive.deep_analysis_coverage import (
            record_deep_analysis_coverage,
        )

        record_deep_analysis_coverage(
            memory,
            technical_explanation=tech,
            feedback_on_answer=llm_out.feedback_on_answer or "",
            follow_up_question=llm_out.follow_up_question or "",
            references=list(llm_out.references or []),
        )
        trace(
            "NODE_DIVE deep_analysis coverage | "
            f"sources={len(memory.deep_analysis_used_source_ids or [])} "
            f"atoms={len(memory.deep_analysis_used_atom_keys or [])} "
            f"chunks={len(memory.deep_analysis_used_chunk_ids or [])} "
            f"digests={len(memory.deep_analysis_prior_digests or [])}"
        )
        try:
            from knowledge_engine.src.node_deep_dive.socratic_poles import (
                persist_socratic_poles_on_commit,
            )

            persist_socratic_poles_on_commit(
                memory,
                curriculum_id=(req.curriculum_id or "").strip(),
                node_id=str(getattr(req.node_data, "node_id", "") or ""),
            )
        except Exception as exc:
            trace(f"NODE_DIVE socratic_poles persist skip | {exc}")

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
            if not qid:
                qid = focus_id
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
        elif is_overlay_eval_kind(choice) or is_overlay_eval_kind(factory_mode):
            from knowledge_engine.src.node_deep_dive.concept_map import (
                start_overlay_push,
            )

            kind = choice if is_overlay_eval_kind(choice) else factory_mode
            qid = (llm_out.question_sub_concept_id or "").strip() or focus_id
            cid = start_overlay_push(memory, kind, concept_id=qid)
            if cid:
                bound = set_pending_evaluation_for_tutor_turn(memory, cid)
                llm_out = llm_out.model_copy(update={"question_sub_concept_id": cid})
                if bound:
                    trace(
                        f"NODE_DIVE overlay pending set | concept={bound} "
                        f"kind={kind}"
                    )

    verified_ids = list_verified_sub_concept_ids(memory)[:8]
    llm_out = llm_out.model_copy(update={"verified_sub_concept_ids": verified_ids})
    tutor = resolve_tutor_display_message(llm_out, tutor)
    if tutor:
        memory.last_tutor_display_message = tutor[:SCHEMA_TUTOR_MESSAGE_MAX]
        memory.last_tutor_follow_up_question = (
            llm_out.follow_up_question or ""
        ).strip()[:SCHEMA_FOLLOW_UP_QUESTION_MAX]

    _warn_if_question_without_pending(memory, llm_out, tutor)

    from knowledge_engine.src.node_deep_dive.drill_orchestrator import (
        consume_completed_drill_latch,
    )

    consume_completed_drill_latch(memory)

    # One-shot Deep Mastery celebration — avoid re-firing Asterisk-question closure copy.
    if (memory.last_eval_directive or "").strip() == "DEEP_MASTERY_EARNED" and (
        not star_task_blocks_transition(memory)
    ):
        memory.last_eval_directive = "PASSED_CLEAN"

    try:
        _emit_commit_telemetry(req, memory, factory_mode, raw_user, t0)
    except Exception:
        pass

    return {
        **state,
        "memory": memory,
        "tutor_message": tutor,
        "llm_out": llm_out,
        "response_verified_sub_concept_ids": verified_ids,
    }
