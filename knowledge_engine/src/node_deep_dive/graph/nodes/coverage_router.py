"""Deterministic router: interaction mode, dense vs tutor, next focus_sub_concept_id."""

from __future__ import annotations

from knowledge_engine.services.curriculum_whitelist_prompt import (
    enrich_node_learning_materials_from_graph,
)
from knowledge_engine.src.node_deep_dive.concept_map import (
    classify_gloss_fork_choice,
    ensure_sub_concept_map,
    select_next_sub_concept,
    set_pending_evaluation_for_tutor_turn,
    start_overlay_push,
)
from knowledge_engine.src.node_deep_dive.graph.state import TutorGraphState
from knowledge_engine.src.node_deep_dive.learning_loop import (
    advance_phase_after_chat,
    set_learning_mode,
)
from knowledge_engine.src.node_deep_dive.lecture_coverage import (
    assess_lecture_coverage,
    build_coverage_dense_output,
    build_coverage_short_message,
)
from knowledge_engine.src.node_deep_dive.lecture_coverage_registry import (
    suggest_uncovered_deep_dive_topics,
)
from knowledge_engine.src.node_deep_dive.lecture_scope import resolve_lecture_scope
from knowledge_engine.src.node_deep_dive.memory_schemas import UserIntent
from knowledge_engine.src.node_deep_dive.session_store import get_session
from knowledge_engine.src.node_deep_dive.step_pipeline import (
    rotate_window_after_message,
)
from knowledge_engine.src.node_deep_dive.tiered_memory import append_to_active_window
from knowledge_engine.src.node_deep_dive.tutor_dialogue import (
    deep_dive_llm_output_from_chat_text,
)
from knowledge_engine.ui.run_log import trace


def _strip_mode_prefixes(raw_user: str) -> tuple[str, bool]:
    from knowledge_engine.src.node_deep_dive.prompt_factory import (
        parse_tutor_mode_prefix,
    )

    lecture_pressed = raw_user.startswith("[mode:lecture]")
    user_msg, _mode = parse_tutor_mode_prefix(raw_user)
    return user_msg or raw_user, lecture_pressed


def _apply_learning_mode_prefixes(memory, raw_user: str) -> None:
    from knowledge_engine.src.node_deep_dive.prompt_factory import (
        parse_tutor_mode_prefix,
    )

    _body, mode = parse_tutor_mode_prefix(raw_user)
    if mode == "lecture" or raw_user.startswith("[mode:lecture]"):
        set_learning_mode(memory, "lecture")
    elif mode == "blitz" or raw_user.startswith("[mode:blitz]"):
        set_learning_mode(memory, "express_blitz")
    elif mode == "socratic" or raw_user.startswith("[mode:socratic]"):
        set_learning_mode(memory, "socratic_point")


def coverage_router_node(state: TutorGraphState) -> TutorGraphState:
    """Set interaction_mode, route, focus_sub_concept_id (no LLM)."""
    from knowledge_engine.src.node_deep_dive.engine import (
        _apply_dense_material,
        _is_explicit_lecture_request,
        _needs_dense_material,
        resolve_interaction_prompt_mode,
    )
    from knowledge_engine.src.node_deep_dive.prompt_factory import (
        is_factory_control_mode,
        parse_tutor_mode_prefix,
    )

    req = state["request"]
    memory = state["memory"]
    node = req.node_data
    anchor = state["anchor"]
    action = (req.user_action or "").strip().lower()
    raw_user = (req.user_message or "").strip()
    _apply_learning_mode_prefixes(memory, raw_user)
    user_msg, lecture_pressed = _strip_mode_prefixes(raw_user)
    _cleaned, factory_mode = parse_tutor_mode_prefix(raw_user)

    intent: UserIntent = state.get("intent") or "ANSWER"
    if (
        intent == "INTENT_EXPLAIN"
        and not lecture_pressed
        and not _is_explicit_lecture_request(user_msg)
    ):
        intent = "ANSWER"
        trace(
            "NODE_DIVE intent coerce | INTENT_EXPLAIN→ANSWER "
            "(развёрнутый ответ, не запрос лекции)"
        )

    ensure_sub_concept_map(memory, node)
    from knowledge_engine.src.node_deep_dive.control_intent import (
        apply_mode_selection_intent,
        classify_control_chip,
    )

    chip = classify_control_chip(raw_user, memory=memory)
    if chip in ("practice", "check", "skip"):
        apply_mode_selection_intent(memory, chip)
    elif chip == "blitz":
        # Free-text/vector-matched blitz request (tag case already handled
        # by _apply_learning_mode_prefixes above) — persistent mode switch,
        # same as the intro "проверка" chip -> express_blitz.
        set_learning_mode(memory, "express_blitz")
    elif chip == "socratic":
        set_learning_mode(memory, "socratic_point")
    choice = classify_gloss_fork_choice(raw_user)
    if choice in ("how", "mech") and chip not in ("practice", "check"):
        from knowledge_engine.src.node_deep_dive.star_task_fsm import start_layer_drill

        start_layer_drill(memory, "HOW" if choice == "how" else "MECH")
    elif choice in ("gloss", "next"):
        from knowledge_engine.src.node_deep_dive.star_task_fsm import clear_layer_drill

        clear_layer_drill(memory)
    elif choice in ("deep_analysis", "advanced_analysis", "deep_design"):
        start_overlay_push(memory, choice)

    focus = select_next_sub_concept(memory)
    from knowledge_engine.src.node_deep_dive.star_task_fsm import (
        layer_drill_is_active,
    )

    if layer_drill_is_active(memory) and focus is None:
        from knowledge_engine.src.node_deep_dive.star_task_fsm import clear_layer_drill

        clear_layer_drill(memory)
        focus = select_next_sub_concept(memory)
    elif (getattr(memory, "active_optional_layer", "") or "").strip().upper() in (
        "HOW",
        "MECHANIC",
    ) and focus is None:
        memory.active_optional_layer = ""
        focus = select_next_sub_concept(memory)
    focus_id = (focus.id if focus else "").strip()
    if not focus_id:
        trace(
            "WARN coverage_router | empty focus_sub_concept_id — "
            "lecture/tutor without map anchor (silent credit loss risk)"
        )
    else:
        # Explicit generation focus before dense/tutor LLM (never lecture «in vacuo»).
        memory.next_question_concept_id = focus_id
        if focus is not None:
            trace(
                f"NODE_DIVE focus pinned | concept={focus_id} "
                f"label={(focus.label or '')[:80]}"
            )
    interaction_mode = resolve_interaction_prompt_mode(memory, intent, user_msg)

    wants_dense = _needs_dense_material(memory, intent, user_msg, lecture_pressed)
    # Prompt Factory chip modes always use isolated tutor prompts (never dense lecture).
    if is_factory_control_mode(factory_mode):
        wants_dense = False
        lecture_pressed = False
        trace(f"PROMPT_FACTORY | route=tutor forced | mode={factory_mode}")
    route = "tutor"
    content = state.get("content")
    llm_out = state.get("llm_out")
    tutor_message = (state.get("tutor_message") or "").strip()

    if wants_dense:
        route = "dense"
        session = get_session(req.curriculum_id, node.node_id)
        node_for_lecture = enrich_node_learning_materials_from_graph(
            node, req.curriculum_id
        )
        lecture_scope, focus_text = resolve_lecture_scope(
            user_msg,
            memory,
            lecture_button_pressed=lecture_pressed,
        )
        content_summary = (content.summary or "").strip() if content else ""
        coverage = assess_lecture_coverage(
            memory,
            session.history,
            user_msg,
            lecture_scope,
            focus_text,
            lecture_pressed,
            content_summary=content_summary,
        )
        trace(
            f"NODE_DIVE coverage | covered={coverage.is_topic_already_covered} "
            f"notice={coverage.should_return_coverage_notice} "
            f"scope={lecture_scope}"
        )
        if coverage.should_return_coverage_notice:
            route = "coverage_notice"
            topics = suggest_uncovered_deep_dive_topics(node_for_lecture, memory)
            tutor_message = build_coverage_short_message(
                node_for_lecture.title,
                topics,
                registry=memory.covered_subtopics,
                matching_keys=list(coverage.matching_subtopic_keys),
            )
            dense = build_coverage_dense_output(node_for_lecture, topics)
            if content is not None:
                content = _apply_dense_material(content, dense)
            llm_out = deep_dive_llm_output_from_chat_text(tutor_message)
            if action in ("chat", "verify") and raw_user:
                append_to_active_window(memory, "user", raw_user)
                rotate_window_after_message(
                    memory, anchor, req.curriculum_id, node.node_id
                )
            if tutor_message:
                append_to_active_window(memory, "tutor", tutor_message)
                rotate_window_after_message(
                    memory, anchor, req.curriculum_id, node.node_id
                )
                set_pending_evaluation_for_tutor_turn(memory, focus_id)
    else:
        advance_phase_after_chat(memory, intent, action)
        trace(
            f"NODE_DIVE ▶ {action} intent={intent} "
            f"phase={memory.learning_phase} mode={memory.learning_mode}"
        )

    return {
        **state,
        "memory": memory,
        "intent": intent,
        "interaction_mode": interaction_mode,
        "route": route,
        "focus_sub_concept_id": focus_id,
        "content": content,
        "tutor_message": tutor_message,
        "llm_out": llm_out,
    }
