"""Step analysis node: deterministic intent (VectorIntentRouter) + concepts_matrix updates (no sub_concept eval)."""

from __future__ import annotations

from typing import Any

from knowledge_engine.schemas.fsm import TutorStage
from knowledge_engine.src.node_deep_dive.control_intent import classify_control_chip
from knowledge_engine.src.node_deep_dive.graph.stage_events import stage_scope
from knowledge_engine.src.node_deep_dive.graph.state import TutorGraphState
from knowledge_engine.src.node_deep_dive.step_pipeline import (
    heuristic_step_analysis,
    resolve_user_intent_from_chip,
    run_step_analysis,
    should_run_step_analysis_llm,
)
from knowledge_engine.src.node_deep_dive.tiered_memory import apply_concept_updates
from knowledge_engine.ui.run_log import trace


def step_analysis_node(
    state: TutorGraphState,
    config: dict[str, Any] | None = None,
) -> TutorGraphState:
    """FSM stage wrapper, см. graph/stage_events.py."""
    with stage_scope(
        state,
        config,
        TutorStage.INTENT_ANALYSIS,
        running_message="Анализируем ваш запрос…",
    ):
        return _step_analysis_node_impl(state)


def _step_analysis_node_impl(state: TutorGraphState) -> TutorGraphState:
    """Deterministic chip→intent (0 LLM calls) + LLM matrix patch; no gap eval, no active_window append."""
    req = state["request"]
    memory = state["memory"]
    anchor = state["anchor"]
    node = req.node_data
    user_message = (req.user_message or "").strip()
    action = (req.user_action or "").strip().lower()

    chip = classify_control_chip(user_message, memory=memory)
    intent = resolve_user_intent_from_chip(chip)
    if chip:
        trace(f"NODE_DIVE step_analysis intent | chip={chip} intent={intent}")

    if should_run_step_analysis_llm(user_message, memory, action):
        try:
            analysis = run_step_analysis(user_message, memory, node, anchor)
        except Exception as exc:
            trace(
                f"NODE_DIVE step_analysis fallback (heuristic) | "
                f"{type(exc).__name__}: {exc}"
            )
            analysis = heuristic_step_analysis(user_message, memory.learning_phase)
    else:
        trace("NODE_DIVE step_analysis skip | heuristic (latency)")
        analysis = heuristic_step_analysis(user_message, memory.learning_phase)

    apply_concept_updates(memory, analysis.concept_updates)
    gap = (analysis.critical_gap or "").strip() or None
    return {
        **state,
        "memory": memory,
        "intent": intent,
        "pipeline_gap": gap,
    }
