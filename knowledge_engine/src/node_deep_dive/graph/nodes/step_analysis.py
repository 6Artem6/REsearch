"""Step analysis node: intent and concepts_matrix updates (no sub_concept eval)."""

from __future__ import annotations

from knowledge_engine.src.node_deep_dive.graph.state import TutorGraphState
from knowledge_engine.src.node_deep_dive.step_pipeline import (
    heuristic_step_analysis,
    run_step_analysis,
    should_run_step_analysis_llm,
)
from knowledge_engine.src.node_deep_dive.tiered_memory import apply_concept_updates
from knowledge_engine.ui.run_log import trace


def step_analysis_node(state: TutorGraphState) -> TutorGraphState:
    """Lite step_analysis + matrix patch; no gap eval, no active_window append."""
    req = state["request"]
    memory = state["memory"]
    anchor = state["anchor"]
    node = req.node_data
    user_message = (req.user_message or "").strip()
    action = (req.user_action or "").strip().lower()

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
        "intent": analysis.intent,
        "pipeline_gap": gap,
    }
