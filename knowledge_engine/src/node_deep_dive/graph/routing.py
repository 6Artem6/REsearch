"""Conditional routing for Node Deep-Dive LangGraph."""

from __future__ import annotations

from typing import Literal

from knowledge_engine.src.node_deep_dive.graph.state import TutorGraphState
from knowledge_engine.src.node_deep_dive.init_context import (
    user_declines_node_equivalence,
)

AfterIngestRoute = Literal["init", "lazy_intro", "equivalence", "step_analysis"]


def route_by_action(state: TutorGraphState) -> AfterIngestRoute:
    """After ingest: init prepare vs chat turn vs lazy intro vs equivalence."""
    req = state.get("request")
    action = (getattr(req, "user_action", None) or "").strip().lower()
    if action == "init":
        return "init"

    raw_user = (getattr(req, "user_message", None) or "").strip()
    node = req.node_data
    from knowledge_engine.src.node_deep_dive.engine import (
        _session_needs_lazy_intro,
    )
    from knowledge_engine.src.node_deep_dive.session_store import get_session

    session = get_session(req.curriculum_id, node.node_id)
    memory = state.get("memory")
    if user_declines_node_equivalence(raw_user) and session.node_status == "unexplored":
        return "equivalence"
    if memory is not None and _session_needs_lazy_intro(session, memory, raw_user):
        return "lazy_intro"
    return "step_analysis"


route_after_ingest = route_by_action


def route_interaction(
    state: TutorGraphState,
) -> Literal["tutor_generate", "dense_lecture", "persist"]:
    """
    After coverage_router: dispatch by ``state["route"]``.

    Router node must set ``route`` without LLM (deterministic).
    """
    route = (state.get("route") or "tutor").strip().lower()
    if route == "dense":
        return "dense_lecture"
    if route in ("coverage_notice", "transition", "skip_llm"):
        return "persist"
    return "tutor_generate"


def after_router(state: TutorGraphState) -> str:
    """Alias for conditional_edges map keys (same as route_interaction return value)."""
    return route_interaction(state)
