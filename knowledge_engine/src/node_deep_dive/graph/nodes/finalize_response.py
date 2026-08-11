"""Build NodeDeepDiveResponse from graph state."""

from __future__ import annotations

from knowledge_engine.src.node_deep_dive.engine import finalize_graph_chat_response
from knowledge_engine.src.node_deep_dive.graph.state import TutorGraphState


async def finalize_response_node(state: TutorGraphState) -> TutorGraphState:
    """Map state → ``NodeDeepDiveResponse`` (post-commit, no window side effects)."""
    resp = await finalize_graph_chat_response(state)
    return {**state, "response": resp}
