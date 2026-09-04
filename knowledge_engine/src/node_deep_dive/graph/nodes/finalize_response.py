"""Build NodeDeepDiveResponse from graph state."""

from __future__ import annotations

from typing import Any

from knowledge_engine.schemas.fsm import TutorStage
from knowledge_engine.src.node_deep_dive.engine import finalize_graph_chat_response
from knowledge_engine.src.node_deep_dive.graph.stage_events import stage_scope
from knowledge_engine.src.node_deep_dive.graph.state import TutorGraphState


async def finalize_response_node(
    state: TutorGraphState,
    config: dict[str, Any] | None = None,
) -> TutorGraphState:
    """Map state → ``NodeDeepDiveResponse`` (post-commit, no window side effects)."""
    with stage_scope(
        state, config, TutorStage.FINALIZE, running_message="Завершаем ход…"
    ):
        resp = await finalize_graph_chat_response(state)
    return {**state, "response": resp}
