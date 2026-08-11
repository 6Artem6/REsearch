"""Init subgraph: lazy grounding + directional RAG (prepare only)."""

from __future__ import annotations

from typing import Any

from knowledge_engine.src.node_deep_dive.graph.state import TutorGraphState


async def init_node(
    state: TutorGraphState,
    config: dict[str, Any] | None = None,
) -> TutorGraphState:
    """Prepare session memory/content; first tutor question on later chat."""
    from knowledge_engine.src.node_deep_dive.engine import run_init_prepare_turn

    _ = config
    return await run_init_prepare_turn(state)
