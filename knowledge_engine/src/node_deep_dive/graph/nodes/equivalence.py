"""User declines node as already known."""

from __future__ import annotations

from typing import Any

from knowledge_engine.src.node_deep_dive.graph.state import TutorGraphState


async def equivalence_node(
    state: TutorGraphState,
    config: dict[str, Any] | None = None,
) -> TutorGraphState:
    from knowledge_engine.src.node_deep_dive.engine import run_equivalence_turn

    _ = config
    return await run_equivalence_turn(state)
