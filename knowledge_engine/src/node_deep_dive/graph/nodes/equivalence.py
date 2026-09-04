"""User declines node as already known."""

from __future__ import annotations

from typing import Any

from knowledge_engine.schemas.fsm import TutorStage
from knowledge_engine.src.node_deep_dive.graph.stage_events import stage_scope
from knowledge_engine.src.node_deep_dive.graph.state import TutorGraphState


async def equivalence_node(
    state: TutorGraphState,
    config: dict[str, Any] | None = None,
) -> TutorGraphState:
    from knowledge_engine.src.node_deep_dive.engine import run_equivalence_turn

    with stage_scope(
        state, config, TutorStage.INIT, running_message="Фиксируем решение…"
    ):
        return await run_equivalence_turn(state)
