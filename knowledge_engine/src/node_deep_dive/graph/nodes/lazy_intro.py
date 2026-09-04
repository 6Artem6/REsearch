"""First chat on unexplored node: intro assessment or fast-track."""

from __future__ import annotations

from typing import Any

from knowledge_engine.schemas.fsm import TutorStage
from knowledge_engine.src.node_deep_dive.graph.stage_events import stage_scope
from knowledge_engine.src.node_deep_dive.graph.state import TutorGraphState


async def lazy_intro_node(
    state: TutorGraphState,
    config: dict[str, Any] | None = None,
) -> TutorGraphState:
    from knowledge_engine.src.node_deep_dive.engine import run_lazy_intro_turn

    with stage_scope(
        state, config, TutorStage.INIT, running_message="Готовим вводный вопрос…"
    ):
        return await run_lazy_intro_turn(state)
