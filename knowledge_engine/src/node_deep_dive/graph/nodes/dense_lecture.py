"""Dense lecture generation node (RAG + generate_dense_material)."""

from __future__ import annotations

from typing import Any

from knowledge_engine.src.node_deep_dive.graph.state import TutorGraphState


def _stream_from_config(config: dict[str, Any] | None) -> Any:
    if not config:
        return None
    return (config.get("configurable") or {}).get("stream_callback")


async def dense_lecture_node(
    state: TutorGraphState,
    config: dict[str, Any] | None = None,
) -> TutorGraphState:
    """RAG + dense LLM; updates content, memory phase, tutor_message → commit_turn."""
    from knowledge_engine.src.node_deep_dive.engine import run_dense_lecture_turn

    return await run_dense_lecture_turn(state, _stream_from_config(config))
