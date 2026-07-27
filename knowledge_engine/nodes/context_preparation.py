"""Сбор Sandwich payload: SLM horizons + блочный контекст (без перегенерации текста)."""

from __future__ import annotations

from typing import Any

from knowledge_engine.nodes.decomposition import decomposition_node
from knowledge_engine.schemas import EngineGraphState, EngineState
from knowledge_engine.services.context_blocks import build_context_blocks
from knowledge_engine.services.context_manager import rolling_summarize_dialogue
from knowledge_engine.services.search.horizons import build_horizon_queries
from knowledge_engine.ui.logger import set_status
from knowledge_engine.ui.run_log import node_end, node_start


def context_preparation_node(state: EngineGraphState) -> dict[str, Any]:
    node_start("context_preparation_node (SLM + blocks)")
    parsed = EngineState.model_validate(state)
    updates: dict[str, Any] = {
        "context_block_selections": {},
        "context_corrected_once": False,
        "gemini_payload": "",
    }

    if not parsed.abstractions:
        set_status("[context_preparation] 7B: CS-абстракции…")
        decomp = decomposition_node(state)
        updates.update(decomp)
        parsed = EngineState.model_validate({**parsed.model_dump(), **decomp})

    set_status("[context_preparation] 1.5B: запросы 3 горизонтов…")
    horizon_queries = build_horizon_queries(
        parsed.user_problem,
        parsed.context_constraints,
        parsed.abstractions,
    )
    updates["search_horizon_queries"] = horizon_queries

    rolling = rolling_summarize_dialogue(parsed)
    updates["dialogue_rolling_summary"] = rolling
    parsed = EngineState.model_validate({**parsed.model_dump(), **updates})

    blocks = build_context_blocks(parsed)
    updates["context_blocks"] = [b.model_dump() for b in blocks]
    set_status(
        f"[context_preparation] блоков контекста: {len(blocks)} "
        f"(профиль по ## в user_profile.md)"
    )

    node_end("context_preparation_node (SLM + blocks)", f"blocks={len(blocks)}")
    return updates
