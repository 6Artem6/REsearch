"""Общая инициализация state для CLI и API."""

from __future__ import annotations

import knowledge_engine.config as cfg
from knowledge_engine.schemas import EngineState


def build_initial_state(
    problem: str,
    constraints: str,
    *,
    discovery_cache_first: bool = False,
) -> dict:
    base = EngineState(
        user_problem=problem,
        context_constraints=constraints,
        discovery_cache_first=discovery_cache_first,
        material_source_urls=[],
    ).model_dump(mode="json")
    version = (cfg.GRAPH_VERSION or "0.4").strip()
    if version in ("0.3", "0.4"):
        base.update(
            {
                "original_query": problem,
                "constraints": constraints,
                "l0_summary": "",
                "pending_urls": [],
                "explored_urls": [],
                "depth": 0,
                "l1_node_ids": [],
                "knowledge_node_ids": [],
            }
        )
    return base
