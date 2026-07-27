"""v0.4: Discovery + Domain Trust + source archive."""

from __future__ import annotations

from typing import Any

from knowledge_engine.schemas import EngineGraphState
from knowledge_engine.services.discovery_collect import discovery_state_updates
from knowledge_engine.ui.logger import set_status
from knowledge_engine.ui.pipeline_phase import pipeline_phase
from knowledge_engine.ui.run_log import node_end, node_start


def discovery_node_v04(state: EngineGraphState) -> dict[str, Any]:
    node_start("discovery_node_v04")
    pipeline_phase("Discovery (SearchRegistry + Domain Trust)")
    queries = list(
        state.get("expanded_search_queries") or state.get("search_queries") or []
    )
    if not queries:
        queries = [state.get("original_query") or state.get("user_problem") or ""]

    depth = int(state.get("depth") or 0)
    if depth > 0:
        gaps = state.get("last_research_gaps") or []
        set_status(f"[discovery] Re-Act {depth} | gaps: {', '.join(gaps[:2]) or '—'}")

    set_status("[discovery] SearchRegistry / archive…")
    updates = discovery_state_updates(state, queries, query_limit=12)
    pending_n = len(updates.get("pending_urls") or [])
    material_n = len(updates.get("material_source_urls") or [])
    node_end("discovery_node_v04", f"pending={pending_n} material_saved={material_n}")
    return updates
