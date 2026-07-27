"""v0.3: Discovery + Domain Trust + source archive."""

from __future__ import annotations

from typing import Any

from knowledge_engine.schemas import EngineGraphState
from knowledge_engine.services.discovery_collect import discovery_state_updates
from knowledge_engine.ui.logger import set_status
from knowledge_engine.ui.run_log import node_end, node_start


def discovery_node(state: EngineGraphState) -> dict[str, Any]:
    node_start("discovery_node (v0.3 search)")
    queries = list(state.get("search_queries") or [])
    if not queries:
        queries = [state.get("original_query") or state.get("user_problem") or ""]

    set_status("[discovery] SearchRegistry: горизонты / API…")
    depth = int(state.get("depth") or 0)
    if depth > 0:
        gaps = state.get("last_research_gaps") or []
        set_status(
            f"[discovery] Re-Act раунд {depth} | gaps: " f"{', '.join(gaps[:3]) or '—'}"
        )
    updates = discovery_state_updates(state, queries, query_limit=4)
    node_end(
        "discovery_node (v0.3 search)",
        f"pending={len(updates.get('pending_urls') or [])}",
    )
    return updates
