"""v0.4: Python + 1.5B контроль Re-Act routing."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from knowledge_engine.config import LOCAL_ROUTER_MODEL, MAX_RESEARCH_DEPTH, MAX_URLS
from knowledge_engine.llm import invoke_logged, structured_chat
from knowledge_engine.schemas import EngineGraphState, RouterDecision
from knowledge_engine.ui.logger import set_status
from knowledge_engine.ui.pipeline_phase import pipeline_phase
from knowledge_engine.ui.run_log import node_end, node_start, trace

_VALID_TARGETS = frozenset(
    {
        "document_fetch_node",
        "query_expansion_node",
        "pre_synthesis_clusterizer_node",
    }
)


def _compute_target(state: EngineGraphState) -> tuple[str, dict[str, Any]]:
    """Жёсткие лимиты в Python; 1.5B только подтверждает, не расширяет loop."""
    pending = list(state.get("pending_urls") or [])
    explored = state.get("explored_urls") or []
    sufficient = bool(state.get("research_sufficient"))
    depth = int(state.get("depth") or 0)
    explored_n = len(explored)
    extra: dict[str, Any] = {}

    at_url_cap = explored_n >= MAX_URLS
    at_depth_cap = depth >= MAX_RESEARCH_DEPTH

    if sufficient:
        return "pre_synthesis_clusterizer_node", extra

    if at_url_cap or at_depth_cap:
        if pending:
            extra["pending_urls"] = []
            trace(
                f"ROUTER stop drain pending={len(pending)} | "
                f"urls={explored_n}/{MAX_URLS} depth={depth}/{MAX_RESEARCH_DEPTH}"
            )
        return "pre_synthesis_clusterizer_node", extra

    if pending:
        return "document_fetch_node", extra

    if depth < MAX_RESEARCH_DEPTH:
        return "query_expansion_node", extra

    return "pre_synthesis_clusterizer_node", extra


def _enforce_hard_limits(
    target: str,
    state: EngineGraphState,
    extra: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    explored_n = len(state.get("explored_urls") or [])
    depth = int(state.get("depth") or 0)
    sufficient = bool(state.get("research_sufficient"))

    if sufficient and target != "pre_synthesis_clusterizer_node":
        trace(f"ROUTER clamp sufficient → pre_synthesis (was {target})")
        return "pre_synthesis_clusterizer_node", extra

    if explored_n >= MAX_URLS and target in (
        "document_fetch_node",
        "query_expansion_node",
    ):
        trace(f"ROUTER clamp url cap → pre_synthesis (was {target})")
        pending = list(state.get("pending_urls") or [])
        if pending:
            extra = dict(extra)
            extra["pending_urls"] = []
        return "pre_synthesis_clusterizer_node", extra

    if depth >= MAX_RESEARCH_DEPTH and target == "query_expansion_node":
        trace(f"ROUTER clamp depth cap → pre_synthesis (was {target})")
        return "pre_synthesis_clusterizer_node", extra

    return target, extra


def decision_router_node(state: EngineGraphState) -> dict[str, Any]:
    node_start("decision_router_node (1.5B/Python)")
    pipeline_phase("Router (1.5B/Python)")
    target, extra = _compute_target(state)
    depth = int(state.get("depth") or 0)
    explored_n = len(state.get("explored_urls") or [])

    if depth >= MAX_RESEARCH_DEPTH:
        trace(f"ROUTER hard limit depth>={MAX_RESEARCH_DEPTH}")
    if explored_n >= MAX_URLS:
        trace(f"ROUTER hard limit urls>={MAX_URLS}")

    rationale = (
        f"target={target} | depth={depth} | urls={explored_n}/{MAX_URLS} | "
        f"sufficient={state.get('research_sufficient')}"
    )
    try:
        llm = structured_chat(
            LOCAL_ROUTER_MODEL, RouterDecision, temperature=0.0, num_predict=256
        )
        msg = invoke_logged(
            llm,
            [
                SystemMessage(
                    content="Подтверди routing Re-Act loop одним JSON RouterDecision (next_node, rationale)."
                ),
                HumanMessage(content=rationale),
            ],
            "router / RouterDecision",
        )
        if isinstance(msg, RouterDecision) and msg.next_node in _VALID_TARGETS:
            target = msg.next_node
            rationale = msg.rationale or rationale
    except Exception as exc:
        trace(f"ROUTER 1.5B skip, Python only | {exc}")

    target, extra = _enforce_hard_limits(target, state, extra)

    set_status(f"[router] → {target}")
    trace(f"ROUTER ✓ {rationale}")
    node_end("decision_router_node", target)
    return {"router_target": target, **extra}
