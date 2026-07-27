"""LangGraph v0.3.1 — Re-Act loop + smart extractor."""

from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from knowledge_engine.config import MAX_RESEARCH_DEPTH, MAX_RESEARCH_SOURCES
from knowledge_engine.graph.nodes.decomposition import decomposition_node
from knowledge_engine.graph.nodes.discovery import discovery_node
from knowledge_engine.graph.nodes.extractor import extractor_node
from knowledge_engine.graph.nodes.matrix import matrix_node
from knowledge_engine.graph.nodes.research_evaluator import research_evaluator_node
from knowledge_engine.graph.nodes.unraveling import unraveling_node
from knowledge_engine.nodes.lancedb_save import lancedb_save_node
from knowledge_engine.schemas import EngineGraphState


def route_after_research_evaluator(
    state: EngineGraphState,
) -> Literal["extractor_node", "discovery_node", "matrix_node"]:
    pending = state.get("pending_urls") or []
    explored = state.get("explored_urls") or []
    sufficient = bool(state.get("research_sufficient"))
    depth = int(state.get("depth") or 0)

    if pending and len(explored) < MAX_RESEARCH_SOURCES:
        return "extractor_node"

    if sufficient:
        return "matrix_node"

    if depth < MAX_RESEARCH_DEPTH and not sufficient:
        return "discovery_node"

    return "matrix_node"


def build_graph_v03():
    workflow = StateGraph(EngineGraphState)

    workflow.add_node("decomposition_node", decomposition_node)
    workflow.add_node("discovery_node", discovery_node)
    workflow.add_node("extractor_node", extractor_node)
    workflow.add_node("research_evaluator_node", research_evaluator_node)
    workflow.add_node("matrix_node", matrix_node)
    workflow.add_node("lancedb_save_node", lancedb_save_node)
    workflow.add_node("unraveling_node", unraveling_node)

    workflow.set_entry_point("decomposition_node")
    workflow.add_edge("decomposition_node", "discovery_node")
    workflow.add_edge("discovery_node", "extractor_node")
    workflow.add_edge("extractor_node", "research_evaluator_node")
    workflow.add_conditional_edges(
        "research_evaluator_node",
        route_after_research_evaluator,
        {
            "extractor_node": "extractor_node",
            "discovery_node": "discovery_node",
            "matrix_node": "matrix_node",
        },
    )
    workflow.add_edge("discovery_node", "extractor_node")
    workflow.add_edge("matrix_node", "lancedb_save_node")
    workflow.add_edge("lancedb_save_node", "unraveling_node")
    workflow.add_edge("unraveling_node", END)

    memory = MemorySaver()
    return workflow.compile(
        checkpointer=memory,
        interrupt_before=["unraveling_node"],
    )
