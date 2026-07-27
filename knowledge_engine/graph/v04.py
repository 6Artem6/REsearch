"""LangGraph v0.4 — 3-tier hybrid pipeline."""

from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from knowledge_engine.graph.nodes.v04_decision_router import decision_router_node
from knowledge_engine.graph.nodes.v04_decomposition import decomposition_node_v04
from knowledge_engine.graph.nodes.v04_deep_extractor import deep_extractor_node
from knowledge_engine.graph.nodes.v04_discovery import discovery_node_v04
from knowledge_engine.graph.nodes.v04_document_fetch import document_fetch_node
from knowledge_engine.graph.nodes.v04_matrix import matrix_node_v04
from knowledge_engine.graph.nodes.v04_pre_synthesis import (
    pre_synthesis_clusterizer_node,
)
from knowledge_engine.graph.nodes.v04_query_expansion import query_expansion_node
from knowledge_engine.graph.nodes.v04_research_evaluator import (
    research_evaluator_node_v04,
)
from knowledge_engine.graph.nodes.v04_structure_filter import (
    junk_and_structure_filter_node,
)
from knowledge_engine.graph.nodes.v04_unraveling import unraveling_node_v04
from knowledge_engine.nodes.lancedb_save import lancedb_save_node
from knowledge_engine.schemas import EngineGraphState


def route_from_decision_router(
    state: EngineGraphState,
) -> Literal[
    "document_fetch_node",
    "query_expansion_node",
    "pre_synthesis_clusterizer_node",
]:
    target = state.get("router_target") or "pre_synthesis_clusterizer_node"
    if target == "discovery_node":
        return "query_expansion_node"
    if target in (
        "document_fetch_node",
        "query_expansion_node",
        "pre_synthesis_clusterizer_node",
    ):
        return target
    return "pre_synthesis_clusterizer_node"


def build_graph_v04():
    workflow = StateGraph(EngineGraphState)

    workflow.add_node("decomposition_node", decomposition_node_v04)
    workflow.add_node("query_expansion_node", query_expansion_node)
    workflow.add_node("discovery_node", discovery_node_v04)
    workflow.add_node("document_fetch_node", document_fetch_node)
    workflow.add_node("junk_and_structure_filter_node", junk_and_structure_filter_node)
    workflow.add_node("deep_extractor_node", deep_extractor_node)
    workflow.add_node("research_evaluator_node", research_evaluator_node_v04)
    workflow.add_node("decision_router_node", decision_router_node)
    workflow.add_node("pre_synthesis_clusterizer_node", pre_synthesis_clusterizer_node)
    workflow.add_node("matrix_node", matrix_node_v04)
    workflow.add_node("lancedb_save_node", lancedb_save_node)
    workflow.add_node("unraveling_node", unraveling_node_v04)

    workflow.set_entry_point("decomposition_node")
    workflow.add_edge("decomposition_node", "query_expansion_node")
    workflow.add_edge("query_expansion_node", "discovery_node")
    workflow.add_edge("discovery_node", "document_fetch_node")
    workflow.add_edge("document_fetch_node", "junk_and_structure_filter_node")
    workflow.add_edge("junk_and_structure_filter_node", "deep_extractor_node")
    workflow.add_edge("deep_extractor_node", "research_evaluator_node")
    workflow.add_edge("research_evaluator_node", "decision_router_node")
    workflow.add_conditional_edges(
        "decision_router_node",
        route_from_decision_router,
        {
            "document_fetch_node": "document_fetch_node",
            "query_expansion_node": "query_expansion_node",
            "pre_synthesis_clusterizer_node": "pre_synthesis_clusterizer_node",
        },
    )
    workflow.add_edge("query_expansion_node", "discovery_node")
    workflow.add_edge("pre_synthesis_clusterizer_node", "matrix_node")
    workflow.add_edge("matrix_node", "lancedb_save_node")
    workflow.add_edge("lancedb_save_node", "unraveling_node")
    workflow.add_edge("unraveling_node", END)

    memory = MemorySaver()
    return workflow.compile(
        checkpointer=memory,
        interrupt_before=["unraveling_node"],
    )
