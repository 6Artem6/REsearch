"""Deep Researcher (Gemini) + Profile Validator (1.5B)."""

from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

import knowledge_engine.config as ke_config
from knowledge_engine.config import (
    MAX_RESEARCH_FIND_ROUNDS,
    MAX_RESEARCH_SOURCES,
    MIN_VALIDATED_SOURCES,
)
from knowledge_engine.nodes.context_evaluator import evaluate_and_refine_context_node
from knowledge_engine.nodes.context_preparation import context_preparation_node
from knowledge_engine.nodes.gemini_heavy_reasoning import gemini_heavy_reasoning_node
from knowledge_engine.nodes.gemini_researcher import (
    gemini_extract_source_node,
    gemini_final_matrix_node,
    gemini_find_sources_node,
)
from knowledge_engine.nodes.intent_and_clarify import intent_and_clarify_node
from knowledge_engine.nodes.lancedb_save import lancedb_save_node
from knowledge_engine.nodes.local_rag_check import local_rag_check_node
from knowledge_engine.nodes.matrix import matrix_node
from knowledge_engine.nodes.multi_search import multi_search_node
from knowledge_engine.nodes.profile_validator import profile_validator_node
from knowledge_engine.nodes.unraveling import unraveling_node
from knowledge_engine.schemas import EngineGraphState


def route_after_find_sources(
    state: EngineGraphState,
) -> Literal["gemini_extract_source_node", "gemini_final_matrix_node"]:
    urls = state.get("research_source_urls") or []
    if not urls:
        return "gemini_final_matrix_node"
    return "gemini_extract_source_node"


def route_after_local_rag(
    state: EngineGraphState,
) -> Literal[
    "context_preparation_node",
    "gemini_find_sources_node",
    "multi_search_node",
]:
    if ke_config.SKIP_GEMINI:
        return "multi_search_node"
    if state.get("is_rag_sufficient"):
        return "context_preparation_node"
    return "gemini_find_sources_node"


def route_after_context_eval(
    state: EngineGraphState,
) -> Literal["gemini_heavy_reasoning_node", "evaluate_and_refine_context_node"]:
    if state.get("is_ready_for_gemini") or state.get("context_corrected_once"):
        return "gemini_heavy_reasoning_node"
    return "evaluate_and_refine_context_node"


def route_after_validator(
    state: EngineGraphState,
) -> Literal[
    "gemini_extract_source_node", "gemini_find_sources_node", "gemini_final_matrix_node"
]:
    validated = int(state.get("validated_source_count") or 0)
    idx = int(state.get("research_source_index") or 0)
    urls = state.get("research_source_urls") or []
    find_rounds = int(state.get("research_find_rounds") or 0)

    if validated >= MIN_VALIDATED_SOURCES:
        return "gemini_final_matrix_node"

    if idx < len(urls) and idx < MAX_RESEARCH_SOURCES:
        return "gemini_extract_source_node"

    if find_rounds < MAX_RESEARCH_FIND_ROUNDS:
        return "gemini_find_sources_node"

    return "gemini_final_matrix_node"


def route_after_multi_search(
    state: EngineGraphState,
) -> Literal["matrix_node"]:
    return "matrix_node"


def build_graph_v02():
    workflow = StateGraph(EngineGraphState)

    workflow.add_node("intent_and_clarify_node", intent_and_clarify_node)
    workflow.add_node("local_rag_check_node", local_rag_check_node)
    workflow.add_node("context_preparation_node", context_preparation_node)
    workflow.add_node(
        "evaluate_and_refine_context_node",
        evaluate_and_refine_context_node,
    )
    workflow.add_node("gemini_heavy_reasoning_node", gemini_heavy_reasoning_node)
    workflow.add_node("gemini_find_sources_node", gemini_find_sources_node)
    workflow.add_node("gemini_extract_source_node", gemini_extract_source_node)
    workflow.add_node("profile_validator_node", profile_validator_node)
    workflow.add_node("gemini_final_matrix_node", gemini_final_matrix_node)
    workflow.add_node("multi_search_node", multi_search_node)
    workflow.add_node("matrix_node", matrix_node)
    workflow.add_node("lancedb_save_node", lancedb_save_node)
    workflow.add_node("unraveling_node", unraveling_node)

    workflow.set_entry_point("intent_and_clarify_node")
    workflow.add_edge("intent_and_clarify_node", "local_rag_check_node")
    workflow.add_conditional_edges(
        "local_rag_check_node",
        route_after_local_rag,
        {
            "context_preparation_node": "context_preparation_node",
            "gemini_find_sources_node": "gemini_find_sources_node",
            "multi_search_node": "multi_search_node",
        },
    )
    workflow.add_edge("context_preparation_node", "evaluate_and_refine_context_node")
    workflow.add_conditional_edges(
        "evaluate_and_refine_context_node",
        route_after_context_eval,
        {
            "gemini_heavy_reasoning_node": "gemini_heavy_reasoning_node",
            "evaluate_and_refine_context_node": "evaluate_and_refine_context_node",
        },
    )
    workflow.add_edge("gemini_heavy_reasoning_node", "lancedb_save_node")
    workflow.add_conditional_edges(
        "gemini_find_sources_node",
        route_after_find_sources,
        {
            "gemini_extract_source_node": "gemini_extract_source_node",
            "gemini_final_matrix_node": "gemini_final_matrix_node",
        },
    )
    workflow.add_edge("gemini_extract_source_node", "profile_validator_node")
    workflow.add_conditional_edges(
        "profile_validator_node",
        route_after_validator,
        {
            "gemini_extract_source_node": "gemini_extract_source_node",
            "gemini_find_sources_node": "gemini_find_sources_node",
            "gemini_final_matrix_node": "gemini_final_matrix_node",
        },
    )
    workflow.add_edge("multi_search_node", "matrix_node")
    workflow.add_edge("matrix_node", "lancedb_save_node")
    workflow.add_edge("gemini_final_matrix_node", "lancedb_save_node")
    workflow.add_edge("lancedb_save_node", "unraveling_node")
    workflow.add_edge("unraveling_node", END)

    memory = MemorySaver()
    return workflow.compile(
        checkpointer=memory,
        interrupt_before=["unraveling_node"],
    )
