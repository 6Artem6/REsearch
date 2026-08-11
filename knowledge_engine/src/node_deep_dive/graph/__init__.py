"""Node Deep-Dive LangGraph orchestration."""

from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from knowledge_engine.src.node_deep_dive.graph.nodes import (
    commit_turn_node,
    coverage_router_node,
    dense_lecture_node,
    equivalence_node,
    finalize_response_node,
    ingest_node,
    lazy_intro_node,
    persist_node,
    step_analysis_node,
    sub_concept_eval_node,
    tutor_generate_node,
)
from knowledge_engine.src.node_deep_dive.graph.routing import (
    route_after_ingest,
    route_interaction,
)
from knowledge_engine.src.node_deep_dive.graph.state import TutorGraphState
from knowledge_engine.src.node_deep_dive.graph.subgraphs.init import init_node


def build_tutor_graph() -> StateGraph:
    """Full Node Deep-Dive pipeline: init, chat/verify, persist, finalize."""
    workflow = StateGraph(TutorGraphState)

    workflow.add_node("ingest", ingest_node)
    workflow.add_node("init", init_node)
    workflow.add_node("lazy_intro", lazy_intro_node)
    workflow.add_node("equivalence", equivalence_node)
    workflow.add_node("step_analysis", step_analysis_node)
    workflow.add_node("sub_concept_eval", sub_concept_eval_node)
    workflow.add_node("coverage_router", coverage_router_node)
    workflow.add_node("tutor_generate", tutor_generate_node)
    workflow.add_node("dense_lecture", dense_lecture_node)
    workflow.add_node("commit_turn", commit_turn_node)
    workflow.add_node("persist", persist_node)
    workflow.add_node("finalize_response", finalize_response_node)

    workflow.set_entry_point("ingest")
    workflow.add_conditional_edges(
        "ingest",
        route_after_ingest,
        {
            "init": "init",
            "lazy_intro": "lazy_intro",
            "equivalence": "equivalence",
            "step_analysis": "step_analysis",
        },
    )
    workflow.add_edge("init", "persist")
    workflow.add_edge("lazy_intro", "commit_turn")
    workflow.add_edge("equivalence", "commit_turn")
    workflow.add_edge("step_analysis", "sub_concept_eval")
    workflow.add_edge("sub_concept_eval", "coverage_router")
    workflow.add_conditional_edges(
        "coverage_router",
        route_interaction,
        {
            "tutor_generate": "tutor_generate",
            "dense_lecture": "dense_lecture",
            "persist": "persist",
        },
    )
    workflow.add_edge("tutor_generate", "commit_turn")
    workflow.add_edge("dense_lecture", "commit_turn")
    workflow.add_edge("commit_turn", "persist")
    workflow.add_edge("persist", "finalize_response")
    workflow.add_edge("finalize_response", END)

    return workflow


@lru_cache(maxsize=1)
def get_compiled_tutor_graph():
    """Compiled graph with in-memory checkpointer (dev)."""
    memory = MemorySaver()
    return build_tutor_graph().compile(checkpointer=memory)


__all__ = ["TutorGraphState", "build_tutor_graph", "get_compiled_tutor_graph"]
