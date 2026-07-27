"""v0.3: Trade-off матрица из иерархического LanceDB + stateless Gemini."""

from __future__ import annotations

from typing import Any

from knowledge_engine.llm_locale import GEMINI_RUSSIAN_ROLE, RUSSIAN_OUTPUT_RULE
from knowledge_engine.schemas import AnalysisReport, EngineGraphState
from knowledge_engine.services.gemini_stateless import global_anchor_from_state
from knowledge_engine.services.hybrid_llm import run_matrix_hybrid
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.ui.logger import set_status
from knowledge_engine.ui.run_log import node_end, node_start


def matrix_node(state: EngineGraphState) -> dict[str, Any]:
    node_start("matrix_node (v0.3 hierarchical Gemini)")
    query = state.get("original_query") or state.get("user_problem") or ""
    constraints = state.get("constraints") or ""
    l0 = state.get("l0_summary") or ""
    anchor = global_anchor_from_state(query, constraints, l0)

    store = VectorStore()
    hits = store.hybrid_search_nodes(f"{query}\n{constraints}", limit=5)
    context_blocks: list[str] = []
    for node in hits:
        ctx = store.get_hierarchical_context(node.id)
        if ctx:
            context_blocks.append(ctx)

    if not context_blocks and state.get("l0_node_id"):
        ctx = store.get_hierarchical_context(state["l0_node_id"])
        if ctx:
            context_blocks.append(ctx)

    hierarchy_text = "\n\n---\n\n".join(context_blocks) or l0

    set_status("[matrix] Stateless Gemini → AnalysisReport JSON…")
    system = (
        f"{GEMINI_RUSSIAN_ROLE} {RUSSIAN_OUTPUT_RULE} "
        "На основе иерархического контекста (L0/L1/L2) построй AnalysisReport: "
        "abstractions + ровно 3 options (id 1,2,3). category: Классика / SOTA (Современное) / Минимализм. "
        "Все поля на русском."
    )
    user = f"ИЕРАРХИЧЕСКИЙ КОНТЕКСТ:\n{hierarchy_text[:14000]}"

    report = run_matrix_hybrid(
        system,
        user,
        anchor,
        AnalysisReport,
        "matrix / AnalysisReport",
    )

    if len(report.options) != 3:
        raise RuntimeError(f"Ожидалось 3 options, получено {len(report.options)}")

    raw = report.model_dump_json(ensure_ascii=False)
    node_end("matrix_node (v0.3 hierarchical Gemini)", "report ok")
    return {
        "report": report.model_dump(),
        "abstractions": [a.model_dump() for a in report.abstractions],
        "gemini_raw_response": raw,
    }
