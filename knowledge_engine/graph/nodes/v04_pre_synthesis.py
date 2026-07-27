"""v0.4: 7B кластеризация L2 перед matrix."""

from __future__ import annotations

from typing import Any

from knowledge_engine.config import LOCAL_HEAVY_MODEL, OLLAMA_STRUCTURE_NUM_PREDICT
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.schemas import EngineGraphState, PreSynthesisDraft
from knowledge_engine.services.gemini_stateless import global_anchor_from_state
from knowledge_engine.services.local_llm_stateless import run_local_structured
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.ui.logger import set_status
from knowledge_engine.ui.pipeline_phase import pipeline_phase
from knowledge_engine.ui.run_log import node_end, node_start


def pre_synthesis_clusterizer_node(state: EngineGraphState) -> dict[str, Any]:
    node_start("pre_synthesis_clusterizer_node (7B)")
    pipeline_phase("Pre-Synthesis (7B)")
    store = VectorStore()
    l2_lines: list[str] = []
    for nid in state.get("knowledge_node_ids") or []:
        node = store.get_knowledge_node(nid)
        if node and node.level == "L2_EVIDENCE":
            l2_lines.append(node.content[:500])

    anchor = global_anchor_from_state(
        state.get("original_query") or state.get("user_problem") or "",
        state.get("constraints") or state.get("context_constraints") or "",
        state.get("l0_summary") or "",
    )
    l0_ctx = state.get("l0_summary") or ""
    hierarchy = ""
    if state.get("l0_node_id"):
        hierarchy = store.get_hierarchical_context(state["l0_node_id"]) or l0_ctx

    set_status("[pre_synthesis] 7B кластеризация L2…")
    system = (
        f"{RUSSIAN_OUTPUT_RULE} "
        "Группируй L2-факты по тегам, удали дубликаты, подготовь matrix_input для Trade-off матрицы. "
        "JSON PreSynthesisDraft."
    )
    user = (
        f"Иерархия:\n{hierarchy[:8000]}\n\n"
        f"L2 raw ({len(l2_lines)}):\n" + "\n---\n".join(l2_lines[:30])
    )

    draft = run_local_structured(
        LOCAL_HEAVY_MODEL,
        PreSynthesisDraft,
        system,
        user,
        anchor,
        "pre_synthesis / PreSynthesisDraft",
        num_predict=OLLAMA_STRUCTURE_NUM_PREDICT,
    )

    matrix_input = draft.matrix_input or draft.deduplicated_summary
    node_end("pre_synthesis_clusterizer_node", f"tags={len(draft.tags)}")
    return {
        "pre_synthesis_draft": matrix_input[:16000],
    }
