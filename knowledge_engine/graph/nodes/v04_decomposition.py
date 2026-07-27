"""v0.4: Gemini-only L0/L1 декомпозиция."""

from __future__ import annotations

from typing import Any

from knowledge_engine.llm_locale import GEMINI_RUSSIAN_ROLE, RUSSIAN_OUTPUT_RULE
from knowledge_engine.schemas import EngineGraphState, L0DecompositionResult
from knowledge_engine.services.gemini_stateless import (
    global_anchor_from_state,
    run_stateless_gemini,
)
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.ui.logger import set_status
from knowledge_engine.ui.pipeline_phase import pipeline_phase
from knowledge_engine.ui.run_log import node_end, node_start


def decomposition_node_v04(state: EngineGraphState) -> dict[str, Any]:
    node_start("decomposition_node_v04 (Gemini)")
    pipeline_phase("Decomposition (Gemini)")
    query = state.get("original_query") or state.get("user_problem") or ""
    constraints = state.get("constraints") or state.get("context_constraints") or ""
    anchor = global_anchor_from_state(query, constraints, "")

    set_status("[decomposition] Gemini → L0 + L1 + базовые запросы…")
    system = (
        f"{GEMINI_RUSSIAN_ROLE} {RUSSIAN_OUTPUT_RULE} "
        "Построй мета-карту Deep Research (L0) и 2–4 L1-паттерна для Trade-off матрицы. "
        "search_queries: 2–3 коротких базовых запроса (SOTA / infra / prod)."
    )
    user = f"Задача:\n{query}\nОграничения:\n{constraints}"

    result = run_stateless_gemini(
        system,
        user,
        anchor,
        response_schema=L0DecompositionResult,
        label="v04 decomposition / L0",
    )

    store = VectorStore()
    l0_id = store.save_knowledge_node("L0_META", result.l0_summary, parent_id=None)
    l1_ids: list[str] = []
    for pat in result.l1_patterns[:6]:
        body = f"{pat.title}\n{pat.description}".strip()
        l1_ids.append(store.save_knowledge_node("L1_PATTERN", body, parent_id=l0_id))

    node_end("decomposition_node_v04", f"L0 + {len(l1_ids)} L1")
    return {
        "l0_summary": result.l0_summary,
        "l0_node_id": l0_id,
        "l1_node_ids": l1_ids,
        "search_queries": result.search_queries[:3],
        "pending_urls": [],
        "explored_urls": [],
        "depth": 0,
        "knowledge_node_ids": [l0_id] + l1_ids,
        "expanded_search_queries": [],
    }
