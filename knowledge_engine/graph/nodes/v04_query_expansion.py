"""v0.4: 7B разворачивает базовые запросы Gemini в 10–15 векторов."""

from __future__ import annotations

from typing import Any

from knowledge_engine.config import LOCAL_HEAVY_MODEL
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.schemas import EngineGraphState, QueryExpansionResult
from knowledge_engine.services.gemini_stateless import global_anchor_from_state
from knowledge_engine.services.local_llm_stateless import run_local_structured
from knowledge_engine.services.query_expander import apply_smart_query_syntax_batch
from knowledge_engine.ui.logger import set_status
from knowledge_engine.ui.pipeline_phase import pipeline_phase
from knowledge_engine.ui.run_log import node_end, node_start


def query_expansion_node(state: EngineGraphState) -> dict[str, Any]:
    node_start("query_expansion_node (7B)")
    pipeline_phase("Query Expansion (7B)")
    base = list(state.get("search_queries") or [])
    if not base:
        base = [state.get("original_query") or state.get("user_problem") or ""]

    anchor = global_anchor_from_state(
        state.get("original_query") or state.get("user_problem") or "",
        state.get("constraints") or state.get("context_constraints") or "",
        state.get("l0_summary") or "",
    )
    set_status("[query_expansion] 7B → 10–15 поисковых векторов…")
    system = (
        f"{RUSSIAN_OUTPUT_RULE} "
        "Разверни 2–3 базовые запросы в 10–15 точных поисковых векторов для технического поиска. "
        "Фокус: architecture, failure mode, tail latency, LanceDB/GraphRAG. "
        "НЕ добавляй site:/минус-слова вручную — post-process добавит операторы SearXNG. "
        "JSON QueryExpansionResult."
    )
    user = "Базовые запросы:\n" + "\n".join(f"- {q}" for q in base[:4])

    result = run_local_structured(
        LOCAL_HEAVY_MODEL,
        QueryExpansionResult,
        system,
        user,
        anchor,
        "query_expansion / QueryExpansionResult",
    )
    expanded = [q.strip() for q in result.expanded_queries if q.strip()][:15]
    if len(expanded) < 5:
        expanded = base + expanded
    expanded = apply_smart_query_syntax_batch(expanded)

    node_end("query_expansion_node", f"vectors={len(expanded)}")
    return {"expanded_search_queries": expanded}
