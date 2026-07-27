"""Проверка LanceDB перед внешним поиском."""

from __future__ import annotations

from typing import Any

from knowledge_engine.config import RAG_HYBRID_LIMIT, RAG_MIN_RELEVANT_HITS
from knowledge_engine.schemas import EngineGraphState, EngineState
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.ui.logger import set_status
from knowledge_engine.ui.run_log import node_end, node_start


def _rag_query(state: EngineState) -> str:
    parts = [state.user_problem, state.context_constraints]
    for a in state.abstractions:
        parts.append(f"{a.title} {a.cs_concept}")
    return "\n".join(parts)


def local_rag_check_node(state: EngineGraphState) -> dict[str, Any]:
    node_start("local_rag_check_node (LanceDB hybrid)")
    parsed = EngineState.model_validate(state)
    set_status("[local_rag_check] LanceDB hybrid_search…")
    store = VectorStore()
    hits = store.hybrid_search(_rag_query(parsed), limit=RAG_HYBRID_LIMIT)

    existing_urls = {s.url for s in parsed.found_summaries}
    merged_summaries = list(parsed.found_summaries)
    for h in hits:
        if h.url not in existing_urls:
            merged_summaries.append(h)
            existing_urls.add(h.url)

    is_sufficient = len(hits) >= RAG_MIN_RELEVANT_HITS
    if is_sufficient:
        set_status(
            f"[LanceDB] найдено {len(hits)} релевантных summary — пропуск внешнего поиска"
        )
    else:
        set_status(f"[LanceDB] мало контекста ({len(hits)}) — нужен внешний поиск")

    facts = list(parsed.found_facts)
    for s in hits:
        for t in s.key_takeaways[:2]:
            if t not in facts:
                facts.append(t)

    node_end(
        "local_rag_check_node (LanceDB hybrid)",
        f"hits={len(hits)}, sufficient={is_sufficient}",
    )
    return {
        "found_summaries": [s.model_dump() for s in merged_summaries],
        "is_rag_sufficient": is_sufficient,
        "is_facts_sufficient": is_sufficient,
        "found_facts": facts,
    }
