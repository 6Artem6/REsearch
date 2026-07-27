"""Knowledge Engine v0.7 — LangGraph orchestrator."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from knowledge_engine.src.analytics.chunker import extract_structured_chunks
from knowledge_engine.src.analytics.profiler import (
    build_concept_graph,
    build_profile_gap_map,
    build_tradeoff_matrix,
)
from knowledge_engine.src.dedup import (
    ChunkDedupStore,
    compute_density_delta,
    ingest_document_chunks,
)
from knowledge_engine.src.guardrails import run_stage_0
from knowledge_engine.src.guardrails.personal_context import PersonalContext
from knowledge_engine.src.retrieval.paper_documents import fetch_all_paper_documents
from knowledge_engine.src.retrieval.semantic_scholar import retrieve_scholarly_papers
from knowledge_engine.src.state import (
    KnowledgeEngineState,
    ScrapedDocument,
    StructuredChunk,
)
from knowledge_engine.ui.run_log import node_end, node_start, trace


def _personal_context_from_state(state: KnowledgeEngineState) -> PersonalContext | None:
    raw = state.get("personal_context")
    if raw is None:
        return None
    if isinstance(raw, PersonalContext):
        return raw
    if isinstance(raw, dict):
        return PersonalContext.model_validate(raw)
    return None


def _documents_from_state(state: KnowledgeEngineState) -> List[ScrapedDocument]:
    out: List[ScrapedDocument] = []
    for item in state.get("documents") or []:
        if isinstance(item, ScrapedDocument):
            out.append(item)
        elif isinstance(item, dict):
            out.append(ScrapedDocument.model_validate(item))
    return out


def _chunks_from_state(state: KnowledgeEngineState) -> List[StructuredChunk]:
    out: List[StructuredChunk] = []
    for item in state.get("structured_chunks") or []:
        if isinstance(item, StructuredChunk):
            out.append(item)
        elif isinstance(item, dict):
            out.append(StructuredChunk.model_validate(item))
    return out


def _global_anchor(state: KnowledgeEngineState) -> str:
    user_query = (state.get("user_query") or "").strip()
    ctx = _personal_context_from_state(state)
    profile = (state.get("user_profile_md") or "")[:1200]
    parts = [f"Задача: {user_query}"]
    if ctx:
        parts.append(f"Personal context: {ctx.summary}")
    parts.append(f"Профиль:\n{profile}")
    return "\n".join(parts)


async def node_personal_context(state: KnowledgeEngineState) -> Dict[str, Any]:
    node_start("context_inject")
    detail = ""
    try:
        user_query = (state.get("user_query") or "").strip()
        profile = state.get("user_profile_md") or ""
        trace("Ollama (локально) › Personal Context Injector")
        ctx = await run_stage_0(user_query, profile)
        detail = f"arch={len(ctx.target_architecture)}"
        return {
            "personal_context": ctx.model_dump(),
            "current_step": "personal_context_complete",
        }
    finally:
        node_end("context_inject", detail)


async def node_scholar_retrieval(state: KnowledgeEngineState) -> Dict[str, Any]:
    node_start("scholar_fetch")
    detail = ""
    try:
        user_query = (state.get("user_query") or "").strip()
        papers = await retrieve_scholarly_papers(user_query)
        paper_dicts = [p.model_dump() for p in papers]

        trace(f"scholar › fetch {len(papers)} paper bodies")
        new_docs = await fetch_all_paper_documents(papers)

        store = ChunkDedupStore()
        scraped_total = 0
        unique_ingested = 0
        for doc in new_docs:
            trace("Ollama+LanceDB › dedup ingest чанков")
            accepted, scraped_n = await ingest_document_chunks(
                store, doc.doc_id, doc.raw_markdown
            )
            scraped_total += scraped_n
            unique_ingested += len(accepted)
            doc.cosine_dedup_passed = len(accepted) > 0
            trace(
                f"dedup ✓ doc={doc.doc_id} | scraped={scraped_n} accepted={len(accepted)}"
            )

        delta = compute_density_delta(unique_ingested, scraped_total)
        detail = f"papers={len(papers)} docs={len(new_docs)} delta={delta:.2f}"

        return {
            "scholarly_papers": paper_dicts,
            "documents": [d.model_dump() for d in new_docs],
            "total_scraped_chunks": scraped_total,
            "unique_chunks_ingested": unique_ingested,
            "density_delta": delta,
            "search_depth": 1,
            "current_step": "scholar_retrieval_complete",
        }
    finally:
        node_end("scholar_fetch", detail)


async def node_chunking(state: KnowledgeEngineState) -> Dict[str, Any]:
    node_start("chunking")
    detail = ""
    try:
        anchor = _global_anchor(state)
        documents = _documents_from_state(state)
        structured: List[StructuredChunk] = list(_chunks_from_state(state))

        for i, doc in enumerate(documents, start=1):
            trace(f"Gemini Lite › chunking doc {i}/{len(documents)} | {doc.doc_id}")
            chunks = await asyncio.to_thread(extract_structured_chunks, doc, anchor)
            structured.extend(chunks)

        detail = f"structured_chunks={len(structured)}"
        trace(f"V07 chunking ✓ | {detail}")
        return {
            "structured_chunks": [c.model_dump() for c in structured],
            "current_step": "chunking_complete",
        }
    finally:
        node_end("chunking", detail)


async def node_profiling(state: KnowledgeEngineState) -> Dict[str, Any]:
    node_start("profiling")
    detail = ""
    try:
        anchor = _global_anchor(state)
        profile = state.get("user_profile_md") or ""
        user_query = (state.get("user_query") or "").strip()
        chunks = _chunks_from_state(state)
        documents = _documents_from_state(state)
        personal = _personal_context_from_state(state)
        papers_raw = state.get("scholarly_papers") or []

        trace("Gemini Flash › L2a ConceptGraph")
        concept_graph = await asyncio.to_thread(
            build_concept_graph,
            chunks,
            anchor,
            documents,
            personal,
            papers_raw,
            user_query,
        )
        trace("Gemini Flash › L2b ProfileGapMap")
        gap_map = await asyncio.to_thread(
            build_profile_gap_map,
            concept_graph,
            profile,
            anchor,
            documents,
            personal,
            papers_raw,
            user_query,
        )
        trace("Gemini Flash › L2c TradeoffMatrix")
        matrix = await asyncio.to_thread(
            build_tradeoff_matrix,
            concept_graph,
            gap_map,
            profile,
            anchor,
            personal,
            papers_raw,
            user_query,
        )

        detail = f"matrix options={len(matrix)}"
        trace(f"V07 profiling ✓ | {detail}")
        return {
            "concept_graph": concept_graph,
            "profile_gap_map": gap_map,
            "tradeoff_matrix": matrix,
            "current_step": "completed",
        }
    finally:
        node_end("profiling", detail)


def _build_workflow() -> StateGraph:
    workflow = StateGraph(KnowledgeEngineState)
    workflow.add_node("context_inject", node_personal_context)
    workflow.add_node("scholar_fetch", node_scholar_retrieval)
    workflow.add_node("chunking", node_chunking)
    workflow.add_node("profiling", node_profiling)

    workflow.set_entry_point("context_inject")
    workflow.add_edge("context_inject", "scholar_fetch")
    workflow.add_edge("scholar_fetch", "chunking")
    workflow.add_edge("chunking", "profiling")
    workflow.add_edge("profiling", END)
    return workflow


_v07_compiled: Any = None


def compile_v07_graph() -> Any:
    global _v07_compiled
    if _v07_compiled is None:
        memory = MemorySaver()
        _v07_compiled = _build_workflow().compile(checkpointer=memory)
    return _v07_compiled


knowledge_engine_v07_graph = compile_v07_graph()


async def run_knowledge_engine_v07(
    user_query: str,
    user_profile_md: str,
    thread_id: str,
) -> KnowledgeEngineState:
    from knowledge_engine.src.state import empty_v07_state

    graph = compile_v07_graph()
    initial = empty_v07_state(thread_id, user_profile_md, user_query=user_query)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 32}
    trace(
        f"GRAPH ▶ v0.7 ainvoke | thread_id={thread_id} | "
        "retrieval=SemanticScholar+arXiv"
    )
    result = await graph.ainvoke(initial, config=config)
    trace(f"GRAPH ✓ v0.7 завершён | step={result.get('current_step')}")
    return result
