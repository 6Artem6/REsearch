"""Knowledge Engine v0.7 core modules (locks, state, fetcher, dedup)."""

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
    should_terminate_search,
)
from knowledge_engine.src.fetcher import fetch_document
from knowledge_engine.src.graph import (
    compile_v07_graph,
    knowledge_engine_v07_graph,
    run_knowledge_engine_v07,
)
from knowledge_engine.src.guardrails import run_personal_context_stage, run_stage_0
from knowledge_engine.src.locks import (
    staged_uma_lock,
    staged_uma_lock_decorator,
    uma_resource_lock,
)
from knowledge_engine.src.state import (
    KnowledgeEngineState,
    PersonalContext,
    ScrapedDocument,
    StructuredChunk,
    ValidatedQuerySpec,
    empty_v07_state,
)

__all__ = [
    "build_concept_graph",
    "build_profile_gap_map",
    "build_tradeoff_matrix",
    "extract_structured_chunks",
    "run_personal_context_stage",
    "run_stage_0",
    "compile_v07_graph",
    "knowledge_engine_v07_graph",
    "run_knowledge_engine_v07",
    "ChunkDedupStore",
    "fetch_document",
    "KnowledgeEngineState",
    "PersonalContext",
    "ScrapedDocument",
    "StructuredChunk",
    "ValidatedQuerySpec",
    "compute_density_delta",
    "empty_v07_state",
    "ingest_document_chunks",
    "should_terminate_search",
    "staged_uma_lock",
    "staged_uma_lock_decorator",
    "uma_resource_lock",
]
