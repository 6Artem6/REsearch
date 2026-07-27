"""Knowledge Engine v0.7 — Stages 4 & 5 (Gemini Lite / Flash analytics)."""

from knowledge_engine.src.analytics.chunker import extract_structured_chunks
from knowledge_engine.src.analytics.profiler import (
    build_concept_graph,
    build_profile_gap_map,
    build_tradeoff_matrix,
)
from knowledge_engine.src.analytics.schemas import (
    ConceptGraph,
    ProfileGapMap,
    TradeoffMatrixResult,
)

__all__ = [
    "ConceptGraph",
    "ProfileGapMap",
    "TradeoffMatrixResult",
    "build_concept_graph",
    "build_profile_gap_map",
    "build_tradeoff_matrix",
    "extract_structured_chunks",
]
