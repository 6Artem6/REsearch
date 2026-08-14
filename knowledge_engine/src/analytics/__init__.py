"""Knowledge Engine v0.7 — Stages 4 & 5 (Gemini Lite / Flash analytics)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
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


def __getattr__(name: str):
    if name == "extract_structured_chunks":
        from knowledge_engine.src.analytics.chunker import extract_structured_chunks

        return extract_structured_chunks
    if name == "build_concept_graph":
        from knowledge_engine.src.analytics.profiler import build_concept_graph

        return build_concept_graph
    if name == "build_profile_gap_map":
        from knowledge_engine.src.analytics.profiler import build_profile_gap_map

        return build_profile_gap_map
    if name == "build_tradeoff_matrix":
        from knowledge_engine.src.analytics.profiler import build_tradeoff_matrix

        return build_tradeoff_matrix
    if name in ("ConceptGraph", "ProfileGapMap", "TradeoffMatrixResult"):
        from knowledge_engine.src.analytics.schemas import (
            ConceptGraph,
            ProfileGapMap,
            TradeoffMatrixResult,
        )

        return {
            "ConceptGraph": ConceptGraph,
            "ProfileGapMap": ProfileGapMap,
            "TradeoffMatrixResult": TradeoffMatrixResult,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
