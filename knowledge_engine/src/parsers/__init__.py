"""PDF structure parsing helpers for academic ingest."""

from knowledge_engine.src.parsers.paper_structure_analyzer import (
    PaperStructureAnalyzer,
    apply_structure_filter,
    local_fallback_analysis,
    prepare_paper_body_for_gemma,
    prepare_paper_body_for_gemma_async,
    run_inbound_ingest_gate,
)
from knowledge_engine.src.parsers.paper_structure_schema import (
    ExtractMode,
    InformationDensity,
    PaperCredibilityAnalysis,
    PaperStructureAnalysis,
    ParagraphAnalysis,
    ParagraphCredibility,
    ParagraphPriority,
    SemanticLevel,
    TechnicalCorrectness,
)
from knowledge_engine.src.parsers.ingest_gate import (
    INGEST_GATE_REJECT_REASON,
    calculate_article_quality,
)

__all__ = [
    "PaperStructureAnalyzer",
    "apply_structure_filter",
    "local_fallback_analysis",
    "prepare_paper_body_for_gemma",
    "prepare_paper_body_for_gemma_async",
    "run_inbound_ingest_gate",
    "calculate_article_quality",
    "INGEST_GATE_REJECT_REASON",
    "PaperStructureAnalysis",
    "PaperCredibilityAnalysis",
    "ParagraphAnalysis",
    "ParagraphCredibility",
    "ParagraphPriority",
    "SemanticLevel",
    "TechnicalCorrectness",
    "InformationDensity",
    "ExtractMode",
]
