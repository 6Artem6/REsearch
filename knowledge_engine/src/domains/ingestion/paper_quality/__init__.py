"""PDF structure parsing helpers for academic ingest."""

from knowledge_engine.src.domains.ingestion.paper_quality.ingest_gate import (
    INGEST_GATE_REJECT_REASON,
    calculate_article_quality,
)
from knowledge_engine.src.domains.ingestion.paper_quality.paper_structure_analyzer import (
    PaperStructureAnalyzer,
    apply_structure_filter,
    local_fallback_analysis,
    prepare_paper_body_for_gemma,
    prepare_paper_body_for_gemma_async,
    run_inbound_ingest_gate,
)
from knowledge_engine.src.domains.ingestion.paper_quality.paper_structure_schema import (
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
