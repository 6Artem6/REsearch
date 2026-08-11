"""v0.7 analytics — Gemini Flash/Lite contracts (aliases с реестром)."""

from knowledge_engine.src.analytics.schemas import (
    ChunkExtractionResult as ChunkExtractionContract,
)
from knowledge_engine.src.analytics.schemas import ConceptGraph as ConceptGraphContract
from knowledge_engine.src.analytics.schemas import (
    ProfileGapMap as ProfileGapMapContract,
)
from knowledge_engine.src.analytics.schemas import (
    TradeoffMatrixResult as TradeoffMatrixContract,
)

__all__ = [
    "ChunkExtractionContract",
    "ConceptGraphContract",
    "ProfileGapMapContract",
    "TradeoffMatrixContract",
]
