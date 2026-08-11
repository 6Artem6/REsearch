"""Article → LanceDB rag_chunks ingestion."""

from knowledge_engine.ingestion.ingest import (
    ingest_document_summary,
    ingest_exa_highlights_fallback,
)

__all__ = ["ingest_document_summary", "ingest_exa_highlights_fallback"]
