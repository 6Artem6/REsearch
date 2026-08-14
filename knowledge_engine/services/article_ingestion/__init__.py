"""Инжест статей: схемы → Mermaid."""

from knowledge_engine.services.article_ingestion.auto_ingest import (
    maybe_ingest_article_diagrams,
)
from knowledge_engine.services.article_ingestion.pipeline import (
    ArticleFormat,
    ArticleIngestionPipeline,
    IngestedDiagram,
)

__all__ = [
    "ArticleFormat",
    "ArticleIngestionPipeline",
    "IngestedDiagram",
    "maybe_ingest_article_diagrams",
]
