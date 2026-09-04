"""Инжест статей: схемы → Mermaid.

Submodule imports must not eagerly load auto_ingest/pipeline (circular with llm.py).
"""

from typing import Any

__all__ = [
    "ArticleFormat",
    "ArticleIngestionPipeline",
    "IngestedDiagram",
    "maybe_ingest_article_diagrams",
]


def __getattr__(name: str) -> Any:
    if name == "maybe_ingest_article_diagrams":
        from knowledge_engine.services.article_ingestion.auto_ingest import (
            maybe_ingest_article_diagrams,
        )

        return maybe_ingest_article_diagrams
    if name in {"ArticleFormat", "ArticleIngestionPipeline", "IngestedDiagram"}:
        from knowledge_engine.services.article_ingestion import pipeline as _pipeline

        return getattr(_pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
