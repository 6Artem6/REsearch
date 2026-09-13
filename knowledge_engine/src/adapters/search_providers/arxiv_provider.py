"""Совместимость: используйте services.search.providers.ArxivProvider."""

from knowledge_engine.src.adapters.search_providers.providers import (
    ArxivProvider as ArxivSearchProvider,
)

__all__ = ["ArxivSearchProvider"]
