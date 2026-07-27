"""Совместимость: используйте services.search.providers.ArxivProvider."""

from knowledge_engine.services.search.providers import (
    ArxivProvider as ArxivSearchProvider,
)

__all__ = ["ArxivSearchProvider"]
