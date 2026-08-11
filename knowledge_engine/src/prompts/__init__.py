"""Общие текстовые константы для system prompts (вне доменных воркеров DAG/RAG)."""

from knowledge_engine.src.prompts.engineering_context import (
    CONTEXT_OVERRIDE_FULLSTACK_PRAGMATISM,
    CONTEXT_OVERRIDE_OPERATIONAL_MAC,
    GLOBAL_ENGINEERING_CRITERIA,
    format_optional_context_overrides,
)

__all__ = [
    "CONTEXT_OVERRIDE_FULLSTACK_PRAGMATISM",
    "CONTEXT_OVERRIDE_OPERATIONAL_MAC",
    "GLOBAL_ENGINEERING_CRITERIA",
    "format_optional_context_overrides",
]
