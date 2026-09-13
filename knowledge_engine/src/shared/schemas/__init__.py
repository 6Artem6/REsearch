"""Shared Pydantic schemas (engine state + lazy LLM contracts)."""

from __future__ import annotations

from typing import Any

_LEGACY: Any = None


def _load_legacy_schemas() -> Any:
    global _LEGACY
    if _LEGACY is None:
        from knowledge_engine.src.shared import schemas_legacy

        _LEGACY = schemas_legacy
    return _LEGACY


def __getattr__(name: str) -> Any:
    if name == "GEMINI_STRUCTURED_CONTRACTS":
        from knowledge_engine.src.shared.schemas.llm_contracts import (
            GEMINI_STRUCTURED_CONTRACTS,
        )

        return GEMINI_STRUCTURED_CONTRACTS
    if name == "StructuredLectureResponse":
        from knowledge_engine.src.domains.grounding.tutor import (
            StructuredLectureResponse,
        )

        return StructuredLectureResponse
    if name == "structured_lecture_to_dense":
        from knowledge_engine.src.domains.grounding.tutor import (
            structured_lecture_to_dense,
        )

        return structured_lecture_to_dense
    if name in {
        "ScopeType",
        "KnowledgeAtom",
        "ParagraphInspectionResult",
        "AggregatedKnowledgeBase",
        "SCOPE_TAGGING_PROMPT_RULES",
        "KNOWLEDGE_TRIANGULATION_TUTOR_RULES",
        "format_takeaways_for_tutor",
        "normalize_knowledge_atoms",
    }:
        from knowledge_engine.src.shared import extraction as _extraction

        return getattr(_extraction, name)
    legacy = _load_legacy_schemas()
    if hasattr(legacy, name):
        return getattr(legacy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    names = set(dir(_load_legacy_schemas()))
    names.update(
        {
            "GEMINI_STRUCTURED_CONTRACTS",
            "StructuredLectureResponse",
            "structured_lecture_to_dense",
            "ScopeType",
            "KnowledgeAtom",
            "ParagraphInspectionResult",
            "AggregatedKnowledgeBase",
            "SCOPE_TAGGING_PROMPT_RULES",
            "KNOWLEDGE_TRIANGULATION_TUTOR_RULES",
            "format_takeaways_for_tutor",
            "normalize_knowledge_atoms",
        }
    )
    return sorted(names)
