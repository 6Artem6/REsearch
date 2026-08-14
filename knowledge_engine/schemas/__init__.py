"""Shared Pydantic schemas (engine state + lazy LLM contracts)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

_LEGACY: Any = None


def _load_legacy_schemas() -> Any:
    global _LEGACY
    if _LEGACY is not None:
        return _LEGACY
    path = Path(__file__).resolve().parent.parent / "schemas.py"
    spec = importlib.util.spec_from_file_location(
        "knowledge_engine._engine_schemas",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load engine schemas from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _LEGACY = mod
    return mod


def __getattr__(name: str) -> Any:
    if name == "GEMINI_STRUCTURED_CONTRACTS":
        from knowledge_engine.schemas.llm_contracts import GEMINI_STRUCTURED_CONTRACTS

        return GEMINI_STRUCTURED_CONTRACTS
    if name == "StructuredLectureResponse":
        from knowledge_engine.schemas.llm_contracts.tutor import (
            StructuredLectureResponse,
        )

        return StructuredLectureResponse
    if name == "structured_lecture_to_dense":
        from knowledge_engine.schemas.llm_contracts.tutor import (
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
        from knowledge_engine.schemas import extraction as _extraction

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
