"""Реэкспорт Prompt Compositor (backward compatibility)."""

from knowledge_engine.src.node_deep_dive.tutor_prompt_builder import (
    COMMON_FORMATTING,
    DEEP_DIVE_TUTOR_SYSTEM,
    DIALOGUE_PEDAGOGICAL_FLOW,
    GROUNDED_ARCHITECTURE_RULE,
    INTRO_ASSESSMENT_SYSTEM,
    TUTOR_DIALOGUE_PEDAGOGICAL_FLOW,
    TUTOR_DIALOGUE_SYSTEM,
    TUTOR_FORMATTING_RULES,
    TUTOR_GROUNDED_ARCHITECTURE_RULE,
    build_dialogue_system,
    build_dense_system,
    build_intro_system,
    build_lecture_chat_system,
)

__all__ = [
    "COMMON_FORMATTING",
    "DEEP_DIVE_TUTOR_SYSTEM",
    "DIALOGUE_PEDAGOGICAL_FLOW",
    "GROUNDED_ARCHITECTURE_RULE",
    "INTRO_ASSESSMENT_SYSTEM",
    "TUTOR_DIALOGUE_PEDAGOGICAL_FLOW",
    "TUTOR_DIALOGUE_SYSTEM",
    "TUTOR_FORMATTING_RULES",
    "TUTOR_GROUNDED_ARCHITECTURE_RULE",
    "build_dialogue_system",
    "build_dense_system",
    "build_intro_system",
    "build_lecture_chat_system",
]
