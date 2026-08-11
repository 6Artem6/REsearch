"""Backward-compatible re-exports from Prompt Compositor.

Dialogue system text: use `dialogue_prompt_en.py` (EN). Constants re-exported here
from `tutor_prompt_builder` are mostly LEGACY_RU_TEMPORARY for intro/lecture only.
"""

from knowledge_engine.src.node_deep_dive.prompt_types import InteractionPromptMode
from knowledge_engine.src.node_deep_dive.tutor_prompt_builder import (
    COMMON_FORMATTING,
    CONCEPT_INTRODUCTION_FRAMEWORK,
    DIALOGUE_PEDAGOGICAL_FLOW,
    GROUNDED_ARCHITECTURE_RULE,
    TUTOR_BEHAVIOR_GUARDRAILS,
    TUTOR_DIALOGUE_PEDAGOGICAL_FLOW,
    TUTOR_FORMATTING_RULES,
    TUTOR_GROUNDED_ARCHITECTURE_RULE,
    build_dense_system,
    build_dialogue_system,
    build_intro_system,
    build_lecture_chat_system,
    compose_system_prompt,
)

__all__ = [
    "COMMON_FORMATTING",
    "CONCEPT_INTRODUCTION_FRAMEWORK",
    "DIALOGUE_PEDAGOGICAL_FLOW",
    "GROUNDED_ARCHITECTURE_RULE",
    "InteractionPromptMode",
    "TUTOR_BEHAVIOR_GUARDRAILS",
    "TUTOR_DIALOGUE_PEDAGOGICAL_FLOW",
    "TUTOR_FORMATTING_RULES",
    "TUTOR_GROUNDED_ARCHITECTURE_RULE",
    "build_dialogue_system",
    "build_dense_system",
    "build_intro_system",
    "build_lecture_chat_system",
    "compose_system_prompt",
]
