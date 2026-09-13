"""Режимы сборки system prompt для Node Deep-Dive (отдельно от memory.learning_mode)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from knowledge_engine.src.domains.grounding.memory_schemas import SessionMemory


class InteractionPromptMode(str, Enum):
    INTRO = "intro"
    DIALOGUE_FEEDBACK = "dialogue_feedback"
    LECTURE_CHAT = "lecture_chat"
    LECTURE_DENSE = "lecture_dense"


@dataclass
class PromptComposeContext:
    memory: SessionMemory | None = None
    targeted: bool = False
    topic_already_covered: bool = False
    interaction_axis: str = "lecture_self_check"
    """"topic_qna" swaps LECTURE_DENSE's mandatory-checkpoint structure rules
    for a variant that cancels PART 2 entirely — see
    _dense_lecture_module_parts / LECTURE_MODE_STRUCTURE_RULES_TOPIC_QNA."""
    last_recency_len: int = field(default=0, repr=False)
    recency_tail: str = field(default="", repr=False)
    """Set by compose_system_prompt(); caller injects into the user payload, not system_instruction."""


# Dialogue system prompt (English in dialogue_prompt_en.py).
# Intro / lecture modes still use temporary Russian blocks in tutor_prompt_builder.py.
DIALOGUE_SYSTEM_PROMPT_MIN_LEN = 5_500
DIALOGUE_SYSTEM_PROMPT_MAX_LEN = 20_500
