"""Prompt Compositor: единый источник правил и сборка system prompts тьютора.

System prompt language policy:
- **English** for all LLM system instructions (cache-friendly, token-efficient).
- **Russian** only for model *output* fields (`feedback_on_answer`, `lecture_body`, …)
  via `RUSSIAN_OUTPUT_RULE` or explicit LANGUAGE blocks in EN modules.

Canonical sources by mode:
- `dialogue_feedback` → `dialogue_prompt_en.py` (fully EN system; stable).
- `intro`, `lecture_dense`, `lecture_chat` → `lecture_prompt_en.py` (EN system; Russian hints as trailing docstrings).

Do not add new Russian system rules here; extend EN modules or add `*_prompt_en.py`.
"""

from __future__ import annotations

import logging

from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.services.curriculum_whitelist_prompt import (
    _TUTOR_NEIGHBORHOOD_RULES,
)
from knowledge_engine.src.node_deep_dive.context_bounded_eval import (
    CONTEXT_BOUNDED_QUESTION_RULES,
)
from knowledge_engine.src.node_deep_dive.dialogue_prompt_en import (
    DIALOGUE_RECENCY_REMINDERS_EN,
    dialogue_base_system_parts,
    dialogue_module_parts,
)
from knowledge_engine.src.node_deep_dive.tutor_field_limits import (
    PROMPT_LECTURE_BODY_TARGET_MAX_WORDS,
)
from knowledge_engine.src.node_deep_dive.interaction_prompt_layout import (
    BLOCK_STATIC_PRESET_HEADER,
)
from knowledge_engine.src.node_deep_dive.lecture_prompt_en import (
    COMMON_FORMATTING,
    CONCEPT_INTRODUCTION_FRAMEWORK,
    CONCEPT_INTRODUCTION_INTRO_RULES,
    CONCEPT_INTRODUCTION_LECTURE_RULE,
    DEEP_DIVE_MECH_RULE,
    DENSE_FUNDAMENTALS_BLOCK,
    DENSE_LECTURE_INTERACTION_MODE,
    DENSE_REFERENCES_WHITELIST,
    DIAGRAM_INTEGRATION_CROSS_REF,
    DIAGRAM_SELECTION_RULES,
    DIALOGUE_PEDAGOGICAL_FLOW,
    DIALOGUE_TUTOR_JSON_CONTRACT,
    EXTERNAL_SEARCH_TOOL_RULE,
    GLOBAL_REGISTRY_PROMPT_RULES,
    GROUNDED_ARCHITECTURE_RULE,
    INTRO_MODULE_CONTEXT_BRIDGE,
    INTRO_MODULE_INTRO_ASSESSMENT,
    INTRO_RECENCY_TAIL,
    KNOWLEDGE_TRIANGULATION_LECTURE_RULES,
    LECTURE_CHAT_INTERACTION_MODE,
    LECTURE_CHAT_TAIL_RULES,
    LECTURE_DENSE_RULES,
    LECTURE_GAP_STEERING_RULES,
    LECTURE_MODE_STRUCTURE_RULES,
    LECTURE_SYSTEM_PROMPT,
    NO_CLOSING_QUESTIONNAIRES,
    NODE_MATERIALS_TOUR_RULES,
    PINNED_DIAGRAMS_GUIDING_RULES,
    SOFT_PITCHING_RULE,
    TARGETED_LECTURE_WORDS,
    TOPIC_ALREADY_COVERED_DENSE,
    TOPIC_COMPLETION_RULE,
    TUTOR_PERSONA,
    VERIFIED_LINK_GROUNDING_RULE,
)
from knowledge_engine.src.node_deep_dive.memory_schemas import SessionMemory
from knowledge_engine.src.node_deep_dive.prompt_types import (
    DIALOGUE_SYSTEM_PROMPT_MAX_LEN,
    DIALOGUE_SYSTEM_PROMPT_MIN_LEN,
    InteractionPromptMode,
    PromptComposeContext,
)
from knowledge_engine.src.node_deep_dive.term_registry import (
    format_already_explained_terms_block,
)
from knowledge_engine.src.processors.question_formation_rules import (
    QUESTION_FORMATION_RULES,
)
from knowledge_engine.src.processors.source_anchors import (
    REASONER_SOURCE_ATTRIBUTION_PROMPT,
)
from knowledge_engine.src.prompts.engineering_context import (
    GLOBAL_ENGINEERING_CRITERIA,
    format_optional_context_overrides,
)
from knowledge_engine.src.source_evaluator.evaluator import (
    format_whitelist_for_reasoner_prompt,
)

logger = logging.getLogger(__name__)

# Backward-compatible aliases (re-export lecture_prompt_en).
TUTOR_FORMATTING_RULES = COMMON_FORMATTING
"""
RU (пояснение): alias COMMON_FORMATTING для legacy импортов.
"""
TUTOR_DIALOGUE_PEDAGOGICAL_FLOW = DIALOGUE_PEDAGOGICAL_FLOW
"""
RU (пояснение): alias DIALOGUE_PEDAGOGICAL_FLOW.
"""
TUTOR_GROUNDED_ARCHITECTURE_RULE = GROUNDED_ARCHITECTURE_RULE
"""
RU (пояснение): alias GROUNDED_ARCHITECTURE_RULE.
"""


def _base_system_prompt_parts() -> list[str]:
    return [
        RUSSIAN_OUTPUT_RULE,
        TUTOR_PERSONA,
        COMMON_FORMATTING,
        GLOBAL_REGISTRY_PROMPT_RULES,
        VERIFIED_LINK_GROUNDING_RULE,
    ]


def _intro_module_parts() -> list[str]:
    return [
        INTRO_MODULE_INTRO_ASSESSMENT,
        INTRO_MODULE_CONTEXT_BRIDGE,
        CONCEPT_INTRODUCTION_INTRO_RULES,
        CONTEXT_BOUNDED_QUESTION_RULES,
        QUESTION_FORMATION_RULES,
    ]


def _dialogue_module_parts() -> list[str]:
    return dialogue_module_parts()


def _lecture_chat_module_parts() -> list[str]:
    return [
        LECTURE_CHAT_INTERACTION_MODE,
        CONCEPT_INTRODUCTION_FRAMEWORK,
        LECTURE_DENSE_RULES,
        LECTURE_MODE_STRUCTURE_RULES,
        KNOWLEDGE_TRIANGULATION_LECTURE_RULES,
        DIAGRAM_INTEGRATION_CROSS_REF,
        DIAGRAM_SELECTION_RULES,
        NO_CLOSING_QUESTIONNAIRES,
        TOPIC_COMPLETION_RULE,
        DEEP_DIVE_MECH_RULE,
        SOFT_PITCHING_RULE,
        LECTURE_CHAT_TAIL_RULES,
        GROUNDED_ARCHITECTURE_RULE,
        NODE_MATERIALS_TOUR_RULES,
        _TUTOR_NEIGHBORHOOD_RULES,
        format_whitelist_for_reasoner_prompt(),
        CONTEXT_BOUNDED_QUESTION_RULES,
        QUESTION_FORMATION_RULES,
        DIALOGUE_TUTOR_JSON_CONTRACT,
    ]


def _dense_lecture_module_parts(ctx: PromptComposeContext) -> list[str]:
    from knowledge_engine.config import LECTURE_MIN_WORDS_TARGET

    min_w = max(1000, LECTURE_MIN_WORDS_TARGET)
    parts: list[str] = [
        f"{BLOCK_STATIC_PRESET_HEADER}\n" f"{LECTURE_DENSE_RULES}\n",
        DENSE_LECTURE_INTERACTION_MODE,
        LECTURE_SYSTEM_PROMPT,
        GROUNDED_ARCHITECTURE_RULE,
    ]
    if ctx.topic_already_covered:
        parts.append(TOPIC_ALREADY_COVERED_DENSE)
    if not ctx.topic_already_covered and not ctx.targeted:
        parts.append(
            f"lecture_body (REQUIRED): ≥{min_w} words for chat; "
            f"target ≤{PROMPT_LECTURE_BODY_TARGET_MAX_WORDS} words unless user asks for more"
        )
    else:
        parts.append(
            "lecture_body: relevant delta only or targeted deep dive (no base repeat)"
        )
    parts.extend(
        [
            DENSE_FUNDAMENTALS_BLOCK,
            NO_CLOSING_QUESTIONNAIRES,
            TOPIC_COMPLETION_RULE,
            DEEP_DIVE_MECH_RULE,
            SOFT_PITCHING_RULE,
            PINNED_DIAGRAMS_GUIDING_RULES,
            DIAGRAM_SELECTION_RULES,
            EXTERNAL_SEARCH_TOOL_RULE,
            CONTEXT_BOUNDED_QUESTION_RULES,
            QUESTION_FORMATION_RULES,
            REASONER_SOURCE_ATTRIBUTION_PROMPT,
            format_whitelist_for_reasoner_prompt(),
            DENSE_REFERENCES_WHITELIST,
        ]
    )
    if ctx.targeted:
        parts.append(TARGETED_LECTURE_WORDS.format(min_w=min_w))
    return parts


def build_critical_rules_recency_tail(
    *,
    mode: InteractionPromptMode | None = None,
    memory: SessionMemory | None = None,
    topic_already_covered: bool = False,
    # Deprecated kwargs (lecture_rag_context migration); prefer `mode`.
    dense_lecture: bool = False,
    dialogue: bool = False,
    lecture_chat: bool = False,
    lite: bool = False,
) -> str:
    """
    Компактный хвост приоритетов — ровно один раз в конце system prompt (не в dynamic suffix).
    """
    if mode is None:
        if dense_lecture:
            mode = InteractionPromptMode.LECTURE_DENSE
        elif lecture_chat:
            mode = InteractionPromptMode.LECTURE_CHAT
        elif dialogue:
            mode = InteractionPromptMode.DIALOGUE_FEEDBACK
        else:
            mode = InteractionPromptMode.INTRO
    if lite:
        pass  # deprecated; ignored

    parts: list[str] = [
        "=== CRITICAL_RULES_RECENCY (highest priority — obey over earlier sections) ===",
        GLOBAL_ENGINEERING_CRITERIA.strip(),
    ]
    overrides = format_optional_context_overrides()
    if overrides:
        parts.append(overrides)

    if mode == InteractionPromptMode.DIALOGUE_FEEDBACK:
        parts.append(DIALOGUE_RECENCY_REMINDERS_EN)
    elif mode == InteractionPromptMode.LECTURE_CHAT:
        parts.extend(
            [
                LECTURE_MODE_STRUCTURE_RULES,
                NO_CLOSING_QUESTIONNAIRES,
                CONCEPT_INTRODUCTION_LECTURE_RULE,
                DIAGRAM_INTEGRATION_CROSS_REF,
                DIAGRAM_SELECTION_RULES,
                PINNED_DIAGRAMS_GUIDING_RULES,
            ]
        )
    elif mode == InteractionPromptMode.LECTURE_DENSE:
        parts.extend(
            [
                LECTURE_MODE_STRUCTURE_RULES,
                LECTURE_GAP_STEERING_RULES,
                NO_CLOSING_QUESTIONNAIRES,
                CONCEPT_INTRODUCTION_LECTURE_RULE,
                DIAGRAM_INTEGRATION_CROSS_REF,
                DIAGRAM_SELECTION_RULES,
                PINNED_DIAGRAMS_GUIDING_RULES,
            ]
        )
        if topic_already_covered:
            parts.append(
                "IS_TOPIC_ALREADY_COVERED=True: Deep Dive On-Demand only — "
                "no base lecture repeat; honor lecture_coverage_registry / COVERAGE_CONTEXT."
            )
        parts.append(
            "lecture_body MUST include extracted_concepts (3–5 micro-topics) in JSON even "
            "when lecture is long; do not drop schema fields due to length."
        )
    elif mode == InteractionPromptMode.INTRO:
        parts.append(INTRO_RECENCY_TAIL)

    terms_block = format_already_explained_terms_block(memory) if memory else ""
    if terms_block:
        parts.append(terms_block)
    return "\n\n".join(p for p in parts if p.strip())


def compose_system_prompt(
    mode: InteractionPromptMode,
    *,
    context: PromptComposeContext | None = None,
) -> str:
    ctx = context or PromptComposeContext()
    if mode == InteractionPromptMode.DIALOGUE_FEEDBACK:
        sections: list[str] = list(dialogue_base_system_parts())
        sections.extend(_dialogue_module_parts())
    else:
        sections = list(_base_system_prompt_parts())
        if mode == InteractionPromptMode.INTRO:
            sections.extend(_intro_module_parts())
        elif mode == InteractionPromptMode.LECTURE_CHAT:
            sections.extend(_lecture_chat_module_parts())
        elif mode == InteractionPromptMode.LECTURE_DENSE:
            sections.extend(_dense_lecture_module_parts(ctx))
        else:
            raise ValueError(f"Unknown InteractionPromptMode: {mode!r}")

    recency = build_critical_rules_recency_tail(
        mode=mode,
        memory=ctx.memory,
        topic_already_covered=ctx.topic_already_covered,
    )
    ctx.last_recency_len = len(recency)
    sections.append(recency)
    text = "\n\n".join(s for s in sections if (s or "").strip())
    total_len = len(text)
    if mode == InteractionPromptMode.DIALOGUE_FEEDBACK:
        if (
            total_len < DIALOGUE_SYSTEM_PROMPT_MIN_LEN
            or total_len > DIALOGUE_SYSTEM_PROMPT_MAX_LEN
        ):
            logger.warning(
                "tutor_prompt dialogue system length=%s outside [%s, %s] "
                "(recency=%s)",
                total_len,
                DIALOGUE_SYSTEM_PROMPT_MIN_LEN,
                DIALOGUE_SYSTEM_PROMPT_MAX_LEN,
                ctx.last_recency_len,
            )
        else:
            logger.info(
                "tutor_prompt dialogue system length=%s recency=%s",
                total_len,
                ctx.last_recency_len,
            )
    else:
        logger.debug(
            "tutor_prompt system composed mode=%s length=%s recency=%s",
            mode.value,
            total_len,
            ctx.last_recency_len,
        )
    return text


def build_intro_system() -> str:
    return compose_system_prompt(InteractionPromptMode.INTRO)


def build_dialogue_system() -> str:
    return compose_system_prompt(InteractionPromptMode.DIALOGUE_FEEDBACK)


def build_lecture_chat_system() -> str:
    return compose_system_prompt(InteractionPromptMode.LECTURE_CHAT)


def build_dense_system(
    targeted: bool = False,
    topic_already_covered: bool = False,
    *,
    memory: SessionMemory | None = None,
) -> str:
    return compose_system_prompt(
        InteractionPromptMode.LECTURE_DENSE,
        context=PromptComposeContext(
            targeted=targeted,
            topic_already_covered=topic_already_covered,
            memory=memory,
        ),
    )


def __getattr__(name: str):
    """Lazy legacy constants (no import-time prompt baking)."""
    if name == "INTRO_ASSESSMENT_SYSTEM":
        return build_intro_system()
    if name in ("TUTOR_DIALOGUE_SYSTEM", "TUTOR_SYSTEM_PROMPT"):
        return build_dialogue_system()
    if name == "DEEP_DIVE_TUTOR_SYSTEM":
        return build_lecture_chat_system()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
