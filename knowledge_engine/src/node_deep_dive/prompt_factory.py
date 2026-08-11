"""Prompt Factory: select isolated system prompts from UI [mode:…] prefixes."""

from __future__ import annotations

import re
from typing import Literal

from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.src.node_deep_dive.deep_dive_how_prompt import (
    DEEP_DIVE_HOW_PROMPT,
)
from knowledge_engine.src.node_deep_dive.deep_dive_mech_prompt import (
    DEEP_DIVE_MECH_PROMPT,
)
from knowledge_engine.src.node_deep_dive.gloss_summary_prompt import (
    GLOSS_SUMMARY_PROMPT,
)

TutorFactoryMode = Literal[
    "default",
    "deep_dive_mech",
    "deep_dive_how",
    "gloss",
    "lecture",
    "blitz",
    "socratic",
]

_MODE_PREFIX_RE = re.compile(
    r"^\[mode:(deep_dive_mech|deep_dive_how|gloss|lecture|blitz|socratic)\]\s*",
    re.I,
)

_CHIP_MODE_TO_CHOICE = {
    "deep_dive_mech": "mech",
    "deep_dive_how": "how",
    "gloss": "gloss",
}

_JSON_CONTRACT_TAIL = (
    "=== JSON OUTPUT (DeepDiveTutorContract) ===\n"
    "Strictly output valid JSON matching DeepDiveTutorContract. "
    "No tutor_message field. "
    "User-facing text fields (feedback_on_answer, technical_explanation, "
    "follow_up_question) MUST be in natural Russian.\n"
)


def parse_tutor_mode_prefix(user_message: str) -> tuple[str, TutorFactoryMode]:
    """
    Strip a leading ``[mode:…]`` prefix if present.

    Returns ``(cleaned_user_message, factory_mode)``.
    """
    raw = (user_message or "").strip()
    if not raw:
        return "", "default"
    m = _MODE_PREFIX_RE.match(raw)
    if not m:
        return raw, "default"
    mode = m.group(1).strip().lower()
    body = raw[m.end() :].strip()
    if mode in (
        "deep_dive_mech",
        "deep_dive_how",
        "gloss",
        "lecture",
        "blitz",
        "socratic",
    ):
        return body, mode  # type: ignore[return-value]
    return body or raw, "default"


def factory_mode_to_gloss_choice(mode: TutorFactoryMode | str) -> str:
    """Map factory mode → classify_gloss_fork_choice token (or empty)."""
    return _CHIP_MODE_TO_CHOICE.get((mode or "").strip().lower(), "")


def is_factory_control_mode(mode: TutorFactoryMode | str) -> bool:
    """Modes that skip Evaluator and use an isolated system prompt."""
    return (mode or "").strip().lower() in (
        "deep_dive_mech",
        "deep_dive_how",
        "gloss",
    )


def select_system_prompt_and_mode(
    user_message: str,
    *,
    default_system_prompt: str = "",
) -> tuple[str, TutorFactoryMode, str]:
    """
    Resolve system prompt override from ``[mode:…]`` prefix.

    Returns ``(system_prompt, mode, cleaned_user_message)``.
    For ``default`` / lecture / blitz / socratic without dedicated isolated prompts,
    returns ``default_system_prompt`` unchanged (caller still owns lecture routing).
    """
    cleaned, mode = parse_tutor_mode_prefix(user_message)
    if mode == "deep_dive_mech":
        system = "\n\n".join(
            [
                DEEP_DIVE_MECH_PROMPT.strip(),
                RUSSIAN_OUTPUT_RULE.strip(),
                _JSON_CONTRACT_TAIL.strip(),
            ]
        )
        return system, mode, cleaned
    if mode == "deep_dive_how":
        system = "\n\n".join(
            [
                DEEP_DIVE_HOW_PROMPT.strip(),
                RUSSIAN_OUTPUT_RULE.strip(),
                _JSON_CONTRACT_TAIL.strip(),
            ]
        )
        return system, mode, cleaned
    if mode == "gloss":
        system = "\n\n".join(
            [
                GLOSS_SUMMARY_PROMPT.strip(),
                RUSSIAN_OUTPUT_RULE.strip(),
                _JSON_CONTRACT_TAIL.strip(),
            ]
        )
        return system, mode, cleaned
    return (default_system_prompt or ""), mode, cleaned


def select_isolated_prompt_for_mode(mode: TutorFactoryMode | str) -> str | None:
    """Return isolated system prompt body for a factory mode, or None."""
    m = (mode or "").strip().lower()
    if m == "deep_dive_mech":
        return DEEP_DIVE_MECH_PROMPT
    if m == "deep_dive_how":
        return DEEP_DIVE_HOW_PROMPT
    if m == "gloss":
        return GLOSS_SUMMARY_PROMPT
    return None
