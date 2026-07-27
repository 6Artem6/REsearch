"""Gemini calls for v0.7 analytics (pinned Lite / Flash models)."""

from __future__ import annotations

from typing import Type, TypeVar

from pydantic import BaseModel

from knowledge_engine.config import (
    GEMINI_FLASH_MODEL,
    GEMINI_LITE_MODEL,
    GEMINI_RPM_PAUSE_SEC,
)
from knowledge_engine.services.gemini_stateless import (
    GeminiUnavailableError,
    is_gemini_available,
    run_gemini_structured_with_chain,
    run_gemini_text_with_chain,
)

T = TypeVar("T", bound=BaseModel)


def run_gemini_lite_structured(
    system_instruction: str,
    user_payload: str,
    global_anchor: str,
    response_schema: Type[T],
    label: str,
) -> T:
    if not is_gemini_available():
        raise GeminiUnavailableError("Gemini недоступен для v0.7 analytics")
    return run_gemini_structured_with_chain(
        GEMINI_LITE_MODEL,
        system_instruction,
        user_payload,
        global_anchor,
        response_schema,
        f"v07 lite / {label}",
        rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
    )


def run_gemini_lite_text(
    system_instruction: str,
    user_payload: str,
    global_anchor: str,
    label: str,
) -> str:
    if not is_gemini_available():
        raise GeminiUnavailableError("Gemini недоступен для v0.7 analytics")
    return run_gemini_text_with_chain(
        GEMINI_LITE_MODEL,
        system_instruction,
        user_payload,
        global_anchor,
        label,
        rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
    )


def run_gemini_flash_structured(
    system_instruction: str,
    user_payload: str,
    global_anchor: str,
    response_schema: Type[T],
    label: str,
) -> T:
    if not is_gemini_available():
        raise GeminiUnavailableError("Gemini недоступен для v0.7 analytics")
    return run_gemini_structured_with_chain(
        GEMINI_FLASH_MODEL,
        system_instruction,
        user_payload,
        global_anchor,
        response_schema,
        f"v07 flash / {label}",
        rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
    )


def run_gemini_flash_text(
    system_instruction: str,
    user_payload: str,
    global_anchor: str,
    label: str,
) -> str:
    """Текстовый ответ Flash (REPL / follow-up) с retry и model chain."""
    if not is_gemini_available():
        raise GeminiUnavailableError("Gemini недоступен для v0.7 analytics")
    return run_gemini_text_with_chain(
        GEMINI_FLASH_MODEL,
        system_instruction,
        user_payload,
        global_anchor,
        label,
        rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
    )
