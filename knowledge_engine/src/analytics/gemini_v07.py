"""Gemini calls for v0.7 analytics (pinned Lite / Flash models)."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Type, TypeVar

from pydantic import BaseModel

from knowledge_engine.config import (
    GEMINI_FLASH_MODEL,
    GEMINI_LITE_MODEL,
    GEMINI_RPM_PAUSE_SEC,
)
from knowledge_engine.services.gemini_stateless import (
    GeminiUnavailableError,
    gemini_lite_model_chain,
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
    stream_callback: Callable[[str], None] | None = None,
    stream_text_field: str | None = None,
) -> T:
    if not is_gemini_available():
        raise GeminiUnavailableError("Gemini недоступен для v0.7 analytics")
    models = gemini_lite_model_chain(GEMINI_LITE_MODEL)
    return run_gemini_structured_with_chain(
        GEMINI_LITE_MODEL,
        system_instruction,
        user_payload,
        global_anchor,
        response_schema,
        f"v07 lite / {label}",
        rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
        stream_callback=stream_callback,
        stream_text_field=stream_text_field,
        models=models,
    )


def run_gemini_lite_text(
    system_instruction: str,
    user_payload: str,
    global_anchor: str,
    label: str,
    stream_callback: Callable[[str], None] | None = None,
) -> str:
    """Deprecated: do not use for new product features.

    Prefer ``run_gemini_lite_structured`` with an explicit Pydantic
    ``response_schema``.
    """
    warnings.warn(
        "run_gemini_lite_text is deprecated; use run_gemini_lite_structured "
        "with a Pydantic response_schema",
        DeprecationWarning,
        stacklevel=2,
    )
    if not is_gemini_available():
        raise GeminiUnavailableError("Gemini недоступен для v0.7 analytics")
    models = gemini_lite_model_chain(GEMINI_LITE_MODEL)
    return run_gemini_text_with_chain(
        GEMINI_LITE_MODEL,
        system_instruction,
        user_payload,
        global_anchor,
        label,
        rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
        stream_callback=stream_callback,
        models=models,
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
    models = gemini_lite_model_chain(GEMINI_FLASH_MODEL)
    return run_gemini_structured_with_chain(
        GEMINI_FLASH_MODEL,
        system_instruction,
        user_payload,
        global_anchor,
        response_schema,
        f"v07 flash / {label}",
        rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
        models=models,
    )


def run_gemini_flash_text(
    system_instruction: str,
    user_payload: str,
    global_anchor: str,
    label: str,
) -> str:
    """Deprecated: do not use for new product features.

    Prefer ``run_gemini_flash_structured`` with an explicit Pydantic
    ``response_schema``.
    """
    warnings.warn(
        "run_gemini_flash_text is deprecated; use run_gemini_flash_structured "
        "with a Pydantic response_schema",
        DeprecationWarning,
        stacklevel=2,
    )
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
