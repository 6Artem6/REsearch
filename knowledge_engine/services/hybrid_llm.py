"""Гибрид: Gemini API (тяжёлые задачи) + Gemma Cloud (Re-Act / fallback)."""

from __future__ import annotations

import time
from typing import TypeVar

from pydantic import BaseModel

from knowledge_engine.config import (
    GEMINI_RPM_PAUSE_SEC,
    LOCAL_L2_MODEL,
    MAIN_MODEL,
    OLLAMA_STRUCTURE_NUM_PREDICT,
    REACT_EVAL_MODEL,
)
from knowledge_engine.services.gemini_stateless import (
    is_gemini_available,
    run_stateless_gemini,
)
from knowledge_engine.services.local_llm_stateless import (
    run_local_structured,
    run_local_text,
)
from knowledge_engine.ui.run_log import trace

T = TypeVar("T", bound=BaseModel)


def run_structured_hybrid(
    system_instruction: str,
    user_payload: str,
    global_anchor: str,
    response_schema: type[T],
    label: str,
    *,
    prefer_gemini: bool = True,
    rpm_pause_sec: float = 0.0,
    local_model: str | None = None,
    local_num_predict: int | None = None,
) -> T:
    """
    Stateless structured JSON: Gemini when available, else / on error — Gemma Cloud.
    rpm_pause_sec — пауза перед облачным вызовом (лимит RPM).
    """
    model = local_model or MAIN_MODEL
    if prefer_gemini and is_gemini_available():
        if rpm_pause_sec > 0:
            time.sleep(rpm_pause_sec)
        try:
            return run_stateless_gemini(
                system_instruction,
                user_payload,
                global_anchor,
                response_schema=response_schema,
                label=label,
            )
        except Exception as exc:
            trace(f"HYBRID fallback local structured | {label} | {exc}")
    return run_local_structured(
        model,
        response_schema,
        system_instruction,
        user_payload,
        global_anchor,
        label,
        num_predict=local_num_predict,
    )


def run_text_hybrid(
    system_instruction: str,
    user_payload: str,
    global_anchor: str,
    label: str,
    *,
    prefer_gemini: bool = True,
    rpm_pause_sec: float = 0.0,
    local_model: str | None = None,
) -> str:
    if prefer_gemini and is_gemini_available():
        if rpm_pause_sec > 0:
            time.sleep(rpm_pause_sec)
        try:
            out = run_stateless_gemini(
                system_instruction,
                user_payload,
                global_anchor,
                response_schema=None,
                label=label,
            )
            if isinstance(out, str):
                return out
        except Exception as exc:
            trace(f"HYBRID fallback local text | {label} | {exc}")
    model = local_model or MAIN_MODEL
    return run_local_text(
        model,
        system_instruction,
        user_payload,
        global_anchor,
        label,
    )


def run_react_evaluation(
    system_instruction: str,
    user_payload: str,
    global_anchor: str,
    response_schema: type[T],
    label: str,
) -> T:
    """Re-Act evaluator: Gemma Cloud (structured)."""
    return run_local_structured(
        REACT_EVAL_MODEL,
        response_schema,
        system_instruction,
        user_payload,
        global_anchor,
        label,
        temperature=0.05,
        num_predict=1024,
    )


def run_l2_extraction_hybrid(
    system_instruction: str,
    user_payload: str,
    global_anchor: str,
    response_schema: type[T],
    label: str,
) -> T:
    return run_structured_hybrid(
        system_instruction,
        user_payload,
        global_anchor,
        response_schema,
        label,
        prefer_gemini=True,
        rpm_pause_sec=GEMINI_RPM_PAUSE_SEC,
        local_model=LOCAL_L2_MODEL,
        local_num_predict=OLLAMA_STRUCTURE_NUM_PREDICT,
    )


def run_matrix_hybrid(
    system_instruction: str,
    user_payload: str,
    global_anchor: str,
    response_schema: type[T],
    label: str,
) -> T:
    return run_structured_hybrid(
        system_instruction,
        user_payload,
        global_anchor,
        response_schema,
        label,
        prefer_gemini=True,
        rpm_pause_sec=0,
        local_model=MAIN_MODEL,
        local_num_predict=OLLAMA_STRUCTURE_NUM_PREDICT,
    )
