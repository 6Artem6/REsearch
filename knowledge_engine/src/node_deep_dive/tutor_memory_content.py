"""Текст реплики тьютора для active_window / fact_manifest (без follow-up вопросов)."""

from __future__ import annotations

from knowledge_engine.src.node_deep_dive.schemas import DeepDiveLLMOutput
from knowledge_engine.src.node_deep_dive.tutor_dialogue import (
    compose_tutor_dialogue_message,
)
from knowledge_engine.src.node_deep_dive.tutor_reply_sanitize import (
    sanitize_evicted_tutor_content_for_manifest,
)


def tutor_content_for_active_window(
    llm_out: DeepDiveLLMOutput | None,
    *,
    fallback_compose_text: str = "",
) -> str:
    """
    Содержимое tutor-реплики для sliding window и fact_manifest extract.

    Использует feedback_on_answer + technical_explanation; не включает follow_up_question.
    """
    if llm_out is not None:
        parts: list[str] = []
        fb = (llm_out.feedback_on_answer or "").strip()
        tech = (llm_out.technical_explanation or "").strip()
        if fb:
            parts.append(fb)
        if tech:
            parts.append(tech)
        combined = "\n\n".join(parts).strip()
        if combined:
            return combined[:12_000]
    fallback = (fallback_compose_text or "").strip()
    if not fallback and llm_out is not None:
        fallback = compose_tutor_dialogue_message(
            feedback_on_answer=llm_out.feedback_on_answer,
            technical_explanation=llm_out.technical_explanation,
        )
    if not fallback:
        return ""
    return sanitize_evicted_tutor_content_for_manifest(fallback)[:12_000]
