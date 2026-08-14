"""Склейка семантических полей реплики тьютора (dialogue)."""

from __future__ import annotations

import json
from typing import Any

from knowledge_engine.src.node_deep_dive.schemas import DeepDiveLLMOutput


def compose_tutor_dialogue_message(
    *,
    feedback_on_answer: str = "",
    technical_explanation: str = "",
    follow_up_question: str = "",
) -> str:
    parts: list[str] = []
    for block in (feedback_on_answer, technical_explanation, follow_up_question):
        t = (block or "").strip()
        if t:
            parts.append(t)
    return "\n\n".join(parts)


def compose_tutor_dialogue_from_output(llm_out: DeepDiveLLMOutput | None) -> str:
    if llm_out is None:
        return ""
    return compose_tutor_dialogue_message(
        feedback_on_answer=llm_out.feedback_on_answer,
        technical_explanation=llm_out.technical_explanation,
        follow_up_question=llm_out.follow_up_question,
    )


def coerce_deep_dive_llm_output(raw: Any) -> DeepDiveLLMOutput | None:
    if raw is None:
        return None
    if isinstance(raw, DeepDiveLLMOutput):
        return raw
    if isinstance(raw, dict):
        try:
            return DeepDiveLLMOutput.model_validate(raw)
        except Exception:
            return None
    return None


def resolve_tutor_display_message(
    llm_out: DeepDiveLLMOutput | Any | None,
    fallback: str = "",
) -> str:
    """Полный текст для UI/history: всегда из semantic fields, не из active_window."""
    from knowledge_engine.web.llm_text_repair import repair_llm_display_text

    llm = coerce_deep_dive_llm_output(llm_out)
    composed = compose_tutor_dialogue_from_output(llm).strip()
    if composed:
        return repair_llm_display_text(composed)
    return repair_llm_display_text((fallback or "").strip())


def deep_dive_llm_output_from_chat_text(
    text: str,
    *,
    node_status: str = "in_progress",
    **extra,
) -> DeepDiveLLMOutput:
    """Legacy/dense/coverage: один текст чата → semantic fields."""
    raw = (text or "").strip()
    technical = raw
    follow_up = ""
    if "**Самопроверка:**" in raw:
        body, _, tail = raw.partition("**Самопроверка:**")
        technical = body.strip()
        follow_up = tail.strip()
    data = {
        "node_status": node_status,
        "feedback_on_answer": "",
        "technical_explanation": technical,
        "follow_up_question": follow_up,
        **extra,
    }
    return DeepDiveLLMOutput.model_validate(data)


def _parse_tutor_model_json(raw: str) -> DeepDiveLLMOutput | None:
    text = (raw or "").strip()
    if not text or "{" not in text:
        return None
    from knowledge_engine.services.gemini_stateless import extract_clean_json

    try:
        data = json.loads(extract_clean_json(text))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if "follow_up_question" in data or "feedback_on_answer" in data:
        try:
            return DeepDiveLLMOutput.model_validate(
                {
                    "node_status": data.get("node_status") or "in_progress",
                    "summary": data.get("summary") or "",
                    "feedback_on_answer": data.get("feedback_on_answer") or "",
                    "technical_explanation": data.get("technical_explanation") or "",
                    "follow_up_question": data.get("follow_up_question") or "",
                    "question_sub_concept_id": data.get("question_sub_concept_id"),
                    "introduced_terms": data.get("introduced_terms") or [],
                    "verified_sub_concept_ids": data.get("verified_sub_concept_ids")
                    or [],
                    "ready_for_transition": bool(data.get("ready_for_transition")),
                    "suggested_next_step": data.get("suggested_next_step"),
                    "quick_replies": list(data.get("quick_replies") or []),
                }
            )
        except Exception:
            return None
    tutor_msg = (data.get("tutor_message") or "").strip()
    if tutor_msg:
        return deep_dive_llm_output_from_chat_text(
            tutor_msg,
            node_status=str(data.get("node_status") or "in_progress"),
        )
    return None


def recover_tutor_display_from_chat_sessions(memory: Any) -> tuple[str, str]:
    """
    Восстановить полный UI-текст из chat_sessions (сырой JSON tutor ответа).
    Используется когда history/active_window урезаны без follow_up_question.
    """
    if memory is None:
        return "", ""
    sessions = getattr(memory, "chat_sessions", None) or {}
    if not isinstance(sessions, dict):
        return "", ""
    labels = [
        "node_deep_dive/tutor",
        "node_deep_dive/chat",
        "node_deep_dive/verify",
    ]
    for label in labels:
        blob = sessions.get(label)
        if not isinstance(blob, dict):
            continue
        turns = blob.get("api_turns") or []
        for turn in reversed(turns):
            role = str(turn.get("role") or "").strip().lower()
            if role not in ("model", "assistant"):
                continue
            llm_out = _parse_tutor_model_json(str(turn.get("content") or ""))
            if llm_out is None:
                continue
            display = resolve_tutor_display_message(llm_out, "")
            fu = (llm_out.follow_up_question or "").strip()
            if display:
                return display, fu
    return "", ""
