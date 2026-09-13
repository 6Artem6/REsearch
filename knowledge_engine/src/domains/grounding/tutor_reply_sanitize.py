"""Санитизация реплик тьютора: transition и fact_manifest."""

from __future__ import annotations

import re

_TRANSITION_CHOICE_MARKERS = (
    "перейд",
    "следующ",
    "нода",
    "лекци",
    "углуб",
    "хочешь",
    "хотите",
    "advanced",
    "deep mode",
    "deep dive",
    "deep_dive",
    "next_node",
    "или разбер",
    "если хочешь",
)

_UNANSWERED_NOTE = "[Unanswered tutor question removed]"


def is_transition_choice_question(fragment: str) -> bool:
    """True when a «?» fragment is a next-step CTA, not a technical quiz."""
    return _is_transition_choice_fragment(fragment)


def _is_transition_choice_fragment(fragment: str) -> bool:
    low = (fragment or "").strip().lower()
    if "?" not in low:
        return False
    return any(m in low for m in _TRANSITION_CHOICE_MARKERS)


def _split_sentences(text: str) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    parts = re.split(r"(?<=[.!?…])\s+", t)
    return [p for p in parts if p.strip()]


def strip_trailing_question_sentences(
    text: str,
    *,
    keep_transition_choice: bool,
) -> tuple[str, bool]:
    """
    Удаляет с конца предложения с «?».
    keep_transition_choice=True — оставляет финальный выбор (next node / углубление).
    """
    body = (text or "").strip()
    if not body or "?" not in body:
        return body, False

    sentences = _split_sentences(body)
    if not sentences:
        if "?" in body:
            return body.rsplit("?", 1)[0].strip() + "?", False
        return body, False

    removed = False
    while sentences:
        last = sentences[-1].strip()
        if "?" not in last:
            break
        if keep_transition_choice and is_transition_choice_question(last):
            break
        sentences.pop()
        removed = True

    if not sentences:
        return "", removed
    out = " ".join(sentences).strip()
    if not out and removed:
        paras = body.split("\n\n")
        while paras:
            tail = paras[-1].strip()
            if "?" in tail and not (
                keep_transition_choice and is_transition_choice_question(tail)
            ):
                paras.pop()
                removed = True
                continue
            break
        out = "\n\n".join(paras).strip()
    return out, removed


def sanitize_tutor_message_for_transition(text: str) -> str:
    """Убрать тех. хвост с «?» при ready_for_transition; сохранить выбор перехода."""
    cleaned, _ = strip_trailing_question_sentences(
        text,
        keep_transition_choice=True,
    )
    return (cleaned or (text or "").strip()).strip()


def sanitize_evicted_tutor_content_for_manifest(text: str) -> str:
    """Убрать неотвеченные вопросы тьютора перед fact_manifest extract."""
    cleaned, removed = strip_trailing_question_sentences(
        text,
        keep_transition_choice=False,
    )
    if not removed:
        return (text or "").strip()
    base = (cleaned or "").strip()
    if base:
        return f"{base}\n\n{_UNANSWERED_NOTE}"
    return _UNANSWERED_NOTE
