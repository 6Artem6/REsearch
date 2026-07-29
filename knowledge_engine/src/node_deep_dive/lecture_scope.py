"""Роутинг full_node_lecture vs targeted_lecture и фокус для диалога."""

from __future__ import annotations

import re
from typing import Literal

from knowledge_engine.src.node_deep_dive.memory_schemas import SessionMemory

LectureScope = Literal["full_node_lecture", "targeted_lecture"]

_GENERIC_LECTURE_STUBS = (
    "дай плотный материал по теме",
    "дай плотный материал",
    "дай лекцию",
    "плотный материал по теме",
    "дай плотный материал по теме.",
    "плотный материал",
)

_MODE_PREFIX_RE = re.compile(r"^\[mode:\w+\]\s*", re.I)


def _norm_stub(text: str) -> str:
    return (text or "").lower().strip().rstrip(".").strip()


def is_generic_lecture_stub(text: str) -> bool:
    t = _norm_stub(text)
    if not t:
        return True
    return t in {_norm_stub(s) for s in _GENERIC_LECTURE_STUBS}


def chat_subtopic_active(memory: SessionMemory) -> bool:
    if memory.learning_phase in (
        "checkpoint",
        "pathway_decision",
        "socratic_focus",
        "dense_material",
    ):
        return True
    user_turns = 0
    for m in memory.active_window:
        if (m.get("role") or "") != "user":
            continue
        c = (m.get("content") or "").strip()
        if len(c) < 8:
            continue
        if c.startswith("[mode:"):
            continue
        if is_generic_lecture_stub(c):
            continue
        user_turns += 1
    return user_turns >= 1


def last_substantive_user_message(memory: SessionMemory) -> str:
    for m in reversed(memory.active_window):
        if (m.get("role") or "") != "user":
            continue
        c = _MODE_PREFIX_RE.sub("", (m.get("content") or "").strip())
        if len(c) < 8 or is_generic_lecture_stub(c):
            continue
        return c[:800]
    return ""


def last_tutor_thread_prompt(memory: SessionMemory) -> str:
    for m in reversed(memory.active_window):
        if (m.get("role") or "") != "tutor":
            continue
        c = (m.get("content") or "").strip()
        if len(c) < 20:
            continue
        return c[:800]
    return ""


def resolve_lecture_scope(
    user_message: str,
    memory: SessionMemory,
    lecture_button_pressed: bool = False,
) -> tuple[LectureScope, str]:
    """
    targeted_lecture: конкретный вопрос пользователя или активная ветка диалога.
    full_node_lecture: обзор по всей ноде.
    """
    focus = _MODE_PREFIX_RE.sub("", (user_message or "").strip())
    if focus and not is_generic_lecture_stub(focus):
        return "targeted_lecture", focus
    if lecture_button_pressed or chat_subtopic_active(memory):
        sub = last_substantive_user_message(memory)
        if not sub:
            sub = last_tutor_thread_prompt(memory)
        if sub:
            return "targeted_lecture", sub
    return "full_node_lecture", focus


def is_topic_question(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 10:
        return False
    if "?" in t or "？" in t:
        return True
    low = t.lower()
    starters = (
        "почему ",
        "зачем ",
        "как ",
        "а почему",
        "а как ",
        "что если",
        "объясни ",
        "расскажи ",
        "why ",
        "how ",
        "what if",
    )
    return any(low.startswith(s) or f" {s.strip()}" in low for s in starters)


def dialogue_focus_text(user_message: str, memory: SessionMemory) -> str:
    """Фокус для mode:dialogue_feedback / mini-lecture без сброса на всю ноду."""
    msg = _MODE_PREFIX_RE.sub("", (user_message or "").strip())
    if msg and not is_generic_lecture_stub(msg):
        if is_topic_question(msg) or len(msg) >= 24:
            return msg[:800]
    sub = last_substantive_user_message(memory)
    if sub and chat_subtopic_active(memory):
        return sub[:800]
    return ""
