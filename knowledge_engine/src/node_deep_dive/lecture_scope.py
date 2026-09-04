"""Lecture scope routing: full_node vs targeted, anchored to active sub-concept."""

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


def strip_mode_prefix(text: str) -> str:
    return _MODE_PREFIX_RE.sub("", (text or "").strip()).strip()


def is_lecture_request_message(text: str) -> bool:
    """
    True for UI [mode:lecture] button / explicit dense-material ask.

    These turns must NOT run the sub-concept gap evaluator (only real answers do).
    Long free-text answers are never treated as lecture requests via fuzzy stems.
    """
    from knowledge_engine.src.node_deep_dive.control_intent import (
        is_short_lecture_request,
    )

    raw = (text or "").strip()
    if not raw:
        return False
    if is_short_lecture_request(raw):
        return True
    # Exact generic stubs (also covered by control_intent, kept for clarity).
    body = strip_mode_prefix(raw)
    return is_generic_lecture_stub(body)


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


def _active_subconcept_focus(memory: SessionMemory) -> tuple[str, str]:
    """Return (id, label) for active generation focus; never from chat_history."""
    from knowledge_engine.src.node_deep_dive.subconcept_invariants import (
        resolve_active_subconcept_id,
        resolve_active_subconcept_label,
    )

    active_id = resolve_active_subconcept_id(memory)
    if not active_id:
        return "", ""
    label = resolve_active_subconcept_label(memory, active_id)
    return active_id, label


def resolve_lecture_scope(
    user_message: str,
    memory: SessionMemory,
    lecture_button_pressed: bool = False,
) -> tuple[LectureScope, str]:
    """
    Resolve lecture scope + focus text.

    Invariant: when a concept-map active_subconcept_id exists, lecture topic is
    taken ONLY from that id — never from chat_history / prior tutor turns.
    """
    focus = _MODE_PREFIX_RE.sub("", (user_message or "").strip())
    active_id, active_label = _active_subconcept_focus(memory)

    if active_id:
        # Hard anchor — chat_history must not choose the lecture topic
        base = f"{active_label} [subconcept_id={active_id}]"
        if focus and not is_generic_lecture_stub(focus):
            return (
                "targeted_lecture",
                f"{base}\nuser_angle: {focus[:500]}",
            )
        return "targeted_lecture", base

    # No concept-map focus yet — legacy fallbacks (intro / empty map)
    if focus and not is_generic_lecture_stub(focus):
        return "targeted_lecture", focus
    if lecture_button_pressed and is_generic_lecture_stub(focus or user_message):
        return "full_node_lecture", focus
    if lecture_button_pressed or chat_subtopic_active(memory):
        sub = last_substantive_user_message(memory)
        if sub:
            return "targeted_lecture", sub
        if not is_generic_lecture_stub(focus or user_message):
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
    """Focus for dialogue_feedback; prefer active sub-concept over chat_history."""
    msg = _MODE_PREFIX_RE.sub("", (user_message or "").strip())
    if msg and not is_generic_lecture_stub(msg):
        if is_topic_question(msg) or len(msg) >= 24:
            return msg[:800]
    active_id, active_label = _active_subconcept_focus(memory)
    if active_id:
        return f"{active_label} [subconcept_id={active_id}]"
    sub = last_substantive_user_message(memory)
    if sub and chat_subtopic_active(memory):
        return sub[:800]
    return ""
