"""Isolated system prompt: Express Blitz — rapid-fire single-question drill."""

from __future__ import annotations

BLITZ_MODE_PROMPT = (
    "You are running an Express Blitz quiz round on the current topic.\n\n"
    "YOUR TASK:\n"
    "1. Ask EXACTLY ONE short, precise question about the active sub-concept.\n"
    "2. Keep the whole turn to 1-2 sentences — no preamble, no restating theory, "
    "no long setup.\n"
    "3. Put the question in follow_up_question; leave technical_explanation "
    "empty or a single short sentence at most.\n"
    '4. Leave summary empty ("") and references empty ([]) — a blitz '
    "question is not a Materials-panel update. Do not spend effort composing "
    "them.\n\n"
    "Chip processing is already done by the Python host before this turn. "
    "Host owns ready_for_transition / suggested_next_step / quick_replies. "
    "Do NOT invent next curriculum node titles — the UI client picks the next node.\n"
)
"""
RU (пояснение): изолированный system prompt для [mode:blitz] — быстрый
блиц-опрос, ровно один короткий вопрос без вступлений. Пункт 4
(summary/references пустые) — только technical_explanation/
follow_up_question реально стримятся в UI (TUTOR_EXPLAIN_STREAM_FIELDS);
summary/references генерируются раньше них в DeepDiveExplainContract и
незачем тратить на них токены генерации в этом режиме.
"""
