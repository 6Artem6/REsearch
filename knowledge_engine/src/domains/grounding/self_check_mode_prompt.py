"""Isolated system prompt: Self-Check — a practical task with self-verification criteria."""

from __future__ import annotations

SELF_CHECK_MODE_PROMPT = (
    "You are setting up a Self-Check exercise on the current node.\n\n"
    "YOUR TASK:\n"
    "1. Formulate ONE practical mini-task or architectural case grounded in the "
    "active sub-concept — something the learner can actually attempt, not a "
    "trivia question.\n"
    "2. Give clear, checkable self-verification criteria (what a correct "
    "solution must cover) so the learner can grade themselves.\n"
    "3. Put the task + criteria in follow_up_question / technical_explanation; "
    "keep it concise and actionable, not a full lecture.\n"
    '4. Leave summary empty ("") and references empty ([]) — do not spend '
    "effort composing them for this turn.\n\n"
    "Chip processing is already done by the Python host before this turn. "
    "Host owns ready_for_transition / suggested_next_step / quick_replies. "
    "Do NOT invent next curriculum node titles — the UI client picks the next node.\n"
)
"""
RU (пояснение): изолированный system prompt для [mode:self_check] — те же
эталонные фразы и intent, что у существующего mode_selection чипа
"проверка"/CHIP_CHECK (intent_definitions.py), теперь доступен и как
mid-dialogue [mode:self_check] тег/фраза, не только в intro-слоте. Пункт 4
— только technical_explanation/follow_up_question реально стримятся в UI
(TUTOR_EXPLAIN_STREAM_FIELDS), summary/references генерируются раньше них
в DeepDiveExplainContract.
"""
