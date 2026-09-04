"""Isolated system prompt: Next Module — close out the node with a short summary."""

from __future__ import annotations

NEXT_MODULE_PROMPT = (
    "The learner requested to move on to the next module.\n\n"
    "YOUR TASK:\n"
    "1. Write a SHORT summary (2-3 sentences) IN technical_explanation of what "
    "was covered in this node.\n"
    "2. Do NOT ask a new technical/evaluative question.\n"
    "3. Do NOT propose curriculum node titles yourself — the host/UI client "
    "owns the node-switch controller and picks the next node.\n"
    '4. Leave the top-level summary field empty ("") and references empty '
    "([]) — the wrap-up text belongs in technical_explanation only.\n\n"
    "Chip processing is already done by the Python host before this turn. "
    "Host owns ready_for_transition / suggested_next_step / quick_replies. "
    "question_sub_concept_id=null. No quiz this turn.\n"
)
"""
RU (пояснение): изолированный system prompt для [mode:next_module] — тот же
intent, что у существующего чипа "Идем дальше" (intent_definitions.py, rule
"next"), теперь доступен и как mid-dialogue [mode:next_module] тег/фраза.
Пункт 4 — только technical_explanation/follow_up_question реально стримятся
в UI (TUTOR_EXPLAIN_STREAM_FIELDS); верхнеуровневые summary/references
DeepDiveExplainContract генерируются раньше них и не нужны для этого хода.
"""
