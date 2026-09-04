"""Schema ceilings vs prompt targets for tutor / lecture prose fields.

Pydantic ``max_length`` uses SCHEMA_* — physical model output bounds plus ~30–50%
headroom for outliers, not unbounded buffers.

System prompts tell the model to stay within PROMPT_* (much smaller).

Token anchors (see ``config.py``):
- ``LECTURE_MAX_OUTPUT_TOKENS=8192`` → ~22k RU chars for dense lecture fields.
- ``GEMINI_INTRO_MAX_OUTPUT_TOKENS=2048`` → ~4.5k RU chars for panel summary.
"""

from __future__ import annotations

# --- Schema ceilings (validation / storage upper bound) ---
# tutor_message: composed chat display (feedback + tech + follow-up)
SCHEMA_TUTOR_MESSAGE_MAX = 6_000
# lecture_body / technical_explanation: LECTURE_MAX_OUTPUT_TOKENS=8192 (~20k RU)
SCHEMA_LECTURE_BODY_MAX = 22_000
SCHEMA_TECHNICAL_EXPLANATION_MAX = 22_000
# follow_up / checkpoint: prompt target 400 chars + ~3× anomaly headroom
SCHEMA_FOLLOW_UP_QUESTION_MAX = 1_500
SCHEMA_CHECKPOINT_PROMPT_MAX = 1_500
SCHEMA_FEEDBACK_ON_ANSWER_MAX = 6_000
# summary: GEMINI_INTRO_MAX_OUTPUT_TOKENS=2048 (~5k RU)
SCHEMA_SUMMARY_MAX = 4_500
SCHEMA_BRIDGE_TO_NEXT_MAX = 1_500
SCHEMA_NEW_GAP_MAX = 1_500

# --- Prompt targets (instruct the model; well below schema ceilings) ---
PROMPT_FOLLOW_UP_MAX_CHARS = 400
PROMPT_CHECKPOINT_MAX_CHARS = 400
PROMPT_INTRO_TUTOR_MESSAGE_MAX_CHARS = 400
PROMPT_LECTURE_BODY_TARGET_MAX_WORDS = 2500

PROMPT_FOLLOW_UP_TARGET_RULE = (
    f"follow_up_question / checkpoint_prompt: ONE technical question, "
    f"≤{PROMPT_FOLLOW_UP_MAX_CHARS} characters (hard schema cap "
    f"{SCHEMA_FOLLOW_UP_QUESTION_MAX} — do not exceed target)."
)

PROMPT_CHECKPOINT_TARGET_RULE = (
    f"checkpoint_prompt: ONE closing self-check question, "
    f"≤{PROMPT_CHECKPOINT_MAX_CHARS} characters; never repeat lecture_body."
)
