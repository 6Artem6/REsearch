"""Smart Selection Prompts — быстрые контекстные вопросы через Gemma Cloud."""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from knowledge_engine.llm import complete_structured_async
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.src.processors.question_formation_rules import (
    QUESTION_FORMATION_RULES,
)
from knowledge_engine.ui.run_log import trace

SELECTION_PROMPT_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE}\n"
    """You are a research-engine assistant. Generate exactly 3 short, precise
engineering questions about the fragment the user selected.

CONTEXT:
- Main topic: {topic}
- Paragraph: {paragraph_context}
- Selected fragment: {selected_text}

RULES:
1. Questions MUST be in Russian.
2. Cover three angles:
   - (1) Cause/details: why or how this mechanism works under the hood
   - (2) Alternative/comparison: what else exists or how it differs
   - (3) Practice/consequence: what it leads to or how to avoid it
3. Each question is short (up to 10 words), dense, professional.
4. Return JSON only:
{{
  "questions": [
    "Question 1?",
    "Question 2?",
    "Question 3?"
  ]
}}
"""
    + QUESTION_FORMATION_RULES
)

DEFAULT_SELECTION_QUESTIONS: list[str] = [
    "Что это значит на практике?",
    "Каковы причины этого?",
    "Какая есть альтернатива?",
]

_MAX_SELECTED = 800
_MAX_PARAGRAPH = 2000
_MAX_TOPIC = 400

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.I)


class SelectionQuestionsJson(BaseModel):
    questions: list[str] = Field(default_factory=list)

    @field_validator("questions", mode="before")
    @classmethod
    def _coerce_questions(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        return []


class SelectionPromptsResult(BaseModel):
    questions: list[str] = Field(min_length=3, max_length=3)
    source: Literal["gemma_cloud", "default"] = "default"


def _clip(text: str, limit: int) -> str:
    s = (text or "").strip()
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


def _build_system_prompt(
    selected_text: str,
    paragraph_context: str,
    topic: str,
) -> str:
    return SELECTION_PROMPT_SYSTEM.format(
        topic=_clip(topic, _MAX_TOPIC) or "(не указана)",
        paragraph_context=_clip(paragraph_context, _MAX_PARAGRAPH) or "(нет)",
        selected_text=_clip(selected_text, _MAX_SELECTED),
    )


def _normalize_questions(raw: list[str]) -> list[str]:
    cleaned: list[str] = []
    for q in raw:
        t = (q or "").strip()
        if not t:
            continue
        if not t.endswith("?"):
            t = t.rstrip(".!") + "?"
        cleaned.append(t)
    idx = 0
    while len(cleaned) < 3:
        cleaned.append(
            DEFAULT_SELECTION_QUESTIONS[idx % len(DEFAULT_SELECTION_QUESTIONS)]
        )
        idx += 1
    return cleaned[:3]


def _parse_questions_payload(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    m = _JSON_FENCE_RE.search(raw)
    if m:
        raw = m.group(1).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict) and "questions" in data:
        return SelectionQuestionsJson.model_validate(data).questions
    return []


async def _gemma_chat_questions(system: str, user: str) -> list[str]:
    parsed = await complete_structured_async(
        SelectionQuestionsJson,
        system,
        user,
        label="selection_prompts",
    )
    if parsed is None:
        return []
    return list(parsed.questions or [])


async def suggest_selection_questions(
    selected_text: str,
    paragraph_context: str,
    topic: str,
) -> SelectionPromptsResult:
    """Async Gemma Cloud call with fallback to default questions."""
    selected = (selected_text or "").strip()
    if len(selected) < 2:
        return SelectionPromptsResult(
            questions=list(DEFAULT_SELECTION_QUESTIONS),
            source="default",
        )

    system = _build_system_prompt(selected, paragraph_context, topic)
    user = "Return JSON with three questions about the selected fragment."

    trace("SELECTION_PROMPTS ▶ Gemma Cloud")
    try:
        raw_questions = await _gemma_chat_questions(system, user)
        questions = _normalize_questions(raw_questions)
        trace(f"SELECTION_PROMPTS ✓ {questions[0][:48]}…")
        return SelectionPromptsResult(questions=questions, source="gemma_cloud")
    except Exception as exc:
        trace(f"SELECTION_PROMPTS ✗ fallback | {exc}")
        return SelectionPromptsResult(
            questions=list(DEFAULT_SELECTION_QUESTIONS),
            source="default",
        )
