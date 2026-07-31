"""Smart Selection Prompts — быстрые контекстные вопросы через локальную Ollama."""

from __future__ import annotations

import json
import re
from typing import Literal

import httpx
from pydantic import BaseModel, Field, field_validator

from knowledge_engine.config import (
    OLLAMA_BASE_URL,
    OLLAMA_ROUTER_KEEP_ALIVE,
    SELECTION_PROMPTS_NUM_PREDICT,
    SELECTION_PROMPTS_OLLAMA_MODEL,
    SELECTION_PROMPTS_TIMEOUT_SEC,
    OLLAMA_ROUTER_NUM_CTX,
)
from knowledge_engine.ui.run_log import trace
from knowledge_engine.src.processors.question_formation_rules import (
    QUESTION_FORMATION_RULES,
)

SELECTION_PROMPT_SYSTEM = """Ты — ассистент исследовательского движка. Твоя задача — сгенерировать 3 коротких, точных и глубоких инженерных вопроса к выделенному пользователем фрагменту текста.

КОНТЕКСТ:
- Главная тема: {topic}
- Абзац: {paragraph_context}
- Выделенный фрагмент: {selected_text}

ПРАВИЛА:
1. Сгенерируй ровно 3 вопроса строго на русском языке.
2. Вопросы должны быть разного характера:
   - Вопрос 1 (Причина/Детали): Почему или как именно работает этот механизм под капотом?
   - Вопрос 2 (Альтернатива/Сравнение): Какая есть альтернатива или в чем отличие от смежного решения?
   - Вопрос 3 (Практика/Следствие): К чему это приводит на практике или как этого избежать?
3. Вопросы должны быть короткими (до 10 слов), емкими и звучащими профессионально.
4. Верни результат строго в формате JSON:
{{
  "questions": [
    "Текст вопроса 1?",
    "Текст вопроса 2?",
    "Текст вопроса 3?"
  ]
}}
""" + QUESTION_FORMATION_RULES

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
    source: Literal["ollama", "default"] = "default"


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


async def _ollama_chat_questions(system: str, user: str) -> list[str]:
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    payload = {
        "model": SELECTION_PROMPTS_OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": SelectionQuestionsJson.model_json_schema(),
        "options": {
            "temperature": 0.25,
            "num_predict": SELECTION_PROMPTS_NUM_PREDICT,
            "num_ctx": OLLAMA_ROUTER_NUM_CTX,
        },
        "keep_alive": OLLAMA_ROUTER_KEEP_ALIVE,
    }
    timeout = httpx.Timeout(SELECTION_PROMPTS_TIMEOUT_SEC)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
    message = data.get("message") or {}
    content = message.get("content") if isinstance(message, dict) else ""
    if not isinstance(content, str):
        content = str(content or "")
    parsed = _parse_questions_payload(content)
    if len(parsed) >= 3:
        return parsed
    try:
        model = SelectionQuestionsJson.model_validate_json(content)
        return model.questions
    except Exception:
        return parsed


async def suggest_selection_questions(
    selected_text: str,
    paragraph_context: str,
    topic: str,
) -> SelectionPromptsResult:
    """Асинхронный вызов Ollama с жёстким таймаутом; fallback на дефолтные вопросы."""
    selected = (selected_text or "").strip()
    if len(selected) < 2:
        return SelectionPromptsResult(
            questions=list(DEFAULT_SELECTION_QUESTIONS),
            source="default",
        )

    system = _build_system_prompt(selected, paragraph_context, topic)
    user = "Сгенерируй JSON с тремя вопросами к выделенному фрагменту."

    trace(
        f"SELECTION_PROMPTS ▶ Ollama {SELECTION_PROMPTS_OLLAMA_MODEL} "
        f"timeout={SELECTION_PROMPTS_TIMEOUT_SEC}s"
    )
    try:
        raw_questions = await _ollama_chat_questions(system, user)
        questions = _normalize_questions(raw_questions)
        trace(f"SELECTION_PROMPTS ✓ {questions[0][:48]}…")
        return SelectionPromptsResult(questions=questions, source="ollama")
    except Exception as exc:
        trace(f"SELECTION_PROMPTS ✗ fallback | {exc}")
        return SelectionPromptsResult(
            questions=list(DEFAULT_SELECTION_QUESTIONS),
            source="default",
        )
