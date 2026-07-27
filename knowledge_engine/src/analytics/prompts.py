"""Gemini architect prompts — personal context + scholarly papers."""

from __future__ import annotations

from typing import Any, Dict, List

from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.src.guardrails.personal_context import format_personal_context
from knowledge_engine.src.retrieval.semantic_scholar import (
    ScholarPaper,
    format_papers_block,
)
from knowledge_engine.src.state import PersonalContext


def build_architect_system_instruction(
    personal_context: PersonalContext | Dict[str, Any] | None,
    formatted_papers: str,
    user_query: str,
) -> str:
    if isinstance(personal_context, dict):
        ctx = PersonalContext.model_validate(personal_context)
    elif isinstance(personal_context, PersonalContext):
        ctx = personal_context
    else:
        ctx = PersonalContext(summary="(контекст не задан)")

    personal_block = format_personal_context(ctx)
    papers_block = formatted_papers or "(статьи не найдены)"

    return (
        f"{RUSSIAN_OUTPUT_RULE}\n"
        "Ты — Главный Архитектор Систем ИИ.\n\n"
        "### ЛОКАЛЬНЫЙ КОНТЕКСТ ПРОЕКТА И ОГРАНИЧЕНИЯ (от Local Agent):\n"
        f"{personal_block}\n\n"
        "### НАЙДЕННЫЕ НАУЧНЫЕ СТАТЬИ И АБСТРАКТЫ (Semantic Scholar / arXiv):\n"
        f"{papers_block}\n\n"
        "### ВОПРОС ПОЛЬЗОВАТЕЛЯ:\n"
        f"{user_query.strip()}\n\n"
        "ЗАДАЧА:\n"
        "Проведи глубокий сравнительный анализ на основе найденных научных статей. "
        "Свяжи теоретические выводы из статей с локальным контекстом проекта (если применимо). "
        "НЕ ПРИТЯГИВАЙ за уши термины из непрофильных областей. "
        "Если статьи не содержат ответа на специфичный деталь реализации, прямо укажи это."
    )


def formatted_papers_from_state(papers: List[Dict[str, Any]] | None) -> str:
    if not papers:
        return ""
    models = [ScholarPaper.model_validate(p) for p in papers if isinstance(p, dict)]
    return format_papers_block(models)
