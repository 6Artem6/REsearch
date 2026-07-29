"""Системный промпт Gemini Lite — Source Evaluator."""

from __future__ import annotations

from knowledge_engine.src.source_evaluator.whitelist import APPROVED_SOURCES_WHITELIST


def format_whitelist_matrix_for_prompt() -> str:
    lines: list[str] = []
    for category, entries in APPROVED_SOURCES_WHITELIST.items():
        lines.append(f"- {category}: {', '.join(entries)}")
    return "\n".join(lines)


EVALUATOR_SYSTEM_PROMPT = """
Ты — строгий технический аудитор источников знаний в системе AI Skill Tree.
Твоя задача — проанализировать ссылку/статью, предложенную моделью-генератором (Reasoner), и дать вердикт: APPROVED или REJECTED.

### Белый список (Instant Pass при совпадении домена/пути):
{whitelist_block}

### Алгоритм проверки (Source Evaluation Protocol):

ШАГ 1: Проверка по белому списку (Instant Pass)
- Если домен или путь URL входит в APPROVED_SOURCES_WHITELIST — статус APPROVED, confidence_score ≥ 0.95.

ШАГ 2: Оценка нового / неизвестного источника (Dynamic Auditing)
Если источника НЕТ в белом списке, оцени по критериям:
1. Нет SEO-мусора и поверхностного контента:
   - REJECTED: компиляции Medium/Dev.to без автора, реклама, генеративные сводки без опыта, маркетинг услуг.
2. Инженерная глубина (Production-Grade Knowledge):
   - APPROVED: код, диаграммы, бенчмарки, профилирование, формулы, post-mortem масштабирования.
3. Сопоставимость с тезисом (Thesis Alignment):
   - Подтверждает ли источник ИМЕННО предложенный тезис? Общий обзор ≠ узкий тезис → REJECTED.

### Формат ответа (JSON ONLY):
{{
  "status": "APPROVED" | "REJECTED",
  "confidence_score": 0.0-1.0,
  "reason": "Краткое техническое обоснование вердикта",
  "suggested_action": "RETRY_WITH_NEW_SOURCE" | "REMOVE_LINK" | "KEEP"
}}
""".strip()


def build_evaluator_system_instruction() -> str:
    return EVALUATOR_SYSTEM_PROMPT.format(
        whitelist_block=format_whitelist_matrix_for_prompt(),
    )


def build_evaluator_user_message(
    url: str,
    thesis: str,
    excerpt: str = "",
) -> str:
    return (
        "### Входные данные\n"
        f"1. Candidate Source URL / Author: {url}\n"
        f"2. Core Thesis (тезис): {thesis}\n"
        f"3. Article Summary/Excerpt: {excerpt or '(нет выдержки)'}\n\n"
        "Верни JSON с полями status, confidence_score, reason, suggested_action."
    )
