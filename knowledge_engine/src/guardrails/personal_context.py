"""Stage 0 — Ollama 7B personal project context (not search query generation)."""

from __future__ import annotations

from knowledge_engine.config import (
    GUARDRAILS_OLLAMA_MODEL,
    OLLAMA_GUARDRAILS_NUM_PREDICT,
)
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.services.local_llm_stateless import run_local_structured
from knowledge_engine.src.locks import run_under_uma_lock
from knowledge_engine.src.state import PersonalContext
from knowledge_engine.ui.run_log import trace


def format_personal_context(ctx: PersonalContext) -> str:
    lines = [
        f"Сводка: {ctx.summary}",
        f"Архитектура: {', '.join(ctx.target_architecture) or '—'}",
        f"Latency / SLA: {ctx.latency_requirements or '—'}",
        f"Ресурсы: {ctx.resource_constraints or '—'}",
        f"Стек: {', '.join(ctx.target_stack) or '—'}",
        f"Фокус проекта: {ctx.project_focus or '—'}",
    ]
    return "\n".join(lines)


def _generate_personal_context_sync(
    user_query: str,
    user_profile_md: str,
    model_name: str,
) -> PersonalContext:
    query = (user_query or "").strip()
    profile = (user_profile_md or "").strip()[:5000]
    system = (
        f"{RUSSIAN_OUTPUT_RULE} "
        "Ты Local Personal Context Agent (Ollama 7B). "
        "НЕ генерируй поисковые запросы для веб-поиска. "
        "НЕ расшифровывай аббревиатуры из словаря — только контекст проекта.\n\n"
        "Проанализируй вопрос пользователя и user_profile.md. "
        "Выдай JSON PersonalContext: summary, target_architecture[], "
        "latency_requirements, resource_constraints, target_stack[], project_focus."
    )
    user_payload = f"Вопрос пользователя:\n{query}\n\n" f"user_profile.md:\n{profile}"
    anchor = f"Задача: {query}"
    trace(f"PERSONAL CONTEXT ▶ Ollama | model={model_name}")
    ctx = run_local_structured(
        model_name,
        PersonalContext,
        system,
        user_payload,
        anchor,
        "personal_context / PersonalContext",
        temperature=0.1,
        num_predict=OLLAMA_GUARDRAILS_NUM_PREDICT,
    )
    trace(f"PERSONAL CONTEXT ✓ | arch={len(ctx.target_architecture)}")
    return ctx


async def run_personal_context_stage(
    user_query: str,
    user_profile_md: str = "",
    model_name: str | None = None,
) -> PersonalContext:
    model = (model_name or GUARDRAILS_OLLAMA_MODEL).strip()
    return await run_under_uma_lock(
        _generate_personal_context_sync,
        user_query,
        user_profile_md,
        model,
    )
