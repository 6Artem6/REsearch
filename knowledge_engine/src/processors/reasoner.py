"""Gemini Flash 3.6 (reasoner) — план и финальный ответ пользователю."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from knowledge_engine.config import GEMINI_REASONER_MODEL, GEMINI_RPM_PAUSE_SEC
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.services.gemini_stateless import (
    GeminiUnavailableError,
    gemini_reasoner_model_chain,
    is_gemini_available,
    run_gemini_structured_with_chain,
)
from knowledge_engine.src.processors.source_anchors import (
    REASONER_SOURCE_ATTRIBUTION_PROMPT,
    format_registry_for_prompt,
    format_valid_docs_for_reasoner,
)
from knowledge_engine.src.source_evaluator.evaluator import format_whitelist_for_reasoner_prompt
from knowledge_engine.src.processors.source_evaluator import (
    MAX_REACT_SOURCE_ITERATIONS,
    audit_answer_sources_react,
)
from knowledge_engine.ui.run_log import trace

FOLLOW_UP_RULES = """
КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО В СЛЕДУЮЩИХ ШАГАХ:
- Запрещено предлагать организационные, командные или менеджерские действия (например: «Провести аудит кодовой базы», «Внедрить CI/CD», «Организовать фасилитацию Event Storming», «Обучить команду»).
- Запрещено давать операционные задачи или «домашние задания» по разработке.

ОБЯЗАТЕЛЬНЫЙ ФОРМАТ ДЛЯ СЛЕДУЮЩИХ ШАГОВ (Навигация по знаниям):
Сформируй ровно 3 исследовательских вектора, которые помогут пользователю углубить понимание темы:

1. 🔬 **Шаг вглубь (Технические детали)**: Предложи разобрать конкретный низкоуровневый механизм, математический компромисс, протокол или внутреннее устройство технологии, упомянутой в ответе.
2. 🔄 **Шаг в сторону (Альтернативы и контрасты)**: Предложи рассмотреть альтернативный архитектурный паттерн, конкурирующую парадигму или крайний случай (edge-case), который контрастирует с текущим решением.
3. 🏛️ **Шаг назад (Фундамент и первопричины)**: Предложи исследовать фундаментальные истоки, историю возникновения или базовый принцип Computer Science, который привёл к появлению этой технологии.

Каждый пункт должен быть сформулирован как увлекательный исследовательский вопрос или тема для следующего запроса, а НЕ как задача для исполнения.
"""

FAST_MODE_REASONER_PROMPT = """Ты — Главный Системный Архитектор. Твоя задача — объяснить архитектурную концепцию или паттерн доступным, наглядным и фактологичным языком.

{whitelist_block}

ПРАВИЛА ОФОРМЛЕНИЯ И НАВИГАЦИИ:
1. **Ссылки на материалы**: Вплетай по тексту ссылки на статьи из белого списка в формате Markdown: `[Название статьи](https://...)`.
2. **Наглядные схемы**: Описывай архитектурные взаимодействия с помощью понятных текстовых блок-схем Mermaid или ASCII-диаграмм.
3. **КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО В СЛЕДУЮЩИХ ШАГАХ**:
   - Не давать менеджерских и организационных поручений ("Провести аудит кодовой базы", "Внедрить CI/CD", "Обучить команду").

СТРУКТУРА ОТВЕТА:
- 📌 **Концепция "на пальцах"**: Суть паттерна + наглядная текстовая схема/диаграмма взаимодействия.
- ⚙️ **Как это работает под капотом**: Детальный разбор механики с ссылками на авторитетные источники.
- ⚖️ **Компромиссы и когда применять**: Плюсы, минусы, узкие места и альтернативы.
- 📚 **Эталонные материалы**: Список из 2-4 конкретных статей/разделов из белого списка для чтения.
- 🧭 **Исследовательские векторы (Следующие шаги)**: 3 вопроса (Шаг вглубь, Шаг в сторону, Шаг назад).
""".format(whitelist_block=format_whitelist_for_reasoner_prompt())

REASONER_REACT_CORRECTION_RULES = """
КОРРЕКЦИЯ ПО РЕЗУЛЬТАТАМ АУДИТА ИСТОЧНИКОВ (Re-Act):
Ниже — системные отклики Source Evaluator (Gemini Lite) по отклонённым ссылкам. Перепиши только проблемные фрагменты:
- замени слабые источники на материалы из Whitelist Matrix (practitioners, ai_pioneers_labs, engineering_blogs, foundational_docs);
- либо убери ссылку (REMOVE_LINK) и объясни тезис через фундаментальные принципы CS.
Сохрани структуру ответа и русский язык. Не добавляй менеджерских рекомендаций.
"""

REASONER_SYSTEM = (
    f"""Ты — Главный Системный Архитектор ИИ-систем.
{RUSSIAN_OUTPUT_RULE}
user_final_answer — только русский (термины EN из источников можно сохранять).

На основе valid_docs (материалы Consensus) и developer_profile_context составь план и ответ.

developer_profile_context может быть пустым — тогда не выдумывай ограничения разработчика.

Изоляция профиля (обязательно):
1. НЕ втирай личный контекст (Jarvis, Apple Silicon, LanceDB, M-series) в общие теоретические объяснения.
   Если вопрос общий (CS / архитектура), ответ только про индустрию и академические trade-off — без локальных проектов и железа.
2. Если личный контекст уместен — сначала объективный общий анализ; ограничения локального окружения только в
   опциональном финальном блоке «Применимость к локальному окружению», не в каждом абзаце.

Требования:
- Связывай теорию с инженерной реализацией; учитывай developer_profile_context только если он задан
  и apply_personal_profile=true в payload.
- Структура user_final_answer:
  контекст → сравнение подходов → риски и trade-off → блок «Направление исследования» (см. ниже)
  → опционально «Применимость к локальному окружению».
- Блок «Рекомендации» в смысле менеджерских поручений НЕ используй. Вместо него — только навигация по знаниям.
{FOLLOW_UP_RULES}
- Формулы и сложность: LaTeX в `$...$` (inline) и `$$...$$` (block), например `$\\mathcal{{O}}(N \\cdot d)$`.
- В формулах только валидный TeX: `\\text{{}}`, `\\frac{{}}{{}}`; без табов, form-feed и дублирования текста.
- Код и псевдокод только где уместно.
- Если данных недостаточно (partial_data_note), явно перечисли пробелы.
- user_final_answer — готовый текст для пользователя без мета-объяснений про пайплайн.
- fact_nuggets — атомарные факты для памяти (короткие, проверяемые); **без** тегов [S1] и URL — только чистый текст для Light RAG.

"""
    + REASONER_SOURCE_ATTRIBUTION_PROMPT
)


class FinalResponsePayload(BaseModel):
    user_final_answer: str = Field(
        description="Готовый глубокий ответ для пользователя"
    )
    fact_nuggets: list[str] = Field(default_factory=list)


def _invoke_reasoner_structured(
    system_instruction: str,
    user_payload: str,
    global_anchor: str,
    label: str,
) -> FinalResponsePayload:
    return run_gemini_structured_with_chain(
        GEMINI_REASONER_MODEL,
        system_instruction,
        user_payload,
        global_anchor,
        FinalResponsePayload,
        label,
        rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
        models=gemini_reasoner_model_chain(),
    )


def _build_user_payload(
    *,
    user_query: str,
    profile_block: str,
    apply_personal_profile: bool,
    mode: str,
    registry_section: str,
    papers_section: str,
    valid_section: str,
    raw_consensus_text: str,
    partial_data_note: str,
    light_rag_block: str,
    react_feedback: str = "",
    previous_answer: str = "",
) -> str:
    parts = [
        f"### user_query\n{user_query}",
        f"### apply_personal_profile\n{apply_personal_profile}",
        f"### developer_profile_context\n{profile_block}",
    ]
    if mode == "fast":
        parts.append("### retrieval_mode\nfast (Consensus не вызывался)")
        parts.append(f"### light_rag_facts\n{light_rag_block}")
    else:
        parts.append(f"### SOURCE REGISTRY\n{registry_section}")
        parts.append(f"### scholarly_papers\n{papers_section}")
        parts.append(f"### valid_docs\n{valid_section}")
        parts.append(f"### consensus_raw (fallback)\n{raw_consensus_text[:10000]}")
    parts.append(f"### partial_data_note\n{partial_data_note or 'none'}")
    if previous_answer:
        parts.append(f"### previous_draft_answer\n{previous_answer[:14000]}")
    if react_feedback:
        parts.append(f"### source_audit_feedback\n{react_feedback}")
        parts.append(REASONER_REACT_CORRECTION_RULES)
    return "\n\n".join(parts)


def run_reasoner(
    valid_docs: list[dict[str, Any]],
    user_query: str,
    user_profile: str,
    global_anchor: str,
    *,
    raw_consensus_text: str = "",
    partial_data_note: str = "",
    papers_block: str = "",
    source_registry: list[dict[str, Any]] | None = None,
    apply_personal_profile: bool = True,
    retrieval_mode: str = "consensus",
    light_rag_context: str = "",
) -> FinalResponsePayload:
    if not is_gemini_available():
        raise GeminiUnavailableError("Gemini reasoner недоступен")
    mode = (retrieval_mode or "consensus").strip().lower()
    registry = source_registry or []
    registry_section = format_registry_for_prompt(registry) if registry else ""
    papers_section = (
        papers_block or registry_section or "(нет структурированного списка статей)"
    )
    valid_section = format_valid_docs_for_reasoner(valid_docs, registry)
    profile_block = user_profile[:8000] if user_profile else "(empty)"
    if not apply_personal_profile:
        profile_block = (
            "(пусто — общий/академический вопрос; apply_personal_profile=false)"
        )
    light_rag_block = (
        light_rag_context or ""
    ).strip() or "(нет релевантных фактов Light RAG)"

    if mode == "fast":
        system_instruction = (
            f"{RUSSIAN_OUTPUT_RULE}\n\n{FAST_MODE_REASONER_PROMPT}\n\n{FOLLOW_UP_RULES}"
        )
        label_prefix = "fast_reasoner"
    else:
        system_instruction = REASONER_SYSTEM
        label_prefix = "consensus_reasoner"

    user_payload = _build_user_payload(
        user_query=user_query,
        profile_block=profile_block,
        apply_personal_profile=apply_personal_profile,
        mode=mode,
        registry_section=registry_section,
        papers_section=papers_section,
        valid_section=valid_section,
        raw_consensus_text=raw_consensus_text,
        partial_data_note=partial_data_note,
        light_rag_block=light_rag_block,
    )
    result = _invoke_reasoner_structured(
        system_instruction,
        user_payload,
        global_anchor,
        f"{label_prefix} / draft",
    )

    accumulated_feedback = ""
    for react_round in range(MAX_REACT_SOURCE_ITERATIONS):
        feedback = audit_answer_sources_react(
            result.user_final_answer,
            registry,
            global_anchor,
        )
        if not feedback:
            trace(f"REACT ✓ источники прошли аудит | round={react_round}")
            break
        trace(f"REACT ▶ коррекция Reasoner | round={react_round + 1}")
        accumulated_feedback = (
            f"{accumulated_feedback}\n{feedback}".strip()
            if accumulated_feedback
            else feedback
        )
        revision_payload = _build_user_payload(
            user_query=user_query,
            profile_block=profile_block,
            apply_personal_profile=apply_personal_profile,
            mode=mode,
            registry_section=registry_section,
            papers_section=papers_section,
            valid_section=valid_section,
            raw_consensus_text=raw_consensus_text,
            partial_data_note=partial_data_note,
            light_rag_block=light_rag_block,
            react_feedback=accumulated_feedback,
            previous_answer=result.user_final_answer,
        )
        result = _invoke_reasoner_structured(
            system_instruction,
            revision_payload,
            global_anchor,
            f"{label_prefix} / react_{react_round + 1}",
        )

    return result
