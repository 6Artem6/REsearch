"""Node 4: deep unraveling of the user-selected trade-off option."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from knowledge_engine.config import MAIN_MODEL
from knowledge_engine.llm import chat_ollama, invoke_logged
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.schemas import EngineGraphState, EngineState
from knowledge_engine.ui.errors import format_error_with_cause
from knowledge_engine.ui.run_log import node_end, node_start


def unraveling_node(state: EngineGraphState) -> dict[str, Any]:
    node_start("unraveling_node (7B → детальный разбор)")
    parsed = EngineState.model_validate(state)

    if parsed.report is None:
        raise RuntimeError("Нет отчёта (report): сначала выполните сборку матрицы.")

    if parsed.selected_option_id is None:
        raise RuntimeError("Не выбран selected_option_id для раскрутки.")

    option = next(
        (o for o in parsed.report.options if o.id == parsed.selected_option_id),
        None,
    )
    if option is None:
        valid = ", ".join(str(o.id) for o in parsed.report.options)
        raise RuntimeError(
            f"Вариант с id={parsed.selected_option_id} не найден. Доступны: {valid}."
        )

    llm = chat_ollama(MAIN_MODEL, temperature=0.25)
    system = SystemMessage(
        content=(
            f"{RUSSIAN_OUTPUT_RULE} "
            "Ты staff-инженер. Дай исчерпывающую раскрутку выбранного архитектурного "
            "варианта: алгоритм, структуры данных, пример кода или конфига, "
            "чек-лист деплоя и типичные failure modes в проде. "
            "Структурированно с заголовками Markdown."
        )
    )
    human = HumanMessage(
        content=(
            f"Исходная задача: {parsed.user_problem}\n"
            f"Ограничения: {parsed.context_constraints or '(нет)'}\n\n"
            f"Выбранный вариант (id={option.id}):\n"
            f"Паттерн: {option.pattern_name}\n"
            f"Категория: {option.category}\n"
            f"Идея: {option.fundamental_idea}\n"
            f"Плюсы: {option.pros}\n"
            f"Риски: {option.cons_and_risks}\n"
            f"Operational cost: {option.operational_cost}\n"
        )
    )

    try:
        response = invoke_logged(llm, [system, human], "unraveling / markdown")
        details = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
    except Exception as exc:
        raise RuntimeError(
            f"Unraveling не удался (проверьте Ollama и модель {MAIN_MODEL}): "
            f"{format_error_with_cause(exc)}"
        ) from exc

    if not details.strip():
        raise RuntimeError("Модель вернула пустой ответ для unraveling.")

    node_end("unraveling_node (7B → детальный разбор)", f"chars={len(details)}")
    return {"unraveled_details": details.strip()}
