"""Node 1: decompose the user problem into CS abstractions."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from knowledge_engine.config import MAIN_MODEL
from knowledge_engine.llm import invoke_logged, structured_chat
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.schemas import CSAbstractionList, EngineGraphState, EngineState
from knowledge_engine.ui.errors import format_error_with_cause
from knowledge_engine.ui.logger import set_status
from knowledge_engine.ui.run_log import node_end, node_start


def decomposition_node(state: EngineGraphState) -> dict[str, Any]:
    node_start("decomposition_node (7B → CS-абстракции)")
    set_status("[decomposition] 7B: CS-абстракции…")
    parsed = EngineState.model_validate(state)
    structured_llm = structured_chat(MAIN_MODEL, CSAbstractionList, temperature=0.2)

    system = SystemMessage(
        content=(
            f"{RUSSIAN_OUTPUT_RULE} "
            "Ты архитектор ПО. Разложи инженерную задачу на 3–6 фундаментальных "
            "CS-абстракций. Отвечай строго в заданной JSON-схеме."
        )
    )
    human = HumanMessage(
        content=(
            f"Задача:\n{parsed.user_problem}\n\n"
            f"Ограничения контекста:\n{parsed.context_constraints or '(не указаны)'}"
        )
    )

    try:
        result: CSAbstractionList | None = invoke_logged(
            structured_llm, [system, human], "decomposition / CSAbstractionList"
        )
        if result is None:
            raise ValueError("structured output returned None")
        abstractions = result.items
    except Exception as exc:
        raise RuntimeError(
            f"Декомпозиция не удалась (проверьте Ollama и модель {MAIN_MODEL}): "
            f"{format_error_with_cause(exc)}"
        ) from exc

    if not abstractions:
        raise RuntimeError("Модель вернула пустой список абстракций.")

    node_end(
        "decomposition_node (7B → CS-абстракции)", f"{len(abstractions)} абстракций"
    )
    return {"abstractions": [a.model_dump() for a in abstractions]}
