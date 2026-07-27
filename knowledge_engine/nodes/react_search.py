"""Node 2: Re-Act search loop — query generation and sufficiency check (router)."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from knowledge_engine.config import ROUTER_MODEL
from knowledge_engine.llm import structured_chat
from knowledge_engine.schemas import (
    EngineGraphState,
    EngineState,
    ReactSearchAssessment,
)
from knowledge_engine.ui.errors import format_error_with_cause


def react_search_node(state: EngineGraphState) -> dict[str, Any]:
    """Generate search queries and assess whether facts are sufficient for the matrix."""
    parsed = EngineState.model_validate(state)
    iteration = parsed.search_iterations + 1

    abstractions_text = "\n".join(
        f"- {a.title} ({a.cs_concept}): {a.description}" for a in parsed.abstractions
    )
    prior_queries = "\n".join(f"- {q}" for q in parsed.search_queries) or "(нет)"
    prior_facts = "\n".join(f"- {f}" for f in parsed.found_facts) or "(нет)"

    structured_llm = structured_chat(
        ROUTER_MODEL, ReactSearchAssessment, temperature=0.1
    )
    system = SystemMessage(
        content=(
            "Ты контроллер Re-Act цикла для архитектурного анализа. "
            "Сгенерируй новые поисковые запросы и смоделируй краткие факты "
            "(как если бы это был локальный knowledge base / веб-поиск). "
            "Оцени, достаточно ли накопленных фактов для Trade-off матрицы."
        )
    )
    human = HumanMessage(
        content=(
            f"Итерация: {iteration}\n"
            f"Задача: {parsed.user_problem}\n"
            f"Ограничения: {parsed.context_constraints or '(нет)'}\n\n"
            f"CS-абстракции:\n{abstractions_text}\n\n"
            f"Уже заданные запросы:\n{prior_queries}\n\n"
            f"Уже найденные факты:\n{prior_facts}\n"
        )
    )

    try:
        assessment: ReactSearchAssessment | None = structured_llm.invoke(
            [system, human]
        )
        if assessment is None:
            raise ValueError("structured output returned None")
    except Exception as exc:
        raise RuntimeError(
            f"Re-Act поиск не удался (проверьте Ollama и модель {ROUTER_MODEL}): "
            f"{format_error_with_cause(exc)}"
        ) from exc

    merged_queries = list(parsed.search_queries)
    for query in assessment.new_queries:
        q = query.strip()
        if q and q not in merged_queries:
            merged_queries.append(q)

    merged_facts = list(parsed.found_facts)
    for fact in assessment.simulated_facts:
        f = fact.strip()
        if f and f not in merged_facts:
            merged_facts.append(f)

    return {
        "search_iterations": iteration,
        "search_queries": merged_queries,
        "found_facts": merged_facts,
        "is_facts_sufficient": assessment.is_facts_sufficient,
    }
