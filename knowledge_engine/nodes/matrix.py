"""Node 3: build Trade-off matrix (AnalysisReport) with MAIN_MODEL."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from knowledge_engine.config import MAIN_MODEL, MATRIX_MAX_SUMMARY_CHARS
from knowledge_engine.llm import invoke_logged, structured_chat
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.schemas import AnalysisReport, EngineGraphState, EngineState
from knowledge_engine.ui.errors import format_error_with_cause
from knowledge_engine.ui.run_log import node_end, node_start


def matrix_node(state: EngineGraphState) -> dict[str, Any]:
    node_start("matrix_node (7B → Trade-off матрица)")
    parsed = EngineState.model_validate(state)

    abstractions_text = "\n".join(
        f"- {a.title} ({a.cs_concept}): {a.description}" for a in parsed.abstractions
    )
    facts_text = "\n".join(f"- {f}" for f in parsed.found_facts) or "(факты не собраны)"
    summaries_text = ""
    for s in parsed.found_summaries:
        block = (
            f"\n### {s.title} ({s.url})\n"
            f"CS: {', '.join(s.cs_concepts)}\n"
            f"Takeaways: {s.key_takeaways}\n"
            f"Failure modes: {s.failure_modes}\n"
        )
        summaries_text += block[:MATRIX_MAX_SUMMARY_CHARS]
    if not summaries_text:
        summaries_text = "(summaries не собраны)"

    structured_llm = structured_chat(MAIN_MODEL, AnalysisReport, temperature=0.3)
    system = SystemMessage(
        content=(
            f"{RUSSIAN_OUTPUT_RULE} "
            "Ты senior архитектор. Собери Trade-off матрицу: ровно 3 варианта решения "
            "(id=1,2,3) с категориями Классика / SOTA (Современное) / Минимализм. "
            "Укажи pros, cons_and_risks (failure modes) и operational_cost. "
            "Поле abstractions должно повторять или уточнить входные абстракции."
        )
    )
    human = HumanMessage(
        content=(
            f"Задача: {parsed.user_problem}\n"
            f"Ограничения: {parsed.context_constraints or '(нет)'}\n\n"
            f"CS-абстракции:\n{abstractions_text}\n\n"
            f"Собранные факты:\n{facts_text}\n\n"
            f"Executive summaries из LanceDB / поиска:\n{summaries_text}\n"
        )
    )

    try:
        report: AnalysisReport | None = invoke_logged(
            structured_llm, [system, human], "matrix / AnalysisReport"
        )
        if report is None:
            raise ValueError("structured output returned None")
    except Exception as exc:
        raise RuntimeError(
            f"Сборка матрицы не удалась (проверьте Ollama и модель {MAIN_MODEL}): "
            f"{format_error_with_cause(exc)}"
        ) from exc

    if len(report.options) != 3:
        raise RuntimeError(
            f"Ожидалось ровно 3 варианта в матрице, получено: {len(report.options)}."
        )

    ids = {opt.id for opt in report.options}
    if ids != {1, 2, 3}:
        raise RuntimeError("ID вариантов должны быть 1, 2 и 3.")

    node_end("matrix_node (7B → Trade-off матрица)", "3 options")
    return {"report": report.model_dump()}
