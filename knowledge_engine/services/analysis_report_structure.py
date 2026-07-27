"""7B: Gemini/текст синтеза → AnalysisReport (с лимитом токенов и retry)."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from knowledge_engine.config import MAIN_MODEL, OLLAMA_STRUCTURE_NUM_PREDICT
from knowledge_engine.llm import invoke_logged, structured_chat
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.schemas import AnalysisReport, EngineState
from knowledge_engine.ui.errors import format_error_with_cause
from knowledge_engine.ui.logger import set_status

_STRUCTURE_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE} "
    "Преобразуй синтез в JSON AnalysisReport. Не выдумывай — только из текста.\n"
    "• abstractions: список CSAbstraction.\n"
    "• options: РОВНО 3 элементы с id=1,2,3.\n"
    "• category для каждого option — ТОЛЬКО одно из: «Классика», «SOTA (Современное)», «Минимализм».\n"
    "• pattern_name — короткое название паттерна; cs_concept — в абстракциях, не в category.\n"
    "• Каждый option ОБЯЗАН содержать: pros (2–4 строк), cons_and_risks (2–4), operational_cost.\n"
    "• JSON должен быть ПОЛНЫМ — не обрезай третий вариант."
)


def _validate_report(report: AnalysisReport) -> AnalysisReport:
    if len(report.options) != 3:
        raise ValueError(f"Ожидалось 3 options, получено {len(report.options)}")
    ids = {o.id for o in report.options}
    if ids != {1, 2, 3}:
        raise ValueError(f"ID вариантов должны быть 1,2,3: {ids}")
    for opt in report.options:
        if not opt.pros or not opt.cons_and_risks or not opt.operational_cost.strip():
            raise ValueError(
                f"option id={opt.id}: неполные pros/cons_and_risks/operational_cost"
            )
    return report


def structure_analysis_report(
    parsed: EngineState,
    synthesis_text: str,
    log_label: str = "structure / AnalysisReport",
) -> AnalysisReport:
    """Structured 7B с увеличенным num_predict и одним retry при обрезке JSON."""
    structured = structured_chat(
        MAIN_MODEL,
        AnalysisReport,
        temperature=0.15,
        num_predict=OLLAMA_STRUCTURE_NUM_PREDICT,
    )
    synthesis = synthesis_text[:12000]
    human = HumanMessage(
        content=f"Задача: {parsed.user_problem}\n\nСинтез:\n{synthesis}"
    )
    system = SystemMessage(content=_STRUCTURE_SYSTEM)

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            if attempt > 0:
                set_status(
                    f"[structure] 7B: повтор JSON ({OLLAMA_STRUCTURE_NUM_PREDICT} tok)…"
                )
            report = invoke_logged(
                structured,
                [system, human],
                log_label if attempt == 0 else f"{log_label} (retry)",
            )
            if report is None:
                raise ValueError("structured output returned None")
            return _validate_report(report)
        except Exception as exc:
            last_error = exc
            err_txt = format_error_with_cause(exc)
            human = HumanMessage(
                content=(
                    f"Задача: {parsed.user_problem}\n\n"
                    f"Синтез:\n{synthesis}\n\n"
                    f"ПРЕДЫДУЩАЯ ПОПЫТКА JSON НЕВАЛИДНА: {err_txt}\n"
                    "Верни ПОЛНЫЙ AnalysisReport. Для option id=3 заполни pros, "
                    "cons_and_risks и operational_cost. category — только Классика/SOTA/Минимализм."
                )
            )

    raise RuntimeError(
        f"Не удалось структурировать синтез в AnalysisReport "
        f"(num_predict={OLLAMA_STRUCTURE_NUM_PREDICT}): "
        f"{format_error_with_cause(last_error)}"
    ) from last_error
