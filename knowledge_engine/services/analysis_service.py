"""Оркестрация графа для CLI и FastAPI."""

from __future__ import annotations

from typing import Any, Optional

from langgraph.types import Command

from knowledge_engine.config import GRAPH_RECURSION_LIMIT, GRAPH_THREAD_ID
from knowledge_engine.graph.initial_state import build_initial_state
from knowledge_engine.graph.runtime import get_compiled_graph
from knowledge_engine.schemas import EngineState
from knowledge_engine.services.job_store import JobStatus, job_store
from knowledge_engine.ui.errors import format_error_with_cause
from knowledge_engine.ui.logger import live_session, set_status
from knowledge_engine.ui.run_log import get_run_log_path, init_run_log, trace


def run_config(thread_id: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": GRAPH_RECURSION_LIMIT,
    }


def _clarification_from_snapshot(snapshot) -> str | None:
    for task in snapshot.tasks or []:
        for intr in task.interrupts or ():
            val = intr.value
            if isinstance(val, dict):
                return str(val.get("question") or val)
            return str(val)
    return None


def invoke_until_pause(
    graph,
    initial: dict,
    config: dict,
    clarify_answer: Optional[str] = None,
) -> tuple[dict, Optional[str]]:
    """
    Запуск до interrupt_before unraveling или до clarify interrupt.
    Возвращает (state_dict, clarify_question если нужен ответ).
    """
    current: Any = initial
    if clarify_answer is not None:
        current = Command(resume=clarify_answer)

    with live_session():
        set_status("[Graph] analyze…")
        result = graph.invoke(current, config=config)

    snapshot = graph.get_state(config)
    clarify_q = _clarification_from_snapshot(snapshot)
    if clarify_q:
        return result, clarify_q
    return result, None


def run_unravel(graph, config: dict, option_id: int) -> dict:
    graph.update_state(config, {"selected_option_id": option_id})
    with live_session():
        set_status(f"[Graph] unraveling {option_id}…")
        return graph.invoke(None, config=config)


def run_analysis_to_matrix(
    problem: str,
    constraints: str,
    thread_id: str,
    clarify_answer: Optional[str] = None,
    discovery_cache_first: bool = False,
) -> tuple[EngineState, Optional[str], Optional[str]]:
    """До матрицы (interrupt перед unraveling)."""
    graph = get_compiled_graph()
    config = run_config(thread_id)
    log_path = init_run_log(problem)
    trace(f"GRAPH ▶ analyze | thread={thread_id}")
    initial = build_initial_state(
        problem,
        constraints,
        discovery_cache_first=discovery_cache_first,
    )

    result, clarify_q = invoke_until_pause(
        graph, initial, config, clarify_answer=clarify_answer
    )
    if clarify_q:
        return EngineState.model_validate(result), clarify_q, str(log_path)

    state = EngineState.model_validate(result)
    trace("GRAPH ✓ matrix ready")
    return state, None, str(log_path)


def run_analysis_job(job_id: str, clarify_answer: Optional[str] = None) -> None:
    job = job_store.get(job_id)
    if not job:
        return
    job_store.update(job_id, status=JobStatus.RUNNING, error=None)
    try:
        saved_log: Optional[str] = job.log_path
        if clarify_answer is not None:
            graph = get_compiled_graph()
            config = run_config(job.thread_id)
            result, clarify_q = invoke_until_pause(
                graph,
                {},
                config,
                clarify_answer=clarify_answer,
            )
            saved_log = saved_log or str(get_run_log_path() or "")
            state = EngineState.model_validate(result)
            if clarify_q:
                job_store.update(
                    job_id,
                    status=JobStatus.RUNNING,
                    clarify_question=clarify_q,
                    log_path=saved_log,
                )
                return
        else:
            state, clarify_q, lp = run_analysis_to_matrix(
                job.problem,
                job.constraints,
                job.thread_id,
                discovery_cache_first=job.discovery_cache_first,
            )
            saved_log = lp or saved_log
            if clarify_q:
                job_store.update(
                    job_id,
                    status=JobStatus.RUNNING,
                    clarify_question=clarify_q,
                    log_path=saved_log,
                )
                return
        if state.report is None:
            job_store.update(
                job_id,
                status=JobStatus.FAILED,
                error="Матрица не сформирована",
            )
            return
        report_dump = state.report.model_dump()
        if job.matrix_only:
            job_store.update(
                job_id,
                status=JobStatus.COMPLETED,
                report=report_dump,
                log_path=saved_log,
                clarify_question=None,
            )
            return
        job_store.update(
            job_id,
            status=JobStatus.MATRIX_READY,
            report=report_dump,
            log_path=saved_log,
            clarify_question=None,
        )
    except Exception as exc:
        job_store.update(
            job_id,
            status=JobStatus.FAILED,
            error=format_error_with_cause(exc),
        )


def run_unravel_for_job(job_id: str, option_id: int) -> None:
    job = job_store.get(job_id)
    if not job:
        return
    job_store.update(job_id, status=JobStatus.RUNNING, selected_option_id=option_id)
    try:
        graph = get_compiled_graph()
        config = run_config(job.thread_id)
        final = run_unravel(graph, config, option_id)
        final_state = EngineState.model_validate(final)
        if not final_state.unraveled_details:
            job_store.update(
                job_id,
                status=JobStatus.FAILED,
                error="Пустой unraveling",
            )
            return
        job_store.update(
            job_id,
            status=JobStatus.COMPLETED,
            unraveled_details=final_state.unraveled_details,
            selected_option_id=option_id,
        )
        trace("GRAPH ✓ unraveling complete (API)")
    except Exception as exc:
        job_store.update(
            job_id,
            status=JobStatus.FAILED,
            error=format_error_with_cause(exc),
        )


def new_thread_id(suffix: str) -> str:
    return f"{GRAPH_THREAD_ID}-{suffix}"
