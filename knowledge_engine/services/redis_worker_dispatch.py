"""Обработка задач из Redis pub/sub."""

from __future__ import annotations

from typing import Any

from knowledge_engine.services.analysis_service import (
    run_analysis_job,
    run_unravel_for_job,
)
from knowledge_engine.services.job_store import JobStatus, job_store
from knowledge_engine.services.v07_run_service import run_v07_job
from knowledge_engine.services.v07_run_store import V07RunStatus, v07_run_store
from knowledge_engine.services.work_handlers import format_work_error, run_work_job
from knowledge_engine.services.work_job_store import work_job_store
from knowledge_engine.ui.run_log import trace


def dispatch_task_message(data: dict[str, Any]) -> bool:
    typ = str(data.get("type") or "").strip()
    if typ == "work_job":
        return _handle_work_job(str(data.get("id") or ""))
    if typ == "analysis":
        return _handle_analysis(
            str(data.get("id") or ""),
            data.get("clarify_answer"),
        )
    if typ == "analysis_unravel":
        return _handle_unravel(
            str(data.get("id") or ""),
            int(data.get("option_id") or 0),
        )
    if typ == "v07":
        return _handle_v07(str(data.get("id") or ""))
    return False


def _handle_work_job(job_id: str) -> bool:
    if not job_id:
        return False
    job = work_job_store.try_claim(job_id)
    if not job:
        return False
    trace(f"WORKER ▶ work_job {job.kind.value} id={job.id}")
    from knowledge_engine.services.worker_busy import worker_busy_scope

    try:
        with worker_busy_scope(f"work_job:{job.id}"):
            result = run_work_job(job)
        work_job_store.complete(job.id, result)
        trace(f"WORKER ✓ work_job id={job.id}")
    except Exception as exc:
        work_job_store.fail(job.id, format_work_error(exc))
        trace(f"WORKER ✗ work_job id={job.id} | {exc}")
    return True


def _handle_analysis(job_id: str, clarify_answer: Any) -> bool:
    if not job_id:
        return False
    job = job_store.try_claim_analysis(job_id)
    if not job:
        return False
    trace(f"WORKER ▶ analysis job={job_id}")
    clar = job.pending_clarify_answer
    if clar:
        job_store.update(job_id, pending_clarify_answer=None)
    elif isinstance(clarify_answer, str) and clarify_answer.strip():
        clar = clarify_answer.strip()
    try:
        run_analysis_job(job_id, clar)
        trace(f"WORKER ✓ analysis job={job_id}")
    except Exception as exc:
        job_store.update(
            job_id,
            status=JobStatus.FAILED,
            error=str(exc),
        )
    return True


def _handle_unravel(job_id: str, option_id: int) -> bool:
    if not job_id or option_id <= 0:
        return False
    job = job_store.try_claim_unravel(job_id, option_id)
    if not job:
        return False
    trace(f"WORKER ▶ unravel job={job_id} opt={option_id}")
    try:
        run_unravel_for_job(job_id, option_id)
        trace(f"WORKER ✓ unravel job={job_id}")
    except Exception as exc:
        job_store.update(job_id, status=JobStatus.FAILED, error=str(exc))
    return True


def _handle_v07(run_id: str) -> bool:
    if not run_id:
        return False
    run = v07_run_store.try_claim(run_id)
    if not run:
        return False
    trace(f"WORKER ▶ v07 run={run_id}")
    try:
        run_v07_job(run_id)
        trace(f"WORKER ✓ v07 run={run_id}")
    except Exception as exc:
        v07_run_store.update(run_id, status=V07RunStatus.FAILED, error=str(exc))
    return True
