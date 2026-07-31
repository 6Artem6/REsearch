"""Постановка задач в очередь worker."""

from __future__ import annotations

from fastapi import HTTPException

from knowledge_engine.config import KE_WORKER_INLINE_FALLBACK

from knowledge_engine.services.work_job_store import (
    WorkJobKind,
    worker_is_alive,
    work_job_store,
)
from knowledge_engine.ui.run_log import trace


def _inline_fallback_allowed() -> bool:
    return KE_WORKER_INLINE_FALLBACK


def require_worker_or_inline() -> None:
    if worker_is_alive():
        return
    if _inline_fallback_allowed():
        trace("WARN: worker offline — KE_WORKER_INLINE_FALLBACK")
        return
    raise HTTPException(
        status_code=503,
        detail=(
            "Worker не запущен. Запустите: python -m knowledge_engine.worker "
            "или make dev (поднимает API + worker)."
        ),
    )


def enqueue_curriculum_generate(payload: dict) -> str:
    require_worker_or_inline()
    if not worker_is_alive():
        from knowledge_engine.services.work_handlers import run_work_job

        job = work_job_store.create(WorkJobKind.CURRICULUM_GENERATE, payload)
        try:
            result = run_work_job(job)
            work_job_store.complete(job.id, result)
        except Exception as exc:
            from knowledge_engine.services.work_handlers import format_work_error

            work_job_store.fail(job.id, format_work_error(exc))
        return job.id
    job = work_job_store.create(WorkJobKind.CURRICULUM_GENERATE, payload)
    return job.id


def enqueue_curriculum_expand(payload: dict) -> str:
    require_worker_or_inline()
    if not worker_is_alive():
        from knowledge_engine.services.work_handlers import run_work_job

        job = work_job_store.create(WorkJobKind.CURRICULUM_EXPAND, payload)
        try:
            result = run_work_job(job)
            work_job_store.complete(job.id, result)
        except Exception as exc:
            from knowledge_engine.services.work_handlers import format_work_error

            work_job_store.fail(job.id, format_work_error(exc))
        return job.id
    job = work_job_store.create(WorkJobKind.CURRICULUM_EXPAND, payload)
    return job.id


def enqueue_node_deep_dive(payload: dict) -> str:
    require_worker_or_inline()
    if not worker_is_alive():
        from knowledge_engine.services.work_handlers import run_work_job

        job = work_job_store.create(WorkJobKind.NODE_DEEP_DIVE, payload)
        try:
            result = run_work_job(job)
            work_job_store.complete(job.id, result)
        except Exception as exc:
            from knowledge_engine.services.work_handlers import format_work_error

            work_job_store.fail(job.id, format_work_error(exc))
        return job.id
    job = work_job_store.create(WorkJobKind.NODE_DEEP_DIVE, payload)
    return job.id
