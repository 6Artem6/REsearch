"""Ожидание завершения этапа анализа (long poll)."""

from __future__ import annotations

import time
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from knowledge_engine.api.dependencies import get_job_store
from knowledge_engine.api.schemas.responses import (
    AnalysisJobResponse,
    AnalysisJobWaitResponse,
)
from knowledge_engine.services.job_store import AnalysisJob, JobStatus, JobStore

router = APIRouter(prefix="/analyses", tags=["analysis-wait"])


def _wait_done(job: AnalysisJob, target: str) -> bool:
    if job.status == JobStatus.FAILED:
        return True
    if job.clarify_question:
        return True
    if target == "matrix":
        return job.status in (JobStatus.MATRIX_READY, JobStatus.COMPLETED)
    if target == "completed":
        return job.status == JobStatus.COMPLETED
    return job.status in (
        JobStatus.MATRIX_READY,
        JobStatus.COMPLETED,
        JobStatus.FAILED,
    )


@router.get("/{job_id}/wait", response_model=AnalysisJobWaitResponse)
async def wait_for_analysis(
    job_id: str,
    store: Annotated[JobStore, Depends(get_job_store)],
    timeout_sec: float = Query(300, ge=1, le=3600, description="Макс. ожидание (сек)"),
    interval_sec: float = Query(
        2, ge=0.5, le=30, description="Интервал проверки внутри запроса"
    ),
    target: Literal["any", "matrix", "completed"] = Query(
        "any",
        description="any=matrix_ready|completed|failed|clarify; matrix=матрица; completed=unravel",
    ),
) -> AnalysisJobWaitResponse:
    """
    Long polling: держит HTTP открытым до готовности или timeout.
    Альтернатива: scripts/poll-analysis.sh (опрос каждые N сек).
    """
    import asyncio

    job = store.get(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Analysis job not found: {job_id}. "
                "См. knowledge_engine/.runs/last-wait-response.json"
            ),
        )

    t0 = time.monotonic()
    deadline = t0 + timeout_sec

    while time.monotonic() < deadline:
        job = store.get(job_id)
        if not job:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Analysis job not found: {job_id}. "
                    "См. knowledge_engine/.runs/last-wait-response.json"
                ),
            )
        if _wait_done(job, target):
            waited = time.monotonic() - t0
            return AnalysisJobWaitResponse(
                job=AnalysisJobResponse.from_job(job),
                done=True,
                timed_out=False,
                waited_sec=round(waited, 2),
            )
        await asyncio.sleep(interval_sec)

    job = store.get(job_id) or job
    waited = time.monotonic() - t0
    done = _wait_done(job, target)
    return AnalysisJobWaitResponse(
        job=AnalysisJobResponse.from_job(job),
        done=done,
        timed_out=not done,
        waited_sec=round(waited, 2),
    )
