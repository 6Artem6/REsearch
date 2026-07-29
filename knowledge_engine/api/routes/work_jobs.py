"""Статус и long-poll для work jobs (Skill Tree / Gemini)."""

from __future__ import annotations

import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from knowledge_engine.api.dependencies import get_work_job_store
from knowledge_engine.services.work_job_store import WorkJobStatus, WorkJobStore

router = APIRouter(prefix="/work-jobs", tags=["work-jobs"])


class WorkJobResponse(BaseModel):
    id: str
    kind: str
    status: str
    created_at: str
    updated_at: str
    result: dict[str, Any] | None = None
    error: str | None = None


class WorkJobWaitResponse(BaseModel):
    job: WorkJobResponse
    done: bool
    timed_out: bool = False
    waited_sec: float = 0.0


def _response(job) -> WorkJobResponse:
    return WorkJobResponse(
        id=job.id,
        kind=job.kind.value,
        status=job.status.value,
        created_at=job.created_at.isoformat(),
        updated_at=job.updated_at.isoformat(),
        result=job.result,
        error=job.error,
    )


def _is_done(job) -> bool:
    return job.status in (WorkJobStatus.COMPLETED, WorkJobStatus.FAILED)


@router.get("/{job_id}", response_model=WorkJobResponse)
def get_work_job(
    job_id: str,
    store: Annotated[WorkJobStore, Depends(get_work_job_store)],
) -> WorkJobResponse:
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Work job not found: {job_id}")
    return _response(job)


@router.get("/{job_id}/wait", response_model=WorkJobWaitResponse)
def wait_work_job(
    job_id: str,
    store: Annotated[WorkJobStore, Depends(get_work_job_store)],
    timeout_sec: float = Query(default=600.0, ge=1.0, le=3600.0),
    poll_interval_sec: float = Query(default=0.5, ge=0.2, le=5.0),
) -> WorkJobWaitResponse:
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Work job not found: {job_id}")

    if _is_done(job):
        return WorkJobWaitResponse(job=_response(job), done=True, waited_sec=0.0)

    deadline = time.monotonic() + timeout_sec
    waited = 0.0
    while time.monotonic() < deadline:
        time.sleep(poll_interval_sec)
        waited += poll_interval_sec
        job = store.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Work job not found: {job_id}")
        if _is_done(job):
            return WorkJobWaitResponse(
                job=_response(job),
                done=True,
                waited_sec=waited,
            )

    job = store.get(job_id) or job
    return WorkJobWaitResponse(
        job=_response(job),
        done=_is_done(job),
        timed_out=not _is_done(job),
        waited_sec=waited,
    )
