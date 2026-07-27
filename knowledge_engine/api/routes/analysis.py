"""Анализ: запуск графа, clarify, unravel."""

from __future__ import annotations

import concurrent.futures
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from knowledge_engine.api.dependencies import get_job_store
from knowledge_engine.api.schemas.requests import (
    AnalyzeCreate,
    ClarifySubmit,
    UnravelRequest,
)
from knowledge_engine.api.schemas.responses import (
    AnalysisJobResponse,
    AnalyzeCreatedResponse,
)
from knowledge_engine.services.analysis_service import (
    run_analysis_job,
    run_unravel_for_job,
)
from knowledge_engine.services.job_store import JobStatus, JobStore

router = APIRouter(prefix="/analyses", tags=["analysis"])

_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="ke-analysis"
)


def _submit(fn, *args) -> None:
    _executor.submit(fn, *args)


@router.post(
    "", response_model=AnalyzeCreatedResponse, status_code=status.HTTP_202_ACCEPTED
)
def create_analysis(
    body: AnalyzeCreate,
    store: Annotated[JobStore, Depends(get_job_store)],
) -> AnalyzeCreatedResponse:
    job = store.create(
        body.problem,
        body.constraints,
        matrix_only=body.matrix_only,
        discovery_cache_first=body.reuse_cached_sources,
    )
    from knowledge_engine.ui.run_log import trace

    trace(f"API ▶ POST /analyses job={job.id} async={body.async_mode}")

    if body.async_mode:
        _submit(run_analysis_job, job.id)
        return AnalyzeCreatedResponse(
            job=AnalysisJobResponse.from_job(job),
            message="Анализ запущен в фоне. GET /analyses/{id} для статуса.",
        )

    run_analysis_job(job.id)
    updated = store.get(job.id)
    if not updated:
        raise HTTPException(status_code=500, detail="Job lost")
    return AnalyzeCreatedResponse(
        job=AnalysisJobResponse.from_job(updated),
        message="Синхронный прогон завершён.",
    )


@router.get("", response_model=list[AnalysisJobResponse])
def list_analyses(
    store: Annotated[JobStore, Depends(get_job_store)],
    limit: int = 20,
) -> list[AnalysisJobResponse]:
    jobs = store.list_recent(limit=limit)
    return [AnalysisJobResponse.from_job(j) for j in jobs]


@router.get("/{job_id}", response_model=AnalysisJobResponse)
def get_analysis(
    job_id: str,
    store: Annotated[JobStore, Depends(get_job_store)],
) -> AnalysisJobResponse:
    job = store.get(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Analysis job not found: {job_id}. "
                "Если API перезапускали (uvicorn reload), откройте сохранённый ответ: "
                "knowledge_engine/.runs/last-wait-response.json или "
                "./knowledge_engine/scripts/view-job.sh --file knowledge_engine/.runs/last-wait-response.json"
            ),
        )
    return AnalysisJobResponse.from_job(job)


@router.post("/{job_id}/clarify", response_model=AnalysisJobResponse)
def submit_clarification(
    job_id: str,
    body: ClarifySubmit,
    store: Annotated[JobStore, Depends(get_job_store)],
) -> AnalysisJobResponse:
    job = store.get(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Analysis job not found: {job_id}. "
                "Если API перезапускали (uvicorn reload), откройте сохранённый ответ: "
                "knowledge_engine/.runs/last-wait-response.json или "
                "./knowledge_engine/scripts/view-job.sh --file knowledge_engine/.runs/last-wait-response.json"
            ),
        )
    if not job.clarify_question:
        raise HTTPException(status_code=400, detail="Нет активного вопроса для clarify")

    _submit(run_analysis_job, job_id, body.answer)
    updated = store.get(job_id)
    return AnalysisJobResponse.from_job(updated or job)


@router.post("/{job_id}/unravel", response_model=AnalysisJobResponse)
def unravel_option(
    job_id: str,
    body: UnravelRequest,
    store: Annotated[JobStore, Depends(get_job_store)],
) -> AnalysisJobResponse:
    job = store.get(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Analysis job not found: {job_id}. "
                "Если API перезапускали (uvicorn reload), откройте сохранённый ответ: "
                "knowledge_engine/.runs/last-wait-response.json или "
                "./knowledge_engine/scripts/view-job.sh --file knowledge_engine/.runs/last-wait-response.json"
            ),
        )
    if job.status not in (JobStatus.MATRIX_READY, JobStatus.COMPLETED):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unravel доступен при status=matrix_ready или повтор при completed; "
                f"сейчас: {job.status.value}"
            ),
        )
    if job.report:
        valid = {o.get("id") for o in job.report.get("options", [])}
        if body.option_id not in valid:
            raise HTTPException(
                status_code=400,
                detail=f"option_id {body.option_id} не в матрице: {sorted(valid)}",
            )

    if (
        job.status == JobStatus.COMPLETED
        and not body.force_rerun
        and job.selected_option_id == body.option_id
        and job.unraveled_details
    ):
        return AnalysisJobResponse.from_job(job)

    if body.async_mode:
        _submit(run_unravel_for_job, job_id, body.option_id)
        return AnalysisJobResponse.from_job(job)

    run_unravel_for_job(job_id, body.option_id)
    updated = store.get(job_id)
    return AnalysisJobResponse.from_job(updated or job)
