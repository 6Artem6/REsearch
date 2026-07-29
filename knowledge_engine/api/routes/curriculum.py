"""Модуль 1 — API генератора учебного графа."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from knowledge_engine.api.helpers.work_enqueue import enqueue_curriculum_generate
from knowledge_engine.services.work_job_store import WorkJobStatus, work_job_store
from knowledge_engine.src.curriculum.schemas import CurriculumGraph
from knowledge_engine.ui.run_log import trace

router = APIRouter(prefix="/curriculum", tags=["curriculum-module-1"])


class CurriculumGenerateRequest(BaseModel):
    target_goal: str = Field(min_length=8, max_length=4000)
    user_level: str = Field(default="Intermediate/Advanced", max_length=120)
    depth_level: str = Field(default="Standard", max_length=40)
    generation_mode: str = Field(
        default="fast",
        description="fast | consensus (UI селектор)",
    )


class CurriculumGenerateAccepted(BaseModel):
    job_id: str
    status: str = "pending"
    message: str = "Генерация в worker. GET /work-jobs/{id}/wait"


@router.post(
    "/generate",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=CurriculumGenerateAccepted,
)
def post_generate_curriculum(body: CurriculumGenerateRequest) -> dict[str, Any]:
    """Поставить генерацию DAG в очередь worker (не блокирует API)."""
    depth = (body.depth_level or "Standard").strip()
    if depth not in ("Overview", "Standard", "Deep Mechanics"):
        raise HTTPException(
            status_code=422,
            detail="depth_level: Overview | Standard | Deep Mechanics",
        )
    gen_mode = (body.generation_mode or "fast").strip().lower()
    if gen_mode in ("deep", "consensus"):
        gen_mode = "consensus"
    else:
        gen_mode = "fast"
    inp = {
        "target_goal": body.target_goal.strip(),
        "user_level": (body.user_level or "Intermediate/Advanced").strip(),
        "depth_level": depth,
        "generation_mode": gen_mode,
    }
    trace(
        f"API ▶ POST /curriculum/generate (queue) | mode={gen_mode} "
        f"{inp['target_goal'][:60]}…"
    )
    job_id = enqueue_curriculum_generate(inp)
    job = work_job_store.get(job_id)
    if job and job.status == WorkJobStatus.COMPLETED and job.result:
        return {
            "job_id": job_id,
            "status": "completed",
            "message": "inline",
            "graph": job.result,
        }
    if job and job.status == WorkJobStatus.FAILED:
        raise HTTPException(status_code=503, detail=job.error or "generate failed")
    return {
        "job_id": job_id,
        "status": "pending",
        "message": "GET /api/v1/work-jobs/{job_id}/wait",
    }


@router.get("/generate/result/{job_id}", response_model=CurriculumGraph)
def get_curriculum_generate_result(job_id: str) -> dict[str, Any]:
    job = work_job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status == WorkJobStatus.FAILED:
        raise HTTPException(status_code=503, detail=job.error or "failed")
    if job.status != WorkJobStatus.COMPLETED or not job.result:
        raise HTTPException(status_code=409, detail=f"status={job.status.value}")
    return job.result
