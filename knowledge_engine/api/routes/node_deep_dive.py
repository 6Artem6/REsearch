"""Модуль 2 — API глубокой проработки ноды."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from knowledge_engine.api.helpers.work_enqueue import enqueue_node_deep_dive
from knowledge_engine.services.work_job_store import WorkJobStatus, work_job_store
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput
from knowledge_engine.ui.run_log import trace

router = APIRouter(prefix="/node-deep-dive", tags=["node-deep-dive-module-2"])


class NodeDeepDiveBody(BaseModel):
    curriculum_id: str = Field(min_length=3, max_length=80)
    node_data: NodeDataInput
    user_action: str = Field(pattern=r"^(init|chat|verify)$")
    user_message: str = Field(default="", max_length=8000)


@router.post("/interact", status_code=status.HTTP_202_ACCEPTED)
def post_node_deep_dive(body: NodeDeepDiveBody) -> dict[str, Any]:
    """Интерактивная сессия ноды через worker."""
    trace(
        f"API ▶ POST /node-deep-dive/interact (queue) "
        f"{body.user_action} | {body.curriculum_id}/{body.node_data.node_id}"
    )
    payload = {
        "curriculum_id": body.curriculum_id.strip(),
        "node_data": body.node_data.model_dump(),
        "user_action": body.user_action,
        "user_message": body.user_message,
    }
    job_id = enqueue_node_deep_dive(payload)
    job = work_job_store.get(job_id)
    if job and job.status == WorkJobStatus.COMPLETED and job.result:
        return job.result
    if job and job.status == WorkJobStatus.FAILED:
        raise HTTPException(status_code=503, detail=job.error or "failed")
    return {"job_id": job_id, "status": "pending"}
