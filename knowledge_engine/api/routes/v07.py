"""v0.7 research runs for web UI."""

from __future__ import annotations

import uuid
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from knowledge_engine.api.helpers.work_enqueue import require_worker_or_inline
from knowledge_engine.services.v07_run_store import V07RunStatus, v07_run_store
from knowledge_engine.ui.run_log import trace
from knowledge_engine.web.present import build_ui_view

router = APIRouter(prefix="/v07/runs", tags=["v07-web"])


class V07RunCreate(BaseModel):
    query: str = Field(min_length=3, max_length=4000)
    thread_id: str | None = None
    retrieval_mode: Literal["fast", "consensus"] = "fast"


class V07RunQuestionLogEntry(BaseModel):
    type: str
    text: str
    ts: str
    snippet: str | None = None


class V07RunResponse(BaseModel):
    id: str
    query: str
    status: str
    current_step: str
    thread_id: str
    created_at: str
    error: str | None = None
    log_path: str | None = None
    has_result: bool = False
    has_partial: bool = False
    questions_log: list[V07RunQuestionLogEntry] = []
    retrieval_mode: str = "fast"


def _run_response(run) -> V07RunResponse:
    has_res = run.result is not None
    completed = run.status == V07RunStatus.COMPLETED
    questions_log = [
        V07RunQuestionLogEntry(
            type=str(item.get("type") or "other"),
            text=str(item.get("text") or ""),
            ts=str(item.get("ts") or ""),
            snippet=item.get("snippet"),
        )
        for item in (run.questions_log or [])
    ]
    return V07RunResponse(
        id=run.id,
        query=run.query,
        status=run.status.value,
        current_step=run.current_step,
        thread_id=run.thread_id,
        created_at=run.created_at.isoformat(),
        error=run.error,
        log_path=run.log_path,
        has_result=has_res and completed,
        has_partial=has_res and not completed,
        questions_log=questions_log,
        retrieval_mode=run.retrieval_mode,
    )


@router.post("", status_code=status.HTTP_202_ACCEPTED)
def create_v07_run(body: V07RunCreate) -> dict[str, Any]:
    thread_id = (body.thread_id or f"web-{uuid.uuid4().hex[:10]}").strip()
    run = v07_run_store.create(
        body.query.strip(),
        thread_id,
        retrieval_mode=body.retrieval_mode,
    )
    trace(f"API ▶ POST /v07/runs id={run.id} mode={run.retrieval_mode}")
    require_worker_or_inline()
    from knowledge_engine.services.redis_client import redis_enabled
    from knowledge_engine.services.redis_tasks import publish_v07_run

    if redis_enabled():
        publish_v07_run(run.id)
    return {
        "run": V07RunResponse(
            id=run.id,
            query=run.query,
            status=run.status.value,
            current_step=run.current_step,
            thread_id=run.thread_id,
            created_at=run.created_at.isoformat(),
            log_path=run.log_path,
            has_result=False,
            has_partial=False,
            questions_log=[],
            retrieval_mode=run.retrieval_mode,
        ).model_dump(),
        "view_url": f"/api/v1/v07/runs/{run.id}/view",
        "poll_url": f"/api/v1/v07/runs/{run.id}",
        "app_url": f"/app?run={run.id}",
    }


@router.get("")
def list_v07_runs(limit: int = 20) -> list[dict[str, Any]]:
    out = []
    for run in v07_run_store.list_recent(limit=limit):
        out.append(_run_response(run).model_dump())
    return out


@router.get("/{run_id}")
def get_v07_run(run_id: str) -> dict[str, Any]:
    run = v07_run_store.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "run": _run_response(run).model_dump(),
        "result": run.result if run.result is not None else None,
    }


@router.get("/{run_id}/view")
def get_v07_run_view(run_id: str) -> dict[str, Any]:
    run = v07_run_store.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if not run.result:
        return {
            "run_id": run_id,
            "status": run.status.value,
            "current_step": run.current_step,
            "ready": False,
            "partial": False,
            "sections": [],
            "toc": [],
            "retrieval_mode": run.retrieval_mode,
        }
    view = build_ui_view(run.result)
    completed = run.status == V07RunStatus.COMPLETED
    view["run_id"] = run_id
    view["status"] = run.status.value
    view["current_step"] = run.current_step
    view["ready"] = completed
    view["partial"] = not completed
    view["query"] = run.query
    view["retrieval_mode"] = (
        run.retrieval_mode or (run.result or {}).get("retrieval_mode") or "fast"
    )
    return view
