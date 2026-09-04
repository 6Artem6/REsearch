"""Постановка задач в очередь worker. API никогда не исполняет ML-jobs inline."""

from __future__ import annotations

import time
from typing import Any

from fastapi import HTTPException

from knowledge_engine.config import KE_WORKER_INLINE_FALLBACK
from knowledge_engine.services.work_job_store import (
    WorkJobKind,
    WorkJobStatus,
    work_job_store,
    worker_is_alive,
)
from knowledge_engine.ui.run_log import trace


def require_worker() -> None:
    if worker_is_alive():
        return
    if KE_WORKER_INLINE_FALLBACK:
        trace(
            "WARN: KE_WORKER_INLINE_FALLBACK ignored — vector/RAG jobs "
            "cannot run in the API process"
        )
    raise HTTPException(
        status_code=503,
        detail=(
            "Worker не запущен. Запустите: python -m knowledge_engine.worker "
            "или make dev (поднимает API + worker)."
        ),
    )


def require_worker_or_inline() -> None:
    """Compat alias: inline ML in API is disabled."""
    require_worker()


def _enqueue(kind: WorkJobKind, payload: dict) -> str:
    require_worker()
    job = work_job_store.create(kind, payload)
    return job.id


def wait_job_result(job_id: str, *, timeout_sec: float) -> dict[str, Any]:
    deadline = time.monotonic() + max(1.0, float(timeout_sec))
    last = None
    while time.monotonic() < deadline:
        job = work_job_store.get(job_id)
        last = job
        if job is None:
            raise HTTPException(status_code=404, detail=f"Work job not found: {job_id}")
        if job.status == WorkJobStatus.COMPLETED:
            return dict(job.result or {})
        if job.status == WorkJobStatus.FAILED:
            raise HTTPException(
                status_code=503,
                detail=job.error or "worker job failed",
            )
        time.sleep(0.05)
    detail = "worker timeout"
    if last is not None:
        detail = f"worker timeout (status={last.status.value})"
    raise HTTPException(status_code=504, detail=detail)


def enqueue_curriculum_generate(payload: dict) -> str:
    return _enqueue(WorkJobKind.CURRICULUM_GENERATE, payload)


def enqueue_curriculum_expand(payload: dict) -> str:
    return _enqueue(WorkJobKind.CURRICULUM_EXPAND, payload)


def enqueue_node_deep_dive(payload: dict) -> str:
    require_worker()
    cid = str(payload.get("curriculum_id") or "").strip()
    nid = str((payload.get("node_data") or {}).get("node_id") or "").strip()
    action = str(payload.get("user_action") or "init").strip().lower()
    if action == "init" and cid and nid:
        existing = work_job_store.find_active_node_deep_dive(
            cid, nid, user_action="init"
        )
        if existing:
            # Coalesce, but re-notify Redis worker: pub/sub is fire-and-forget,
            # so a pending job after restart / missed message would hang forever.
            if existing.status == WorkJobStatus.PENDING:
                try:
                    from knowledge_engine.services.redis_tasks import publish_work_job

                    publish_work_job(existing.id)
                    trace(
                        f"WORK enqueue ↻ republish pending init | {cid}/{nid} "
                        f"job={existing.id}"
                    )
                except Exception as exc:
                    trace(
                        f"WORK enqueue republish failed | {cid}/{nid} "
                        f"job={existing.id} | {exc}"
                    )
            else:
                trace(
                    f"WORK enqueue ⊘ duplicate node init | {cid}/{nid} "
                    f"→ existing job={existing.id} status={existing.status.value}"
                )
            return existing.id
    job = work_job_store.create(WorkJobKind.NODE_DEEP_DIVE, payload)
    return job.id


def enqueue_rag_gateway(payload: dict) -> str:
    return _enqueue(WorkJobKind.RAG_GATEWAY, payload)


def enqueue_node_explain(payload: dict) -> str:
    return _enqueue(WorkJobKind.NODE_EXPLAIN, payload)
