"""Skill Tree — упрощённые эндпоинты ноды (Модуль 2)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from knowledge_engine.api.helpers.work_enqueue import (
    enqueue_node_deep_dive,
    enqueue_node_explain,
    wait_job_result,
)
from knowledge_engine.config import KE_NODE_DIVE_TIMEOUT_SEC
from knowledge_engine.services.job_stream import iter_job_stream_events
from knowledge_engine.services.node_session_reset import (
    reset_node_deep_dive_persistence,
)
from knowledge_engine.services.node_source_registry import registry_for_curriculum_node
from knowledge_engine.services.work_job_store import WorkJobStatus, work_job_store
from knowledge_engine.src.node_deep_dive.engine import complete_node_prepare_response
from knowledge_engine.src.node_deep_dive.schemas import (
    NodeDataInput,
    NodeDeepDiveRequest,
)
from knowledge_engine.src.node_deep_dive.session_store import (
    get_all_sessions_for_curriculum,
    get_node_statuses_for_curriculum,
    get_session,
)
from knowledge_engine.src.processors.selection_prompts import (
    suggest_selection_questions,
)
from knowledge_engine.ui.run_log import trace

router = APIRouter(prefix="/node", tags=["skill-tree-node"])


class NodeSessionBody(BaseModel):
    curriculum_id: str = Field(min_length=3, max_length=80)
    node_data: NodeDataInput


class NodeChatBody(NodeSessionBody):
    user_message: str = Field(min_length=1, max_length=8000)


class NodeSelectionBody(NodeSessionBody):
    selected_text: str = Field(min_length=2, max_length=8000)
    surrounding_paragraph: str = Field(default="", max_length=12000)
    user_question: str = Field(default="", max_length=2000)


class NodeJobAccepted(BaseModel):
    job_id: str
    status: str


def _session_init_ready(curriculum_id: str, node_id: str) -> bool:
    """True when init prepare already persisted memory for this node."""
    session = get_session(curriculum_id, node_id)
    return session.memory is not None


def _build_init_result_from_session(
    curriculum_id: str,
    node_data: NodeDataInput,
) -> dict[str, Any] | None:
    """Rebuild NodeDeepDiveResponse from a prepared session (no new worker job)."""
    cid = curriculum_id.strip()
    nid = node_data.node_id.strip()
    if not _session_init_ready(cid, nid):
        return None
    blob = get_all_sessions_for_curriculum(cid).get(nid) or {}
    labels = [str(x) for x in (blob.get("rag_fact_labels") or []) if str(x).strip()]
    rag_count = len(labels)
    req = NodeDeepDiveRequest(
        curriculum_id=cid,
        node_data=node_data,
        user_action="init",
        user_message="",
    )
    resp = asyncio.run(complete_node_prepare_response(req, rag_count, labels))
    return resp.model_dump()


def _resolve_ready_init_result(
    curriculum_id: str,
    node_data: NodeDataInput,
) -> dict[str, Any] | None:
    """
    Immediate init payload when work already finished:
    1) session memory from a prior init
    2) else latest completed init job.result (non-orphan)
    Also completes any active orphan init job so waiters/duplicates unlock.
    """
    cid = curriculum_id.strip()
    nid = node_data.node_id.strip()
    result = _build_init_result_from_session(cid, node_data)
    if result is None:
        done = work_job_store.find_latest_completed_node_deep_dive(
            cid, nid, user_action="init"
        )
        if done and isinstance(done.result, dict) and done.result:
            # Skip synthetic results from cancel_work_job --complete orphans.
            if not done.result.get("closed_orphan_job"):
                result = dict(done.result)

    if result is None:
        return None

    active = work_job_store.find_active_node_deep_dive(cid, nid, user_action="init")
    if active is not None:
        work_job_store.complete(active.id, result)
        trace(
            f"WORK init ready → complete orphan | {cid}/{nid} "
            f"job={active.id} was={active.status.value}"
        )
    return result


def _enqueue_and_maybe_inline(
    action: str,
    body: NodeSessionBody,
    user_message: str = "",
) -> dict[str, Any]:
    payload = {
        "curriculum_id": body.curriculum_id.strip(),
        "node_data": body.node_data.model_dump(),
        "user_action": action,
        "user_message": user_message,
    }
    job_id = enqueue_node_deep_dive(payload)
    job = work_job_store.get(job_id)
    if job and job.status == WorkJobStatus.COMPLETED and job.result:
        return job.result
    if job and job.status == WorkJobStatus.FAILED:
        raise HTTPException(status_code=503, detail=job.error or "node job failed")
    return {"job_id": job_id, "status": "pending"}


@router.post("/init")
def post_node_init(body: NodeSessionBody, response: Response) -> dict[str, Any]:
    cid = body.curriculum_id.strip()
    nid = body.node_data.node_id.strip()
    ready = _resolve_ready_init_result(cid, body.node_data)
    if ready is not None:
        trace(f"API ▶ POST /node/init (ready) | {cid}/{nid}")
        response.status_code = status.HTTP_200_OK
        return ready
    trace(f"API ▶ POST /node/init (queue) | {cid}/{nid}")
    out = _enqueue_and_maybe_inline("init", body)
    if "job_id" in out:
        response.status_code = status.HTTP_202_ACCEPTED
        return out
    response.status_code = status.HTTP_200_OK
    return out


@router.post("/init-stream")
async def post_node_init_stream(body: NodeSessionBody) -> StreamingResponse:
    """Тот же SSE-паттерн, что /chat-stream (job_stream.py relay), но для
    подготовки ноды (action=init) — по требованию пользователя FSM-стадии
    (см. schemas/fsm.py) должны стримиться и на "подготовку ноды", не только
    на ответ тьютора. _run_node_deep_dive_stream в work_handlers.py уже
    action-агностичен (просто прокидывает user_action из payload) — никаких
    изменений в воркере не требуется, только этот роут."""
    cid = body.curriculum_id.strip()
    nid = body.node_data.node_id.strip()
    # Тот же fast-path, что /node/init (_resolve_ready_init_result) — без
    # него КАЖДОЕ повторное открытие уже проинициализированной ноды шло бы
    # через полный enqueue+worker цикл вместо мгновенного ответа.
    ready = _resolve_ready_init_result(cid, body.node_data)

    async def event_stream():
        if ready is not None:
            trace(f"API ▶ POST /node/init-stream (ready) | {cid}/{nid}")
            yield f"data: {json.dumps({'type': 'complete', 'result': ready}, ensure_ascii=False)}\n\n"
            return
        trace(f"API ▶ POST /node/init-stream (queue SSE) | {cid}/{nid}")
        payload = {
            "curriculum_id": cid,
            "node_data": body.node_data.model_dump(),
            "user_action": "init",
            "user_message": "",
            "stream": True,
        }
        job_id = enqueue_node_deep_dive(payload)
        try:
            async for evt in iter_job_stream_events(
                job_id,
                timeout_sec=KE_NODE_DIVE_TIMEOUT_SEC,
            ):
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
        except Exception as exc:
            from knowledge_engine.ui.errors import trace_exception

            detail = trace_exception(exc, "NODE_DIVE init-stream proxy")
            err = {
                "type": "error",
                "detail": detail,
                "error_type": type(exc).__name__,
            }
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/restart", status_code=status.HTTP_202_ACCEPTED)
def post_node_restart(body: NodeSessionBody) -> dict[str, Any]:
    """
    Сброс прогресса и материалов ноды + повторный init (RAG, memory, registry).
    """
    cid = body.curriculum_id.strip()
    nid = body.node_data.node_id.strip()
    trace(f"API ▶ POST /node/restart (queue) | {cid}/{nid}")
    reset_node_deep_dive_persistence(cid, nid)
    out = _enqueue_and_maybe_inline("init", body)
    if "job_id" in out:
        return out
    return out


@router.post("/chat", status_code=status.HTTP_202_ACCEPTED)
def post_node_chat(body: NodeChatBody) -> dict[str, Any]:
    trace(
        f"API ▶ POST /node/chat (queue) | {body.curriculum_id}/{body.node_data.node_id}"
    )
    out = _enqueue_and_maybe_inline("chat", body, body.user_message.strip())
    if "job_id" in out:
        return out
    return out


@router.post("/chat-stream")
async def post_node_chat_stream(body: NodeChatBody) -> StreamingResponse:
    trace(
        f"API ▶ POST /node/chat-stream (queue SSE) | "
        f"{body.curriculum_id}/{body.node_data.node_id}"
    )
    payload = {
        "curriculum_id": body.curriculum_id.strip(),
        "node_data": body.node_data.model_dump(),
        "user_action": "chat",
        "user_message": body.user_message.strip(),
        "stream": True,
    }
    job_id = enqueue_node_deep_dive(payload)

    async def event_stream():
        try:
            async for evt in iter_job_stream_events(
                job_id,
                timeout_sec=KE_NODE_DIVE_TIMEOUT_SEC,
            ):
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
        except Exception as exc:
            from knowledge_engine.ui.errors import trace_exception

            detail = trace_exception(exc, "NODE_DIVE chat-stream proxy")
            err = {
                "type": "error",
                "detail": detail,
                "error_type": type(exc).__name__,
            }
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/verify", status_code=status.HTTP_202_ACCEPTED)
def post_node_verify(body: NodeChatBody) -> dict[str, Any]:
    trace(
        f"API ▶ POST /node/verify (queue) | {body.curriculum_id}/{body.node_data.node_id}"
    )
    out = _enqueue_and_maybe_inline("verify", body, body.user_message.strip())
    if "job_id" in out:
        return out
    return out


@router.get("/statuses/{curriculum_id}")
def get_node_statuses(curriculum_id: str) -> dict[str, Any]:
    statuses = get_node_statuses_for_curriculum(curriculum_id.strip())
    return {"curriculum_id": curriculum_id, "statuses": statuses}


@router.get("/source-registry/{curriculum_id}/{node_id}")
def get_node_source_registry(curriculum_id: str, node_id: str) -> dict[str, Any]:
    """Реестр [Sx] строго из текущих mapped_source_ids (без stale session JSON)."""
    cid = curriculum_id.strip()
    nid = node_id.strip()
    registry = _node_source_registry(cid, nid)
    return {
        "curriculum_id": cid,
        "node_id": nid,
        "source_registry": registry,
    }


def _node_source_registry(curriculum_id: str, node_id: str) -> list[dict[str, Any]]:
    return registry_for_curriculum_node(curriculum_id, node_id)


@router.post("/suggest-questions")
async def post_node_suggest_questions(body: NodeSelectionBody) -> dict[str, Any]:
    trace(
        f"API ▶ POST /node/suggest-questions | "
        f"{body.curriculum_id}/{body.node_data.node_id}"
    )
    topic = (body.node_data.title or "").strip()
    result = await suggest_selection_questions(
        body.selected_text,
        body.surrounding_paragraph,
        topic,
    )
    return result.model_dump()


@router.post("/explain-selection")
def post_node_explain_selection(body: NodeSelectionBody) -> dict[str, Any]:
    trace(
        f"API ▶ POST /node/explain-selection (queue) | "
        f"{body.curriculum_id}/{body.node_data.node_id}"
    )
    payload = {
        "curriculum_id": body.curriculum_id.strip(),
        "node_data": body.node_data.model_dump(),
        "selected_text": body.selected_text,
        "surrounding_paragraph": body.surrounding_paragraph,
        "user_question": body.user_question or "",
        "stream": False,
    }
    job_id = enqueue_node_explain(payload)
    return wait_job_result(job_id, timeout_sec=KE_NODE_DIVE_TIMEOUT_SEC)


@router.post("/explain-selection-stream")
async def post_node_explain_selection_stream(
    body: NodeSelectionBody,
) -> StreamingResponse:
    trace(
        f"API ▶ POST /node/explain-selection-stream (queue SSE) | "
        f"{body.curriculum_id}/{body.node_data.node_id}"
    )
    payload = {
        "curriculum_id": body.curriculum_id.strip(),
        "node_data": body.node_data.model_dump(),
        "selected_text": body.selected_text,
        "surrounding_paragraph": body.surrounding_paragraph,
        "user_question": body.user_question or "",
        "stream": True,
    }
    job_id = enqueue_node_explain(payload)

    async def event_stream():
        try:
            async for evt in iter_job_stream_events(
                job_id,
                timeout_sec=KE_NODE_DIVE_TIMEOUT_SEC,
            ):
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
        except Exception as exc:
            err = {"type": "error", "detail": str(exc)}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
