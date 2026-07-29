"""Skill Tree — упрощённые эндпоинты ноды (Модуль 2)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from knowledge_engine.api.helpers.work_enqueue import enqueue_node_deep_dive
from knowledge_engine.services.gemini_stateless import GeminiUnavailableError
from knowledge_engine.services.llm_markdown_service import llm_markdown_to_html
from knowledge_engine.services.node_selection_explain import run_node_selection_explain
from knowledge_engine.services.node_source_registry import build_registry_from_references
from knowledge_engine.services.work_job_store import WorkJobStatus, work_job_store
from knowledge_engine.src.node_deep_dive.schemas import (
    NodeDataInput,
    NodeDeepDiveResponse,
)
from knowledge_engine.src.node_deep_dive.session_store import (
    _load_all,
    _session_key,
    get_node_statuses_for_curriculum,
    get_session,
)
from knowledge_engine.src.processors.explainer import DEFAULT_EXPLAIN_QUESTION
from knowledge_engine.src.processors.selection_prompts import suggest_selection_questions
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


@router.post("/init", status_code=status.HTTP_202_ACCEPTED)
def post_node_init(body: NodeSessionBody) -> dict[str, Any]:
    trace(f"API ▶ POST /node/init (queue) | {body.curriculum_id}/{body.node_data.node_id}")
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


def _node_source_registry(curriculum_id: str, node_id: str) -> list[dict[str, Any]]:
    key = _session_key(curriculum_id.strip(), node_id.strip())
    blob = _load_all().get(key) or {}
    registry = list(blob.get("source_registry") or [])
    if registry:
        return registry
    session = get_session(curriculum_id, node_id)
    return build_registry_from_references(session.content.references)


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
        f"API ▶ POST /node/explain-selection | "
        f"{body.curriculum_id}/{body.node_data.node_id}"
    )
    session = get_session(body.curriculum_id.strip(), body.node_data.node_id)
    registry = _node_source_registry(body.curriculum_id, body.node_data.node_id)
    memory = session.memory
    rag_profile = (memory.rag_profile_compressed or "") if memory else ""
    anchor = f"node_deep_dive:{body.curriculum_id}:{body.node_data.node_id}"
    try:
        result = run_node_selection_explain(
            body.node_data.title,
            body.selected_text,
            body.user_question,
            body.surrounding_paragraph,
            session.content.summary,
            rag_profile,
            registry,
            anchor,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GeminiUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    explanation_html = llm_markdown_to_html(result.explanation, registry)
    return {
        "explanation": result.explanation,
        "explanation_html": explanation_html,
        "source_ref": {
            "title": result.source_ref.title,
            "url": result.source_ref.url,
            "source_id": result.source_ref.source_id,
        },
        "default_question": DEFAULT_EXPLAIN_QUESTION,
    }
