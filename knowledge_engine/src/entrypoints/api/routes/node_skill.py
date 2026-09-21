"""Skill Tree — упрощённые эндпоинты ноды (Модуль 2)."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from knowledge_engine.src.config.settings import KE_NODE_DIVE_TIMEOUT_SEC
from knowledge_engine.src.core.run_log import trace
from knowledge_engine.src.domains.grounding.engine import complete_node_prepare_response
from knowledge_engine.src.domains.grounding.node_session_reset import (
    reset_node_deep_dive_persistence,
)
from knowledge_engine.src.domains.grounding.node_source_registry import (
    registry_for_curriculum_node,
)
from knowledge_engine.src.domains.grounding.schemas import (
    NodeDataInput,
    NodeDeepDiveRequest,
)
from knowledge_engine.src.domains.grounding.session_store import (
    get_all_sessions_for_curriculum,
    get_node_statuses_for_curriculum,
    get_session,
)
from knowledge_engine.src.entrypoints.api.helpers.work_enqueue import (
    enqueue_node_deep_dive,
    enqueue_node_explain,
    wait_job_result,
)
from knowledge_engine.src.processors.selection_prompts import (
    suggest_selection_questions,
)
from knowledge_engine.src.shared.job_queue.job_stream import iter_job_stream_events
from knowledge_engine.src.shared.job_queue.work_job_store import (
    WorkJobStatus,
    work_job_store,
)
from knowledge_engine.src.shared.node_grounding.node_gate_contracts import (
    NodeGate1ApprovePayload,
    NodeGate2ApprovePayload,
)

router = APIRouter(prefix="/node", tags=["skill-tree-node"])


class NodeSessionBody(BaseModel):
    curriculum_id: str = Field(min_length=3, max_length=80)
    node_data: NodeDataInput
    interaction_axis: str = Field(
        default="lecture_self_check",
        max_length=32,
        description="lecture_self_check | topic_qna — см. NodeDeepDiveRequest.",
    )


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
        "interaction_axis": body.interaction_axis,
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
            "interaction_axis": body.interaction_axis,
            "stream": True,
        }
        job_id = enqueue_node_deep_dive(payload)
        # Фронт по job_id раз в 30 с опрашивает /work-jobs/{id} (статус ноды).
        yield f"data: {json.dumps({'type': 'job', 'job_id': job_id})}\n\n"
        try:
            async for evt in iter_job_stream_events(
                job_id,
                timeout_sec=KE_NODE_DIVE_TIMEOUT_SEC,
            ):
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
        except Exception as exc:
            from knowledge_engine.src.core.errors import trace_exception

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
        "interaction_axis": body.interaction_axis,
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
            from knowledge_engine.src.core.errors import trace_exception

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


@router.post("/grounding-discover")
async def post_node_grounding_discover(body: NodeSessionBody) -> dict[str, Any]:
    """Node Grounding Gate — Этап 0+1 (add-only, см.

    docs/STEERING_AND_TOPIC_QNA_ROADMAP.md): единый поисковый профиль +
    сбор/дедуп/паспорта кандидатов для Gate 1 ОДНОЙ ноды. НЕ вызывается из
    ``/init``/``/init-stream`` и не меняет lazy-grounding в engine.py —
    автоматическая сшивка в живой поток открытия ноды (плюс Gate 2 и сам
    Map-Reduce инжест) — следующий шаг, не в этом эндпоинте. Safe/idempotent
    для любой ноды: BASE или уже прогруженная DEEP — просто пустой ответ,
    без Flash Lite вызовов."""
    from knowledge_engine.src.domains.curriculum.academic_consensus import (
        consensus_allowed_for_policy,
        is_sota_rd_node,
    )
    from knowledge_engine.src.domains.curriculum.schemas import CurriculumGraph
    from knowledge_engine.src.domains.curriculum.services.node_candidate_collection_service import (
        collect_node_candidates,
    )
    from knowledge_engine.src.domains.curriculum.source_policy import (
        resolve_source_policy,
    )
    from knowledge_engine.src.shared.node_grounding.node_gate_contracts import (
        NodeGate1DiscoveryResponse,
        NodeSearchProfile,
    )
    from knowledge_engine.src.shared.node_grounding.node_search_profile_service import (
        build_node_academic_plan,
        generate_node_search_profile,
    )
    from knowledge_engine.src.shared.skill_tree_store import (
        get_curriculum_graph,
        get_curriculum_meta,
    )

    raw = get_curriculum_graph(body.curriculum_id)
    if not raw:
        raise HTTPException(
            status_code=404, detail=f"curriculum {body.curriculum_id} not found"
        )
    graph = CurriculumGraph.model_validate(raw)
    node = next((n for n in graph.nodes if n.node_id == body.node_data.node_id), None)
    if not node:
        raise HTTPException(
            status_code=404, detail=f"node {body.node_data.node_id} not found"
        )

    node_status = (node.grounding_status or "").strip()
    if node.node_risk_kind != "DEEP" or (node_status == "grounded" and node.source_ref):
        trace(
            f"API ▶ POST /node/grounding-discover ⊘ | {body.curriculum_id}/"
            f"{node.node_id} | risk={node.node_risk_kind} status={node_status} "
            "— no gate needed"
        )
        return NodeGate1DiscoveryResponse(
            curriculum_id=body.curriculum_id,
            node_id=node.node_id,
            profile=NodeSearchProfile(),
            candidates=[],
        ).model_dump()

    meta = get_curriculum_meta(body.curriculum_id) or {}
    policy = resolve_source_policy(
        meta.get("source_policy"),
        str(meta.get("generation_mode") or "fast"),
        default="hybrid",
    )
    sota_override = policy == "practical_only" and is_sota_rd_node(node)
    effective_policy = "hybrid" if sota_override else policy
    include_academic = consensus_allowed_for_policy(effective_policy)

    trace(
        f"API ▶ POST /node/grounding-discover | {body.curriculum_id}/{node.node_id} "
        f"policy={policy} sota_override={sota_override} include_academic={include_academic}"
    )

    profile = await generate_node_search_profile(
        node_title=node.title,
        node_summary=node.brief_summary,
        core_concepts=node.core_concepts,
        include_academic=include_academic,
        curriculum_id=body.curriculum_id,
        node_id=node.node_id,
    )

    academic_plan = None
    if include_academic:
        # Реюз существующего Academic Query Architect (см. docs/
        # ACADEMIC_AND_CONSENSUS.md) — даёт academic_query_en + arxiv_params
        # вместе; профиль получает query для отображения на Gate 1, сбор —
        # оба поля для _primary_academic_hits/harvest_consensus_for_node.
        academic_plan = await build_node_academic_plan(
            node.title,
            node.brief_summary,
            anchor=f"node_gate_academic_plan:{body.curriculum_id}:{node.node_id}",
        )
        if (academic_plan.academic_query_en or "").strip():
            profile = profile.model_copy(
                update={"academic_queries": [academic_plan.academic_query_en]}
            )

    candidates = await collect_node_candidates(
        profile,
        curriculum_id=body.curriculum_id,
        node_id=node.node_id,
        node=node,
        academic_plan=academic_plan,
    )
    return NodeGate1DiscoveryResponse(
        curriculum_id=body.curriculum_id,
        node_id=node.node_id,
        profile=profile,
        candidates=candidates,
    ).model_dump()


@router.post("/grounding-digest")
async def post_node_grounding_digest(body: NodeGate1ApprovePayload) -> dict[str, Any]:
    """Node Grounding Gate — Этап 2 (add-only, см.

    docs/STEERING_AND_TOPIC_QNA_ROADMAP.md): по ≤NODE_GATE1_MAX_APPROVED
    URL, утверждённым на Gate 1 (``body.approved_urls``, схема переиспользует
    ``NodeGate1ApprovePayload`` — та же форма запроса), качает полный текст,
    BGE-M3+Cross-Encoder отбирают top-K смысловых блоков под тему ноды,
    Gemma cloud делает точечный дайджест (Архитектура/Практический
    кейс/Ограничения) — для показа на Gate 2. Не персистит ничего (ни в
    LanceDB/pgvector, ни в registry ноды) — это Этап 3, после Gate 2."""
    from knowledge_engine.src.domains.curriculum.schemas import CurriculumGraph
    from knowledge_engine.src.domains.curriculum.services.node_digest_service import (
        generate_node_digests,
    )
    from knowledge_engine.src.shared.skill_tree_store import get_curriculum_graph

    raw = get_curriculum_graph(body.curriculum_id)
    if not raw:
        raise HTTPException(
            status_code=404, detail=f"curriculum {body.curriculum_id} not found"
        )
    graph = CurriculumGraph.model_validate(raw)
    node = next((n for n in graph.nodes if n.node_id == body.node_id), None)
    if not node:
        raise HTTPException(status_code=404, detail=f"node {body.node_id} not found")

    trace(
        f"API ▶ POST /node/grounding-digest | {body.curriculum_id}/{node.node_id} "
        f"approved={len(body.approved_urls)}"
    )
    t_job = time.perf_counter()
    digests = await generate_node_digests(
        body.approved_urls,
        node_title=node.title,
        node_summary=node.brief_summary,
        curriculum_id=body.curriculum_id,
        node_id=node.node_id,
        anchor=f"node_gate_digest:{body.curriculum_id}:{node.node_id}",
    )
    trace(
        f"API ▶ POST /node/grounding-digest ✓ | {body.curriculum_id}/"
        f"{node.node_id} | {time.perf_counter() - t_job:.1f}s"
    )
    return digests.model_dump()


@router.post("/grounding-finalize")
async def post_node_grounding_finalize(body: NodeGate2ApprovePayload) -> dict[str, Any]:
    """Node Grounding Gate — Этап 3 (add-only, см.

    docs/STEERING_AND_TOPIC_QNA_ROADMAP.md): финальное согласие пользователя
    на Gate 2 (``body.approved_urls``) → реальный тяжёлый Map-Reduce инжест
    (ТОТ ЖЕ, что уже используется обычным lazy-grounding —
    ``summarize_whitelist_blog_hits_async``, ничего нового не изобретается)
    → узел помечается ``grounding_status="grounded"`` с реальными
    ``source_ref``/``mapped_source_ids``. После этого штатный
    ``POST /node/init`` для этой ноды НЕ запускает поиск повторно —
    существующая проверка в ``_apply_lazy_grounding_for_init`` (engine.py,
    не изменялась) видит ``grounded``+``source_ref`` и делает только
    refresh диаграмм."""
    from knowledge_engine.src.domains.curriculum.schemas import CurriculumGraph
    from knowledge_engine.src.domains.curriculum.source_policy import (
        resolve_source_policy,
    )
    from knowledge_engine.src.shared.node_grounding.node_grounding_finalize_service import (
        finalize_node_grounding,
    )
    from knowledge_engine.src.shared.skill_tree_store import (
        get_curriculum_graph,
        get_curriculum_meta,
        save_curriculum_record,
    )

    raw = get_curriculum_graph(body.curriculum_id)
    if not raw:
        raise HTTPException(
            status_code=404, detail=f"curriculum {body.curriculum_id} not found"
        )
    graph = CurriculumGraph.model_validate(raw)
    node = next((n for n in graph.nodes if n.node_id == body.node_id), None)
    if not node:
        raise HTTPException(status_code=404, detail=f"node {body.node_id} not found")

    meta = get_curriculum_meta(body.curriculum_id) or {}
    target_goal = str(meta.get("target_goal") or graph.description or "").strip()
    generation_mode = str(meta.get("generation_mode") or "fast")
    source_policy = resolve_source_policy(
        meta.get("source_policy"), generation_mode, default="hybrid"
    )

    trace(
        f"API ▶ POST /node/grounding-finalize | {body.curriculum_id}/{body.node_id} "
        f"approved={len(body.approved_urls)}"
    )
    t_job = time.perf_counter()
    graph, updated_node = await finalize_node_grounding(
        graph, node, body.approved_urls, target_goal=target_goal
    )
    trace(
        f"API ▶ POST /node/grounding-finalize ✓ | {body.curriculum_id}/"
        f"{body.node_id} | {time.perf_counter() - t_job:.1f}s"
    )
    save_curriculum_record(
        graph,
        target_goal=target_goal,
        generation_mode=generation_mode,
        depth_level=str(meta.get("depth_level") or "Standard"),
        user_level=str(meta.get("user_level") or "Intermediate/Advanced"),
        source_policy=source_policy,
    )
    return {
        "curriculum_id": body.curriculum_id,
        "node_id": updated_node.node_id,
        "grounding_status": updated_node.grounding_status,
        "mapped_source_ids": list(updated_node.mapped_source_ids or []),
        "resource_urls": list(updated_node.resource_urls or []),
    }


@router.post("/ensure-steering-sources")
async def post_node_ensure_steering_sources(body: NodeSessionBody) -> dict[str, Any]:
    """При первом открытии Штурвал-сгенерированной ноды — реальный ingest

    примапленных источников вместо дешёвого Gate-2 дайджеста (см.
    docs/STEERING_AND_TOPIC_QNA_ROADMAP.md — расхождение между названием
    ``WorkJobKind.STEERING_MAP_REDUCE``/"тяжёлый Map-Reduce" в докстрингах
    и фактическим отсутствием Map-Reduce в
    ``generate_curriculum_from_steering_digests``, подтверждено
    ``log_profiler.py`` на реальном прогоне). Instant no-op для любой
    ноды без источников ``source_tier="steering_approved"`` — то есть для
    подавляющего большинства нод (Autopilot, уже доингещенные Штурвал-
    ноды) это мгновенный пустой ответ без единого сетевого вызова.
    Вызывается ФРОНТЕНДОМ перед ``POST /node/init`` — тот же паттерн, что
    уже используется для Node Grounding Gate."""
    from knowledge_engine.src.domains.curriculum.schemas import CurriculumGraph
    from knowledge_engine.src.domains.steering.services.steering_lazy_ingest_service import (
        ensure_node_steering_sources_ingested,
    )
    from knowledge_engine.src.shared.skill_tree_store import (
        get_curriculum_graph,
        get_curriculum_meta,
        save_curriculum_record,
    )

    raw = get_curriculum_graph(body.curriculum_id)
    if not raw:
        raise HTTPException(
            status_code=404, detail=f"curriculum {body.curriculum_id} not found"
        )
    graph = CurriculumGraph.model_validate(raw)
    node = next((n for n in graph.nodes if n.node_id == body.node_data.node_id), None)
    if not node:
        raise HTTPException(
            status_code=404, detail=f"node {body.node_data.node_id} not found"
        )

    meta = get_curriculum_meta(body.curriculum_id) or {}
    target_goal = str(meta.get("target_goal") or graph.description or "").strip()

    t_job = time.perf_counter()
    updated_graph, updated_node = await ensure_node_steering_sources_ingested(
        graph, node, target_goal=target_goal
    )
    if updated_node is node:
        return {"ingested": False}

    trace(
        f"API ▶ POST /node/ensure-steering-sources ✓ | {body.curriculum_id}/"
        f"{node.node_id} | {time.perf_counter() - t_job:.1f}s"
    )
    save_curriculum_record(
        updated_graph,
        target_goal=target_goal,
        generation_mode=str(meta.get("generation_mode") or "fast"),
        depth_level=str(meta.get("depth_level") or "Standard"),
        user_level=str(meta.get("user_level") or "Intermediate/Advanced"),
        source_policy=str(meta.get("source_policy") or "hybrid"),
    )
    return {
        "ingested": True,
        "node_id": updated_node.node_id,
        "grounding_status": updated_node.grounding_status,
    }
