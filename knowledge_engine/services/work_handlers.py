"""Выполнение задач worker (Gemini / LangGraph / Skill Tree)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
from typing import Any

from knowledge_engine.services.analysis_service import (
    run_analysis_job,
    run_unravel_for_job,
)
from knowledge_engine.services.gemini_stateless import GeminiUnavailableError
from knowledge_engine.services.job_store import job_store
from knowledge_engine.services.v07_run_service import run_v07_job
from knowledge_engine.services.v07_run_store import v07_run_store
from knowledge_engine.services.work_job_store import WorkJob, WorkJobKind
from knowledge_engine.src.curriculum.generator import generate_curriculum_graph
from knowledge_engine.src.curriculum.schemas import CurriculumGenerateInput
from knowledge_engine.src.node_deep_dive.engine import (
    complete_node_init_gemini,
    prepare_node_init_rag,
    run_node_deep_dive,
)
from knowledge_engine.src.node_deep_dive.schemas import (
    NodeDataInput,
    NodeDeepDiveRequest,
)
from knowledge_engine.config import GEMINI_TUTOR_TIMEOUT_SEC, KE_RAG_TIMEOUT_SEC
from knowledge_engine.ui.errors import format_error_with_cause


def run_work_job(job: WorkJob) -> dict[str, Any]:
    if job.kind == WorkJobKind.CURRICULUM_GENERATE:
        return _run_curriculum_generate(job.payload)
    if job.kind == WorkJobKind.NODE_DEEP_DIVE:
        return _run_node_deep_dive(job.payload)
    raise ValueError(f"Unknown work job kind: {job.kind}")


def _run_curriculum_generate(payload: dict[str, Any]) -> dict[str, Any]:
    mode_raw = str(payload.get("generation_mode") or "fast").strip().lower()
    if mode_raw in ("deep", "consensus"):
        generation_mode = "consensus"
    else:
        generation_mode = "fast"
    inp = CurriculumGenerateInput(
        target_goal=str(payload.get("target_goal") or "").strip(),
        user_level=str(payload.get("user_level") or "Intermediate/Advanced").strip(),
        depth_level=str(payload.get("depth_level") or "Standard").strip(),
        generation_mode=generation_mode,
    )
    graph = generate_curriculum_graph(inp)
    from knowledge_engine.services.skill_tree_store import save_curriculum_record

    depth = inp.depth_level
    save_curriculum_record(
        graph,
        target_goal=inp.target_goal,
        generation_mode=inp.generation_mode,
        depth_level=depth,
        user_level=inp.user_level,
    )
    out = graph.model_dump()
    meta = dict(out.get("meta") or {})
    meta["generation_mode"] = inp.generation_mode
    out["meta"] = meta
    return out


def _run_node_deep_dive(payload: dict[str, Any]) -> dict[str, Any]:
    from knowledge_engine.ui.run_log import trace
    import time

    action = str(payload.get("user_action") or "init")
    cid = str(payload.get("curriculum_id") or "")
    nid = (payload.get("node_data") or {}).get("node_id", "")
    trace(f"WORKER node_deep_dive ▶ {action} | {cid}/{nid}")
    t_job = time.perf_counter()
    node_raw = payload.get("node_data") or {}
    req = NodeDeepDiveRequest(
        curriculum_id=str(payload.get("curriculum_id") or "").strip(),
        node_data=NodeDataInput.model_validate(node_raw),
        user_action=str(payload.get("user_action") or "init"),
        user_message=str(payload.get("user_message") or ""),
    )
    timeout = float(os.getenv("KE_NODE_DIVE_TIMEOUT_SEC", "900"))

    dive_async_timeout = float(
        os.getenv(
            "KE_NODE_DIVE_ASYNC_TIMEOUT_SEC",
            str(max(GEMINI_TUTOR_TIMEOUT_SEC * 3, 120)),
        )
    )

    async def _run_init_rag() -> tuple[str, str, Any, int, list[str]]:
        return await asyncio.wait_for(
            prepare_node_init_rag(req),
            timeout=KE_RAG_TIMEOUT_SEC,
        )

    async def _run_dive() -> Any:
        return await asyncio.wait_for(
            run_node_deep_dive(req),
            timeout=dive_async_timeout,
        )

    def _in_thread() -> dict[str, Any]:
        try:
            if req.user_action == "init":
                intro_payload, anchor, chat_mgr, rag_facts_count, rag_fact_labels = (
                    asyncio.run(_run_init_rag())
                )
                result = complete_node_init_gemini(
                    req,
                    intro_payload,
                    anchor,
                    chat_mgr,
                    rag_facts_count,
                    rag_fact_labels,
                )
                return result.model_dump()
            result = asyncio.run(_run_dive())
            return result.model_dump()
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"Node Deep-Dive async timeout "
                f"(RAG≤{KE_RAG_TIMEOUT_SEC:.0f}s / dive≤{dive_async_timeout:.0f}s)"
            ) from exc

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_in_thread)
        result = fut.result(timeout=timeout)
    trace(
        f"WORKER node_deep_dive ✓ {action} | {cid}/{nid} | "
        f"{time.perf_counter() - t_job:.1f}s"
    )
    return result


def process_pending_analysis_jobs() -> bool:
    """Один analysis / unravel / clarify из job_store."""
    job = job_store.claim_next_pending_work()
    if not job:
        return False
    try:
        if job.pending_unravel_option_id is not None:
            opt = job.pending_unravel_option_id
            job_store.update(job.id, pending_unravel_option_id=None)
            run_unravel_for_job(job.id, opt)
        else:
            clar = job.pending_clarify_answer
            if clar:
                job_store.update(job.id, pending_clarify_answer=None)
            run_analysis_job(job.id, clar)
    except Exception as exc:
        from knowledge_engine.services.job_store import JobStatus

        job_store.update(
            job.id,
            status=JobStatus.FAILED,
            error=format_error_with_cause(exc),
        )
    return True


def process_pending_v07_run() -> bool:
    run = v07_run_store.claim_next_pending()
    if not run:
        return False
    try:
        run_v07_job(run.id)
    except Exception:
        pass
    return True


def format_work_error(exc: BaseException) -> str:
    if isinstance(exc, (concurrent.futures.TimeoutError, TimeoutError)):
        return (
            f"Node Deep-Dive timeout "
            f"(outer={os.getenv('KE_NODE_DIVE_TIMEOUT_SEC', '900')}s, "
            f"RAG≤{KE_RAG_TIMEOUT_SEC:.0f}s). "
            "Проверьте Gemini RPM/RPD и что запущен один KE worker."
        )
    if isinstance(exc, GeminiUnavailableError):
        return str(exc)
    return format_error_with_cause(exc)
