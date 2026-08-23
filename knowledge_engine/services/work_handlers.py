"""Выполнение задач worker (Gemini / LangGraph / Skill Tree)."""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any

from knowledge_engine.config import (
    KE_NODE_DIVE_ASYNC_TIMEOUT_SEC,
    KE_NODE_DIVE_INIT_ASYNC_TIMEOUT_SEC,
    KE_NODE_DIVE_TIMEOUT_SEC,
)
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
from knowledge_engine.src.node_deep_dive.engine import run_node_deep_dive
from knowledge_engine.src.node_deep_dive.schemas import (
    NodeDataInput,
    NodeDeepDiveRequest,
)
from knowledge_engine.ui.errors import format_error_with_cause


def run_work_job(job: WorkJob) -> dict[str, Any]:
    if job.kind == WorkJobKind.CURRICULUM_GENERATE:
        return _run_curriculum_generate(job.payload)
    if job.kind == WorkJobKind.CURRICULUM_EXPAND:
        return _run_curriculum_expand(job.payload)
    if job.kind == WorkJobKind.NODE_DEEP_DIVE:
        if job.payload.get("stream"):
            return _run_node_deep_dive_stream(job)
        return _run_node_deep_dive(job.payload)
    if job.kind == WorkJobKind.RAG_GATEWAY:
        return _run_rag_gateway(job.payload)
    if job.kind == WorkJobKind.NODE_EXPLAIN:
        if job.payload.get("stream"):
            from knowledge_engine.services.node_explain_job import (
                run_node_explain_stream_job,
            )

            return run_node_explain_stream_job(job.id, job.payload)
        from knowledge_engine.services.node_explain_job import run_node_explain_job

        return run_node_explain_job(job.payload)
    raise ValueError(f"Unknown work job kind: {job.kind}")


def _run_curriculum_generate(payload: dict[str, Any]) -> dict[str, Any]:
    from knowledge_engine.src.curriculum.source_policy import (
        depth_for_source_policy,
        resolve_source_policy,
    )
    from knowledge_engine.ui.run_log import get_run_log_path, init_run_log, trace

    goal_preview = str(payload.get("target_goal") or "")[:56]
    init_run_log(f"curriculum generate | {goal_preview}")
    trace(f"WORKER curriculum generate | log={get_run_log_path()}")

    mode_raw = str(payload.get("generation_mode") or "fast").strip().lower()
    if mode_raw in ("deep", "consensus"):
        generation_mode = "consensus"
    else:
        generation_mode = "fast"
    source_policy = resolve_source_policy(
        payload.get("source_policy"),
        generation_mode,
        default="practical_only",
    )
    depth = str(payload.get("depth_level") or "").strip()
    if not depth:
        depth = depth_for_source_policy(source_policy)
    inp = CurriculumGenerateInput(
        target_goal=str(payload.get("target_goal") or "").strip(),
        user_level=str(payload.get("user_level") or "Intermediate/Advanced").strip(),
        depth_level=depth,
        generation_mode=generation_mode,
        source_policy=source_policy,
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
        source_policy=inp.source_policy,
    )
    out = graph.model_dump()
    meta = dict(out.get("meta") or {})
    meta["generation_mode"] = inp.generation_mode
    meta["source_policy"] = inp.source_policy
    log_path = get_run_log_path()
    if log_path is not None:
        meta["run_log_path"] = str(log_path)
    out["meta"] = meta
    return out


def _run_curriculum_expand(payload: dict[str, Any]) -> dict[str, Any]:
    from knowledge_engine.services.curriculum_service import expand_curriculum

    cid = str(payload.get("curriculum_id") or "").strip()
    prompt = str(payload.get("expansion_prompt") or "").strip()
    from knowledge_engine.src.curriculum.source_policy import resolve_source_policy

    mode_raw = str(payload.get("generation_mode") or "fast").strip().lower()
    generation_mode = "consensus" if mode_raw in ("deep", "consensus") else "fast"
    source_policy = resolve_source_policy(
        payload.get("source_policy"),
        generation_mode,
        default="practical_only",
    )
    graph = expand_curriculum(
        cid,
        prompt,
        generation_mode=generation_mode,
        source_policy=source_policy,
    )
    return graph.model_dump()


def _run_node_deep_dive(payload: dict[str, Any]) -> dict[str, Any]:
    import time

    from knowledge_engine.ui.run_log import trace

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
    dive_async_timeout = (
        KE_NODE_DIVE_INIT_ASYNC_TIMEOUT_SEC
        if action == "init"
        else KE_NODE_DIVE_ASYNC_TIMEOUT_SEC
    )

    async def _run_dive() -> Any:
        if action == "init":
            trace(
                f"NODE_DIVE worker init ▶ | no asyncio job timeout "
                f"(grounding budget≤{dive_async_timeout:.0f}s in graph)"
            )
            return await run_node_deep_dive(req)
        try:
            return await asyncio.wait_for(
                run_node_deep_dive(req),
                timeout=dive_async_timeout,
            )
        except asyncio.TimeoutError:
            try:
                from knowledge_engine.src.retrieval.consensus_session import (
                    shutdown_shared_consensus_session,
                )

                await shutdown_shared_consensus_session()
            except Exception as exc:
                trace(f"Consensus shutdown after dive timeout ⊘ | {exc}")
            raise

    def _in_thread() -> dict[str, Any]:
        try:
            result = asyncio.run(_run_dive())
            return result.model_dump()
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"Node Deep-Dive async timeout (dive≤{dive_async_timeout:.0f}s)"
            ) from exc
        except Exception as exc:
            from knowledge_engine.ui.errors import trace_exception

            trace_exception(exc, "NODE_DIVE worker")
            raise

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_in_thread)
        if action == "init":
            result = fut.result()
        else:
            result = fut.result(timeout=KE_NODE_DIVE_TIMEOUT_SEC)
    trace(
        f"WORKER node_deep_dive ✓ {action} | {cid}/{nid} | "
        f"{time.perf_counter() - t_job:.1f}s"
    )
    return result


def _run_node_deep_dive_stream(job: WorkJob) -> dict[str, Any]:
    import time

    from knowledge_engine.services.job_stream import append_job_stream_event
    from knowledge_engine.src.node_deep_dive.engine import (
        iter_node_deep_dive_chat_stream,
    )
    from knowledge_engine.ui.run_log import trace

    payload = job.payload
    action = str(payload.get("user_action") or "chat")
    cid = str(payload.get("curriculum_id") or "")
    nid = (payload.get("node_data") or {}).get("node_id", "")
    trace(f"WORKER node_deep_dive stream ▶ {action} | {cid}/{nid} job={job.id}")
    t_job = time.perf_counter()
    node_raw = payload.get("node_data") or {}
    req = NodeDeepDiveRequest(
        curriculum_id=str(payload.get("curriculum_id") or "").strip(),
        node_data=NodeDataInput.model_validate(node_raw),
        user_action=str(payload.get("user_action") or "chat"),
        user_message=str(payload.get("user_message") or ""),
    )

    async def _run() -> dict[str, Any]:
        result: dict[str, Any] = {}
        async for evt in iter_node_deep_dive_chat_stream(req):
            append_job_stream_event(job.id, evt)
            if evt.get("type") == "complete" and isinstance(evt.get("result"), dict):
                result = evt["result"]
            if evt.get("type") == "error":
                raise RuntimeError(str(evt.get("detail") or "chat stream error"))
        return result

    try:
        out = asyncio.run(
            asyncio.wait_for(_run(), timeout=KE_NODE_DIVE_ASYNC_TIMEOUT_SEC)
        )
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"Node Deep-Dive async timeout (dive≤{KE_NODE_DIVE_ASYNC_TIMEOUT_SEC:.0f}s)"
        ) from exc
    trace(
        f"WORKER node_deep_dive stream ✓ {action} | {cid}/{nid} | "
        f"{time.perf_counter() - t_job:.1f}s"
    )
    return out


def _run_rag_gateway(payload: dict[str, Any]) -> dict[str, Any]:
    from knowledge_engine.src.rag_gateway.gateway import (
        query_directional_rag,
        save_user_fact_request,
    )
    from knowledge_engine.src.rag_gateway.schemas import (
        DirectionalRAGQuery,
        SaveUserFactRequest,
    )

    op = str(payload.get("op") or "").strip().lower()
    body = payload.get("body") or {}
    if op == "query":
        req = DirectionalRAGQuery.model_validate(body)
        result = asyncio.run(query_directional_rag(req))
        return result.model_dump()
    if op == "facts":
        req = SaveUserFactRequest.model_validate(body)
        n = asyncio.run(save_user_fact_request(req))
        return {"indexed": n, "node_id": req.node_id}
    raise ValueError(f"Unknown rag_gateway op: {op}")


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
        init_hint = ""
        if isinstance(exc, TimeoutError) and "Node Deep-Dive async timeout" in str(exc):
            init_hint = (
                " (init grounding — увеличьте KE_NODE_DIVE_INIT_ASYNC_TIMEOUT_SEC)"
            )
        return (
            f"Node Deep-Dive timeout "
            f"(outer={KE_NODE_DIVE_TIMEOUT_SEC:.0f}s, "
            f"async≤{KE_NODE_DIVE_ASYNC_TIMEOUT_SEC:.0f}s){init_hint}. "
            "Проверьте Gemini RPM/RPD и что запущен один KE worker."
        )
    if isinstance(exc, GeminiUnavailableError):
        return str(exc)
    return format_error_with_cause(exc)
