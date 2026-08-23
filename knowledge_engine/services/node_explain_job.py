"""Worker-side node selection explain (vector RAG + Gemini)."""

from __future__ import annotations

import asyncio
from typing import Any

from knowledge_engine.services.job_stream import append_job_stream_event
from knowledge_engine.services.node_selection_explain import (
    explain_result_to_api_dict,
    iter_node_selection_explain_stream,
    run_node_selection_explain,
)
from knowledge_engine.services.node_source_registry import registry_for_curriculum_node
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput
from knowledge_engine.src.node_deep_dive.session_store import get_session
from knowledge_engine.ui.run_log import trace


def _explain_context(payload: dict[str, Any]) -> dict[str, Any]:
    cid = str(payload.get("curriculum_id") or "").strip()
    node = NodeDataInput.model_validate(payload.get("node_data") or {})
    session = get_session(cid, node.node_id)
    memory = session.memory
    registry = registry_for_curriculum_node(cid, node.node_id)
    rag_profile = (memory.rag_profile_compressed or "") if memory else ""
    return {
        "curriculum_id": cid,
        "node": node,
        "session": session,
        "memory": memory,
        "registry": registry,
        "rag_profile": rag_profile,
        "anchor": f"node_deep_dive:{cid}:{node.node_id}",
        "selected_text": str(payload.get("selected_text") or ""),
        "surrounding_paragraph": str(payload.get("surrounding_paragraph") or ""),
        "user_question": str(payload.get("user_question") or ""),
    }


def run_node_explain_job(payload: dict[str, Any]) -> dict[str, Any]:
    ctx = _explain_context(payload)
    node = ctx["node"]
    result = run_node_selection_explain(
        node.title,
        ctx["selected_text"],
        ctx["user_question"],
        ctx["surrounding_paragraph"],
        ctx["session"].content.summary,
        ctx["rag_profile"],
        ctx["registry"],
        ctx["anchor"],
        memory=ctx["memory"],
        curriculum_id=ctx["curriculum_id"],
        node=node,
    )
    return explain_result_to_api_dict(result, ctx["registry"])


def run_node_explain_stream_job(job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    ctx = _explain_context(payload)
    node = ctx["node"]

    async def _run() -> dict[str, Any]:
        result: dict[str, Any] = {}
        async for evt in iter_node_selection_explain_stream(
            node.title,
            ctx["selected_text"],
            ctx["user_question"],
            ctx["surrounding_paragraph"],
            ctx["session"].content.summary,
            ctx["rag_profile"],
            ctx["registry"],
            ctx["anchor"],
            memory=ctx["memory"],
            curriculum_id=ctx["curriculum_id"],
            node=node,
        ):
            append_job_stream_event(job_id, evt)
            if evt.get("type") == "complete" and isinstance(evt.get("result"), dict):
                result = evt["result"]
            if evt.get("type") == "error":
                raise RuntimeError(str(evt.get("detail") or "explain stream error"))
        return result

    trace(f"WORKER node_explain stream ▶ job={job_id}")
    return asyncio.run(_run())
