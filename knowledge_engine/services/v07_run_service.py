"""Background execution of v0.7 LangGraph for web UI."""

from __future__ import annotations

import asyncio

from knowledge_engine.config import PACKAGE_ROOT
from knowledge_engine.services.v07_run_store import V07RunStatus, v07_run_store
from knowledge_engine.src.agent.local_orchestrator import run_knowledge_engine_v08
from knowledge_engine.src.graph import run_knowledge_engine_v07
from knowledge_engine.ui.logger import live_session, print_timing_summary, set_status
from knowledge_engine.ui.run_log import init_run_log, trace


def _read_profile() -> str:
    path = PACKAGE_ROOT / "user_profile.md"
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return ""


def run_v07_job(run_id: str) -> None:
    run = v07_run_store.get(run_id)
    if not run:
        return

    from knowledge_engine.config import get_graph_version

    v08 = get_graph_version() == "0.8"

    v07_run_store.update(run_id, status=V07RunStatus.RUNNING, current_step="starting")
    log_path = init_run_log(run.query)
    v07_run_store.update(run_id, log_path=str(log_path))

    try:
        with live_session():
            set_status(
                "v0.7 web run…" if not v08 else f"v0.8 {run.retrieval_mode} run…"
            )
            if v08:
                result = asyncio.run(
                    run_knowledge_engine_v08(
                        run.query,
                        _read_profile(),
                        run.thread_id,
                        web_run_id=run_id,
                        retrieval_mode=run.retrieval_mode,
                    )
                )
            else:
                result = asyncio.run(
                    run_knowledge_engine_v07(
                        run.query,
                        _read_profile(),
                        run.thread_id,
                    )
                )
            print_timing_summary()

        state_dict = dict(result)
        v07_run_store.update(
            run_id,
            status=V07RunStatus.COMPLETED,
            current_step=str(state_dict.get("current_step") or "completed"),
            result=state_dict,
        )
        trace(f"V07 WEB ✓ run={run_id}")
    except Exception as exc:
        from knowledge_engine.ui.errors import format_error_with_cause

        err = format_error_with_cause(exc)
        trace(f"V07 WEB ✗ run={run_id} | {err}")
        v07_run_store.update(
            run_id,
            status=V07RunStatus.FAILED,
            error=err,
            current_step="failed",
        )
