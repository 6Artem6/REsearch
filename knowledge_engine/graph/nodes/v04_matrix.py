"""v0.4: Gemini matrix из pre_synthesis черновика."""

from __future__ import annotations

from typing import Any

from knowledge_engine.llm_locale import GEMINI_RUSSIAN_ROLE, RUSSIAN_OUTPUT_RULE
from knowledge_engine.schemas import EngineGraphState
from knowledge_engine.schemas.llm_contracts.v04_gemini import AnalysisReportContract
from knowledge_engine.services.gemini_stateless import (
    global_anchor_from_state,
    run_stateless_gemini,
)
from knowledge_engine.ui.logger import set_status
from knowledge_engine.ui.pipeline_phase import pipeline_phase
from knowledge_engine.ui.run_log import node_end, node_start


def matrix_node_v04(state: EngineGraphState) -> dict[str, Any]:
    node_start("matrix_node_v04 (Gemini)")
    pipeline_phase("Matrix (Gemini)")
    query = state.get("original_query") or state.get("user_problem") or ""
    constraints = state.get("constraints") or ""
    l0 = state.get("l0_summary") or ""
    anchor = global_anchor_from_state(query, constraints, l0)

    draft = state.get("pre_synthesis_draft") or l0
    set_status("[matrix] Gemini → AnalysisReport…")
    system = (
        f"{GEMINI_RUSSIAN_ROLE} {RUSSIAN_OUTPUT_RULE} "
        "На основе черновика построй AnalysisReport: abstractions + ровно 3 options (id 1,2,3). "
        "category: Классика / SOTA (Современное) / Минимализм."
    )
    user = f"ЧЕРНОВИК PRE-SYNTHESIS:\n{draft[:14000]}"

    report = run_stateless_gemini(
        system,
        user,
        anchor,
        response_schema=AnalysisReportContract,
        label="v04 matrix / AnalysisReport",
    )

    if len(report.options) != 3:
        raise RuntimeError(f"Ожидалось 3 options, получено {len(report.options)}")

    raw = report.model_dump_json(ensure_ascii=False)
    node_end("matrix_node_v04", "ok")
    return {
        "report": report.model_dump(),
        "abstractions": [a.model_dump() for a in report.abstractions],
        "gemini_raw_response": raw,
    }
