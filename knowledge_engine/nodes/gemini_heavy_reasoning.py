"""Heavy reasoner: Gemini (Playwright) или локальный 7B fallback."""

from __future__ import annotations

from typing import Any

from knowledge_engine.config import SKIP_GEMINI
from knowledge_engine.nodes.matrix import matrix_node
from knowledge_engine.schemas import EngineGraphState, EngineState
from knowledge_engine.services.ai_dialogue.gemini_session import (
    BrowserGeminiDialogueSession,
)
from knowledge_engine.services.analysis_report_structure import (
    structure_analysis_report,
)
from knowledge_engine.ui.logger import set_status
from knowledge_engine.ui.run_log import node_end, node_start


def gemini_heavy_reasoning_node(state: EngineGraphState) -> dict[str, Any]:
    node_start("gemini_heavy_reasoning_node (heavy LLM)")
    parsed = EngineState.model_validate(state)

    if SKIP_GEMINI or not parsed.gemini_payload:
        set_status("[gemini_heavy] SKIP_GEMINI — локальный 7B matrix")
        node_end("gemini_heavy_reasoning_node (heavy LLM)", "fallback matrix_node")
        return matrix_node(state)

    set_status("[gemini_heavy] Playwright → Gemini (payload only)…")
    session = BrowserGeminiDialogueSession()
    try:
        raw = session.ask_gemini(parsed.gemini_payload)
        history = session.as_chat_dicts()
    finally:
        session.close()

    report = structure_analysis_report(
        parsed, raw, log_label="gemini_heavy / AnalysisReport structure"
    )

    node_end("gemini_heavy_reasoning_node (heavy LLM)", f"response={len(raw)} sym")
    return {
        "gemini_raw_response": raw,
        "external_ai_dialogue_history": history,
        "report": report.model_dump(),
        "abstractions": [a.model_dump() for a in report.abstractions],
    }
