"""Сохранение heavy-синтеза Gemini в LanceDB как DocumentSummary."""

from __future__ import annotations

from typing import Any

from knowledge_engine.schemas import DocumentSummary, EngineGraphState, EngineState
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.ui.logger import set_status
from knowledge_engine.ui.run_log import node_end, node_start


def lancedb_save_node(state: EngineGraphState) -> dict[str, Any]:
    node_start("lancedb_save_node")
    parsed = EngineState.model_validate(state)
    if not parsed.gemini_raw_response and not parsed.report:
        node_end("lancedb_save_node", "skip (empty)")
        return {}

    text = parsed.gemini_raw_response or ""
    takeaways: list[str] = []
    failures: list[str] = []
    if parsed.report:
        for opt in parsed.report.options:
            takeaways.append(f"{opt.pattern_name}: {opt.fundamental_idea[:200]}")
            failures.extend(opt.cons_and_risks[:2])

    summary = DocumentSummary(
        title="Gemini heavy synthesis — Trade-off matrix",
        url="gemini-heavy-reasoning",
        cs_concepts=[
            a.cs_concept for a in (parsed.report.abstractions if parsed.report else [])
        ],
        key_takeaways=takeaways[:12] or [text[:500]],
        failure_modes=failures[:12],
        diagram_descriptions=[],
    )

    set_status("[lancedb_save] embed → LanceDB…")
    store = VectorStore()
    store.save_summary(summary)

    merged = list(parsed.found_summaries)
    if not any(s.url == summary.url for s in merged):
        merged.append(summary)

    node_end("lancedb_save_node", summary.url)
    return {"found_summaries": [s.model_dump() for s in merged]}
