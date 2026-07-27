"""v0.4: 7B junk + выбор секций / code / media."""

from __future__ import annotations

from typing import Any

from knowledge_engine.config import LOCAL_HEAVY_MODEL
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.schemas import (
    DocumentStructure,
    EngineGraphState,
    StructureFilterResult,
)
from knowledge_engine.services.gemini_stateless import global_anchor_from_state
from knowledge_engine.services.local_llm_stateless import run_local_structured
from knowledge_engine.ui.logger import set_status
from knowledge_engine.ui.pipeline_phase import pipeline_phase
from knowledge_engine.ui.run_log import node_end, node_start


def junk_and_structure_filter_node(state: EngineGraphState) -> dict[str, Any]:
    node_start("junk_and_structure_filter_node (7B)")
    pipeline_phase("Structure Filter (7B)")
    raw = state.get("document_structure")
    if not raw:
        node_end("junk_and_structure_filter_node", "no structure")
        return {
            "structure_filter": StructureFilterResult(
                is_relevant=False,
                reject_reason="нет HTML/структуры",
            ).model_dump(),
        }

    doc = DocumentStructure.model_validate(raw)
    anchor = global_anchor_from_state(
        state.get("original_query") or state.get("user_problem") or "",
        state.get("constraints") or state.get("context_constraints") or "",
        state.get("l0_summary") or "",
    )
    toc_preview = "\n".join(f"H{e.level}: {e.title}" for e in doc.toc[:20])
    system = (
        f"{RUSSIAN_OUTPUT_RULE} "
        "Отсеки SEO-мусор. Если страница релевантна задаче — выбери ключи секций из sections, "
        "индексы code_artifacts и media_artifacts для глубокого разбора. JSON StructureFilterResult."
    )
    user = (
        f"URL: {doc.source_url}\n"
        f"Meta: {doc.meta_summary.title} | {doc.meta_summary.description}\n"
        f"Abstract:\n{doc.abstract[:2000]}\n\nTOC:\n{toc_preview}\n\n"
        f"Sections keys: {list(doc.sections.keys())[:24]}"
    )

    set_status("[structure_filter] 7B релевантность + отбор секций…")
    filt = run_local_structured(
        LOCAL_HEAVY_MODEL,
        StructureFilterResult,
        system,
        user,
        anchor,
        "structure_filter / StructureFilterResult",
        num_predict=2048,
    )

    node_end(
        "junk_and_structure_filter_node",
        f"relevant={filt.is_relevant} sections={len(filt.selected_section_keys)}",
    )
    return {"structure_filter": filt.model_dump()}
