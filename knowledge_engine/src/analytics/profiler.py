"""Stage 5 — Gemini Flash L2a / L2b / L2c (research synthesis)."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from knowledge_engine.services.context_manager import load_personal_orchestrator_focus
from knowledge_engine.src.analytics.gemini_v07 import run_gemini_flash_structured
from knowledge_engine.src.analytics.prompts import (
    build_architect_system_instruction,
    formatted_papers_from_state,
)
from knowledge_engine.src.analytics.schemas import (
    ConceptGraph,
    ProfileGapMap,
    TradeoffMatrixResult,
)
from knowledge_engine.src.processors.source_anchors import (
    SOURCE_ANCHOR_RETENTION_PROMPT,
    format_registry_for_prompt,
)
from knowledge_engine.src.state import ScrapedDocument, StructuredChunk

ConceptGraphContract = ConceptGraph
ProfileGapMapContract = ProfileGapMap
TradeoffMatrixContract = TradeoffMatrixResult

_MAX_CHUNKS_IN_PROMPT = 28
_MAX_CHUNK_TEXT = 900
_TOP_K_FULL_DOCS = 3
_MAX_FULL_DOC_CHARS = 12_000

_RESEARCH_PRIORITY = (
    "MAIN PRIORITY: research synthesis and comparison of sources. "
    "Compare approaches across different articles/documents: mechanics, algorithms, data "
    "structures, architectural assumptions. Surface non-obvious engineering nuances and "
    "authors' pitfalls. Connect theory (papers/arXiv) to the task's practical implications. "
    "FORBIDDEN: high-level abstract summaries with no technical detail; "
    "compressing a description into one short line where the field allows extended text."
)
"""
RU (пояснение): L2a ConceptGraph — приоритет сравнению источников и
техническим нюансам, запрет на abstract-сводки без деталей.
"""

_CONTEXT_SECONDARY = (
    "SECONDARY PRIORITY: user profile and hardware (UMA/Mac, etc.) — "
    "only as one of the context_flags/conditions, NOT as a hard filter and NOT to rule out solutions."
)
"""
RU (пояснение): L2b ProfileGapMap — личный профиль/железо вторичны, не
жёсткий фильтр и не повод отбрасывать решения.
"""


def _source_anchor_block(source_registry: List[Dict[str, Any]] | None) -> str:
    if source_registry:
        return (
            format_registry_for_prompt(source_registry)
            + "\n\n"
            + SOURCE_ANCHOR_RETENTION_PROMPT
        )
    return SOURCE_ANCHOR_RETENTION_PROMPT


def _chunks_payload(chunks: List[StructuredChunk]) -> str:
    rows: list[dict[str, Any]] = []
    for ch in chunks[:_MAX_CHUNKS_IN_PROMPT]:
        rows.append(
            {
                "chunk_id": ch.chunk_id,
                "doc_id": ch.doc_id,
                "source_anchor": ch.source_anchor or "",
                "concepts": ch.concepts[:16],
                "code_snippets": ch.code_snippets[:6],
                "p99_relevance_score": ch.p99_relevance_score,
                "text_preview": (ch.text or "")[:_MAX_CHUNK_TEXT],
            }
        )
    return json.dumps(rows, ensure_ascii=False, indent=2)


def _full_documents_payload(documents: List[ScrapedDocument]) -> str:
    if not documents:
        return ""
    ranked = sorted(
        documents,
        key=lambda d: len((d.raw_markdown or "")),
        reverse=True,
    )
    rows: list[dict[str, Any]] = []
    for doc in ranked[:_TOP_K_FULL_DOCS]:
        rows.append(
            {
                "doc_id": doc.doc_id,
                "title": doc.title or "",
                "source_url": doc.source_url,
                "is_pdf": doc.is_pdf,
                "clean_text": (doc.raw_markdown or "")[:_MAX_FULL_DOC_CHARS],
            }
        )
    return json.dumps(rows, ensure_ascii=False, indent=2)


def build_concept_graph(
    chunks: List[StructuredChunk],
    global_anchor: str = "",
    source_documents: List[ScrapedDocument] | None = None,
    personal_context: Any = None,
    scholarly_papers: List[Dict[str, Any]] | None = None,
    user_query: str = "",
    source_registry: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """L2a: ConceptGraph — синтез и сравнение идей из источников."""
    papers_fmt = formatted_papers_from_state(scholarly_papers)
    has_papers = papers_fmt and papers_fmt != "(нет статей в retrieval)"
    has_docs = bool(source_documents)

    if not chunks and not has_papers and not has_docs:
        return ConceptGraph(task_summary="Нет источников для анализа").model_dump()

    system = build_architect_system_instruction(
        personal_context, papers_fmt, user_query
    )
    system += (
        "\n\nДополнительно: построй JSON ConceptGraph "
        "(nodes, edges, cross_source_contrasts, engineering_pitfalls, "
        "theory_practice_bridges, research_synthesis, invariants, contrasts). "
        f"{_RESEARCH_PRIORITY}\n\n" + _source_anchor_block(source_registry)
    )

    user_parts: list[str] = []
    if chunks:
        user_parts.append(
            "Структурированные чанки из найденных документов:\n"
            + _chunks_payload(chunks)
        )
    if has_papers:
        user_parts.append(
            "Научные публикации (Consensus / Semantic Scholar — абстракты и TLDR):\n"
            + papers_fmt
        )
    full_docs = _full_documents_payload(source_documents or [])
    if full_docs:
        user_parts.append(
            "Полные очищенные тексты top-K документов (PDF/HTML после cascade):\n"
            + full_docs
        )
    user = "\n\n".join(user_parts)
    graph = run_gemini_flash_structured(
        system,
        user,
        global_anchor,
        ConceptGraph,
        "L2a / ConceptGraph",
    )
    return graph.model_dump()


def build_profile_gap_map(
    concept_graph: Dict[str, Any],
    user_profile_md: str,
    global_anchor: str = "",
    source_documents: List[ScrapedDocument] | None = None,
    personal_context: Any = None,
    scholarly_papers: List[Dict[str, Any]] | None = None,
    user_query: str = "",
    source_registry: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """L2b: условия, допущения и столкновения теории с контекстом задачи."""
    profile = load_personal_orchestrator_focus()[:2000]
    graph_json = json.dumps(concept_graph, ensure_ascii=False, indent=2)[:16_000]

    papers_fmt = formatted_papers_from_state(scholarly_papers)
    system = build_architect_system_instruction(
        personal_context, papers_fmt, user_query
    )
    system += (
        f"\n\n{_CONTEXT_SECONDARY}\n"
        "L2b: JSON ProfileGapMap — context_synthesis, assumption_clashes, "
        "context_flags, gaps (с source_basis и inline [Sx]).\n\n"
        + _source_anchor_block(source_registry)
    )
    user = (
        f"## Личный фокус (оркестратор)\n{profile}\n\n" f"## ConceptGraph\n{graph_json}"
    )
    full_docs = _full_documents_payload(source_documents or [])
    if full_docs:
        user += f"\n\n## Cleaned source documents (top-K)\n{full_docs}"
    gap_map = run_gemini_flash_structured(
        system,
        user,
        global_anchor,
        ProfileGapMap,
        "L2b / ProfileGapMap",
    )
    return gap_map.model_dump()


def build_tradeoff_matrix(
    concept_graph: Dict[str, Any],
    profile_gap_map: Dict[str, Any],
    user_profile_md: str,
    global_anchor: str = "",
    personal_context: Any = None,
    scholarly_papers: List[Dict[str, Any]] | None = None,
    user_query: str = "",
    source_registry: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """
    L2c: Classical vs SOTA vs Minimalist — развернутый архитектурный разбор.
    """
    profile = load_personal_orchestrator_focus()[:2000]
    graph_json = json.dumps(concept_graph, ensure_ascii=False)[:12_000]
    gap_json = json.dumps(profile_gap_map, ensure_ascii=False)[:12_000]

    papers_fmt = formatted_papers_from_state(scholarly_papers)
    system = build_architect_system_instruction(
        personal_context, papers_fmt, user_query
    )
    system += (
        "\n\nСинтезируй JSON TradeoffMatrixResult: ровно 3 options "
        "(classical | sota | minimalist). Развернутые mechanics_detail, "
        "fundamental_limits, applicability; aligning_sources with [Sx] tags. "
        f"{_RESEARCH_PRIORITY}\n\n" + _source_anchor_block(source_registry)
    )
    user = (
        f"## Личный фокус (оркестратор)\n{profile}\n\n"
        f"## ConceptGraph\n{graph_json}\n\n"
        f"## ProfileGapMap\n{gap_json}"
    )
    matrix = run_gemini_flash_structured(
        system,
        user,
        global_anchor,
        TradeoffMatrixResult,
        "L2c / TradeoffMatrixResult",
    )
    options: List[Dict[str, Any]] = []
    for idx, opt in enumerate(matrix.options, start=1):
        row = opt.model_dump()
        row["id"] = idx
        options.append(row)
    return options
