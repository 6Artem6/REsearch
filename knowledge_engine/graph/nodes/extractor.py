"""v0.3: L2 extraction — smart web fetch + Gemini (fallback local)."""

from __future__ import annotations

from typing import Any

from knowledge_engine.llm_locale import GEMINI_RUSSIAN_ROLE, RUSSIAN_OUTPUT_RULE
from knowledge_engine.schemas import EngineGraphState, L2EvidenceExtraction
from knowledge_engine.services.gemini_stateless import global_anchor_from_state
from knowledge_engine.services.hybrid_llm import run_l2_extraction_hybrid
from knowledge_engine.services.page_filter import is_obviously_useless_page_text
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.services.web_extract import smart_fetch_page_text
from knowledge_engine.ui.logger import set_status
from knowledge_engine.ui.run_log import node_end, node_start, trace


def _pick_l1_parent(l1_ids: list[str], hint: str, store: VectorStore) -> str | None:
    if not l1_ids:
        return None
    hint_l = hint.lower()
    for lid in l1_ids:
        node = store.get_knowledge_node(lid)
        if node and hint_l and hint_l in node.content.lower():
            return lid
    return l1_ids[0]


def extractor_node(state: EngineGraphState) -> dict[str, Any]:
    node_start("extractor_node (v0.3 L2 hybrid)")
    pending = list(state.get("pending_urls") or [])
    explored = list(state.get("explored_urls") or [])
    if not pending:
        node_end("extractor_node", "no pending url")
        return {}

    url = pending[0]
    remaining = pending[1:]
    explored.append(url)

    set_status(f"[extractor] smart fetch {url[:55]}…")
    page_text, method = smart_fetch_page_text(url)
    if not page_text.strip():
        node_end("extractor_node", f"skip empty ({method})")
        return {
            "pending_urls": remaining,
            "explored_urls": explored,
        }

    if is_obviously_useless_page_text(page_text):
        trace(f"extractor skip junk page | {url[:80]} | {method}")
        node_end("extractor_node", f"skip junk ({method})")
        return {
            "pending_urls": remaining,
            "explored_urls": explored,
        }

    set_status(f"[extractor] {method} → L2 hybrid ({len(page_text)} sym)…")
    anchor = global_anchor_from_state(
        state.get("original_query") or state.get("user_problem") or "",
        state.get("constraints") or state.get("context_constraints") or "",
        state.get("l0_summary") or "",
    )
    system = (
        f"{GEMINI_RUSSIAN_ROLE} {RUSSIAN_OUTPUT_RULE} "
        "Из текста страницы извлеки атомарные L2-факты, failure modes и метрики. "
        "Только JSON L2EvidenceExtraction."
    )
    user = f"URL: {url}\n\nТекст страницы:\n{page_text[:10000]}"

    extraction = run_l2_extraction_hybrid(
        system,
        user,
        anchor,
        L2EvidenceExtraction,
        "extractor / L2EvidenceExtraction",
    )

    store = VectorStore()
    l1_ids = list(state.get("l1_node_ids") or [])
    parent_l1 = _pick_l1_parent(l1_ids, extraction.l1_title_hint, store)
    new_ids = list(state.get("knowledge_node_ids") or [])

    for item in extraction.evidences[:12]:
        chunk = (
            f"FACT: {item.fact}\nFAILURE: {item.failure_mode}\nMETRIC: {item.metric}"
        ).strip()
        nid = store.save_knowledge_node(
            "L2_EVIDENCE",
            chunk,
            parent_id=parent_l1,
            source_url=url,
        )
        new_ids.append(nid)

    node_end(
        "extractor_node (v0.3 L2 hybrid)",
        f"L2={len(extraction.evidences)} via {method}",
    )
    return {
        "pending_urls": remaining,
        "explored_urls": explored,
        "knowledge_node_ids": new_ids,
    }
