"""v0.4: fetch HTML + DocumentStructure (Python)."""

from __future__ import annotations

from typing import Any

from knowledge_engine.schemas import EngineGraphState
from knowledge_engine.services.document_parser import parse_html_to_structure
from knowledge_engine.services.web_extract import smart_fetch_page_html
from knowledge_engine.ui.logger import set_status
from knowledge_engine.ui.pipeline_phase import pipeline_phase
from knowledge_engine.ui.run_log import node_end, node_start


def document_fetch_node(state: EngineGraphState) -> dict[str, Any]:
    node_start("document_fetch_node")
    pending = list(state.get("pending_urls") or [])
    explored = list(state.get("explored_urls") or [])
    if not pending:
        node_end("document_fetch_node", "no pending")
        return {"document_structure": None, "current_page_url": ""}

    url = pending[0]
    remaining = pending[1:]
    explored.append(url)

    set_status(f"[document_fetch] {url[:55]}…")
    html, method = smart_fetch_page_html(url)
    from knowledge_engine.config import SOURCE_ARCHIVE_ENABLED
    from knowledge_engine.db.source_links import get_source_link_archive

    if SOURCE_ARCHIVE_ENABLED:
        get_source_link_archive().mark_explored(url, bool(html.strip()), method)
    if not html.strip():
        node_end("document_fetch_node", f"empty ({method})")
        return {
            "pending_urls": remaining,
            "explored_urls": explored,
            "document_structure": None,
            "current_page_url": url,
        }

    structure = parse_html_to_structure(html, url)
    pipeline_phase(f"Structure parse ({method})")
    node_end("document_fetch_node", f"sections={len(structure.sections)}")
    return {
        "pending_urls": remaining,
        "explored_urls": explored,
        "current_page_url": url,
        "document_structure": structure.model_dump(),
    }
