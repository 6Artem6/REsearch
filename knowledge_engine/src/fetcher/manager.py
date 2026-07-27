"""Fetch orchestration: academic cascade → HTTP/trafilatura."""

from __future__ import annotations

from knowledge_engine.src.fetcher.academic import (
    is_academic_url,
    is_challenge_or_empty,
    resolve_academic_document,
)
from knowledge_engine.src.fetcher.http_basic import fetch_http_document
from knowledge_engine.src.state import ScrapedDocument
from knowledge_engine.ui.run_log import trace


def fetch_document(url: str) -> ScrapedDocument | None:
    """
    Ingestion entry: academic cascade (doi/Unpaywall/Sci-Hub/PDF clean)
    then standard HTTP extraction.
    """
    url = (url or "").strip()
    if not url.startswith("http"):
        return None

    if is_academic_url(url):
        trace(f"FETCH route academic | {url[:100]}")
        cleaned = resolve_academic_document(url)
        if cleaned is not None:
            doc = cleaned.to_scraped_document()
            trace(
                f"FETCH ✓ academic | doc_id={doc.doc_id} "
                f"pdf={doc.is_pdf} chars={len(doc.raw_markdown)}"
            )
            return doc

    doc = fetch_http_document(url)
    if doc is not None and not is_challenge_or_empty(doc.raw_markdown, 80):
        trace(f"FETCH ✓ http | chars={len(doc.raw_markdown)}")
        return doc

    if doc is None or is_challenge_or_empty(doc.raw_markdown or "", 80):
        trace(f"FETCH retry academic cascade | {url[:80]}")
        cleaned = resolve_academic_document(url)
        if cleaned is not None:
            out = cleaned.to_scraped_document()
            trace(f"FETCH ✓ academic retry | chars={len(out.raw_markdown)}")
            return out

    return doc
