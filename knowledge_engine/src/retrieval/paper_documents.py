"""Convert ScholarPaper → ScrapedDocument with optional PDF fetch."""

from __future__ import annotations

import asyncio
import hashlib

from knowledge_engine.src.fetcher import fetch_document
from knowledge_engine.src.retrieval.semantic_scholar import (
    ScholarPaper,
    paper_to_document_text,
)
from knowledge_engine.src.state import ScrapedDocument
from knowledge_engine.ui.run_log import trace


def _doc_id(url: str, prefix: str) -> str:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


async def fetch_paper_document(paper: ScholarPaper) -> ScrapedDocument | None:
    url = (paper.pdf_url or paper.source_url or "").strip()
    text = ""
    is_pdf = False
    title = paper.title or "paper"

    if paper.pdf_url:
        trace(f"scholar fetch PDF | {paper.pdf_url[:90]}")
        doc = await asyncio.to_thread(fetch_document, paper.pdf_url)
        if doc and len(doc.raw_markdown or "") >= 80:
            doc.title = title
            return doc

    if paper.source_url and "arxiv.org" in paper.source_url:
        trace(f"scholar fetch arXiv | {paper.source_url[:90]}")
        doc = await asyncio.to_thread(fetch_document, paper.source_url)
        if doc and len(doc.raw_markdown or "") >= 80:
            doc.title = title
            return doc

    text = paper_to_document_text(paper)
    if len(text) < 80:
        return None

    return ScrapedDocument(
        doc_id=_doc_id(url or title, "scholar"),
        source_url=paper.source_url or url,
        source_type="academic_pdf" if is_pdf else "trafilatura",
        raw_markdown=text,
        title=title,
        is_pdf=is_pdf,
        cosine_dedup_passed=False,
    )


async def fetch_all_paper_documents(
    papers: list[ScholarPaper],
) -> list[ScrapedDocument]:
    docs: list[ScrapedDocument] = []
    for p in papers:
        doc = await fetch_paper_document(p)
        if doc is not None:
            docs.append(doc)
    return docs
