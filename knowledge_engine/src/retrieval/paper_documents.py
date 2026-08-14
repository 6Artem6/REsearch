"""Convert ScholarPaper → ScrapedDocument with optional PDF fetch."""

from __future__ import annotations

import asyncio
import hashlib

from knowledge_engine.src.fetcher import fetch_document
from knowledge_engine.src.fetcher.academic import extract_doi
from knowledge_engine.src.retrieval.semantic_scholar import (
    ScholarPaper,
    paper_to_document_text,
)
from knowledge_engine.src.state import ScrapedDocument
from knowledge_engine.ui.run_log import trace


def _doc_id(url: str, prefix: str) -> str:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


async def fetch_paper_document(
    paper: ScholarPaper,
    *,
    abstract_only: bool = False,
) -> ScrapedDocument | None:
    from knowledge_engine.src.fetcher.context import fast_academic_fetch_enabled

    if abstract_only or fast_academic_fetch_enabled():
        text = paper_to_document_text(paper)
        if len(text.strip()) < 80:
            return None
        url = (paper.source_url or paper.pdf_url or "").strip()
        return ScrapedDocument(
            doc_id=_doc_id(url or paper.title or "paper", "scholar"),
            source_url=paper.source_url or url,
            source_type="trafilatura",
            raw_markdown=text[:14000],
            title=(paper.title or "paper")[:400],
            is_pdf=False,
            cosine_dedup_passed=False,
        )

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
        trace("scholar fetch PDF ⊘ | try source_url / doi cascade")

    fetch_urls: list[str] = []
    if paper.source_url:
        fetch_urls.append(paper.source_url.strip())
    doi = extract_doi(paper.pdf_url or "") or extract_doi(paper.source_url or "")
    if doi:
        fetch_urls.append(f"https://doi.org/{doi}")
    seen_u: set[str] = set()
    for u in fetch_urls:
        key = u.lower().rstrip("/")
        if not key or key in seen_u:
            continue
        seen_u.add(key)
        trace(f"scholar fetch cascade | {u[:90]}")
        doc = await asyncio.to_thread(fetch_document, u)
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
