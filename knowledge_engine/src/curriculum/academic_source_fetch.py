"""Академический сбор: Semantic Scholar → arXiv, без 7B когда есть TLDR/abstract."""

from __future__ import annotations

import asyncio

from knowledge_engine.config import (
    CURRICULUM_ACADEMIC_ABSTRACT_MIN_CHARS,
    CURRICULUM_ACADEMIC_ARXIV_LIMIT,
    CURRICULUM_ACADEMIC_SS_LIMIT,
)
from knowledge_engine.schemas import DocumentSummary
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.src.curriculum.curriculum_v08_harvest import _deep_extract_blocks
from knowledge_engine.src.curriculum.schemas import CurriculumSearchHit
from knowledge_engine.src.curriculum.search_query_builder import build_search_queries
from knowledge_engine.src.retrieval.paper_documents import fetch_paper_document
from knowledge_engine.src.retrieval.semantic_scholar import (
    ScholarPaper,
    search_arxiv_fallback,
    search_semantic_scholar,
)
from knowledge_engine.ui.run_log import trace


def _paper_url(paper: ScholarPaper) -> str:
    return (
        (paper.source_url or "").strip()
        or (paper.pdf_url or "").strip()
        or f"https://www.semanticscholar.org/paper/{paper.paper_id}"
    )


def _has_ready_scholar_text(paper: ScholarPaper) -> bool:
    tldr = (paper.tldr or "").strip()
    abstract = (paper.abstract or "").strip()
    if tldr and len(tldr) >= 40:
        return True
    return len(abstract) >= CURRICULUM_ACADEMIC_ABSTRACT_MIN_CHARS


def _document_summary_from_paper(paper: ScholarPaper) -> DocumentSummary:
    url = _paper_url(paper)
    takeaways: list[str] = []
    if (paper.tldr or "").strip():
        takeaways.append(paper.tldr.strip())
    abstract = (paper.abstract or "").strip()
    if abstract:
        chunks = _deep_extract_blocks([], [], [abstract], min_words=60, max_words=200)
        if not chunks:
            chunks = [abstract[:1200]]
        for c in chunks:
            if c not in takeaways:
                takeaways.append(c)
    return DocumentSummary(
        title=(paper.title or url)[:400],
        url=url,
        key_takeaways=takeaways[:8],
        failure_modes=[],
        cs_concepts=[],
        diagram_descriptions=[],
    )


def _hit_extracts_from_summary(ds: DocumentSummary) -> list[str]:
    return _deep_extract_blocks(
        list(ds.key_takeaways or []),
        list(ds.failure_modes or []),
        [],
        min_words=80,
        max_words=300,
    )[:8]


async def _ingest_ready_paper(paper: ScholarPaper) -> CurriculumSearchHit:
    ds = _document_summary_from_paper(paper)
    VectorStore().save_summary(ds)
    extracts = _hit_extracts_from_summary(ds)
    snippet = (paper.abstract or paper.tldr or "")[:1200]
    tier = "arxiv" if paper.source == "arxiv" else "semantic_scholar"
    trace(
        f"CURRICULUM academic ready ✓ | {tier} | {ds.url[:70]} "
        "(LanceDB без 7B Summarizer)"
    )
    return CurriculumSearchHit(
        url=ds.url,
        title=(paper.title or ds.title)[:400],
        snippet=snippet,
        key_extracts=extracts,
        source_tier=tier,
    )


async def _hit_needs_summarizer(paper: ScholarPaper) -> CurriculumSearchHit | None:
    doc = await fetch_paper_document(paper)
    if not doc or len((doc.raw_markdown or "").strip()) < 200:
        trace(
            f"CURRICULUM academic skip | no text | "
            f"{_paper_url(paper)[:70]}"
        )
        return None
    url = _paper_url(paper)
    tier = "arxiv" if paper.source == "arxiv" else "semantic_scholar"
    trace(f"CURRICULUM academic fetch ✓ | {tier} PDF/HTML → 7B later | {url[:70]}")
    return CurriculumSearchHit(
        url=url,
        title=(paper.title or url)[:400],
        snippet=(doc.raw_markdown or "")[:1200],
        key_extracts=[],
        source_tier=tier,
    )


async def _process_paper(paper: ScholarPaper) -> CurriculumSearchHit | None:
    if paper.source == "arxiv" and (paper.abstract or "").strip():
        return await _ingest_ready_paper(paper)
    if _has_ready_scholar_text(paper):
        return await _ingest_ready_paper(paper)
    return await _hit_needs_summarizer(paper)


async def fetch_academic_sources_async(expansion_vector: str) -> list[CurriculumSearchHit]:
    vec = (expansion_vector or "").strip()
    if len(vec) < 8:
        return []

    built = build_search_queries(vec)
    q = built.academic_query
    trace(f"CURRICULUM academic ▶ | query={q[:100]}")

    papers = await search_semantic_scholar(
        q,
        limit=CURRICULUM_ACADEMIC_SS_LIMIT,
        ignore_enabled_flag=True,
    )
    if not papers:
        trace("CURRICULUM academic | Semantic Scholar 0 → arXiv fallback")
        papers = await search_arxiv_fallback(q, limit=CURRICULUM_ACADEMIC_ARXIV_LIMIT)
    else:
        trace(f"CURRICULUM academic | Semantic Scholar papers={len(papers)}")

    hits: list[CurriculumSearchHit] = []
    seen: set[str] = set()
    for paper in papers:
        hit = await _process_paper(paper)
        if not hit:
            continue
        key = hit.url.strip().rstrip("/").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        hits.append(hit)

    trace(f"CURRICULUM academic ✓ | hits={len(hits)}")
    return hits


def fetch_academic_sources(expansion_vector: str) -> list[CurriculumSearchHit]:
    return asyncio.run(fetch_academic_sources_async(expansion_vector))

