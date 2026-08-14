"""Hydrate ScholarPaper metadata via arXiv Atom `id_list` batches."""

from __future__ import annotations

import re
from typing import List, Sequence

from knowledge_engine.config import (
    ARXIV_ID_LIST_CHUNK,
    CURRICULUM_ACADEMIC_ABSTRACT_MIN_CHARS,
)
from knowledge_engine.services.search.arxiv_client import (
    ArxivEntry,
    get_arxiv_client,
    normalize_arxiv_id,
)
from knowledge_engine.src.curriculum.academic_url_canonicalizer import arxiv_id_from_url
from knowledge_engine.src.retrieval.semantic_scholar import ScholarPaper
from knowledge_engine.ui.run_log import trace

_ARXIV_ID_BARE = re.compile(
    r"^(?:arxiv:)?(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+/\d{7}(?:v\d+)?)$",
    re.I,
)
_ARXIV_DOI = re.compile(
    r"10\.48550/arxiv\.([^\s/?#]+)",
    re.I,
)


def extract_arxiv_id_from_paper(paper: ScholarPaper) -> str:
    """Resolve arXiv id from explicit field, URLs, DOI, or bare paper_id."""
    if paper.arxiv_id:
        return normalize_arxiv_id(paper.arxiv_id)

    for blob in (paper.source_url, paper.pdf_url, paper.doi, paper.paper_id):
        text = (blob or "").strip()
        if not text:
            continue
        m = _ARXIV_DOI.search(text)
        if m:
            return normalize_arxiv_id(m.group(1))
        from_url = arxiv_id_from_url(text)
        if from_url:
            return normalize_arxiv_id(from_url)
        bare = _ARXIV_ID_BARE.match(text)
        if bare:
            return normalize_arxiv_id(bare.group(1))
    return ""


def _prefer_text(current: str, incoming: str, *, min_prefer: int = 0) -> str:
    cur = (current or "").strip()
    inc = (incoming or "").strip()
    if not inc:
        return cur
    if not cur:
        return inc
    if len(cur) < min_prefer and len(inc) > len(cur):
        return inc
    return cur


def _year_from_published(published: str) -> int | None:
    raw = (published or "").strip()
    if len(raw) >= 4 and raw[:4].isdigit():
        year = int(raw[:4])
        if 1990 <= year <= 2100:
            return year
    return None


def merge_arxiv_entry_into_paper(
    paper: ScholarPaper, entry: ArxivEntry
) -> ScholarPaper:
    """Fill canonical arXiv metadata without wiping richer Consensus/S2 fields."""
    abs_min = CURRICULUM_ACADEMIC_ABSTRACT_MIN_CHARS
    abstract = _prefer_text(paper.abstract, entry.abstract, min_prefer=abs_min)
    title = _prefer_text(paper.title, entry.title)
    pdf_url = (paper.pdf_url or "").strip() or entry.pdf_url
    source_url = (paper.source_url or "").strip()
    if not source_url or "arxiv.org" not in source_url.lower():
        # Prefer abs URL when we only had a generic Consensus/S2 link.
        lowered = source_url.lower()
        if not source_url or "consensus" in lowered or "semanticscholar" in lowered:
            source_url = entry.abs_url or source_url or entry.pdf_url
        elif not source_url:
            source_url = entry.abs_url or entry.pdf_url
    year = paper.year or _year_from_published(entry.published)
    src = (paper.source or "").strip()
    if src and "arxiv" not in src.lower():
        src = f"{src}+arxiv"
    elif not src:
        src = "arxiv"
    return paper.model_copy(
        update={
            "title": title or paper.title,
            "abstract": abstract,
            "pdf_url": pdf_url,
            "source_url": source_url or paper.source_url,
            "arxiv_id": entry.arxiv_id or paper.arxiv_id,
            "year": year,
            "source": src,
            "paper_id": paper.paper_id or entry.arxiv_id,
        }
    )


async def hydrate_papers_by_arxiv_ids(
    ids: Sequence[str],
    *,
    chunk: int | None = None,
) -> dict[str, ArxivEntry]:
    """Fetch Atom entries for arXiv ids; returns map normalized_id → entry."""
    client = get_arxiv_client()
    entries = await client.fetch_by_ids(ids, chunk_size=chunk or ARXIV_ID_LIST_CHUNK)
    out: dict[str, ArxivEntry] = {}
    for entry in entries:
        key = normalize_arxiv_id(entry.arxiv_id)
        if key:
            out[key.lower()] = entry
    return out


async def hydrate_scholar_papers(
    papers: List[ScholarPaper],
    *,
    chunk: int | None = None,
) -> List[ScholarPaper]:
    """Enrich papers that expose an arXiv id via batched id_list requests."""
    if not papers:
        return papers

    id_by_index: dict[int, str] = {}
    for i, paper in enumerate(papers):
        aid = extract_arxiv_id_from_paper(paper)
        if aid:
            id_by_index[i] = aid

    if not id_by_index:
        return list(papers)

    unique_ids = list(dict.fromkeys(id_by_index.values()))
    trace(f"arXiv hydrate ▶ papers={len(id_by_index)} unique_ids={len(unique_ids)}")
    try:
        by_id = await hydrate_papers_by_arxiv_ids(unique_ids, chunk=chunk)
    except Exception as exc:
        trace(f"arXiv hydrate ✗ {exc}")
        return list(papers)

    if not by_id:
        trace("arXiv hydrate ⊘ empty response")
        return list(papers)

    out: List[ScholarPaper] = []
    merged_n = 0
    for i, paper in enumerate(papers):
        aid = id_by_index.get(i)
        entry = by_id.get(aid.lower()) if aid else None
        if entry is None:
            out.append(paper)
            continue
        out.append(merge_arxiv_entry_into_paper(paper, entry))
        merged_n += 1
    trace(f"arXiv hydrate ✓ merged={merged_n}/{len(papers)}")
    return out
