"""Semantic Scholar + arXiv fallback for academic retrieval."""

from __future__ import annotations

import asyncio
import re
import xml.etree.ElementTree as ET
from typing import List, Optional
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel, Field

from knowledge_engine.config import (
    SEMANTIC_SCHOLAR_API_KEY,
    SEMANTIC_SCHOLAR_ENABLED,
    SEMANTIC_SCHOLAR_LIMIT,
    SEMANTIC_SCHOLAR_429_BACKOFF_SEC,
    SEMANTIC_SCHOLAR_MIN_INTERVAL_SEC,
    SEMANTIC_SCHOLAR_TIMEOUT_SEC,
)
from knowledge_engine.src.retrieval.semantic_scholar_rate_limit import (
    acquire_semantic_scholar_slot_async,
    semantic_scholar_pause_before_retry_async,
)
from knowledge_engine.ui.run_log import trace

_SS_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_SS_PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper"
_ARXIV_API_URL = "https://export.arxiv.org/api/query"
_ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}


class ScholarPaper(BaseModel):
    paper_id: str = ""
    title: str = ""
    year: Optional[int] = None
    tldr: str = ""
    abstract: str = ""
    citation_count: int = 0
    venue: str = ""
    pdf_url: str = ""
    source_url: str = ""
    source: str = Field(default="semantic_scholar")


def _ss_headers() -> dict[str, str]:
    headers = {"User-Agent": "KnowledgeEngine/0.7 (+research)"}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
    return headers


async def _ss_http_get(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict | None = None,
) -> httpx.Response:
    """Один GET с throttle; retry после 429 с паузой 1–1.5s."""
    await acquire_semantic_scholar_slot_async()
    resp = await client.get(url, params=params or {})
    if resp.status_code == 429:
        backoff = max(1.0, min(1.5, SEMANTIC_SCHOLAR_429_BACKOFF_SEC))
        trace(
            f"Semantic Scholar ⊘ 429 — wait {backoff:.2f}s "
            f"(rate limit, min_interval={SEMANTIC_SCHOLAR_MIN_INTERVAL_SEC:.2f}s) "
            "and retry once"
        )
        await semantic_scholar_pause_before_retry_async(backoff)
        resp = await client.get(url, params=params or {})
    return resp


async def get_semantic_scholar_paper_by_id(
    paper_id: str,
    *,
    fields: str = "title,url,abstract,tldr,paperId",
) -> tuple[int, dict | None]:
    pid = (paper_id or "").strip()
    if not pid:
        return 0, None
    url = f"{_SS_PAPER_URL}/{pid}"
    timeout = httpx.Timeout(SEMANTIC_SCHOLAR_TIMEOUT_SEC)
    try:
        async with httpx.AsyncClient(timeout=timeout, headers=_ss_headers()) as client:
            resp = await _ss_http_get(client, url, params={"fields": fields})
            if resp.status_code != 200:
                return resp.status_code, None
            return resp.status_code, resp.json()
    except Exception as exc:
        trace(f"Semantic Scholar ✗ paper/{pid[:12]} | {exc}")
        return 0, None


async def search_semantic_scholar(
    query: str,
    limit: int | None = None,
    *,
    ignore_enabled_flag: bool = False,
) -> List[ScholarPaper]:
    q = (query or "").strip()
    if not q:
        return []
    if not SEMANTIC_SCHOLAR_ENABLED and not ignore_enabled_flag:
        trace("Semantic Scholar ⊘ disabled (SEMANTIC_SCHOLAR_ENABLED=false)")
        return []
    if ignore_enabled_flag:
        from knowledge_engine.services.curriculum_api_quota_store import (
            can_use_semantic_scholar,
            record_semantic_scholar_result,
        )

        allowed, why = can_use_semantic_scholar()
        if not allowed:
            trace(f"Semantic Scholar ⊘ curriculum | {why}")
            return []
    lim = min(limit or SEMANTIC_SCHOLAR_LIMIT, 20)
    params = {
        "query": q,
        "limit": lim,
        "fields": "title,abstract,tldr,citationCount,year,openAccessPdf,venue,url,paperId,externalIds",
    }
    trace(f"Semantic Scholar ▶ search | {q[:120]}")
    timeout = httpx.Timeout(SEMANTIC_SCHOLAR_TIMEOUT_SEC)
    try:
        async with httpx.AsyncClient(timeout=timeout, headers=_ss_headers()) as client:
            resp = await _ss_http_get(client, _SS_SEARCH_URL, params=params)
            if ignore_enabled_flag and resp.status_code in (429, 503):
                from knowledge_engine.services.curriculum_api_quota_store import (
                    record_semantic_scholar_result,
                )

                record_semantic_scholar_result(
                    ok=False,
                    http_status=resp.status_code,
                    quota_exhausted=True,
                )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        trace(f"Semantic Scholar ✗ {exc}")
        if ignore_enabled_flag:
            from knowledge_engine.services.curriculum_api_quota_store import (
                record_semantic_scholar_result,
            )

            record_semantic_scholar_result(ok=False)
        return []

    if ignore_enabled_flag:
        from knowledge_engine.services.curriculum_api_quota_store import (
            record_semantic_scholar_result,
        )

        record_semantic_scholar_result(ok=True)

    papers: List[ScholarPaper] = []
    for item in data.get("data", [])[:lim]:
        pid = str(item.get("paperId") or "")
        oa = item.get("openAccessPdf") or {}
        pdf_url = (oa.get("url") or "").strip() if isinstance(oa, dict) else ""
        url = (item.get("url") or "").strip()
        if not url and pid:
            url = f"https://www.semanticscholar.org/paper/{pid}"
        tldr_raw = item.get("tldr")
        tldr = ""
        if isinstance(tldr_raw, dict):
            tldr = str(tldr_raw.get("text") or "").strip()
        elif isinstance(tldr_raw, str):
            tldr = tldr_raw.strip()

        papers.append(
            ScholarPaper(
                paper_id=pid,
                title=str(item.get("title") or "").strip(),
                year=item.get("year"),
                tldr=tldr,
                abstract=str(item.get("abstract") or "").strip(),
                citation_count=int(item.get("citationCount") or 0),
                venue=str(item.get("venue") or "").strip(),
                pdf_url=pdf_url,
                source_url=url,
                source="semantic_scholar",
            )
        )
    trace(f"Semantic Scholar ✓ papers={len(papers)}")
    return papers


async def search_arxiv_fallback(query: str, limit: int = 5) -> List[ScholarPaper]:
    q = (query or "").strip()
    if not q:
        return []
    if len(q) >= 2 and q[0] == q[-1] == '"':
        q = q[1:-1].strip()
    if q.lower().startswith("all:"):
        search_query = q
    else:
        search_query = f"all:{q}"
    params = urlencode({"search_query": search_query, "start": 0, "max_results": limit})
    url = f"{_ARXIV_API_URL}?{params}"
    trace(f"arXiv API ▶ fallback | {q[:100]}")
    root = None
    try:
        async with httpx.AsyncClient(timeout=25.0, follow_redirects=True) as client:
            for attempt in range(2):
                resp = await client.get(url)
                if resp.status_code == 503 and attempt == 0:
                    trace("arXiv API ⊘ 503 — retry once")
                    await asyncio.sleep(3.0)
                    continue
                resp.raise_for_status()
                if resp.text.strip().startswith("<!DOCTYPE") or "<feed" not in resp.text[:500]:
                    trace(f"arXiv API ✗ non-atom body (HTTP {resp.status_code})")
                    return []
                root = ET.fromstring(resp.text)
                break
    except Exception as exc:
        trace(f"arXiv API ✗ {exc}")
        return []

    if root is None:
        return []

    papers: List[ScholarPaper] = []
    for entry in root.findall("a:entry", _ATOM_NS):
        title = (
            entry.findtext("a:title", default="", namespaces=_ATOM_NS) or ""
        ).strip()
        abstract = (
            entry.findtext("a:summary", default="", namespaces=_ATOM_NS) or ""
        ).strip()
        entry_id = (
            entry.findtext("a:id", default="", namespaces=_ATOM_NS) or ""
        ).strip()
        arxiv_id_m = re.search(r"arxiv\.org/abs/([^/]+)", entry_id, re.I)
        arxiv_id = arxiv_id_m.group(1) if arxiv_id_m else ""
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else ""
        if not title:
            continue
        papers.append(
            ScholarPaper(
                paper_id=arxiv_id or entry_id,
                title=title,
                abstract=abstract,
                pdf_url=pdf_url,
                source_url=entry_id or pdf_url,
                source="arxiv",
            )
        )
    trace(f"arXiv API ✓ papers={len(papers)}")
    return papers


async def retrieve_scholarly_papers(
    query: str,
    limit: int | None = None,
) -> List[ScholarPaper]:
    """Semantic Scholar (если включён) + arXiv; при выключенном SS — только arXiv."""
    lim = limit or SEMANTIC_SCHOLAR_LIMIT
    papers: List[ScholarPaper] = []
    if SEMANTIC_SCHOLAR_ENABLED:
        papers = await search_semantic_scholar(query, limit=lim)
        min_ok = max(3, lim // 2)
        if len(papers) < min_ok:
            trace(f"Semantic Scholar ⊘ few results ({len(papers)}) — arXiv fallback")
    else:
        trace("retrieve_scholarly_papers › arXiv only (Semantic Scholar off)")
    arxiv = await search_arxiv_fallback(query, limit=lim)
    seen_titles = {p.title.lower() for p in papers}
    for p in arxiv:
        if p.title.lower() not in seen_titles:
            papers.append(p)
            seen_titles.add(p.title.lower())
    return papers[:lim]


def format_papers_block(papers: List[ScholarPaper]) -> str:
    if not papers:
        return "(нет статей в retrieval)"
    blocks: List[str] = []
    for i, p in enumerate(papers, start=1):
        lines = [
            f"[{i}] {p.title}",
            f"Year: {p.year or '—'} | Venue: {p.venue or '—'} | Citations: {p.citation_count}",
            f"URL: {p.source_url}",
        ]
        if p.pdf_url:
            lines.append(f"Open PDF: {p.pdf_url}")
        if p.tldr:
            lines.append(f"TLDR: {p.tldr}")
        if p.abstract:
            lines.append(f"Abstract: {p.abstract[:2500]}")
        blocks.append("\n".join(lines))
    return "\n\n---\n\n".join(blocks)


def paper_to_document_text(paper: ScholarPaper) -> str:
    parts = [
        f"# {paper.title}",
        f"Source: {paper.source} | {paper.source_url}",
    ]
    if paper.year:
        parts.append(f"Year: {paper.year}")
    if paper.venue:
        parts.append(f"Venue: {paper.venue}")
    if paper.tldr:
        parts.append(f"\n## TLDR\n{paper.tldr}")
    if paper.abstract:
        parts.append(f"\n## Abstract\n{paper.abstract}")
    return "\n\n".join(parts)
