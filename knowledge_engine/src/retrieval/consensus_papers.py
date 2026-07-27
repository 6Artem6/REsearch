"""Извлечение карточек публикаций из UI Consensus (DOM + текст ответа)."""

from __future__ import annotations

import re
from typing import Any, List
from urllib.parse import unquote

from playwright.async_api import Page

from knowledge_engine.src.retrieval.consensus_capture import (
    is_generic_consensus_url,
    normalize_paper_urls,
)
from knowledge_engine.src.retrieval.semantic_scholar import ScholarPaper
from knowledge_engine.ui.run_log import trace

_URL_RE = re.compile(r"https?://[^\s\]<\"')]+")
_ACADEMIC_HOST = re.compile(
    r"(doi\.org|arxiv\.org|semanticscholar\.org|pubmed|ncbi\.nlm\.gov|openreview\.net|aclweb\.org|springer|ieee\.org|acm\.org)",
    re.I,
)
_CONSENSUS_PAPER_PATH = re.compile(r"consensus\.app/(papers?|p)/", re.I)
_ARXIV_ABS = re.compile(r"arxiv\.org/abs/([\d.]+v?\d*)", re.I)
_ARXIV_PDF = re.compile(r"arxiv\.org/pdf/([\d.]+v?\d*)", re.I)


def _clean_title(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    if len(t) > 220:
        t = t[:217] + "…"
    return t


def _papers_from_urls(
    urls: list[str], context_map: dict[str, str]
) -> List[ScholarPaper]:
    papers: List[ScholarPaper] = []
    seen: set[str] = set()
    for url in urls:
        u = unquote(url.rstrip(".,);]"))
        if u in seen:
            continue
        if not _ACADEMIC_HOST.search(u) and not _CONSENSUS_PAPER_PATH.search(u):
            continue
        if is_generic_consensus_url(u):
            continue
        seen.add(u)
        title = _clean_title(context_map.get(u, "") or u)
        pdf_url = ""
        m = _ARXIV_ABS.search(u) or _ARXIV_PDF.search(u)
        if m:
            pdf_url = f"https://arxiv.org/pdf/{m.group(1)}.pdf"
        papers.append(
            ScholarPaper(
                paper_id=m.group(1) if m else u,
                title=title if title != u else f"Publication ({u[:48]})",
                source_url=u,
                pdf_url=pdf_url,
                source="consensus",
            )
        )
    return papers


async def extract_paper_cards_from_page(page: Page) -> List[ScholarPaper]:
    """Собрать ссылки на публикации из последнего блока ответа Consensus."""
    try:
        raw_links: list[dict[str, Any]] = await page.evaluate(
            """() => {
                const sel = 'a[href*="doi.org"], a[href*="arxiv.org"], a[href*="semanticscholar"], '
                    + 'a[href*="pubmed"], a[href*="consensus.app/papers"], a[href*="consensus.app/paper"], '
                    + '[data-testid*="source"] a, [data-testid*="citation"] a, [data-testid*="paper"] a';
                const anchors = Array.from(document.querySelectorAll(sel));
                const main = document.querySelector('main') || document.body;
                const extra = Array.from(main.querySelectorAll('a[href]'));
                const all = [...anchors, ...extra];
                const seen = new Set();
                const out = [];
                for (const a of all) {
                    const href = a.href;
                    if (!href || seen.has(href)) continue;
                    seen.add(href);
                    const block = (a.closest('[data-testid*="source"], article, li, [class*="Paper"]')
                        || a.parentElement)?.innerText?.slice(0, 800) || '';
                    out.push({
                        href,
                        text: (a.innerText || a.textContent || '').trim().slice(0, 300),
                        block
                    });
                }
                return out;
            }"""
        )
    except Exception as exc:
        trace(f"Consensus ⊘ DOM paper extract | {exc}")
        raw_links = []

    context_map: dict[str, str] = {}
    urls: list[str] = []
    for item in raw_links:
        href = str(item.get("href") or "").strip()
        if not href.startswith("http"):
            continue
        text = str(item.get("text") or "").strip()
        block = str(item.get("block") or "").strip()
        snippet = text or block.split("\n")[0]
        context_map[href] = snippet
        urls.append(href)

    papers = normalize_paper_urls(_papers_from_urls(urls, context_map))
    trace(f"Consensus ✓ DOM papers={len(papers)}")
    return papers


def extract_papers_from_text(raw_text: str) -> List[ScholarPaper]:
    urls = _URL_RE.findall(raw_text or "")
    context_map: dict[str, str] = {}
    for line in (raw_text or "").splitlines():
        for u in _URL_RE.findall(line):
            context_map[u] = line.strip()[:300]
    papers = normalize_paper_urls(
        _papers_from_urls(list(dict.fromkeys(urls)), context_map)
    )
    if papers:
        trace(f"Consensus ✓ text papers={len(papers)}")
    return papers


def merge_scholar_papers(
    primary: List[ScholarPaper],
    extra: List[ScholarPaper],
) -> List[ScholarPaper]:
    out: List[ScholarPaper] = []
    seen: set[str] = set()

    def key(p: ScholarPaper) -> str:
        return (p.source_url or p.title or "").lower().strip()

    for batch in (primary, extra):
        for p in batch:
            k = key(p)
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(p)
    return out


def consensus_docs_to_papers(docs: list[dict[str, Any]]) -> List[ScholarPaper]:
    papers: List[ScholarPaper] = []
    for d in docs:
        url = str(d.get("url") or "").strip()
        title = _clean_title(str(d.get("title") or ""))
        snippet = str(d.get("snippet") or "").strip()
        if is_generic_consensus_url(url):
            url = ""
        if not url and not title:
            continue
        papers.append(
            ScholarPaper(
                title=title or "publication",
                abstract=snippet[:4000],
                source_url=url,
                source="consensus_validator",
            )
        )
    return papers


async def enrich_papers_metadata(papers: List[ScholarPaper]) -> List[ScholarPaper]:
    """Дополнить abstract/tldr через Semantic Scholar по заголовку (если API включён)."""
    from knowledge_engine.config import SEMANTIC_SCHOLAR_ENABLED

    if not SEMANTIC_SCHOLAR_ENABLED:
        trace("Consensus enrich ⊘ Semantic Scholar disabled — keep extracted metadata")
        return papers

    from knowledge_engine.src.retrieval.semantic_scholar import search_semantic_scholar

    enriched: List[ScholarPaper] = []
    for p in papers:
        if (p.abstract or p.tldr) and len(p.title) > 8:
            enriched.append(p)
            continue
        if len(p.title) < 8:
            enriched.append(p)
            continue
        hits = await search_semantic_scholar(p.title, limit=1)
        if not hits:
            enriched.append(p)
            continue
        h = hits[0]
        if not h.title or h.title.lower()[:20] not in p.title.lower()[:40]:
            enriched.append(p)
            continue
        enriched.append(
            ScholarPaper(
                paper_id=h.paper_id or p.paper_id,
                title=p.title or h.title,
                year=h.year or p.year,
                tldr=h.tldr or p.tldr,
                abstract=h.abstract or p.abstract,
                citation_count=h.citation_count,
                venue=h.venue or p.venue,
                pdf_url=h.pdf_url or p.pdf_url,
                source_url=p.source_url or h.source_url,
                source="consensus+semantic_scholar",
            )
        )
    return enriched
