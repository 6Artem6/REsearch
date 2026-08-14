"""Извлечение карточек публикаций из UI Consensus (DOM + текст ответа)."""

from __future__ import annotations

import asyncio
import re
from typing import Any, List
from urllib.parse import unquote

from playwright.async_api import Page

from knowledge_engine.config import (
    CURRICULUM_ACADEMIC_ABSTRACT_MIN_CHARS,
    SEMANTIC_SCHOLAR_ENABLED,
    SEMANTIC_SCHOLAR_ENRICH_TIMEOUT_SEC,
)
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
                arxiv_id=m.group(1) if m else "",
            )
        )
    return papers


async def extract_paper_cards_from_page(page: Page) -> List[ScholarPaper]:
    """Собрать ссылки на публикации из UI (quick search / thread answer)."""
    try:
        raw_links: list[dict[str, Any]] = await page.evaluate(
            """() => {
                const sel = 'a[href*="doi.org"], a[href*="arxiv.org"], a[href*="semanticscholar"], '
                    + 'a[href*="pubmed"], a[href*="consensus.app/papers"], a[href*="consensus.app/paper"], '
                    + '[data-testid*="source"] a, [data-testid*="citation"] a, [data-testid*="paper"] a, '
                    + '[data-testid*="Paper"] a, [data-testid*="result"] a';
                const anchors = Array.from(document.querySelectorAll(sel));
                const main = document.querySelector('main') || document.body;
                const extra = Array.from(main.querySelectorAll('a[href]'));
                const cardRoots = Array.from(
                    main.querySelectorAll(
                        '[data-testid*="paper"], [data-testid*="Paper"], '
                        + '[data-testid*="result"], article, li'
                    )
                );
                const all = [...anchors, ...extra];
                const seen = new Set();
                const out = [];
                for (const a of all) {
                    const href = a.href;
                    if (!href || seen.has(href)) continue;
                    seen.add(href);
                    const block = (a.closest('[data-testid*="paper"], [data-testid*="Paper"], article, li')
                        || a.parentElement)?.innerText?.slice(0, 800) || '';
                    out.push({
                        href,
                        text: (a.innerText || a.textContent || '').trim().slice(0, 300),
                        block
                    });
                }
                for (const card of cardRoots.slice(0, 120)) {
                    const t = (card.innerText || '').trim().slice(0, 900);
                    if (t.length < 40) continue;
                    const innerLinks = card.querySelectorAll('a[href]');
                    for (const a of innerLinks) {
                        const href = a.href;
                        if (!href || seen.has(href)) continue;
                        seen.add(href);
                        out.push({
                            href,
                            text: (a.innerText || '').trim().slice(0, 300),
                            block: t
                        });
                    }
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
        snippet = block if len(block) > len(text) else text
        if not snippet and block:
            snippet = block.split("\n")[0]
        context_map[href] = snippet
        urls.append(href)

    papers = normalize_paper_urls(_papers_from_urls(urls, context_map))
    enriched: List[ScholarPaper] = []
    for p in papers:
        blk = context_map.get((p.source_url or "").strip(), "")
        if blk and len(blk) > 60 and len((p.abstract or "")) < 40:
            enriched.append(p.model_copy(update={"abstract": blk[:800]}))
        else:
            enriched.append(p)
    trace(f"Consensus ✓ DOM papers={len(enriched)}")
    return enriched


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


def _paper_has_usable_abstract(paper: ScholarPaper) -> bool:
    return len((paper.abstract or "").strip()) >= CURRICULUM_ACADEMIC_ABSTRACT_MIN_CHARS


async def _ss_enrich_one_paper(
    paper: ScholarPaper,
    *,
    ignore_enabled_flag: bool,
) -> ScholarPaper:
    if _paper_has_usable_abstract(paper):
        return paper
    title = (paper.title or "").strip()
    if len(title) < 8:
        return paper

    from knowledge_engine.src.retrieval.semantic_scholar import search_semantic_scholar

    timeout = max(0.5, SEMANTIC_SCHOLAR_ENRICH_TIMEOUT_SEC)
    try:
        hits = await asyncio.wait_for(
            search_semantic_scholar(
                title,
                limit=1,
                ignore_enabled_flag=ignore_enabled_flag,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        trace(
            f"Consensus enrich ⊘ SS timeout ({timeout:.1f}s) | "
            f"{title[:60]} — keep as-is"
        )
        return paper
    except Exception as exc:
        trace(f"Consensus enrich ⊘ SS | {title[:50]} | {exc}")
        return paper

    if not hits:
        return paper
    h = hits[0]
    if not h.title or h.title.lower()[:20] not in title.lower()[:40]:
        return paper
    return ScholarPaper(
        paper_id=h.paper_id or paper.paper_id,
        title=paper.title or h.title,
        year=h.year or paper.year,
        tldr=h.tldr or paper.tldr,
        abstract=h.abstract or paper.abstract,
        citation_count=h.citation_count,
        venue=h.venue or paper.venue,
        pdf_url=h.pdf_url or paper.pdf_url,
        source_url=paper.source_url or h.source_url,
        source="consensus+semantic_scholar",
        arxiv_id=paper.arxiv_id or h.arxiv_id,
        doi=paper.doi or h.doi,
    )


async def enrich_papers_metadata(
    papers: List[ScholarPaper],
    *,
    ignore_enabled_flag: bool = False,
) -> List[ScholarPaper]:
    """Дополнить abstract через SS по заголовку; затем arXiv id_list hydrate."""
    from knowledge_engine.src.retrieval.arxiv_hydrate import hydrate_scholar_papers

    if not papers:
        return papers

    enriched: List[ScholarPaper] = list(papers)
    if not SEMANTIC_SCHOLAR_ENABLED and not ignore_enabled_flag:
        trace("Consensus enrich ⊘ Semantic Scholar disabled — keep extracted metadata")
    else:
        need = sum(1 for p in papers if not _paper_has_usable_abstract(p))
        skip_n = len(papers) - need
        if skip_n:
            trace(f"Consensus enrich ⊘ SS skip | abstract_ok={skip_n}/{len(papers)}")
        if need > 0:
            if ignore_enabled_flag:
                from knowledge_engine.services.curriculum_api_quota_store import (
                    can_use_semantic_scholar,
                )

                allowed, why = can_use_semantic_scholar()
                if not allowed:
                    trace(
                        f"Consensus enrich ⊘ SS quota | {why} — skip {need} title lookups"
                    )
                    return await hydrate_scholar_papers(enriched)

            trace(
                f"Consensus enrich ▶ SS | lookups={need} "
                f"timeout={SEMANTIC_SCHOLAR_ENRICH_TIMEOUT_SEC:.1f}s per title"
            )
            enriched = []
            for p in papers:
                if _paper_has_usable_abstract(p):
                    enriched.append(p)
                    continue
                enriched.append(
                    await _ss_enrich_one_paper(
                        p, ignore_enabled_flag=ignore_enabled_flag
                    )
                )

    return await hydrate_scholar_papers(enriched)
