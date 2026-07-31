"""Академический сбор: Semantic Scholar → arXiv, без 7B когда есть TLDR/abstract."""

from __future__ import annotations

import asyncio

from knowledge_engine.config import (
    CURRICULUM_ACADEMIC_ABSTRACT_MIN_CHARS,
    CURRICULUM_ACADEMIC_ARXIV_LIMIT,
    CURRICULUM_ACADEMIC_SEARXNG_LIMIT,
    CURRICULUM_ACADEMIC_SS_LIMIT,
    CURRICULUM_DEEP_NODE_MAX_HITS,
    CURRICULUM_SEARCH_TARGET_HITS,
    CURRICULUM_USE_V08_CONSENSUS,
)
from knowledge_engine.schemas import DocumentSummary
from knowledge_engine.src.curriculum.academic_consensus import (
    harvest_consensus_for_node,
    is_sota_rd_node,
)
from knowledge_engine.src.curriculum.schemas import CurriculumNode
from knowledge_engine.src.curriculum.academic_searxng_search import collect_searxng_academic_rows
from knowledge_engine.src.curriculum.curriculum_v08_harvest import _deep_extract_blocks
from knowledge_engine.src.curriculum.schemas import CurriculumSearchHit
from knowledge_engine.src.curriculum.lite_search_pipeline import build_academic_search_query
from knowledge_engine.src.retrieval.paper_documents import fetch_paper_document
from knowledge_engine.src.retrieval.semantic_scholar import (
    ScholarPaper,
    search_arxiv_fallback,
    search_semantic_scholar,
)
from knowledge_engine.ui.run_log import trace

_PAPER_PROCESS_SEM = asyncio.Semaphore(3)


async def _collect_paper_hits_bounded(
    papers: list[ScholarPaper],
    seen: set[str],
) -> list[CurriculumSearchHit]:
    """До 3 параллельных PDF/fetch; SS search остаётся с глобальным throttle."""

    async def _one(paper: ScholarPaper) -> CurriculumSearchHit | None:
        async with _PAPER_PROCESS_SEM:
            return await _process_paper(paper)

    if not papers:
        return []
    results = await asyncio.gather(*[_one(p) for p in papers])
    hits: list[CurriculumSearchHit] = []
    for hit in results:
        if not hit:
            continue
        key = hit.url.strip().rstrip("/").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        hits.append(hit)
    return hits


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
    extracts = _hit_extracts_from_summary(ds)
    snippet = (paper.abstract or paper.tldr or "")[:1200]
    tier = "arxiv" if paper.source == "arxiv" else "semantic_scholar"
    trace(
        f"CURRICULUM academic ready ✓ | {tier} | {ds.url[:70]} "
        "(LanceDB после batch approve)"
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


def _hit_from_searxng_academic_row(row: dict[str, str]) -> CurriculumSearchHit:
    snippet = (row.get("snippet") or "").strip()
    title = (row.get("title") or row.get("url") or "")[:400]
    url = row["url"]
    tier = (row.get("source_tier") or "searxng_science").strip()
    extracts = _deep_extract_blocks([], [], [snippet], min_words=40, max_words=200)
    if not extracts and snippet:
        extracts = [snippet[:800]]
    return CurriculumSearchHit(
        url=url,
        title=title,
        snippet=snippet[:1200],
        key_extracts=extracts[:8],
        source_tier=tier,
    )


async def _primary_academic_hits(query: str) -> list[CurriculumSearchHit]:
    """Semantic Scholar → SearXNG science → arXiv (без Consensus)."""
    papers = await search_semantic_scholar(
        query,
        limit=CURRICULUM_ACADEMIC_SS_LIMIT,
        ignore_enabled_flag=True,
    )
    if papers:
        trace(f"CURRICULUM academic | Semantic Scholar papers={len(papers)}")
    else:
        trace("CURRICULUM academic | Semantic Scholar 0")

    hits: list[CurriculumSearchHit] = []
    seen: set[str] = set()

    hits.extend(await _collect_paper_hits_bounded(papers, seen))

    need_more = len(hits) < CURRICULUM_ACADEMIC_SS_LIMIT
    if need_more:
        sx_rows = await collect_searxng_academic_rows(
            query,
            limit=CURRICULUM_ACADEMIC_SEARXNG_LIMIT,
        )
        for row in sx_rows:
            key = row["url"].strip().rstrip("/").lower()
            if not key or key in seen:
                continue
            seen.add(key)
            hits.append(_hit_from_searxng_academic_row(row))

    if not hits:
        trace("CURRICULUM academic | SearXNG science empty → arXiv API fallback")
        papers = await search_arxiv_fallback(query, limit=CURRICULUM_ACADEMIC_ARXIV_LIMIT)
        hits.extend(await _collect_paper_hits_bounded(papers, seen))
    return hits


def _merge_dedupe_hits(
    parts: list[list[CurriculumSearchHit]],
    cap: int,
) -> list[CurriculumSearchHit]:
    seen: set[str] = set()
    out: list[CurriculumSearchHit] = []
    for batch in parts:
        for h in batch:
            key = h.url.strip().rstrip("/").lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(h)
            if len(out) >= cap:
                return out
    return out


async def fetch_academic_sources_async(
    expansion_vector: str,
    *,
    node: CurriculumNode | None = None,
    anchor: str = "",
    min_hits: int | None = None,
    allow_consensus: bool = True,
    on_demand: bool = False,
    registry_entries: list | None = None,
    exclude_url_keys: set[str] | None = None,
) -> list[CurriculumSearchHit]:
    vec = (expansion_vector or "").strip()
    if len(vec) < 8:
        return []

    from knowledge_engine.config import CURRICULUM_ON_DEMAND_V08_MAX_PAPERS
    from knowledge_engine.src.fetcher.context import fast_academic_fetch_scope

    exclude = set(exclude_url_keys or [])

    async def _body() -> list[CurriculumSearchHit]:
        if on_demand and node:
            from knowledge_engine.src.curriculum.on_demand_reuse import (
                merge_on_demand_reuse_hits,
            )

            reused = merge_on_demand_reuse_hits(
                vec,
                list(registry_entries or []),
                cap=CURRICULUM_ON_DEMAND_V08_MAX_PAPERS,
                exclude_url_keys=exclude,
            )
            if len(reused) >= CURRICULUM_ON_DEMAND_V08_MAX_PAPERS:
                trace(
                    f"CURRICULUM academic on_demand ✓ | reuse only hits={len(reused)} "
                    "(skip Consensus Playwright)"
                )
                return reused

        q = await build_academic_search_query(
            vec,
            anchor=anchor or f"curriculum_academic:{vec[:400]}",
        )
        if not q:
            return []
        trace(f"CURRICULUM academic ▶ | query={q[:100]}")
        if not on_demand:
            trace(
                "CURRICULUM academic | Consensus hits use enrich_papers_metadata "
                f"(SS skip if abstract≥{CURRICULUM_ACADEMIC_ABSTRACT_MIN_CHARS}, "
                "SS enrich timeout guard)"
            )

        target = min_hits if min_hits is not None else (
            CURRICULUM_DEEP_NODE_MAX_HITS if node else CURRICULUM_ACADEMIC_SS_LIMIT
        )
        cap = max(target, CURRICULUM_DEEP_NODE_MAX_HITS + 2) if node else (
            CURRICULUM_SEARCH_TARGET_HITS
        )
        if on_demand:
            cap = min(cap, CURRICULUM_ON_DEMAND_V08_MAX_PAPERS + 2)

        sota = node is not None and is_sota_rd_node(node)
        parts: list[list[CurriculumSearchHit]] = []

        if on_demand and node:
            from knowledge_engine.src.curriculum.on_demand_reuse import (
                merge_on_demand_reuse_hits,
            )

            pre = merge_on_demand_reuse_hits(
                vec,
                list(registry_entries or []),
                cap=CURRICULUM_ON_DEMAND_V08_MAX_PAPERS,
                exclude_url_keys=exclude,
            )
            if pre:
                parts.append(pre)

        if (
            node
            and sota
            and allow_consensus
            and CURRICULUM_USE_V08_CONSENSUS
            and (not on_demand or sum(len(p) for p in parts) < CURRICULUM_ON_DEMAND_V08_MAX_PAPERS)
        ):
            parts.append(
                await harvest_consensus_for_node(
                    node,
                    vec,
                    anchor,
                    "sota_required",
                    on_demand=on_demand,
                )
            )

        if not on_demand or sum(len(p) for p in parts) < cap:
            parts.append(await _primary_academic_hits(q))
        hits = _merge_dedupe_hits(parts, cap)

        if (
            not on_demand
            and node
            and allow_consensus
            and CURRICULUM_USE_V08_CONSENSUS
            and not sota
            and len(hits) < target
        ):
            extra = await harvest_consensus_for_node(
                node, vec, anchor, "academic_fallback", on_demand=False
            )
            hits = _merge_dedupe_hits([hits, extra], cap)

        if (
            not on_demand
            and not node
            and allow_consensus
            and CURRICULUM_USE_V08_CONSENSUS
            and not hits
        ):
            trace("CURRICULUM consensus ▶ | node=— reason=academic_fallback (bulk)")
            from knowledge_engine.src.curriculum.curriculum_v08_harvest import (
                harvest_curriculum_sources_v08,
            )

            try:
                bulk = await harvest_curriculum_sources_v08(
                    vec,
                    anchor or f"curriculum_academic:{vec[:400]}",
                )
                hits = _merge_dedupe_hits([hits, bulk], cap)
                trace(
                    f"CURRICULUM consensus ✓ | node=— reason=academic_fallback "
                    f"hits={len(bulk)}"
                )
            except Exception as exc:
                trace(f"CURRICULUM consensus ✗ | node=— reason=academic_fallback | {exc}")

        trace(f"CURRICULUM academic ✓ | hits={len(hits)}")
        return hits

    if on_demand:
        with fast_academic_fetch_scope(True):
            return await _body()
    return await _body()


def fetch_academic_sources(expansion_vector: str) -> list[CurriculumSearchHit]:
    return asyncio.run(fetch_academic_sources_async(expansion_vector))

