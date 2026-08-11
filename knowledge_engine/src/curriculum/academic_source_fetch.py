"""Академический сбор: Semantic Scholar → arXiv, без 7B когда есть TLDR/abstract."""

from __future__ import annotations

import asyncio

from knowledge_engine.config import (
    CURRICULUM_ACADEMIC_ABSTRACT_MIN_CHARS,
    CURRICULUM_ACADEMIC_ARXIV_LIMIT,
    CURRICULUM_ACADEMIC_MIN_VALID_REUSE_AFTER_LITE,
    CURRICULUM_ACADEMIC_SEARXNG_LIMIT,
    CURRICULUM_ACADEMIC_SS_LIMIT,
    CURRICULUM_DEEP_NODE_MAX_HITS,
    CURRICULUM_ON_DEMAND_V08_POOL_SIZE,
    CURRICULUM_SEARCH_TARGET_HITS,
    CURRICULUM_USE_V08_CONSENSUS,
)
from knowledge_engine.schemas import DocumentSummary
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.src.curriculum.academic_consensus import (
    harvest_consensus_for_node,
    is_sota_rd_node,
)
from knowledge_engine.src.curriculum.academic_searxng_search import (
    collect_searxng_academic_rows,
)
from knowledge_engine.src.curriculum.academic_url_canonicalizer import (
    academic_source_dedupe_key,
    canonicalize_curriculum_hit,
)
from knowledge_engine.src.curriculum.curriculum_v08_harvest import _deep_extract_blocks
from knowledge_engine.src.curriculum.lite_search_pipeline import (
    batch_lite_eval_curriculum_hits,
    build_academic_search_plan,
)
from knowledge_engine.src.curriculum.schemas import CurriculumNode, CurriculumSearchHit
from knowledge_engine.src.curriculum.source_material_pipeline import (
    _ingest_academic_hit_async,
    _ingest_blog_hit_async,
)
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
        trace(f"CURRICULUM academic skip | no text | " f"{_paper_url(paper)[:70]}")
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


def hit_from_searxng_academic_row(row: dict[str, str]) -> CurriculumSearchHit:
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


def _hit_from_searxng_academic_row(row: dict[str, str]) -> CurriculumSearchHit:
    return hit_from_searxng_academic_row(row)


async def _primary_academic_hits(
    query: str,
    *,
    arxiv_params=None,
    min_hits: int | None = None,
) -> list[CurriculumSearchHit]:
    """Semantic Scholar → SearXNG science → arXiv with relaxation cascade."""
    from knowledge_engine.config import ACADEMIC_RELAXATION_MIN_HITS
    from knowledge_engine.src.retrieval.academic_rerank import (
        RelaxationLevel,
        relax_arxiv_params,
        relaxation_levels,
        should_relax,
        signals_from_scholar_paper,
        sort_by_hybrid_score,
    )
    from knowledge_engine.src.retrieval.arxiv_hydrate import hydrate_scholar_papers

    threshold = (
        max(1, int(min_hits))
        if min_hits is not None
        else max(1, int(ACADEMIC_RELAXATION_MIN_HITS))
    )

    papers = await search_semantic_scholar(
        query,
        limit=CURRICULUM_ACADEMIC_SS_LIMIT,
        ignore_enabled_flag=True,
    )
    if papers:
        papers = await hydrate_scholar_papers(papers)
        ranked = sort_by_hybrid_score(
            papers,
            signals_of=lambda p: signals_from_scholar_paper(
                p,
                relevance_sim=0.8,
                trust_score=1.0,
            ),
            level=RelaxationLevel.STRICT,
        )
        if ranked:
            papers = ranked
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

    if should_relax(hit_count=len(hits), min_hits=threshold) or not hits:
        for level in relaxation_levels():
            if len(hits) >= threshold and hits:
                break
            relaxed = relax_arxiv_params(arxiv_params, level)
            trace(
                f"CURRICULUM academic | arXiv relax L{int(level)} ▶ "
                f"hits={len(hits)}<{threshold} "
                f"cats={relaxed.categories[:3]} "
                f"years={relaxed.start_year}-{relaxed.end_year}"
            )
            arxiv_papers = await search_arxiv_fallback(
                query,
                limit=max(CURRICULUM_ACADEMIC_ARXIV_LIMIT, threshold),
                arxiv_params=relaxed,
            )
            ranked = sort_by_hybrid_score(
                arxiv_papers,
                signals_of=lambda p, level=level: signals_from_scholar_paper(
                    p,
                    relevance_sim=(
                        0.7 if level < RelaxationLevel.BROAD_RELEVANCE else 0.85
                    ),
                    trust_score=0.5,
                ),
                level=level,
            )
            # When rerank flag off, ranked == input; when on and gated empty, try next level.
            to_add = (
                ranked
                if ranked
                else (arxiv_papers if level >= RelaxationLevel.BROAD_RELEVANCE else [])
            )
            before = len(hits)
            hits.extend(await _collect_paper_hits_bounded(to_add, seen))
            trace(
                f"CURRICULUM academic | arXiv relax L{int(level)} ✓ "
                f"+{len(hits) - before} hits total={len(hits)}"
            )
            if not should_relax(hit_count=len(hits), min_hits=threshold):
                break

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
        reuse_pool_cap = max(
            CURRICULUM_ON_DEMAND_V08_POOL_SIZE,
            CURRICULUM_ON_DEMAND_V08_MAX_PAPERS + 4,
        )
        valid_reuse: list[CurriculumSearchHit] = []
        if on_demand and node:
            from knowledge_engine.src.curriculum.on_demand_reuse import (
                merge_on_demand_reuse_hits,
            )

            reuse_raw = merge_on_demand_reuse_hits(
                vec,
                list(registry_entries or []),
                cap=reuse_pool_cap,
                exclude_url_keys=exclude,
            )
            if reuse_raw:
                lite_anchor = anchor or f"curriculum_academic:{vec[:400]}"
                valid_reuse = await batch_lite_eval_curriculum_hits(
                    reuse_raw,
                    vec,
                    anchor=f"{lite_anchor}:on_demand_reuse:{node.node_id}",
                    strict=False,
                )
                trace(
                    f"CURRICULUM academic on_demand reuse lite | "
                    f"raw={len(reuse_raw)} approved={len(valid_reuse)} "
                    f"node={node.node_id}"
                )

        min_reuse = max(1, CURRICULUM_ACADEMIC_MIN_VALID_REUSE_AFTER_LITE)
        force_live_academic = (
            on_demand and node is not None and len(valid_reuse) < min_reuse
        )
        if (
            on_demand
            and node
            and not force_live_academic
            and len(valid_reuse) >= min_reuse
        ):
            trace(
                f"CURRICULUM academic on_demand ✓ | reuse after lite hits={len(valid_reuse)} "
                f"(skip live Consensus)"
            )
            target_early = (
                min_hits if min_hits is not None else CURRICULUM_DEEP_NODE_MAX_HITS
            )
            cap_early = max(target_early, CURRICULUM_DEEP_NODE_MAX_HITS + 2)
            return _merge_dedupe_hits([valid_reuse], cap_early)

        if force_live_academic:
            trace(
                f"CURRICULUM academic force live ▶ | node={node.node_id} "
                f"valid_reuse={len(valid_reuse)} < {min_reuse}"
            )

        plan = await build_academic_search_plan(
            vec,
            anchor=anchor or f"curriculum_academic:{vec[:400]}",
        )
        q = (plan.academic_query_en or "").strip()
        if not q:
            return []
        trace(f"CURRICULUM academic ▶ | query={q[:100]}")
        if not on_demand:
            trace(
                "CURRICULUM academic | Consensus hits use enrich_papers_metadata "
                f"(SS skip if abstract≥{CURRICULUM_ACADEMIC_ABSTRACT_MIN_CHARS}, "
                "SS enrich timeout guard)"
            )

        target = (
            min_hits
            if min_hits is not None
            else (
                CURRICULUM_DEEP_NODE_MAX_HITS if node else CURRICULUM_ACADEMIC_SS_LIMIT
            )
        )
        cap = (
            max(target, CURRICULUM_DEEP_NODE_MAX_HITS + 2)
            if node
            else (CURRICULUM_SEARCH_TARGET_HITS)
        )
        if on_demand:
            cap = min(cap, CURRICULUM_ON_DEMAND_V08_MAX_PAPERS + 2)

        sota = node is not None and is_sota_rd_node(node)
        parts: list[list[CurriculumSearchHit]] = []
        if valid_reuse:
            parts.append(valid_reuse)

        live_on_demand = on_demand and not force_live_academic

        if (
            node
            and sota
            and allow_consensus
            and CURRICULUM_USE_V08_CONSENSUS
            and (
                not live_on_demand
                or force_live_academic
                or sum(len(p) for p in parts) < CURRICULUM_ON_DEMAND_V08_MAX_PAPERS
            )
        ):
            parts.append(
                await harvest_consensus_for_node(
                    node,
                    vec,
                    anchor,
                    "sota_required",
                    on_demand=live_on_demand and not force_live_academic,
                )
            )

        if (
            not live_on_demand
            or force_live_academic
            or sum(len(p) for p in parts) < cap
        ):
            parts.append(
                await _primary_academic_hits(
                    q,
                    arxiv_params=plan.arxiv_params,
                    min_hits=target,
                )
            )
        hits = _merge_dedupe_hits(parts, cap)

        if (
            (not on_demand or force_live_academic)
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
                trace(
                    f"CURRICULUM consensus ✗ | node=— reason=academic_fallback | {exc}"
                )

        trace(f"CURRICULUM academic ✓ | hits={len(hits)}")
        return hits

    if on_demand:
        with fast_academic_fetch_scope(True):
            return await _body()
    return await _body()


def fetch_academic_sources(expansion_vector: str) -> list[CurriculumSearchHit]:
    return asyncio.run(fetch_academic_sources_async(expansion_vector))


HITS_QUEUE_SENTINEL: None = None

_STREAM_ACADEMIC_TIERS = frozenset(
    {
        "consensus",
        "arxiv",
        "semantic_scholar",
        "searxng_science",
        "openalex",
        "academic",
    }
)


def _stream_is_academic_hit(hit: CurriculumSearchHit) -> bool:
    tier = (hit.source_tier or "").strip().lower()
    return tier in _STREAM_ACADEMIC_TIERS or tier.startswith("consensus")


def _stream_hit_extract_words(hit: CurriculumSearchHit) -> int:
    return sum(len((e or "").split()) for e in (hit.key_extracts or []))


def _stream_skip_lite_validate(hit: CurriculumSearchHit) -> bool:
    """Exa уже прошёл Lite на этапе exa_transform; academic — всегда Lite в consumer."""
    tier = (hit.source_tier or "").strip().lower()
    if tier == "exa" and hit.skip_ollama_summary:
        return True
    return False


def _stream_skip_practical_ingest(hit: CurriculumSearchHit) -> bool:
    """Exa/lite path already produced usable extracts — avoid serial map-reduce."""
    if not hit.skip_ollama_summary:
        return False
    return _stream_hit_extract_words(hit) >= 80


def paper_to_stream_discovery_hit(paper: ScholarPaper) -> CurriculumSearchHit | None:
    url = _paper_url(paper)
    if not url.startswith("http"):
        return None
    title = (paper.title or url)[:400]
    snippet = (paper.abstract or paper.tldr or "")[:1200]
    extracts = _deep_extract_blocks([], [], [snippet], min_words=80, max_words=300)
    if not extracts:
        return None
    tier = "arxiv" if paper.source == "arxiv" else "semantic_scholar"
    return CurriculumSearchHit(
        url=url[:2000],
        title=title,
        snippet=snippet[:1200],
        key_extracts=extracts[:8],
        source_tier=tier,
        skip_ollama_summary=True,
    )


async def _stream_lite_validate_hit(
    hit: CurriculumSearchHit,
    goal: str,
    anchor: str,
    node_id: str,
) -> CurriculumSearchHit | None:
    if _stream_skip_lite_validate(hit):
        return hit
    strict = not _stream_is_academic_hit(hit)
    approved = await batch_lite_eval_curriculum_hits(
        [hit],
        goal,
        anchor=f"{anchor}:stream:{node_id}",
        strict=strict,
    )
    return approved[0] if approved else None


async def stream_hit_from_paper(paper: ScholarPaper) -> CurriculumSearchHit | None:
    hit = paper_to_stream_discovery_hit(paper)
    if hit is not None:
        return hit
    return await _process_paper(paper)


def _stream_should_paper_structure_ingest(hit: CurriculumSearchHit) -> bool:
    from knowledge_engine.src.parsers.paper_structure_analyzer import (
        is_academic_pdf_url,
    )

    return _stream_is_academic_hit(hit) or is_academic_pdf_url(hit.url)


async def _stream_ingest_hit(
    hit: CurriculumSearchHit,
    goal: str,
) -> CurriculumSearchHit | None:
    from knowledge_engine.services.academic_gemma_ingest import (
        ingest_academic_body_gemma,
    )
    from knowledge_engine.src.parsers.paper_structure_analyzer import (
        try_fetch_pdf_bytes_for_url,
    )

    if _stream_should_paper_structure_ingest(hit):
        body = "\n\n".join(
            [hit.snippet or "", "\n\n".join(hit.key_extracts or [])]
        ).strip()
        if len(body) < 80:
            updated = await _ingest_academic_hit_async(hit)
            return updated
        pdf_bytes = await asyncio.to_thread(try_fetch_pdf_bytes_for_url, hit.url)
        store = VectorStore()
        ing = await ingest_academic_body_gemma(
            hit.title or hit.url,
            hit.url,
            body,
            store,
            target_topic=goal,
            pdf_bytes=pdf_bytes,
            gemma_budget_blocking=True,
        )
        if ing is None:
            return await _ingest_academic_hit_async(hit)
        extracts = _hit_extracts_from_summary(ing.summary)
        return hit.model_copy(
            update={
                "title": (ing.summary.title or hit.title)[:400],
                "key_extracts": extracts,
            }
        )
    if _stream_skip_practical_ingest(hit):
        return hit
    return await _ingest_blog_hit_async(hit)


async def process_hits_stream(
    hits_queue: asyncio.Queue,
    node: CurriculumNode,
    *,
    goal: str,
    anchor: str,
    source_policy: str,
    out_hits: list[CurriculumSearchHit],
    pool_cap: int,
    seen_urls: set[str],
    seen_lock: asyncio.Lock,
) -> None:
    trace(
        f"CURRICULUM stream consumer ▶ | node={node.node_id} " f"policy={source_policy}"
    )
    while True:
        item = await hits_queue.get()
        try:
            if item is HITS_QUEUE_SENTINEL:
                break
            hit = await canonicalize_curriculum_hit(item)
            url_key = academic_source_dedupe_key(hit.url)
            if not url_key:
                continue
            async with seen_lock:
                if not url_key or url_key in seen_urls:
                    continue
                if len(out_hits) >= pool_cap:
                    continue
                seen_urls.add(url_key)

            validated = await _stream_lite_validate_hit(hit, goal, anchor, node.node_id)
            if validated is None:
                continue

            # Gemma map-reduce только после replenish (lazy_ground summarize).
            processed = validated
            if _stream_is_academic_hit(validated) or not _stream_skip_practical_ingest(
                validated
            ):
                trace(
                    f"CURRICULUM stream ingest defer | post-replenish | "
                    f"{validated.url[:72]}"
                )

            async with seen_lock:
                if len(out_hits) < pool_cap:
                    out_hits.append(processed)
        except Exception as exc:
            trace(f"CURRICULUM stream consumer skip | {exc}")
        finally:
            hits_queue.task_done()

    trace(f"CURRICULUM stream consumer ✓ | node={node.node_id} hits={len(out_hits)}")
