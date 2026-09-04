"""Hybrid Search Pipeline: LanceDB enrich, Gemini Grounding blogs, Summarizer."""

from __future__ import annotations

import asyncio
import re
import threading

from knowledge_engine.config import (
    CURRICULUM_CONSENSUS_MIN_APPROVED_ACADEMIC,
    CURRICULUM_GEMINI_GROUNDING_ENABLED,
    CURRICULUM_GEMINI_WEB_HARVEST_ENABLED,
    CURRICULUM_LITE_SITE_SUGGEST_ENABLED,
    CURRICULUM_SEARCH_MIN_HITS,
    CURRICULUM_SEARCH_TARGET_HITS,
    CURRICULUM_URL_VALIDATE_TIMEOUT_SEC,
    CURRICULUM_USE_V08_CONSENSUS,
    KE_INGEST_URL_CONCURRENCY,
)
from knowledge_engine.db.domain_blocklist import add_blocked_domain
from knowledge_engine.ingestion.ingest import ingest_exa_highlights_fallback
from knowledge_engine.services.academic_gemma_ingest import ingest_academic_body_gemma
from knowledge_engine.services.gemini_search_grounding import (
    search_grounded_whitelist_blogs_detailed,
)
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.services.web_extract import (
    is_anti_bot_fetch_result,
    smart_fetch_page_html,
    smart_fetch_page_text,
)
from knowledge_engine.src.curriculum.curriculum_v08_harvest import _deep_extract_blocks
from knowledge_engine.src.curriculum.schemas import CurriculumGraph, CurriculumSearchHit
from knowledge_engine.src.curriculum.search_prestep import _normalize_url_key
from knowledge_engine.src.curriculum.source_hit_curation import (
    collect_archived_practical_hits,
    curate_practical_hits,
)
from knowledge_engine.src.curriculum.url_validate import validate_and_filter_urls
from knowledge_engine.ui.run_log import trace

DEEP_BLOG_EXTRACT_WORDS = 120
PRE_MAP_MIN_BODY_WORDS = 80

_BLOG_SOURCE_TIERS = frozenset(
    {
        "whitelist_blog",
        "gemini_grounding",
        "gemini_web",
        "lite_suggested_site",
        "archive",
        "google_cse",
        "searxng",
        "ddgs",
        "exa",
    }
)

_ACADEMIC_SOURCE_TIERS = frozenset(
    {
        "semantic_scholar",
        "arxiv",
        "consensus",
        "searxng_science",
    }
)

_CHUNK_SPLIT = re.compile(r"\n\s*\n|(?<=[.!?])\s+")


def _split_extracts(text: str, max_chunks: int = 6, chunk_len: int = 600) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    parts: list[str] = []
    for block in _CHUNK_SPLIT.split(raw):
        s = block.strip()
        if len(s) < 40:
            continue
        if len(s) > chunk_len:
            s = s[:chunk_len].rsplit(" ", 1)[0] + "…"
        parts.append(s)
        if len(parts) >= max_chunks:
            break
    return parts


def _extracts_from_document_summary(ds) -> list[str]:
    """Полноценные выжимки из LanceDB (Consensus summarizer)."""
    extra: list[str] = []
    exec_sum = (getattr(ds, "executive_summary", None) or "").strip()
    if exec_sum:
        extra.append(exec_sum)
    return _deep_extract_blocks(
        list(ds.key_takeaways or []),
        list(ds.failure_modes or []),
        extra,
        min_words=80,
        max_words=300,
    )


def _extract_word_total(extracts: list[str]) -> int:
    return sum(len((e or "").split()) for e in extracts)


async def enrich_search_hits_with_extracts_async(
    hits: list[CurriculumSearchHit],
    target_goal: str = "",
) -> list[CurriculumSearchHit]:
    """key_extracts из LanceDB (takeaways + failure_modes) или v0.8 harvest."""
    if not hits:
        return hits

    urls = [h.url for h in hits if h.url.startswith("http")]
    by_url: dict[str, object] = {}
    try:
        store = VectorStore()
        for ds in await store.fetch_summaries_by_urls(urls, limit=len(urls) + 2):
            key = (ds.url or "").strip().rstrip("/").lower()
            if key:
                by_url[key] = ds
    except Exception as exc:
        trace(f"CURRICULUM extracts LanceDB skip | {exc}")

    enriched: list[CurriculumSearchHit] = []
    for i, hit in enumerate(hits, start=1):
        key_lower = hit.url.strip().rstrip("/").lower()
        ds = by_url.get(key_lower)
        extracts: list[str] = list(hit.key_extracts or [])
        mandatory_academic = hit_requires_mandatory_academic_ingest(hit)

        if hit.skip_ollama_summary and _extract_word_total(extracts) >= 100:
            pass
        elif ds and not mandatory_academic:
            lance_extracts = _extracts_from_document_summary(ds)
            if _extract_word_total(lance_extracts) > _extract_word_total(extracts):
                extracts = lance_extracts
            if not hit.title and ds.title:
                hit = hit.model_copy(update={"title": ds.title[:400]})

        if not extracts or _extract_word_total(extracts) < 120:
            if hit.skip_ollama_summary:
                pass
            elif ds and not mandatory_academic:
                extracts = _extracts_from_document_summary(ds) or extracts
            if not extracts and hit.snippet:
                extracts = _deep_extract_blocks([], [], [hit.snippet], 150, 300)
            if not extracts and hit.snippet:
                extracts = _split_extracts(hit.snippet, max_chunks=2, chunk_len=800)

        seen: set[str] = set()
        deduped: list[str] = []
        for e in extracts:
            e = e.strip()
            if not e or e in seen:
                continue
            seen.add(e)
            deduped.append(e[:2000])

        enriched.append(
            hit.model_copy(
                update={
                    "source_id": hit.source_id or f"src_{i}",
                    "key_extracts": deduped[:8],
                }
            )
        )

    deep = sum(1 for h in enriched if _extract_word_total(h.key_extracts) >= 150)
    trace(
        f"CURRICULUM extracts enrich ✓ | hits={len(enriched)} "
        f"deep_context={deep}/{len(enriched)}"
    )
    return enriched


def enrich_search_hits_with_extracts(
    hits: list[CurriculumSearchHit],
    target_goal: str = "",
) -> list[CurriculumSearchHit]:
    """Sync wrapper — legitimate top-level asyncio.run() bridge for callers
    rooted in the synchronous worker process (search-first pipeline, no event
    loop — see knowledge_engine/worker/__main__.py). An already-async caller
    must await enrich_search_hits_with_extracts_async(...) directly."""
    if not hits:
        return hits
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(enrich_search_hits_with_extracts_async(hits, target_goal))
    raise RuntimeError(
        "enrich_search_hits_with_extracts() called from inside a running "
        "event loop — await enrich_search_hits_with_extracts_async(...) "
        "directly instead"
    )


def collect_practical_blog_hits(
    target_goal: str,
    *,
    context_vector: str = "",
    limit_per_provider: int = 4,
    grounding_only: bool = False,
    defer_lite_batch: bool = False,
) -> list[CurriculumSearchHit]:
    """Практика: архив → CSE → SearXNG → (опц.) DDGS / Gemini web → Lite batch."""
    from knowledge_engine.src.curriculum.practical_source_fetch import (
        fetch_practical_sources,
    )

    exclude: set[str] = set()
    out: list[CurriculumSearchHit] = []
    anchor = f"curriculum_blogs:{(target_goal or '').strip()[:500]}"
    search_vec = (context_vector or target_goal or "").strip()

    for h in collect_archived_practical_hits(
        target_goal,
        exclude_url_keys=exclude,
        limit=4,
        strict=True,
    ):
        from knowledge_engine.src.curriculum.practical_url_filters import (
            practical_url_reject_reason,
        )

        reason = practical_url_reject_reason(h.url)
        if reason:
            trace(f"CURRICULUM practical filter ⊘ | {h.url[:70]} | {reason}")
            continue
        key = _normalize_url_key(h.url)
        if key:
            exclude.add(key)
            out.append(h)

    cap = CURRICULUM_SEARCH_TARGET_HITS
    min_blog = max(3, CURRICULUM_SEARCH_MIN_HITS // 2)

    try:
        api_hits = fetch_practical_sources(search_vec, max_hits=cap)
        for h in api_hits:
            key = _normalize_url_key(h.url)
            if not key or key in exclude:
                continue
            exclude.add(key)
            out.append(h)
    except Exception as exc:
        trace(f"CURRICULUM practical api skip | {exc}")

    if CURRICULUM_GEMINI_WEB_HARVEST_ENABLED:
        from knowledge_engine.src.curriculum.gemini_web_blog_harvest import (
            collect_gemini_web_practical_hits,
        )

        try:
            for h in collect_gemini_web_practical_hits(
                target_goal,
                context_vector=context_vector,
            ):
                key = _normalize_url_key(h.url)
                if not key or key in exclude:
                    continue
                exclude.add(key)
                out.append(h)
        except Exception as exc:
            trace(f"CURRICULUM gemini_web skip | {exc}")

    if CURRICULUM_LITE_SITE_SUGGEST_ENABLED and len(out) < cap:
        from knowledge_engine.src.curriculum.source_discovery_expand import (
            collect_lite_suggested_site_hits,
        )

        added = collect_lite_suggested_site_hits(
            target_goal,
            out,
            exclude_url_keys=exclude,
            limit_per_provider=limit_per_provider,
            max_hits=cap - len(out),
            anchor=anchor,
        )
        out.extend(added)
    elif len(out) < cap:
        trace(
            "CURRICULUM lite site suggest ⊘ | disabled "
            "(Lite Search Query Architect уже формирует site: запросы)"
        )

    if CURRICULUM_GEMINI_GROUNDING_ENABLED:
        try:
            gr = search_grounded_whitelist_blogs_detailed(
                target_goal,
                context_vector=context_vector,
            )
            for g in gr.hits:
                key = _normalize_url_key(g.url)
                if not key or key in exclude:
                    continue
                exclude.add(key)
                out.append(
                    CurriculumSearchHit(
                        url=g.url,
                        title=g.title,
                        snippet=g.snippet,
                        source_tier="gemini_grounding",
                    )
                )
        except Exception as exc:
            trace(f"CURRICULUM gemini_grounding skip | {exc}")
    else:
        trace(
            "CURRICULUM gemini_grounding ⊘ | disabled "
            "(CURRICULUM_GEMINI_GROUNDING_ENABLED=false)"
        )

    need_searx = len(out) < min_blog or (grounding_only and len(out) == 0)
    if need_searx and not any(h.source_tier == "searxng" for h in out):
        from knowledge_engine.src.curriculum.search_prestep import (
            _collect_whitelist_blog_hits,
        )

        trace(f"CURRICULUM searxng fallback ▶ | have={len(out)} need>={min_blog}")
        added = _collect_whitelist_blog_hits(
            target_goal,
            limit_per_provider=limit_per_provider,
            max_hits=cap - len(out),
            exclude_url_keys=exclude,
        )
        for h in added:
            key = _normalize_url_key(h.url)
            if key:
                exclude.add(key)
        out.extend(added)

    lite_cap = 16
    out = curate_practical_hits(
        out,
        target_goal,
        anchor=anchor,
        max_out=cap,
        lite_review_cap=lite_cap,
        lite_batch=not defer_lite_batch,
    )
    trace(
        f"CURRICULUM practical blogs ✓ | total={len(out)} "
        f"google_cse={sum(1 for h in out if h.source_tier == 'google_cse')} "
        f"exa={sum(1 for h in out if h.source_tier == 'exa')} "
        f"searxng={sum(1 for h in out if h.source_tier == 'searxng')} "
        f"ddgs={sum(1 for h in out if h.source_tier == 'ddgs')} "
        f"api_grounding={sum(1 for h in out if h.source_tier == 'gemini_grounding')} "
        f"gemini_web={sum(1 for h in out if h.source_tier == 'gemini_web')} "
        f"archive={sum(1 for h in out if h.source_tier == 'archive')} "
        f"searxng_fallback={sum(1 for h in out if h.source_tier == 'whitelist_blog')} "
    )
    from knowledge_engine.src.curriculum.curriculum_lancedb_persist import (
        persist_approved_curriculum_hits_to_lancedb,
    )

    persist_approved_curriculum_hits_to_lancedb(out, label="practical_post_batch")
    return out[:cap]


def _count_academic_hits(hits: list[CurriculumSearchHit]) -> int:
    return sum(
        1 for h in hits if (h.source_tier or "").strip() in _ACADEMIC_SOURCE_TIERS
    )


def _run_consensus_harvest(
    target_goal: str,
    context_vector: str,
) -> list[CurriculumSearchHit]:
    from knowledge_engine.src.curriculum.curriculum_v08_harvest import (
        harvest_curriculum_sources_v08,
    )

    goal = (target_goal or "").strip()
    vec = (context_vector or goal).strip()
    anchor = f"curriculum:{goal[:500]}"
    try:
        return asyncio.run(harvest_curriculum_sources_v08(vec or goal, anchor))
    except Exception as exc:
        trace(f"CURRICULUM consensus ✗ | {exc}")
        return []


def _supplement_academic_from_consensus(
    hits: list[CurriculumSearchHit],
    target_goal: str,
    context_vector: str,
    *,
    stage: str,
    force: bool = False,
) -> list[CurriculumSearchHit]:
    if not CURRICULUM_USE_V08_CONSENSUS:
        trace("CURRICULUM consensus ⊘ | CURRICULUM_USE_V08_CONSENSUS=false")
        return hits
    approved_academic = _count_academic_hits(hits)
    min_academic = CURRICULUM_CONSENSUS_MIN_APPROVED_ACADEMIC
    min_pool = CURRICULUM_SEARCH_MIN_HITS
    pool_thin = len(hits) < min_pool
    if not force and not pool_thin and approved_academic >= min_academic:
        trace(
            f"CURRICULUM consensus ⊘ | stage={stage} "
            f"pool={len(hits)} academic={approved_academic} "
            f"(need pool<{min_pool} or academic<{min_academic})"
        )
        return hits
    trace(
        f"CURRICULUM consensus ▶ | node=— reason=pool_fallback stage={stage} "
        f"force={force} pool={len(hits)} academic={approved_academic}"
    )
    extra = _run_consensus_harvest(target_goal, context_vector)
    if not extra:
        trace("CURRICULUM consensus ⊘ | harvest returned 0 hits")
        return hits
    merged = _merge_hit_lists([hits, extra])
    trace(
        f"CURRICULUM consensus ✓ | stage={stage} "
        f"added={len(extra)} merged={len(merged)}"
    )
    return merged


def _finalize_collected_hits(
    hits: list[CurriculumSearchHit],
    *,
    label: str,
) -> list[CurriculumSearchHit]:
    from knowledge_engine.src.curriculum.curriculum_lancedb_persist import (
        persist_approved_curriculum_hits_to_lancedb,
    )

    persist_approved_curriculum_hits_to_lancedb(hits, label=label)
    return hits


def _batch_filter_curriculum_hits(
    hits: list[CurriculumSearchHit],
    target_goal: str,
    *,
    anchor_suffix: str = "policy",
) -> list[CurriculumSearchHit]:
    if not hits:
        return hits
    from knowledge_engine.src.curriculum.lite_search_pipeline import (
        batch_lite_eval_curriculum_hits,
    )

    goal = (target_goal or "").strip()
    anchor = f"curriculum_sources:{goal[:500]}:{anchor_suffix}"

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        filtered = asyncio.run(
            batch_lite_eval_curriculum_hits(hits, goal, anchor=anchor)
        )
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1) as pool:
            filtered = pool.submit(
                asyncio.run,
                batch_lite_eval_curriculum_hits(hits, goal, anchor=anchor),
            ).result()

    cap = CURRICULUM_SEARCH_TARGET_HITS
    trace(
        f"CURRICULUM lite batch policy ✓ | suffix={anchor_suffix} "
        f"in={len(hits)} out={len(filtered)}"
    )
    return filtered[:cap]


def collect_academic_source_hits(
    target_goal: str,
    *,
    context_vector: str = "",
) -> list[CurriculumSearchHit]:
    """Semantic Scholar → arXiv; Consensus только fallback при пустом API (bulk)."""
    import asyncio

    from knowledge_engine.src.curriculum.academic_source_fetch import (
        fetch_academic_sources_async,
    )

    vec = (context_vector or target_goal or "").strip()
    if len(vec) < 8:
        return []

    hits = asyncio.run(
        fetch_academic_sources_async(
            vec,
            anchor=f"curriculum_academic:{vec[:400]}",
            allow_consensus=True,
        )
    )
    cap = CURRICULUM_SEARCH_TARGET_HITS
    trace(f"CURRICULUM academic ✓ | bulk hits={len(hits)}")
    return hits[:cap]


def _merge_hit_lists(
    parts: list[list[CurriculumSearchHit]],
    cap: int | None = None,
) -> list[CurriculumSearchHit]:
    cap = cap if cap is not None else CURRICULUM_SEARCH_TARGET_HITS
    seen: set[str] = set()
    out: list[CurriculumSearchHit] = []
    for batch in parts:
        for h in batch:
            key = _normalize_url_key(h.url)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(h)
            if len(out) >= cap:
                return out
    return out


def collect_sources_by_policy(
    target_goal: str,
    *,
    source_policy: str = "hybrid",
    context_vector: str = "",
    limit_per_provider: int = 4,
    grounding_only: bool = False,
) -> list[CurriculumSearchHit]:
    """
    hybrid — practical blogs, затем Consensus (последовательно, один Playwright profile).
    practical_only — whitelist blogs / web / search.
    academic_only — Consensus harvest only.
    """
    from knowledge_engine.src.curriculum.source_policy import normalize_source_policy

    policy = normalize_source_policy(source_policy, default="hybrid")
    trace(f"CURRICULUM sources ▶ | policy={policy}")

    if policy == "academic_only":
        hits = collect_academic_source_hits(
            target_goal,
            context_vector=context_vector or target_goal,
        )
        filtered = _batch_filter_curriculum_hits(
            hits,
            target_goal,
            anchor_suffix="academic_only",
        )
        filtered = _supplement_academic_from_consensus(
            filtered,
            target_goal,
            context_vector or target_goal,
            stage="academic_only",
        )
        return _finalize_collected_hits(filtered, label="academic_only")

    if policy == "practical_only":
        return collect_practical_blog_hits(
            target_goal,
            context_vector=context_vector,
            limit_per_provider=limit_per_provider,
            grounding_only=grounding_only,
        )

    practical = collect_practical_blog_hits(
        target_goal,
        context_vector=context_vector,
        limit_per_provider=limit_per_provider,
        grounding_only=grounding_only,
        defer_lite_batch=True,
    )
    academic = collect_academic_source_hits(
        target_goal,
        context_vector=context_vector or target_goal,
    )
    merged = _merge_hit_lists([practical, academic])
    filtered = _batch_filter_curriculum_hits(
        merged,
        target_goal,
        anchor_suffix="hybrid",
    )
    filtered = _supplement_academic_from_consensus(
        filtered,
        target_goal,
        context_vector or target_goal,
        stage="hybrid",
    )
    filtered = _finalize_collected_hits(filtered, label="hybrid")
    trace(
        f"CURRICULUM hybrid ✓ | academic={len(academic)} "
        f"practical={len(practical)} merged={len(merged)} "
        f"after_batch={len(filtered)} academic_in_final={_count_academic_hits(filtered)}"
    )
    return filtered


def collect_sources_for_expand(
    expansion_vector: str,
    *,
    source_policy: str = "practical_only",
) -> list[CurriculumSearchHit]:
    vec = (expansion_vector or "").strip()
    if len(vec) < 8:
        return []
    return collect_sources_by_policy(
        vec,
        source_policy=source_policy,
        context_vector=vec,
        grounding_only=False,
    )


def registry_hits_from_graph(graph: CurriculumGraph) -> list[CurriculumSearchHit]:
    hits: list[CurriculumSearchHit] = []
    for e in graph.curriculum_sources_registry or []:
        url = (e.url or "").strip()
        if not url.startswith("http"):
            continue
        tier = (e.source_tier or "").strip()
        hits.append(
            CurriculumSearchHit(
                source_id=e.source_id,
                url=url,
                title=e.title,
                snippet=(e.snippet or e.why_read or "")[:1200],
                key_extracts=list(e.key_extracts or [])[:12],
                source_tier=tier,
            )
        )
    return hits


def merge_expansion_source_pool(
    graph: CurriculumGraph,
    new_hits: list[CurriculumSearchHit],
) -> list[CurriculumSearchHit]:
    """Объединённый пул: реестр курса + новые grounding hits (для Flash expand)."""
    merged = registry_hits_from_graph(graph)
    seen = {_normalize_url_key(h.url) for h in merged}
    for h in new_hits:
        k = _normalize_url_key(h.url)
        if k and k not in seen:
            seen.add(k)
            merged.append(h)
    out: list[CurriculumSearchHit] = []
    for i, h in enumerate(merged, start=1):
        sid = (h.source_id or "").strip() or f"src_{i}"
        out.append(h.model_copy(update={"source_id": sid[:16]}))
    trace(
        f"CURRICULUM expand pool ✓ | registry+new={len(out)} "
        f"new_grounding={len(new_hits)}"
    )
    return out


def _highlights_text_from_hit(hit: CurriculumSearchHit) -> str:
    parts = [str(x).strip() for x in (hit.key_extracts or []) if str(x).strip()]
    if parts:
        return "\n\n".join(parts)
    return (hit.snippet or "").strip()


async def _ingest_exa_highlights_fallback(
    hit: CurriculumSearchHit,
) -> tuple[list[str], str] | None:
    text = _highlights_text_from_hit(hit)
    if len(text) < 40:
        return None
    n = await ingest_exa_highlights_fallback(hit, body_text=text)
    if n <= 0:
        return None
    extracts = _deep_extract_blocks([], [], [text[:3000]], 80, 300)
    if not extracts:
        extracts = [text[:1500]]
    return extracts[:8], (hit.title or hit.url)[:400]


def _summary_to_extracts_and_title(
    hit: CurriculumSearchHit,
    summary,
    *,
    source_text: str = "",
) -> tuple[list[str], str]:
    extra: list[str] = []
    exec_sum = (getattr(summary, "executive_summary", None) or "").strip()
    if exec_sum:
        extra.append(exec_sum)
    extracts = _deep_extract_blocks(
        list(summary.key_takeaways or []),
        list(summary.failure_modes or []),
        extra,
        min_words=80,
        max_words=300,
    )
    if (
        _extract_word_count(extracts) < DEEP_BLOG_EXTRACT_WORDS
        and (source_text or "").strip()
    ):
        extracts = _deep_extract_blocks(
            list(summary.key_takeaways or []),
            list(summary.failure_modes or []),
            extra + [source_text.strip()],
            min_words=80,
            max_words=300,
        )
    if not extracts:
        fallback = (hit.snippet or "").strip() or " ".join(summary.key_takeaways or [])[
            :2000
        ]
        extracts = _deep_extract_blocks([], [], [fallback], 80, 300)
    return extracts[:8], (summary.title or hit.title)[:400]


async def _ingest_url_with_spatial_map_reduce(
    hit: CurriculumSearchHit,
    html: str,
    *,
    tier_label: str,
) -> tuple[list[str], str]:
    """Gemma Cloud BLOG_SPATIAL map-reduce; structured Gemma fallback writes *_map_* windows."""
    from knowledge_engine.services.article_ingestion.blog_spatial_pipeline import (
        ingest_blog_with_spatial_mapping,
        persist_gemma_cloud_map_fallback,
    )

    sid = (hit.source_id or "").strip()
    annotated, summary, _saved = await ingest_blog_with_spatial_mapping(
        hit.title or hit.url,
        hit.url,
        sid,
        raw_html=html,
        save_lancedb=True,
    )
    if summary is None:
        trace(
            f"CURRICULUM spatial map-reduce ⊘ | {tier_label} | "
            f"fallback Gemma Cloud MAP | {hit.url[:55]}"
        )
        text, _method = await asyncio.to_thread(smart_fetch_page_text, hit.url)
        body = (text or "").strip() or (html or "").strip()
        if len(body) < 200:
            return [], hit.title
        summary = await persist_gemma_cloud_map_fallback(
            hit.title or hit.url, hit.url, _text_for_summarizer(body)
        )
        if summary is None:
            return [], hit.title
        return _summary_to_extracts_and_title(hit, summary, source_text=body)
    source = (annotated.annotated_markdown if annotated else "") or html
    return _summary_to_extracts_and_title(hit, summary, source_text=source)


async def _ingest_blog_url(hit: CurriculumSearchHit) -> tuple[list[str], str]:
    from knowledge_engine.src.parsers.paper_structure_analyzer import (
        is_academic_pdf_url,
    )

    if is_academic_pdf_url(hit.url):
        trace(
            f"PAPER_STRUCTURE route | academic PDF ingest (not BLOG_SPATIAL map) | "
            f"{hit.url[:60]}"
        )
        text, _method = await asyncio.to_thread(smart_fetch_page_text, hit.url)
        if len((text or "").strip()) < 200:
            fb = await _ingest_exa_highlights_fallback(hit)
            if fb:
                return fb
            return [], hit.title
        store = VectorStore()
        ing = await ingest_academic_body_gemma(
            hit.title or hit.url,
            hit.url,
            text,
            store,
            target_topic=hit.title or hit.url,
        )
        if ing is None:
            return [], hit.title
        return _summary_to_extracts_and_title(hit, ing.summary, source_text=text)

    early, title, html = await _ingest_blog_url_precheck(hit)
    if early is not None:
        return early, title
    return await _ingest_url_with_spatial_map_reduce(hit, html, tier_label="blog")


async def _ingest_blog_url_precheck(
    hit: CurriculumSearchHit,
) -> tuple[list[str] | None, str, str | None]:
    """Cache-reuse / fetch / anti-bot / thin-content checks, factored out of
    _ingest_blog_url so a batch of hits can run this independently per-hit
    and THEN submit every MAP+REDUCE-eligible one to a single pooled
    ingest_blog_spatial_mapping_batch_async call (BATCH POOLING AGGREGATION
    task) instead of each hit driving its own single-article MAP+REDUCE.

    Returns (early_exit_extracts, title, html):
    - early_exit_extracts is not None => no MAP+REDUCE needed/possible for
      this hit, use (early_exit_extracts, title) directly as the result.
    - early_exit_extracts is None => html is the fetched page, ready for
      MAP+REDUCE (single-hit or batched)."""
    cached = await _extracts_from_lancedb_url(hit.url)
    if cached and _lancedb_has_map_windows(hit.url):
        trace(f"CURRICULUM summarizer reuse LanceDB ⊘ blog | {hit.url[:60]}")
        await asyncio.to_thread(_try_blog_spatial_diagrams, hit)
        extracts, title = cached
        return extracts, title, None

    from knowledge_engine.src.curriculum.pre_flight_triage import pop_preflight_html

    preflight_html = pop_preflight_html(hit.url)
    if preflight_html is not None:
        html, fetch_method = preflight_html, "httpx"
        trace(f"CURRICULUM blog fetch ⊘ reuse pre-flight html | {hit.url[:60]}")
    else:
        html, fetch_method = await asyncio.to_thread(smart_fetch_page_html, hit.url)
    from knowledge_engine.ingest.pipeline_audit import pipeline_audit

    pipeline_audit(
        "Fetch",
        hit.url,
        html or "",
        extra=f"method={fetch_method} blog_ingest",
    )
    if is_anti_bot_fetch_result("", fetch_method, html=html):
        add_blocked_domain(hit.url, "anti_bot_detected")
        trace(f"CURRICULUM ingest anti_bot → exa highlights | blog | {hit.url[:60]}")
        fb = await _ingest_exa_highlights_fallback(hit)
        if fb:
            return fb[0], fb[1], None
        await asyncio.to_thread(_try_blog_spatial_diagrams, hit)
        return [], hit.title, None
    if len((html or "").strip()) < 200:
        trace(f"[Triage Pre-MAP] Skip MAP {hit.url} due to: fetched body < 200 chars")
        await asyncio.to_thread(_try_blog_spatial_diagrams, hit)
        return [], hit.title, None
    body_words = len((html or "").split())
    if body_words < PRE_MAP_MIN_BODY_WORDS:
        trace(
            f"[Triage Pre-MAP] Skip MAP {hit.url} due to: "
            f"body_words={body_words} < {PRE_MAP_MIN_BODY_WORDS}"
        )
        fb = await _ingest_exa_highlights_fallback(hit)
        if fb:
            return fb[0], fb[1], None
        await asyncio.to_thread(_try_blog_spatial_diagrams, hit)
        return [], hit.title, None

    return None, hit.title, html


async def _pre_map_dedup_batch_items(
    batch_items: list[tuple[str, str, str, str | None]],
) -> tuple[list[tuple[str, str, str, str | None]], dict[str, str]]:
    """Runs Pre-MAP Dedup (src/deduplication/pre_map_deduplicator.py) over one
    MAP+REDUCE-eligible batch. Returns (canonical_items, alias_url ->
    canonical_url) — alias URLs are dropped from the returned item list (so
    they never re-run MAP+REDUCE) but the mapping lets the caller reuse the
    canonical's extracts for them and record alias_of for grounding. Fail-open:
    on any error, or if disabled, returns the batch unchanged with no aliases."""
    from knowledge_engine.config import PRE_MAP_DEDUP_ENABLED

    if not PRE_MAP_DEDUP_ENABLED or len(batch_items) < 2:
        return batch_items, {}

    from knowledge_engine.src.deduplication.pre_map_deduplicator import (
        PreMapCandidate,
        deduplicate_before_map_reduce,
    )

    candidates = [
        PreMapCandidate(id=url, url=url, text=html)
        for _title, url, _sid, html in batch_items
        if html
    ]
    if len(candidates) < 2:
        return batch_items, {}

    try:
        result = await deduplicate_before_map_reduce(candidates)
    except Exception as exc:
        trace(f"CURRICULUM pre_map_dedup ✗ | {type(exc).__name__}: {exc}")
        return batch_items, {}

    alias_of_url = {
        alias_id: canonical_id
        for canonical_id, aliases in result.alias_map.items()
        for alias_id in aliases
    }
    if not alias_of_url:
        return batch_items, {}

    canonical_items = [item for item in batch_items if item[1] not in alias_of_url]
    trace(
        f"CURRICULUM pre_map_dedup ✓ | in={len(batch_items)} "
        f"canonical={len(canonical_items)} alias={len(alias_of_url)}"
    )
    return canonical_items, alias_of_url


async def _ingest_blog_hits_batch_async(
    blog_hits: list[CurriculumSearchHit],
    *,
    backfill_margin: int = 0,
    desired_count: int | None = None,
) -> list[CurriculumSearchHit]:
    """Batched sibling of spawning one _ingest_blog_hit_async task per hit:
    runs each hit's precheck (cache-reuse / fetch / anti-bot / thin-content)
    independently, then runs Pre-MAP Dedup (_pre_map_dedup_batch_items) to
    collapse near-duplicates, then submits every remaining CANONICAL hit to
    ONE ingest_blog_spatial_mapping_batch_async(articles=N) pooled call
    instead of N separate single-article map_reduce_jobs_pooled_async calls
    (BATCH POOLING AGGREGATION task).

    ``backfill_margin`` (0 by default — see DEEP_INGEST_BACKFILL_MARGIN):
    when 0, ALIAS hits skip MAP+REDUCE but still return under their own URL,
    reusing their canonical's extracts (old behaviour — saves MAP+REDUCE
    compute, keeps every source citable). When > 0, ALIAS hits are dropped
    entirely instead of duplicating content under a second URL, hits that
    failed the credibility gate / came back with empty key_extracts are
    dropped too, and the survivors are ranked by exa_relevance_score before
    being cut down to the real target.

    ``desired_count`` is that real target (the node's actual quota cap, e.g.
    ``min(CURRICULUM_DEEP_NODE_MAX_HITS, quota.total_max)`` — see the caller
    in targeted_node_search.py) and is the authoritative cap whenever given.
    Without it, the target falls back to ``len(blog_hits) - backfill_margin``
    — but that inference silently assumes ``blog_hits`` already contains a
    full cap+margin pool, which is false whenever the upstream candidate pool
    was too small to fill the margin (the common case for a narrow node):
    replenish_valid_hits_until_cap then hands back exactly ``cap`` hits, not
    ``cap + margin``, and subtracting the margin a second time here shrank an
    already-correct result below the real target. Pass ``desired_count``
    explicitly whenever the caller knows it."""
    if not blog_hits:
        return []

    def _finish(out: list[CurriculumSearchHit]) -> list[CurriculumSearchHit]:
        if backfill_margin <= 0 and desired_count is None:
            return out
        target = (
            desired_count
            if desired_count is not None
            else max(0, len(blog_hits) - backfill_margin)
        )
        # RU: раньше срез [:target] брал первые N по позиции в out — мог
        # оставить провалившийся (empty key_extracts, не прошёл
        # credibility-гейт) кандидат и выкинуть уже полностью обработанный
        # (MAP+REDUCE+LanceDB) успешный, только потому что тот стоял позже
        # в исходном порядке поиска. Сначала фильтруем неудачные, затем
        # ранжируем оставшиеся по exa_relevance_score — и только потом режем.
        valid = [h for h in out if h.key_extracts]
        dropped = len(out) - len(valid)
        valid.sort(
            key=lambda h: (
                h.exa_relevance_score if h.exa_relevance_score is not None else -1.0
            ),
            reverse=True,
        )
        if len(valid) <= target:
            if dropped:
                trace(
                    f"CURRICULUM pre_map_dedup backfill ⊘ | dropped {dropped} "
                    f"empty-extract candidate(s) | in={len(out)} valid={len(valid)}"
                )
            return valid
        trace(
            f"CURRICULUM pre_map_dedup backfill ✓ | pool={len(blog_hits)} "
            f"margin={backfill_margin} desired_count={desired_count} "
            f"in={len(out)} valid={len(valid)} dropped_empty={dropped} "
            f"trimmed_to={target}"
        )
        return valid[:target]

    async def _pre(h: CurriculumSearchHit):
        return h, await _ingest_blog_url_precheck(h)

    prechecked = await asyncio.gather(*[_pre(h) for h in blog_hits])

    out: list[CurriculumSearchHit] = []
    batch_items: list[tuple[str, str, str, str | None]] = []
    batch_hits: dict[str, CurriculumSearchHit] = {}
    for h, (early, title, html) in prechecked:
        if early is not None:
            _log_post_map_extract_quality(h.url, early)
            out.append(
                h.model_copy(update={"title": title, "key_extracts": early})
                if early
                else h
            )
            continue
        batch_items.append((title, h.url, "", html))
        batch_hits[h.url] = h

    if not batch_items:
        return _finish(out)

    canonical_items, alias_of_url = await _pre_map_dedup_batch_items(batch_items)

    from knowledge_engine.services.article_ingestion.blog_spatial_pipeline import (
        ingest_blog_spatial_mapping_batch_async,
    )

    results = await ingest_blog_spatial_mapping_batch_async(canonical_items)

    resolved: dict[str, tuple[list[str], str]] = {}
    for _title, url, _sid, html in canonical_items:
        h = batch_hits[url]
        annotated, summary, _saved = results.get(url, (None, None, 0))
        if summary is None:
            _log_post_map_extract_quality(h.url, [])
            resolved[url] = ([], h.title)
            continue
        source_text = (annotated.annotated_markdown if annotated else "") or (
            html or ""
        )
        extracts, title = _summary_to_extracts_and_title(
            h, summary, source_text=source_text
        )
        _log_post_map_extract_quality(h.url, extracts)
        resolved[url] = (extracts, title)

    for url, h in batch_hits.items():
        if backfill_margin > 0 and url in alias_of_url:
            trace(f"CURRICULUM pre_map_dedup backfill ⊘ | alias dropped | {url[:70]}")
            continue
        canonical_url = alias_of_url.get(url, url)
        extracts, title = resolved.get(canonical_url, ([], h.title))
        update: dict[str, object] = {}
        if extracts:
            update["title"] = title
            update["key_extracts"] = extracts
        if canonical_url != url:
            update["alias_of"] = canonical_url
        out.append(h.model_copy(update=update) if update else h)

    return _finish(out)


def _try_blog_spatial_diagrams(hit: CurriculumSearchHit) -> None:
    try:
        from knowledge_engine.services.article_ingestion.blog_spatial_pipeline import (
            run_blog_spatial_diagram_ingest,
        )

        sid = (hit.source_id or "").strip()
        run_blog_spatial_diagram_ingest(sid, hit.url)
    except Exception as exc:
        trace(f"BLOG_SPATIAL ingest ⊘ | {hit.url[:60]} | {exc}")


def _run_auto_article_diagrams_sync(hit: CurriculumSearchHit) -> None:
    try:
        from knowledge_engine.services.article_ingestion.auto_ingest import (
            maybe_ingest_article_diagrams,
        )
        from knowledge_engine.services.article_ingestion.pipeline import ArticleFormat
        from knowledge_engine.src.parsers.paper_structure_analyzer import (
            get_cached_prefetch_pdf_bytes,
        )

        sid = (hit.source_id or "").strip()
        pdf = get_cached_prefetch_pdf_bytes(hit.url)
        kwargs: dict[str, object] = {}
        if pdf:
            kwargs["data"] = pdf
            kwargs["content_type"] = ArticleFormat.PDF
        maybe_ingest_article_diagrams(sid, hit.url, **kwargs)
    except Exception as exc:
        trace(f"ARTICLE_AUTO_INGEST ⊘ harvest | {hit.url[:60]} | {exc}")


def _spawn_auto_article_diagrams_daemon(hit: CurriculumSearchHit) -> None:
    """Diagram harvest off the asyncio loop (does not block asyncio.run / init job)."""
    url_preview = (hit.url or "")[:60]
    thread = threading.Thread(
        target=_run_auto_article_diagrams_sync,
        args=(hit,),
        name=f"article_auto_ingest:{url_preview[:32]}",
        daemon=True,
    )
    thread.start()
    trace(
        f"ARTICLE_AUTO_INGEST ▶ daemon | {url_preview} "
        "(isolated — init/work-jobs not blocked)"
    )


def _try_auto_article_diagrams(hit: CurriculumSearchHit) -> None:
    try:
        _spawn_auto_article_diagrams_daemon(hit)
    except Exception as exc:
        trace(f"ARTICLE_AUTO_INGEST ⊘ harvest | {hit.url[:60]} | {exc}")


def _hit_extract_words(hit: CurriculumSearchHit) -> int:
    return sum(len((e or "").split()) for e in hit.key_extracts)


def _extract_word_count(extracts: list[str] | None) -> int:
    return sum(len((e or "").split()) for e in (extracts or []))


def _log_post_map_drop(url: str, extracts: list[str], *, reason: str) -> None:
    trace(f"[Triage Post-MAP] Dropped {url} due to: {reason}")


def _log_post_map_extract_quality(url: str, extracts: list[str]) -> None:
    from knowledge_engine.ingest.pipeline_audit import pipeline_audit

    joined = "\n".join(extracts or [])
    pipeline_audit("MAP", url, joined, extra="curriculum key_extracts after REDUCE")
    words = _extract_word_count(extracts)
    if extracts and words >= DEEP_BLOG_EXTRACT_WORDS:
        return
    if not extracts:
        reason = "empty extracts after MAP/fallback"
    else:
        reason = (
            f"extract_words={words} < {DEEP_BLOG_EXTRACT_WORDS} "
            "(takeaways/executive_summary too short for deep_blogs)"
        )
    _log_post_map_drop(url, extracts, reason=reason)


def hit_requires_mandatory_academic_ingest(hit: CurriculumSearchHit) -> bool:
    """Quota-approved academic / arXiv PDF — full body ingest, not abstract-only."""
    from knowledge_engine.src.parsers.paper_structure_analyzer import (
        is_academic_pdf_url,
    )

    url = (hit.url or "").strip()
    if not url.startswith("http"):
        return False
    tier = (hit.source_tier or "").strip().lower()
    if is_academic_pdf_url(url):
        return True
    if tier in _ACADEMIC_SOURCE_TIERS or tier.startswith("consensus"):
        return True
    if "arxiv.org" in url.lower():
        return True
    return False


async def ingest_mandatory_academic_hits_async(
    hits: list[CurriculumSearchHit],
    *,
    label: str = "post_replenish",
    defer_missing: bool = False,
) -> list[CurriculumSearchHit]:
    """
    Post-replenish academic/PDF body ingest.

    Always prefers LanceDB reuse when the URL is already ingested.
    ``defer_missing`` (on-demand / lazy init): do not block on Gemma for
    missing URLs — keep snippet extracts and spawn background full ingest.
    """
    if not hits:
        return hits

    out = list(hits)
    need_full: list[tuple[int, CurriculumSearchHit]] = []
    reused = 0
    for i, h in enumerate(hits):
        if not hit_requires_mandatory_academic_ingest(h):
            continue
        cached = await _extracts_from_lancedb_url(h.url)
        if cached:
            extracts, title = cached
            out[i] = h.model_copy(update={"title": title, "key_extracts": extracts})
            reused += 1
            trace(f"CURRICULUM post-replenish reuse LanceDB | {label} | {h.url[:60]}")
            _try_auto_article_diagrams(out[i])
            continue
        need_full.append((i, h))

    if not need_full:
        if reused:
            trace(
                f"CURRICULUM post-replenish ingest ✓ | {label} | "
                f"reused_lancedb={reused} full=0"
            )
        return out

    if defer_missing:
        bg_hits = [h for _, h in need_full]
        _spawn_mandatory_academic_ingest_daemon(bg_hits, label=label)
        trace(
            f"CURRICULUM post-replenish ingest ▶ defer | {label} | "
            f"reused_lancedb={reused} background_full={len(bg_hits)} "
            "(init not blocked)"
        )
        return out

    trace(
        f"CURRICULUM post-replenish ingest ▶ | {label} | "
        f"mandatory_academic={len(need_full)} reused_lancedb={reused}"
    )
    tasks = [
        asyncio.create_task(_ingest_academic_hit_async(h, force_full_ingest=True))
        for _, h in need_full
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    ok = 0
    for (idx, _), res in zip(need_full, results):
        if isinstance(res, BaseException):
            trace(
                f"CURRICULUM post-replenish ingest ⊘ | {label} | "
                f"{out[idx].url[:60]} | {res}"
            )
            continue
        if res is not None:
            out[idx] = res
            ok += 1
    trace(
        f"CURRICULUM post-replenish ingest ✓ | {label} | "
        f"ingested={ok}/{len(need_full)} reused_lancedb={reused}"
    )
    return out


def _spawn_mandatory_academic_ingest_daemon(
    hits: list[CurriculumSearchHit],
    *,
    label: str,
) -> None:
    """Full Gemma ingest off the init critical path."""
    if not hits:
        return
    snapshot = list(hits)

    def _run() -> None:
        try:
            asyncio.run(
                ingest_mandatory_academic_hits_async(
                    snapshot,
                    label=f"{label}:bg",
                    defer_missing=False,
                )
            )
        except Exception as exc:
            trace(f"CURRICULUM post-replenish bg ⊘ | {label} | {exc}")

    thread = threading.Thread(
        target=_run,
        name=f"mandatory_academic:{label[:40]}",
        daemon=True,
    )
    thread.start()


async def _extracts_from_lancedb_url(url: str) -> tuple[list[str], str] | None:
    u = (url or "").strip()
    if not u.startswith("http"):
        return None
    summaries = await VectorStore().fetch_summaries_by_urls([u], limit=1)
    if not summaries:
        return None
    s = summaries[0]
    extracts = _deep_extract_blocks(
        list(s.key_takeaways or []),
        list(s.failure_modes or []),
        [],
    )
    if not extracts:
        return None
    title = (s.title or u)[:400]
    return extracts[:8], title


def _lancedb_has_map_windows(url: str) -> bool:
    """True when rag_chunks already has MAP windows for this URL (not passport-only)."""
    u = (url or "").strip()
    if not u.startswith("http"):
        return False
    try:
        from knowledge_engine.db.rag_chunks_schema import COL_CHUNK_ID

        store = VectorStore()
        doc_id = store.doc_id_for_url(u)
        return any(
            "_map_" in str(row.get(COL_CHUNK_ID) or "")
            for row in store.fetch_rag_chunks_by_doc_id(doc_id)
        )
    except Exception:
        return False


_INGEST_URL_SEM = asyncio.Semaphore(max(1, KE_INGEST_URL_CONCURRENCY))
_SUMMARIZER_MAX_INPUT_CHARS = 500_000


def _text_for_summarizer(text: str) -> str:
    t = (text or "").strip()
    return t[:_SUMMARIZER_MAX_INPUT_CHARS] if t else ""


async def _ingest_academic_hit_async(
    hit: CurriculumSearchHit,
    *,
    force_full_ingest: bool = False,
) -> CurriculumSearchHit | None:
    async with _INGEST_URL_SEM:
        try:
            from knowledge_engine.src.curriculum.academic_url_canonicalizer import (
                coerce_arxiv_url_to_pdf,
            )
            from knowledge_engine.src.parsers.paper_structure_analyzer import (
                cache_prefetch_pdf_bytes,
                is_academic_pdf_url,
                try_fetch_pdf_bytes_for_url,
            )

            canon_url = coerce_arxiv_url_to_pdf(hit.url)
            if canon_url != (hit.url or "").strip():
                hit = hit.model_copy(update={"url": canon_url})

            cached = await _extracts_from_lancedb_url(hit.url)
            if cached:
                # force_full_ingest means "ensure body in LanceDB", not "re-run Gemma".
                trace(
                    f"CURRICULUM summarizer reuse LanceDB ⊘ academic | {hit.url[:60]}"
                )
                _try_auto_article_diagrams(hit)
                extracts, title = cached
            elif is_academic_pdf_url(hit.url):
                pdf_bytes = await asyncio.to_thread(
                    try_fetch_pdf_bytes_for_url, hit.url
                )
                if not pdf_bytes:
                    fb = await _ingest_exa_highlights_fallback(hit)
                    if fb:
                        _try_auto_article_diagrams(hit)
                        extracts, title = fb
                    else:
                        return None
                else:
                    cache_prefetch_pdf_bytes(hit.url, pdf_bytes)
                    try:
                        from knowledge_engine.services.parsers.article_manifest import (
                            ArticleResourceManifest,
                        )
                        from knowledge_engine.services.parsers.article_resource_discoverer import (
                            get_cached_manifest,
                            store_manifest,
                        )

                        manifest = get_cached_manifest(hit.url)
                        if manifest is None:
                            manifest = ArticleResourceManifest(
                                source_id=(hit.source_id or "").strip(),
                                canonical_url=hit.url,
                            )
                        manifest.fetched_pdf_bytes = pdf_bytes
                        store_manifest(manifest)
                    except Exception:
                        pass
                    body = (hit.snippet or hit.title or hit.url or "")[:800]
                    store = VectorStore()
                    ing = await ingest_academic_body_gemma(
                        hit.title or hit.url,
                        hit.url,
                        body,
                        store,
                        target_topic=hit.title or hit.url,
                        pdf_bytes=pdf_bytes,
                    )
                    if ing is None:
                        return None
                    extracts, title = _summary_to_extracts_and_title(
                        hit, ing.summary, source_text=body
                    )
                    _try_auto_article_diagrams(hit)
            else:
                html, fetch_method = await asyncio.to_thread(
                    smart_fetch_page_html, hit.url
                )
                if is_anti_bot_fetch_result("", fetch_method, html=html):
                    add_blocked_domain(hit.url, "anti_bot_detected")
                    fb = await _ingest_exa_highlights_fallback(hit)
                    if fb:
                        _try_auto_article_diagrams(hit)
                        extracts, title = fb
                    else:
                        return None
                else:
                    from knowledge_engine.services.web_extract import (
                        smart_fetch_page_text,
                    )

                    text, _ = await asyncio.to_thread(smart_fetch_page_text, hit.url)
                    if len((text or "").strip()) < 200:
                        fb = await _ingest_exa_highlights_fallback(hit)
                        if fb:
                            extracts, title = fb
                        else:
                            return None
                    else:
                        store = VectorStore()
                        from knowledge_engine.src.parsers.paper_structure_analyzer import (
                            try_fetch_pdf_bytes_for_url,
                        )

                        pdf_bytes = await asyncio.to_thread(
                            try_fetch_pdf_bytes_for_url, hit.url
                        )
                        ing = await ingest_academic_body_gemma(
                            hit.title or hit.url,
                            hit.url,
                            text,
                            store,
                            target_topic=hit.title or hit.url,
                            pdf_bytes=pdf_bytes,
                        )
                        if ing is None:
                            return None
                        extracts, title = _summary_to_extracts_and_title(
                            hit, ing.summary, source_text=text
                        )
                        _try_auto_article_diagrams(hit)
            if not extracts:
                return None
            return hit.model_copy(update={"title": title, "key_extracts": extracts})
        except Exception as exc:
            trace(f"CURRICULUM academic summarizer skip | {hit.url[:50]} | {exc}")
            return None


async def _spatial_blog_diagrams_batch_async(hits: list[CurriculumSearchHit]) -> None:
    if not hits:
        return
    try:
        from knowledge_engine.services.article_ingestion.blog_spatial_pipeline import (
            prepare_spatial_diagram_job,
            run_spatial_diagram_ingest_jobs_async,
        )

        jobs = []
        for hit in hits:
            sid = (hit.source_id or "").strip()
            prepared = await asyncio.to_thread(
                prepare_spatial_diagram_job,
                sid,
                hit.url,
            )
            if prepared is not None:
                jobs.append(prepared)
        if jobs:
            await run_spatial_diagram_ingest_jobs_async(jobs)
    except Exception as exc:
        trace(f"BLOG_SPATIAL batch ⊘ | {exc}")


async def _spatial_blog_diagrams_async(hit: CurriculumSearchHit) -> None:
    await _spatial_blog_diagrams_batch_async([hit])


async def _ingest_blog_hit_async(hit: CurriculumSearchHit) -> CurriculumSearchHit:
    from knowledge_engine.src.parsers.paper_structure_analyzer import (
        is_academic_pdf_url,
    )

    if is_academic_pdf_url(hit.url):
        result = await _ingest_academic_hit_async(hit)
        return result if result is not None else hit
    async with _INGEST_URL_SEM:
        try:
            extracts, title = await _ingest_blog_url(hit)
            _log_post_map_extract_quality(hit.url, extracts)
            if not extracts:
                return hit
            return hit.model_copy(update={"title": title, "key_extracts": extracts})
        except Exception as exc:
            trace(f"CURRICULUM blog summarizer skip | {hit.url[:50]} | {exc}")
            return hit


async def summarize_whitelist_blog_hits_async(
    hits: list[CurriculumSearchHit],
    target_goal: str = "",
    *,
    backfill_margin: int = 0,
    desired_count: int | None = None,
) -> list[CurriculumSearchHit]:
    """Последовательный ingest URL (Semaphore 1); Cloud LLM Pipeline map-reduce
    + structured-JSON fallback (без локальных инстансов).

    ``backfill_margin`` / ``desired_count`` — прокидываются в
    _ingest_blog_hits_batch_async (см. его докстринг): при backfill_margin > 0
    ALIAS-хиты дропаются вместо дублирования контента под своим URL;
    ``desired_count`` — реальная цель (quota cap ноды), приоритетнее вывода
    из длины пула."""
    from knowledge_engine.src.parsers.paper_structure_analyzer import (
        is_academic_pdf_url,
    )

    blog_hits = [h for h in hits if (h.source_tier or "").strip() in _BLOG_SOURCE_TIERS]
    academic_hits = [
        h for h in hits if (h.source_tier or "").strip() in _ACADEMIC_SOURCE_TIERS
    ]
    other_hits = [
        h
        for h in hits
        if (h.source_tier or "").strip() not in _BLOG_SOURCE_TIERS
        and (h.source_tier or "").strip() not in _ACADEMIC_SOURCE_TIERS
    ]

    if blog_hits:
        valid_blog, broken_blog = validate_and_filter_urls(
            blog_hits,
            timeout=CURRICULUM_URL_VALIDATE_TIMEOUT_SEC,
        )
        if broken_blog:
            trace(
                f"CURRICULUM summarizer gate ⊘ | skip broken urls={len(broken_blog)} "
                "(не попадут в 7B / LanceDB)"
            )
        blog_hits = valid_blog

    out: list[CurriculumSearchHit] = list(other_hits)

    academic_tasks: list[asyncio.Task] = []
    for h in academic_hits:
        if hit_requires_mandatory_academic_ingest(h):
            academic_tasks.append(
                asyncio.create_task(
                    _ingest_academic_hit_async(h, force_full_ingest=True)
                )
            )
            continue
        if _hit_extract_words(h) >= DEEP_BLOG_EXTRACT_WORDS:
            out.append(h)
            continue
        academic_tasks.append(asyncio.create_task(_ingest_academic_hit_async(h)))

    diagram_ingest_hits: list[CurriculumSearchHit] = list(academic_hits)
    blog_batch_hits: list[CurriculumSearchHit] = []
    for h in blog_hits:
        if is_academic_pdf_url(h.url):
            academic_tasks.append(
                asyncio.create_task(
                    _ingest_academic_hit_async(h, force_full_ingest=True)
                )
            )
            continue
        blog_batch_hits.append(h)

    async def _collect_academic() -> list[CurriculumSearchHit | None]:
        if not academic_tasks:
            return []
        return list(await asyncio.gather(*academic_tasks))

    async def _collect_blog() -> list[CurriculumSearchHit]:
        # Все хиты блогов, годные для MAP+REDUCE, отправляются вместе ОДНИМ
        # пуловым вызовом map_reduce_jobs_pooled_async(articles=N) вместо N
        # отдельных вызовов по одной статье (задача BATCH POOLING AGGREGATION).
        return await _ingest_blog_hits_batch_async(
            blog_batch_hits,
            backfill_margin=backfill_margin,
            desired_count=desired_count,
        )

    ac_done, blog_done = await asyncio.gather(
        _collect_academic(),
        _collect_blog(),
    )
    if diagram_ingest_hits:
        seen_url: set[str] = set()
        deduped: list[CurriculumSearchHit] = []
        for h in diagram_ingest_hits:
            k = _normalize_url_key(h.url)
            if not k or k in seen_url:
                continue
            seen_url.add(k)
            deduped.append(h)
        await _spatial_blog_diagrams_batch_async(deduped)
    for done in ac_done:
        if done is not None:
            out.append(done)
    out.extend(blog_done)

    trace(
        f"CURRICULUM summarizer parallel ✓ | ingest_tasks="
        f"academic={len(academic_tasks)} blog={len(blog_batch_hits)} sem=2"
    )

    blog_deep = sum(
        1
        for h in out
        if h.source_tier in _BLOG_SOURCE_TIERS
        and _hit_extract_words(h) >= DEEP_BLOG_EXTRACT_WORDS
    )
    academic_deep = sum(
        1
        for h in out
        if h.source_tier in _ACADEMIC_SOURCE_TIERS
        and _hit_extract_words(h) >= DEEP_BLOG_EXTRACT_WORDS
    )
    trace(
        f"CURRICULUM summarizer ✓ | deep_blogs={blog_deep} deep_academic={academic_deep}"
    )
    return out


def summarize_whitelist_blog_hits(
    hits: list[CurriculumSearchHit],
    target_goal: str = "",
) -> list[CurriculumSearchHit]:
    """Sync wrapper — legitimate top-level asyncio.run() bridge for callers
    that are genuinely synchronous (worker job dispatch has no event loop at
    all — see knowledge_engine/worker/__main__.py). A caller that is ALREADY
    async has no business going through this sync facade: it must await
    summarize_whitelist_blog_hits_async(...) directly — a plain sync function
    cannot bridge into async work from inside an already-running loop."""
    if not hits:
        return hits
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(summarize_whitelist_blog_hits_async(hits, target_goal))
    raise RuntimeError(
        "summarize_whitelist_blog_hits() called from inside a running event "
        "loop — await summarize_whitelist_blog_hits_async(...) directly instead"
    )


summarize_practical_blog_hits = summarize_whitelist_blog_hits
