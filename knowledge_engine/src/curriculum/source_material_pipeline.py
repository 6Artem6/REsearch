"""Hybrid Search Pipeline: LanceDB enrich, Gemini Grounding blogs, Summarizer."""

from __future__ import annotations

import asyncio
import re

from knowledge_engine.config import (
    CURRICULUM_GEMINI_GROUNDING_ENABLED,
    CURRICULUM_GEMINI_WEB_HARVEST_ENABLED,
    CURRICULUM_LITE_SITE_SUGGEST_ENABLED,
    CURRICULUM_SEARCH_MIN_HITS,
    CURRICULUM_SEARCH_TARGET_HITS,
    CURRICULUM_URL_VALIDATE_TIMEOUT_SEC,
)
from knowledge_engine.services.gemini_search_grounding import (
    search_grounded_whitelist_blogs_detailed,
)
from knowledge_engine.services.summarizer import summarize_article
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.services.web_extract import smart_fetch_page_text
from knowledge_engine.src.curriculum.curriculum_v08_harvest import _deep_extract_blocks
from knowledge_engine.src.curriculum.schemas import CurriculumGraph, CurriculumSearchHit
from knowledge_engine.src.curriculum.source_hit_curation import (
    collect_archived_practical_hits,
    curate_practical_hits,
)
from knowledge_engine.src.curriculum.search_prestep import _normalize_url_key
from knowledge_engine.src.curriculum.url_validate import validate_and_filter_urls
from knowledge_engine.ui.run_log import trace

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
    }
)

_ACADEMIC_SOURCE_TIERS = frozenset(
    {
        "semantic_scholar",
        "arxiv",
        "consensus",
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
    return _deep_extract_blocks(
        list(ds.key_takeaways or []),
        list(ds.failure_modes or []),
        [],
        min_words=150,
        max_words=300,
    )


def _extract_word_total(extracts: list[str]) -> int:
    return sum(len((e or "").split()) for e in extracts)


def enrich_search_hits_with_extracts(
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
        for ds in store.fetch_summaries_by_urls(urls, limit=len(urls) + 2):
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

        if ds:
            lance_extracts = _extracts_from_document_summary(ds)
            if _extract_word_total(lance_extracts) > _extract_word_total(extracts):
                extracts = lance_extracts
            if not hit.title and ds.title:
                hit = hit.model_copy(update={"title": ds.title[:400]})

        if not extracts or _extract_word_total(extracts) < 120:
            if ds:
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


def collect_practical_blog_hits(
    target_goal: str,
    *,
    context_vector: str = "",
    limit_per_provider: int = 4,
    grounding_only: bool = False,
    defer_lite_batch: bool = False,
) -> list[CurriculumSearchHit]:
    """Практика: архив → CSE → SearXNG → (опц.) DDGS / Gemini web → Lite batch."""
    from knowledge_engine.src.curriculum.practical_source_fetch import fetch_practical_sources

    exclude: set[str] = set()
    out: list[CurriculumSearchHit] = []
    grounding_exhausted = False
    anchor = f"curriculum_blogs:{(target_goal or '').strip()[:500]}"
    search_vec = (context_vector or target_goal or "").strip()

    for h in collect_archived_practical_hits(
        target_goal,
        exclude_url_keys=exclude,
        limit=4,
        strict=True,
    ):
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

    if (
        CURRICULUM_LITE_SITE_SUGGEST_ENABLED
        and len(out) < cap
    ):
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
            grounding_exhausted = gr.gemini_exhausted
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
            grounding_exhausted = True
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

        trace(
            f"CURRICULUM searxng fallback ▶ | have={len(out)} need>={min_blog}"
        )
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
        f"searxng={sum(1 for h in out if h.source_tier == 'searxng')} "
        f"ddgs={sum(1 for h in out if h.source_tier == 'ddgs')} "
        f"api_grounding={sum(1 for h in out if h.source_tier == 'gemini_grounding')} "
        f"gemini_web={sum(1 for h in out if h.source_tier == 'gemini_web')} "
        f"archive={sum(1 for h in out if h.source_tier == 'archive')} "
        f"searxng_fallback={sum(1 for h in out if h.source_tier == 'whitelist_blog')} "
    )
    return out[:cap]


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
    """Semantic Scholar → arXiv; опционально Consensus Playwright при пустом API."""
    from knowledge_engine.config import CURRICULUM_USE_V08_CONSENSUS
    from knowledge_engine.src.curriculum.academic_source_fetch import fetch_academic_sources

    vec = (context_vector or target_goal or "").strip()
    if len(vec) < 8:
        return []

    hits = fetch_academic_sources(vec)
    if hits:
        trace(
            f"CURRICULUM academic ✓ | api hits={len(hits)} "
            "(Consensus не вызывается — API-first)"
        )
        return hits

    if not CURRICULUM_USE_V08_CONSENSUS:
        trace("CURRICULUM academic ⊘ | API empty, Consensus disabled")
        return []

    import asyncio

    from knowledge_engine.src.curriculum.curriculum_v08_harvest import (
        harvest_curriculum_sources_v08,
    )

    goal = (target_goal or "").strip()
    anchor = f"curriculum:{goal[:500]}"
    trace("CURRICULUM academic ▶ | Consensus Playwright fallback")
    try:
        hits = asyncio.run(harvest_curriculum_sources_v08(goal, anchor))
    except Exception as exc:
        trace(f"CURRICULUM academic ✗ | {exc}")
        hits = []
    cap = CURRICULUM_SEARCH_TARGET_HITS
    trace(f"CURRICULUM academic ✓ | consensus hits={len(hits)}")
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
        return _batch_filter_curriculum_hits(
            hits,
            target_goal,
            anchor_suffix="academic_only",
        )

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
    trace(
        f"CURRICULUM hybrid ✓ | academic={len(academic)} "
        f"practical={len(practical)} merged={len(merged)} "
        f"after_batch={len(filtered)}"
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


def _ingest_blog_url(hit: CurriculumSearchHit) -> tuple[list[str], str]:
    text, _method = smart_fetch_page_text(hit.url)
    if len((text or "").strip()) < 200:
        return [], hit.title
    summary = summarize_article(hit.title or hit.url, hit.url, text[:14000])
    VectorStore().save_summary(summary)
    extracts = _deep_extract_blocks(
        list(summary.key_takeaways or []),
        list(summary.failure_modes or []),
        [],
    )
    if not extracts:
        extracts = _deep_extract_blocks([], [], [hit.snippet or text[:2000]], 80, 300)
    return extracts[:8], (summary.title or hit.title)[:400]


def _ingest_academic_url(hit: CurriculumSearchHit) -> tuple[list[str], str]:
    text = (hit.snippet or "").strip()
    if len(text) < 200:
        text, _method = smart_fetch_page_text(hit.url)
    if len((text or "").strip()) < 200:
        return [], hit.title
    summary = summarize_article(hit.title or hit.url, hit.url, text[:14000])
    VectorStore().save_summary(summary)
    extracts = _deep_extract_blocks(
        list(summary.key_takeaways or []),
        list(summary.failure_modes or []),
        [],
    )
    return extracts[:8], (summary.title or hit.title)[:400]


def _hit_extract_words(hit: CurriculumSearchHit) -> int:
    return sum(len((e or "").split()) for e in hit.key_extracts)


def summarize_whitelist_blog_hits(
    hits: list[CurriculumSearchHit],
    target_goal: str = "",
) -> list[CurriculumSearchHit]:
    """LanceDB ingest: готовые выжимки без 7B; живой URL → 7B Summarizer."""
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

    for h in academic_hits:
        if _hit_extract_words(h) >= 120:
            out.append(h)
            continue
        try:
            extracts, title = _ingest_academic_url(h)
            if not extracts:
                continue
            out.append(h.model_copy(update={"title": title, "key_extracts": extracts}))
        except Exception as exc:
            trace(f"CURRICULUM academic summarizer skip | {h.url[:50]} | {exc}")

    for h in blog_hits:
        if h.key_extracts and _hit_extract_words(h) >= 120:
            out.append(h)
            continue
        try:
            extracts, title = _ingest_blog_url(h)
            if not extracts:
                out.append(h)
                continue
            out.append(h.model_copy(update={"title": title, "key_extracts": extracts}))
        except Exception as exc:
            trace(f"CURRICULUM blog summarizer skip | {h.url[:50]} | {exc}")
            out.append(h)
    blog_deep = sum(
        1
        for h in out
        if h.source_tier in _BLOG_SOURCE_TIERS
        and _hit_extract_words(h) >= 120
    )
    academic_deep = sum(
        1
        for h in out
        if h.source_tier in _ACADEMIC_SOURCE_TIERS
        and _hit_extract_words(h) >= 120
    )
    trace(
        f"CURRICULUM summarizer ✓ | deep_blogs={blog_deep} deep_academic={academic_deep}"
    )
    return out


summarize_practical_blog_hits = summarize_whitelist_blog_hits
