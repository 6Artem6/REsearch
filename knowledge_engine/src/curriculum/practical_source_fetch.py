"""Практический сбор: Google CSE → SearXNG → (опц.) DDGS."""

from __future__ import annotations

import asyncio

import httpx

from knowledge_engine.config import (
    CURRICULUM_GOOGLE_CSE_ENABLED,
    CURRICULUM_PRACTICAL_CSE_LIMIT,
    CURRICULUM_PRACTICAL_DDGS_ENABLED,
    CURRICULUM_PRACTICAL_DDGS_LIMIT,
    CURRICULUM_PRACTICAL_SEARXNG_LIMIT,
    CURRICULUM_PRACTICAL_SNIPPET_MIN_CHARS,
    CURRICULUM_URL_VALIDATE_TIMEOUT_SEC,
    GOOGLE_CSE_API_KEY,
    GOOGLE_CSE_ID,
)
from knowledge_engine.src.curriculum.curriculum_v08_harvest import _deep_extract_blocks
from knowledge_engine.src.curriculum.practical_searxng_search import (
    collect_searxng_practical_rows,
)
from knowledge_engine.src.curriculum.practical_url_filters import (
    filter_practical_search_row,
)
from knowledge_engine.src.curriculum.schemas import CurriculumSearchHit
from knowledge_engine.src.curriculum.search_query_builder import build_search_queries
from knowledge_engine.src.curriculum.url_validate import validate_and_filter_urls_async
from knowledge_engine.src.source_evaluator.curriculum_source_pool import (
    is_collectible_article_url,
)
from knowledge_engine.ui.run_log import trace

_GOOGLE_CSE_URL = "https://www.googleapis.com/customsearch/v1"


def _normalize_href(href: str) -> str:
    u = (href or "").strip().rstrip(".,);]")
    if not u.startswith("http"):
        return ""
    return u.split("#")[0].rstrip("/")


async def _search_google_cse(
    query: str, limit: int
) -> tuple[list[dict[str, str]], bool]:
    """(rows, quota_exhausted)"""
    if not CURRICULUM_GOOGLE_CSE_ENABLED:
        trace(
            "CURRICULUM google_cse ⊘ | disabled "
            "(CURRICULUM_GOOGLE_CSE_ENABLED=false — практика через SearXNG)"
        )
        return [], False
    if not GOOGLE_CSE_API_KEY or not GOOGLE_CSE_ID:
        trace("CURRICULUM google_cse ⊘ | GOOGLE_CSE_API_KEY / GOOGLE_CSE_ID not set")
        return [], False

    from knowledge_engine.services.curriculum_api_quota_store import (
        can_use_google_cse,
        record_google_cse_result,
    )

    allowed, why = can_use_google_cse()
    if not allowed:
        trace(f"CURRICULUM google_cse ⊘ | {why} — SearXNG fallback")
        return [], True

    params = {
        "q": query,
        "key": GOOGLE_CSE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "num": min(max(1, limit), 10),
    }
    trace(f"CURRICULUM google_cse ▶ | {query[:120]}")
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(_GOOGLE_CSE_URL, params=params)
            if resp.status_code == 429:
                trace("CURRICULUM google_cse ✗ | 429 quota")
                record_google_cse_result(
                    ok=False, http_status=429, quota_exhausted=True
                )
                return [], True
            if resp.status_code >= 400:
                trace(f"CURRICULUM google_cse ✗ | HTTP {resp.status_code}")
                record_google_cse_result(ok=False, http_status=resp.status_code)
                return [], resp.status_code in (403, 429)
            data = resp.json()
    except Exception as exc:
        trace(f"CURRICULUM google_cse ✗ | {exc}")
        record_google_cse_result(ok=False)
        return [], False

    record_google_cse_result(ok=True)

    rows: list[dict[str, str]] = []
    for item in data.get("items", [])[:limit]:
        link = _normalize_href(str(item.get("link") or ""))
        if not link:
            continue
        rows.append(
            {
                "url": link,
                "title": str(item.get("title") or link)[:400],
                "snippet": str(item.get("snippet") or "")[:1200],
            }
        )
    trace(f"CURRICULUM google_cse ✓ | hits={len(rows)}")
    return rows, False


def _search_ddgs(query: str, limit: int) -> list[dict[str, str]]:
    trace(f"CURRICULUM ddgs ▶ | {query[:120]}")
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        trace("CURRICULUM ddgs ✗ | install duckduckgo-search")
        return []

    rows: list[dict[str, str]] = []
    try:
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=limit):
                if not isinstance(item, dict):
                    continue
                href = _normalize_href(str(item.get("href") or item.get("url") or ""))
                if not href:
                    continue
                rows.append(
                    {
                        "url": href,
                        "title": str(item.get("title") or href)[:400],
                        "snippet": str(item.get("body") or item.get("snippet") or "")[
                            :1200
                        ],
                    }
                )
    except Exception as exc:
        trace(f"CURRICULUM ddgs ✗ | {exc}")
        return []

    trace(f"CURRICULUM ddgs ✓ | hits={len(rows)}")
    return rows


def _snippet_ready(snippet: str) -> bool:
    return len((snippet or "").strip()) >= CURRICULUM_PRACTICAL_SNIPPET_MIN_CHARS


def _row_to_hit(row: dict[str, str], tier: str) -> CurriculumSearchHit:
    url = row["url"]
    title = row["title"]
    snippet = row.get("snippet") or ""
    extracts: list[str] = []
    if _snippet_ready(snippet):
        extracts = _deep_extract_blocks([], [], [snippet], min_words=60, max_words=220)
        if not extracts:
            extracts = [snippet[:800]]
    return CurriculumSearchHit(
        url=url,
        title=title,
        snippet=snippet,
        key_extracts=extracts[:8],
        source_tier=tier,
    )


def _merge_row_lists(
    parts: list[tuple[list[dict[str, str]], str]],
    cap: int,
) -> list[CurriculumSearchHit]:
    hits: list[CurriculumSearchHit] = []
    seen: set[str] = set()
    for rows, tier in parts:
        for row in rows:
            url = row.get("url") or ""
            if not is_collectible_article_url(url):
                trace(f"CURRICULUM practical filter ⊘ | {url[:70]} | not_collectible")
                continue
            if not filter_practical_search_row(row):
                continue
            key = url.lower()
            if key in seen:
                continue
            seen.add(key)
            hits.append(_row_to_hit(row, tier))
            if len(hits) >= cap:
                return hits
    return hits


async def _search_exa_practical(query: str, limit: int) -> list[CurriculumSearchHit]:
    from knowledge_engine.services.search.exa_transform import (
        fetch_exa_curriculum_hits_simple,
    )

    return await fetch_exa_curriculum_hits_simple(
        query,
        limit=limit,
        anchor="curriculum:practical_bulk",
    )


async def fetch_practical_sources_async(
    expansion_vector: str,
    *,
    max_hits: int = 8,
) -> list[CurriculumSearchHit]:
    vec = (expansion_vector or "").strip()
    if len(vec) < 8:
        return []

    built = build_search_queries(vec)
    q = built.practical_query
    cap = max(1, max_hits)

    merged: list[CurriculumSearchHit] = []
    seen_urls: set[str] = set()

    exa_hits = await _search_exa_practical(q, cap)
    for h in exa_hits:
        key = h.url.lower()
        if key in seen_urls:
            continue
        seen_urls.add(key)
        merged.append(h)

    parts: list[tuple[list[dict[str, str]], str]] = []
    need_fallback = len(merged) < cap

    cse_rows: list[dict[str, str]] = []
    cse_exhausted = False
    if need_fallback and CURRICULUM_GOOGLE_CSE_ENABLED:
        cse_rows, cse_exhausted = await _search_google_cse(
            q, CURRICULUM_PRACTICAL_CSE_LIMIT
        )
    if cse_rows:
        parts.append((cse_rows, "google_cse"))

    need_more = need_fallback and (len(cse_rows) < cap - len(merged) or cse_exhausted)
    if need_more:
        sx_rows = await collect_searxng_practical_rows(
            vec,
            limit=CURRICULUM_PRACTICAL_SEARXNG_LIMIT,
        )
        if sx_rows:
            parts.append((sx_rows, "searxng"))

    total_rows = sum(len(p[0]) for p in parts)
    if (
        need_fallback
        and total_rows < cap - len(merged)
        and CURRICULUM_PRACTICAL_DDGS_ENABLED
    ):
        ddgs_rows = _search_ddgs(q, CURRICULUM_PRACTICAL_DDGS_LIMIT)
        if ddgs_rows:
            parts.append((ddgs_rows, "ddgs"))

    if parts:
        extra_cap = max(0, cap - len(merged))
        for h in _merge_row_lists(parts, extra_cap):
            key = h.url.lower()
            if key in seen_urls:
                continue
            seen_urls.add(key)
            merged.append(h)

    if not merged:
        return []

    valid, broken = await validate_and_filter_urls_async(
        merged,
        timeout=CURRICULUM_URL_VALIDATE_TIMEOUT_SEC,
    )
    if broken:
        trace(f"CURRICULUM practical validate ⊘ | broken={len(broken)}")
    tiers = {h.source_tier for h in valid}
    trace(f"CURRICULUM practical ✓ | valid={len(valid)} tiers={sorted(tiers)}")
    return valid[:cap]


def fetch_practical_sources(
    expansion_vector: str,
    *,
    max_hits: int = 8,
) -> list[CurriculumSearchHit]:
    return asyncio.run(
        fetch_practical_sources_async(expansion_vector, max_hits=max_hits)
    )
