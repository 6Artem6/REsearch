"""Общая логика discovery: архив, cache-first, domain trust, v0.6 SearXNG."""

from __future__ import annotations

from typing import Any

from knowledge_engine.config import (
    DISCOVERY_MODE,
    MAX_URLS,
    SEARXNG_DISCOVERY_CATEGORIES,
    SOURCE_ARCHIVE_ENABLED,
)
from knowledge_engine.db.source_links import get_source_link_archive
from knowledge_engine.schemas import EngineGraphState
from knowledge_engine.services.discovery_trust import apply_domain_trust_to_urls
from knowledge_engine.services.query_expander import apply_smart_query_syntax_batch
from knowledge_engine.services.search.registry import default_registry
from knowledge_engine.ui.run_log import trace


def merge_pending_from_discovery(
    state: EngineGraphState,
    queries: list[str],
    query_limit: int,
    hits_per_query: int = 2,
) -> tuple[list[str], list[str]]:
    """
    Собирает URL: опционально архив (cache_first), SearXNG, domain trust, приоритет trusted.
    Возвращает (pending_urls, archived_material_urls для state).
    """
    explored = set(state.get("explored_urls") or [])
    pending_existing = list(state.get("pending_urls") or [])
    problem = state.get("original_query") or state.get("user_problem") or ""
    material: list[str] = list(state.get("material_source_urls") or [])

    archive_candidates: list[str] = []
    cache_first = (
        bool(state.get("discovery_cache_first")) or DISCOVERY_MODE == "cache_first"
    )
    if SOURCE_ARCHIVE_ENABLED and cache_first:
        archive_candidates = get_source_link_archive().get_reusable_urls(
            problem,
            explored,
            limit=10,
            min_trust=0.4,
        )
        if archive_candidates:
            trace(f"ARCHIVE cache_first | candidates={len(archive_candidates)}")

    raw_urls: list[str] = []
    url_searxng_engines: dict[str, str] = {}
    if DISCOVERY_MODE != "archive_only":
        registry = default_registry()
        smart_queries = apply_smart_query_syntax_batch(queries[:query_limit])
        categories = list(SEARXNG_DISCOVERY_CATEGORIES) or None
        if categories:
            trace(f"SEARXNG targeted | categories={','.join(categories)}")
        for q in smart_queries:
            hits = registry.multi_search_sync(
                q,
                limit_per_provider=hits_per_query,
                searxng_categories=categories,
            )
            for h in hits:
                url = (h.url or "").strip()
                if not url.startswith("http"):
                    continue
                raw_urls.append(url)
                if h.engine:
                    url_searxng_engines[url] = h.engine

    accepted: list[str] = []
    rejected: list[str] = []
    if raw_urls:
        accepted, rejected, _ = apply_domain_trust_to_urls(
            raw_urls,
            problem[:200],
            url_searxng_engines=url_searxng_engines,
        )
    elif archive_candidates and DISCOVERY_MODE == "cache_first":
        accepted = list(archive_candidates)

    merged: list[str] = []
    for url in archive_candidates + accepted + pending_existing:
        if url in explored or url in merged:
            continue
        merged.append(url)
        if url not in material:
            material.append(url)

    for url in raw_urls:
        if url not in material:
            material.append(url)

    capped = merged[: max(1, MAX_URLS)]
    return capped, material


def discovery_state_updates(
    state: EngineGraphState,
    queries: list[str],
    query_limit: int,
) -> dict[str, Any]:
    pending, material = merge_pending_from_discovery(state, queries, query_limit)
    return {
        "pending_urls": pending,
        "material_source_urls": material,
    }
