"""Практический поиск через локальный SearXNG (Lite site: queries)."""

from __future__ import annotations

from typing import Any

from knowledge_engine.config import (
    CURRICULUM_PRACTICAL_SEARXNG_LIMIT,
    CURRICULUM_PRACTICAL_SEARXNG_QUERIES,
    SEARXNG_DISCOVERY_CATEGORIES,
    SEARXNG_ENABLED,
)
from knowledge_engine.services.searxng_client import searxng_search_json
from knowledge_engine.src.curriculum.lite_search_pipeline import (
    build_search_queries,
    flatten_whitelist_domains,
)
from knowledge_engine.src.curriculum.search_query_builder import build_practical_searxng_queries
from knowledge_engine.ui.run_log import trace


def _normalize_href(href: str) -> str:
    u = (href or "").strip().rstrip(".,);]")
    if not u.startswith("http"):
        return ""
    return u.split("#")[0].rstrip("/")


async def _searxng_queries_for_goal(
    expansion_vector: str,
    *,
    max_queries: int,
    anchor: str | None = None,
) -> list[str]:
    goal = (expansion_vector or "").strip()
    domains = flatten_whitelist_domains()
    plan = await build_search_queries(
        goal,
        domains,
        anchor=anchor or f"curriculum_searxng:{goal[:400]}",
    )
    queries = [q.strip() for q in (plan.queries or []) if (q or "").strip()]
    if len(queries) < 2:
        queries = build_practical_searxng_queries(goal, max_queries=max_queries)
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out[:max(1, max_queries)]


async def collect_searxng_practical_rows(
    expansion_vector: str,
    *,
    limit: int | None = None,
    anchor: str | None = None,
) -> list[dict[str, str]]:
    """JSON hits из SearXNG: categories it/science/general, дедуп URL."""
    if not SEARXNG_ENABLED:
        trace("CURRICULUM searxng ⊘ | SEARXNG_ENABLED=false (docker compose up -d searxng)")
        return []

    cap = limit if limit is not None else CURRICULUM_PRACTICAL_SEARXNG_LIMIT
    max_q = CURRICULUM_PRACTICAL_SEARXNG_QUERIES
    queries = await _searxng_queries_for_goal(
        expansion_vector,
        max_queries=max_q,
        anchor=anchor,
    )
    if not queries:
        return []

    categories = list(SEARXNG_DISCOVERY_CATEGORIES) or ["it", "science", "general"]
    per_q = max(2, (cap + len(queries) - 1) // len(queries))
    rows: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    trace(
        f"CURRICULUM searxng ▶ | queries={len(queries)} "
        f"categories={','.join(categories)} cap={cap}"
    )

    for q in queries:
        if len(rows) >= cap:
            break
        raw: list[dict[str, Any]] = await searxng_search_json(
            q,
            limit=per_q,
            categories=categories,
        )
        for item in raw:
            if item.get("error"):
                continue
            url = _normalize_href(str(item.get("url") or ""))
            if not url or url.lower() in seen_urls:
                continue
            seen_urls.add(url.lower())
            rows.append(
                {
                    "url": url,
                    "title": str(item.get("title") or url)[:400],
                    "snippet": str(item.get("snippet") or "")[:1200],
                }
            )
            if len(rows) >= cap:
                break

    trace(f"CURRICULUM searxng ✓ | hits={len(rows)}")
    return rows
