"""Практический поиск через локальный SearXNG (Lite site: queries, category general)."""

from __future__ import annotations

from typing import Any

from knowledge_engine.config import (
    CURRICULUM_PRACTICAL_SEARXNG_ENGINES,
    CURRICULUM_PRACTICAL_SEARXNG_LIMIT,
    CURRICULUM_PRACTICAL_SEARXNG_QUERIES,
    SEARXNG_ENABLED,
)
from knowledge_engine.services.searxng_client import searxng_search_json
from knowledge_engine.src.curriculum.lite_search_pipeline import (
    build_search_queries,
    flatten_whitelist_domains,
)
from knowledge_engine.src.curriculum.practical_url_filters import filter_practical_search_row
from knowledge_engine.src.curriculum.search_query_builder import build_practical_searxng_queries
from knowledge_engine.ui.run_log import trace

# Жёстко: только general → Google/Bing (не science / arXiv).
_PRACTICAL_SEARXNG_CATEGORIES: list[str] = ["general"]


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
    lite_query_plan: bool = True,
    max_queries: int | None = None,
) -> list[dict[str, str]]:
    """SearXNG practical: categories=['general'], engines google,bing."""
    if not SEARXNG_ENABLED:
        trace("CURRICULUM searxng ⊘ | SEARXNG_ENABLED=false (docker compose up -d searxng)")
        return []

    cap = limit if limit is not None else CURRICULUM_PRACTICAL_SEARXNG_LIMIT
    max_q = max_queries if max_queries is not None else CURRICULUM_PRACTICAL_SEARXNG_QUERIES
    if lite_query_plan:
        queries = await _searxng_queries_for_goal(
            expansion_vector,
            max_queries=max_q,
            anchor=anchor,
        )
    else:
        goal = (expansion_vector or "").strip()
        queries = build_practical_searxng_queries(goal, max_queries=max(1, max_q))
    if not queries:
        return []

    categories = list(_PRACTICAL_SEARXNG_CATEGORIES)
    engines = (CURRICULUM_PRACTICAL_SEARXNG_ENGINES or "google,bing").strip()
    per_q = max(2, (cap + len(queries) - 1) // len(queries))
    rows: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    rejected = 0

    trace(
        f"CURRICULUM searxng practical ▶ | queries={len(queries)} "
        f"engines={engines} categories={categories} cap={cap}"
    )

    for q in queries:
        if len(rows) >= cap:
            break
        raw: list[dict[str, Any]] = await searxng_search_json(
            q,
            limit=per_q,
            categories=categories,
            engines=engines,
        )
        for item in raw:
            if item.get("error"):
                continue
            url = _normalize_href(str(item.get("url") or ""))
            if not url or url.lower() in seen_urls:
                continue
            row = {
                "url": url,
                "title": str(item.get("title") or url)[:400],
                "snippet": str(item.get("snippet") or "")[:1200],
                "engine": str(item.get("engine") or ""),
            }
            if not filter_practical_search_row(row):
                rejected += 1
                continue
            seen_urls.add(url.lower())
            rows.append(row)
            if len(rows) >= cap:
                break

    trace(
        f"CURRICULUM searxng practical ✓ | hits={len(rows)} "
        f"post_filter_rejected={rejected}"
    )
    return rows
