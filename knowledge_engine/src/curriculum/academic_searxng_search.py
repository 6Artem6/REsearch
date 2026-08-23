"""Академический поиск через SearXNG: category science + engines arxiv/scholar."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from knowledge_engine.config import (
    CURRICULUM_ACADEMIC_SEARXNG_CATEGORIES,
    CURRICULUM_ACADEMIC_SEARXNG_ENGINES,
    CURRICULUM_ACADEMIC_SEARXNG_LIMIT,
    SEARXNG_ENABLED,
)
from knowledge_engine.services.searxng_client import searxng_search_json
from knowledge_engine.ui.run_log import trace


def _academic_categories() -> list[str]:
    raw = (CURRICULUM_ACADEMIC_SEARXNG_CATEGORIES or "science").strip()
    cats = [c.strip() for c in raw.split(",") if c.strip()]
    return cats or ["science"]


def _academic_engines() -> str:
    return (CURRICULUM_ACADEMIC_SEARXNG_ENGINES or "arxiv,google scholar").strip()


def _normalize_href(href: str) -> str:
    u = (href or "").strip().rstrip(".,);]")
    if not u.startswith("http"):
        return ""
    return u.split("#")[0].rstrip("/")


def _academic_url_accept_reason(url: str) -> str | None:
    """None = accept; string = reject reason."""
    u = (url or "").strip().lower()
    if not u.startswith("http"):
        return "not_http"
    host = (urlparse(u).netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    allowed = (
        "arxiv.org",
        "export.arxiv.org",
        "semanticscholar.org",
        "doi.org",
        "scholar.google.com",
    )
    for suffix in allowed:
        if host == suffix or host.endswith(f".{suffix}"):
            return None
    return f"not_academic_host:{host[:40]}"


def _tier_for_academic_url(url: str, engine: str) -> str:
    u = (url or "").lower()
    eng = (engine or "").lower()
    if "arxiv" in u or "arxiv" in eng:
        return "arxiv"
    if "semantic" in u or "scholar" in eng:
        return "semantic_scholar"
    return "searxng_science"


async def collect_searxng_academic_rows(
    query: str,
    *,
    limit: int | None = None,
) -> list[dict[str, str]]:
    """SearXNG academic: categories=science, engines=arxiv,google scholar.

    Never falls back to bing/google (those pollute with non-academic hosts).
    Category ``it`` (github/hn/stackoverflow) is intentionally not used here —
    that belongs to practical/community search, not the academic track.
    """
    if not SEARXNG_ENABLED:
        trace("CURRICULUM searxng academic ⊘ | SEARXNG_ENABLED=false")
        return []

    q = (query or "").strip()
    if len(q) < 4:
        return []

    cap = limit if limit is not None else CURRICULUM_ACADEMIC_SEARXNG_LIMIT
    categories = _academic_categories()
    engines = _academic_engines()
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    rejected = 0

    trace(
        f"CURRICULUM searxng academic ▶ | query={q[:100]} "
        f"engines={engines} categories={','.join(categories)} cap={cap}"
    )

    raw: list[dict[str, Any]] = await searxng_search_json(
        q,
        limit=cap,
        categories=categories,
        engines=engines,
    )
    for item in raw:
        if item.get("error"):
            continue
        url = _normalize_href(str(item.get("url") or ""))
        if not url or url.lower() in seen:
            continue
        reason = _academic_url_accept_reason(url)
        if reason:
            trace(f"CURRICULUM searxng academic filter ⊘ | {url[:70]} | {reason}")
            rejected += 1
            continue
        seen.add(url.lower())
        engine = str(item.get("engine") or "")
        rows.append(
            {
                "url": url,
                "title": str(item.get("title") or url)[:400],
                "snippet": str(item.get("snippet") or "")[:1200],
                "engine": engine,
                "source_tier": _tier_for_academic_url(url, engine),
            }
        )
        if len(rows) >= cap:
            break

    trace(
        f"CURRICULUM searxng academic ✓ | hits={len(rows)} "
        f"post_filter_rejected={rejected}"
    )
    return rows
