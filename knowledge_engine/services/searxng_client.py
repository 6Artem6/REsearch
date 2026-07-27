"""HTTP-клиент SearXNG (JSON): categories, engines, поле engine в результатах."""

from __future__ import annotations

from typing import Any, Optional

import httpx

from knowledge_engine.config import (
    SEARXNG_BASE_URL,
    SEARXNG_DEFAULT_ENGINES,
    SEARXNG_ENABLED,
    SEARXNG_REQUEST_HEADERS,
    SEARXNG_TIMEOUT_SEC,
)
from knowledge_engine.ui.run_log import trace

# Инжиниринговые движки SearXNG — приоритет trust / очереди
_PRIORITY_SEARXNG_ENGINES: frozenset[str] = frozenset(
    {
        "hackernews",
        "hacker_news",
        "github",
        "stackoverflow",
        "arxiv",
        "google scholar",
        "google_scholar",
        "semantic_scholar",
    }
)


def normalize_searxng_engine(engine: str) -> str:
    return (engine or "").strip().lower().replace(" ", "_")


def is_priority_searxng_engine(engine: str) -> bool:
    key = normalize_searxng_engine(engine)
    if key in _PRIORITY_SEARXNG_ENGINES:
        return True
    return any(p in key for p in ("github", "hackernews", "stackoverflow", "arxiv"))


def engine_trust_hint(engine: str) -> tuple[float, str] | None:
    """Стартовый trust для ссылок из IT/science движков (до Domain Profiler)."""
    if not is_priority_searxng_engine(engine):
        return None
    key = normalize_searxng_engine(engine)
    if "arxiv" in key or "scholar" in key:
        return (0.85, "academic")
    return (0.8, "tech_blog")


async def searxng_search_json(
    query: str,
    limit: int = 5,
    *,
    categories: Optional[list[str]] = None,
    engines: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Поиск через локальный SearXNG.
    Каждый элемент: title, url, snippet, source, engine.
    """
    if not SEARXNG_ENABLED or not query.strip():
        return []

    timeout = httpx.Timeout(SEARXNG_TIMEOUT_SEC)
    base = f"{SEARXNG_BASE_URL}/search"

    engines_chain: list[str] = []
    if engines:
        engines_chain.append(engines)
    engines_chain.extend(["bing", "google", SEARXNG_DEFAULT_ENGINES, ""])

    categories_param = ",".join(c for c in (categories or []) if c.strip())

    last_exc: Exception | None = None
    for eng in engines_chain:
        params: dict[str, str] = {"q": query, "format": "json"}
        if eng:
            params["engines"] = eng
        if categories_param:
            params["categories"] = categories_param
        try:
            async with httpx.AsyncClient(
                timeout=timeout, headers=SEARXNG_REQUEST_HEADERS
            ) as client:
                res = await client.get(base, params=params)
                res.raise_for_status()
                data = res.json()
            errors = data.get("errors") or []
            raw = data.get("results", [])[:limit]
            if not raw and errors:
                last_exc = RuntimeError(f"errors={errors[:2]}")
                continue
            results: list[dict[str, Any]] = []
            for item in raw:
                engine_name = str(
                    item.get("engine") or item.get("engines") or "searxng"
                )
                results.append(
                    {
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "snippet": item.get("content"),
                        "source": "google_meta",
                        "engine": engine_name,
                    }
                )
            if categories_param and results:
                trace(
                    f"SEARXNG categories={categories_param} | "
                    f"engines_sample={results[0].get('engine')}"
                )
            return results
        except Exception as exc:
            last_exc = exc
            continue

    return [
        {"error": f"SearXNG error: {last_exc}", "source": "google_meta", "engine": ""}
    ]
