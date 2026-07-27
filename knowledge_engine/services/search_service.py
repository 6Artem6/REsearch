"""Поиск для API (SearchRegistry / горизонты)."""

from __future__ import annotations

from typing import Any

from knowledge_engine.services.search.horizons import (
    HORIZON_LABELS,
    HORIZON_PROVIDERS,
    SearchHorizon,
    build_horizon_queries,
)
from knowledge_engine.services.search.registry import default_registry
from knowledge_engine.services.search.searxng_health import check_searxng


def searxng_health() -> tuple[bool, str]:
    return check_searxng()


def search_flat(query: str, limit_per_provider: int = 2) -> list[dict[str, Any]]:
    registry = default_registry()
    hits = registry.multi_search_sync(query, limit_per_provider=limit_per_provider)
    return [
        {
            "source": h.source,
            "title": h.title,
            "url": h.url,
            "horizon": getattr(h, "horizon", None),
        }
        for h in hits
    ]


def search_horizons(
    query: str,
    constraints: str,
    limit_per_provider: int = 3,
) -> dict[str, Any]:
    registry = default_registry()
    fake_abs = [
        {
            "title": query[:80],
            "cs_concept": query,
            "description": "API test-search",
        }
    ]
    horizon_queries = build_horizon_queries(query, constraints, fake_abs)
    hits, _ = registry.multi_search_horizons_sync(
        query,
        constraints,
        fake_abs,
        limit_per_provider=limit_per_provider,
    )
    buckets: dict[str, list[dict[str, Any]]] = {}
    for horizon in SearchHorizon:
        buckets[horizon.value] = [
            {
                "source": h.source,
                "title": h.title,
                "url": h.url,
            }
            for h in hits
            if h.horizon == horizon.value
        ]
    meta = {
        "horizon_labels": {h.value: HORIZON_LABELS[h] for h in SearchHorizon},
        "horizon_providers": {
            h.value: list(HORIZON_PROVIDERS[h]) for h in SearchHorizon
        },
        "horizon_queries": {h.value: horizon_queries[h] for h in SearchHorizon},
        "unique_url_count": len(hits),
    }
    return {"meta": meta, "results": buckets}
