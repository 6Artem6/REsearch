"""Централизованный реестр поисковых провайдеров."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from knowledge_engine.config import SEARCH_ACTIVE_PROVIDERS
from knowledge_engine.services.search.base import (
    BaseSearchProvider,
    SearchResult,
    search_result_from_dict,
)
from knowledge_engine.services.search.horizons import (
    HORIZON_PROVIDERS,
    SearchHorizon,
    build_horizon_queries,
)
from knowledge_engine.services.search.providers import (
    ArxivProvider,
    ConsensusSearchProvider,
    CrossrefProvider,
    HabrSearchProvider,
    SearXNGProvider,
    SemanticScholarProvider,
)


class SearchRegistry:
    def __init__(self) -> None:
        self.providers: Dict[str, BaseSearchProvider] = {}
        self.register(SearXNGProvider())
        self.register(SemanticScholarProvider())
        self.register(HabrSearchProvider())
        self.register(ConsensusSearchProvider())
        self.register(ArxivProvider())
        self.register(CrossrefProvider())

    def register(self, provider: BaseSearchProvider) -> None:
        self.providers[provider.name] = provider

    async def multi_search(
        self,
        query: str,
        active_providers: Optional[List[str]] = None,
        limit_per_provider: int = 3,
        searxng_categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        targets = active_providers or list(SEARCH_ACTIVE_PROVIDERS)
        aggregated: list[dict[str, Any]] = []
        searxng_kw: dict[str, Any] = {}
        if searxng_categories:
            searxng_kw["categories"] = list(searxng_categories)
        for name in targets:
            provider = self.providers.get(name)
            if provider is None:
                continue
            try:
                kw = dict(searxng_kw) if name == "google_meta" else {}
                res = await provider.search(query, limit=limit_per_provider, **kw)
                aggregated.extend(res)
            except Exception as exc:
                aggregated.append({"error": str(exc), "source": name})
        return {"query": query, "results": aggregated}

    def multi_search_sync(
        self,
        query: str,
        active_providers: Optional[List[str]] = None,
        limit_per_provider: int = 3,
        searxng_categories: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """Синхронная обёртка для узлов LangGraph."""
        payload = asyncio.run(
            self.multi_search(
                query,
                active_providers=active_providers,
                limit_per_provider=limit_per_provider,
                searxng_categories=searxng_categories,
            )
        )
        merged: list[SearchResult] = []
        seen: set[str] = set()
        for item in payload.get("results", []):
            hit = search_result_from_dict(item)
            if hit is None or hit.url in seen:
                continue
            seen.add(hit.url)
            merged.append(hit)
        return merged

    def multi_search_horizons_sync(
        self,
        user_problem: str,
        context_constraints: str,
        abstractions: list[Any],
        limit_per_provider: int = 2,
        horizon_queries: dict[SearchHorizon, str] | None = None,
    ) -> tuple[list[SearchResult], dict[str, str]]:
        """Поиск по трём горизонтам SOTA / Infra / Prod с разными провайдерами."""
        built = horizon_queries or build_horizon_queries(
            user_problem, context_constraints, abstractions
        )
        merged: list[SearchResult] = []
        seen: set[str] = set()
        flat_queries: dict[str, str] = {}

        for horizon, query in built.items():
            flat_queries[horizon.value] = query
            provider_names = list(HORIZON_PROVIDERS[horizon])
            hits = self.multi_search_sync(
                query,
                active_providers=provider_names,
                limit_per_provider=limit_per_provider,
            )
            for hit in hits:
                hit.horizon = horizon.value
                if hit.url in seen:
                    continue
                seen.add(hit.url)
                merged.append(hit)

        return merged, flat_queries


def default_registry() -> SearchRegistry:
    return SearchRegistry()
