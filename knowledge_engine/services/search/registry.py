"""Централизованный реестр поисковых провайдеров."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from knowledge_engine.config import (
    EXA_API_KEY,
    EXA_SEARCH_ENABLED,
    resolved_search_active_providers,
)
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
    ExaSearchProvider,
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
        if EXA_API_KEY and EXA_SEARCH_ENABLED:
            self.register(ExaSearchProvider())

    def register(self, provider: BaseSearchProvider) -> None:
        self.providers[provider.name] = provider

    async def multi_search(
        self,
        query: str,
        active_providers: Optional[List[str]] = None,
        limit_per_provider: int = 3,
        searxng_categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        targets = active_providers or list(resolved_search_active_providers())

        async def _run_provider(name: str) -> list[dict[str, Any]]:
            provider = self.providers.get(name)
            if provider is None:
                return []
            try:
                kw: dict[str, Any] = {}
                if name == "google_meta" and searxng_categories:
                    kw["categories"] = list(searxng_categories)
                res = await provider.search(query, limit=limit_per_provider, **kw)
                return list(res) if res else []
            except Exception as exc:
                return [{"error": str(exc), "source": name}]

        chunks = await asyncio.gather(*[_run_provider(n) for n in targets])
        aggregated: list[dict[str, Any]] = []
        for chunk in chunks:
            aggregated.extend(chunk)
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

    async def multi_search_queries_batch(
        self,
        queries: list[str],
        *,
        active_providers: Optional[List[str]] = None,
        limit_per_provider: int = 3,
        searxng_categories: Optional[List[str]] = None,
        concurrency: int = 6,
    ) -> List[SearchResult]:
        """Параллельный batch запросов (open search / fallback)."""
        qs = [q.strip() for q in queries if (q or "").strip()]
        if not qs:
            return []
        sem = asyncio.Semaphore(max(1, concurrency))

        async def _one(query: str) -> List[SearchResult]:
            async with sem:
                payload = await self.multi_search(
                    query,
                    active_providers=active_providers,
                    limit_per_provider=limit_per_provider,
                    searxng_categories=searxng_categories,
                )
            out: list[SearchResult] = []
            seen_local: set[str] = set()
            for item in payload.get("results", []):
                hit = search_result_from_dict(item)
                if hit is None or hit.url in seen_local:
                    continue
                seen_local.add(hit.url)
                out.append(hit)
            return out

        batches = await asyncio.gather(*[_one(q) for q in qs])
        merged: list[SearchResult] = []
        seen: set[str] = set()
        for hits in batches:
            for hit in hits:
                if hit.url in seen:
                    continue
                seen.add(hit.url)
                merged.append(hit)
        return merged

    def multi_search_queries_batch_sync(
        self,
        queries: list[str],
        *,
        active_providers: Optional[List[str]] = None,
        limit_per_provider: int = 3,
        searxng_categories: Optional[List[str]] = None,
        concurrency: int = 6,
    ) -> List[SearchResult]:
        return asyncio.run(
            self.multi_search_queries_batch(
                queries,
                active_providers=active_providers,
                limit_per_provider=limit_per_provider,
                searxng_categories=searxng_categories,
                concurrency=concurrency,
            )
        )

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
