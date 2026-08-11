"""SearXNG, академические API, Habr, Consensus (dorks)."""

from __future__ import annotations

from typing import Any

import httpx

from knowledge_engine.config import (
    CROSSREF_API_URL,
    HABR_API_URL,
    LECTURE_EXTERNAL_SEARCH_HTTP_TIMEOUT_SEC,
    SEARXNG_BASE_URL,
    SEARXNG_ENABLED,
    SEARXNG_REQUEST_HEADERS,
    SEARXNG_TIMEOUT_SEC,
    SEMANTIC_SCHOLAR_API_URL,
)
from knowledge_engine.services.search.base import BaseSearchProvider
from knowledge_engine.services.searxng_client import searxng_search_json


class SearXNGProvider(BaseSearchProvider):
    """Google/Bing + IT/science engines через локальный SearXNG (JSON)."""

    name = "google_meta"

    async def search(self, query: str, limit: int = 5, **kwargs: Any) -> list[dict]:
        categories = kwargs.get("categories")
        engines = kwargs.get("engines")
        if categories is not None and not isinstance(categories, list):
            categories = None
        if engines is not None and not isinstance(engines, str):
            engines = None
        return await searxng_search_json(
            query,
            limit=limit,
            categories=categories,
            engines=engines,
        )


class HabrSearchProvider(BaseSearchProvider):
    """Хабр: SearXNG dork site:habr.com (надёжнее, чем только Kairos API)."""

    name = "habr"

    async def search(self, query: str, limit: int = 5, **kwargs: Any) -> list[dict]:
        if not SEARXNG_ENABLED:
            return await self._habr_api_fallback(query, limit)
        timeout = httpx.Timeout(SEARXNG_TIMEOUT_SEC)
        try:
            async with httpx.AsyncClient(
                timeout=timeout, headers=SEARXNG_REQUEST_HEADERS
            ) as client:
                res = await client.get(
                    f"{SEARXNG_BASE_URL}/search",
                    params={
                        "q": f"site:habr.com/ru/ {query}",
                        "format": "json",
                        "engines": "bing",
                    },
                )
                res.raise_for_status()
                data = res.json()
            return [
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "snippet": item.get("content"),
                    "source": "habr",
                }
                for item in data.get("results", [])[:limit]
            ]
        except Exception:
            return await self._habr_api_fallback(query, limit)

    async def _habr_api_fallback(self, query: str, limit: int) -> list[dict]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(
                    HABR_API_URL,
                    params={"q": query, "per_page": limit},
                )
                res.raise_for_status()
                payload = res.json()
        except Exception:
            return []
        articles = payload.get("articles") or payload.get("data") or []
        out: list[dict] = []
        for art in articles[:limit]:
            url = art.get("url") or f"https://habr.com/ru/post/{art.get('id', '')}/"
            out.append(
                {
                    "title": art.get("title", ""),
                    "url": url,
                    "snippet": (
                        art.get("leadData", {}).get("text") or art.get("snippet") or ""
                    )[:500],
                    "source": "habr_api",
                }
            )
        return out


class SemanticScholarProvider(BaseSearchProvider):
    name = "semantic_scholar"

    async def search(self, query: str, limit: int = 5, **kwargs: Any) -> list[dict]:
        from knowledge_engine.src.retrieval.semantic_scholar_rate_limit import (
            acquire_semantic_scholar_slot_async,
        )

        try:
            async with httpx.AsyncClient(
                timeout=LECTURE_EXTERNAL_SEARCH_HTTP_TIMEOUT_SEC,
            ) as client:
                await acquire_semantic_scholar_slot_async()
                res = await client.get(
                    SEMANTIC_SCHOLAR_API_URL,
                    params={
                        "query": query,
                        "limit": limit,
                        "fields": "title,abstract,url,paperId",
                    },
                )
                res.raise_for_status()
                data = res.json()
        except Exception:
            return []
        out: list[dict] = []
        for paper in data.get("data", [])[:limit]:
            pid = paper.get("paperId")
            url = paper.get("url") or (
                f"https://www.semanticscholar.org/paper/{pid}" if pid else ""
            )
            if not url:
                continue
            out.append(
                {
                    "title": paper.get("title"),
                    "url": url,
                    "snippet": (paper.get("abstract") or "")[:500],
                    "source": "semantic_scholar",
                }
            )
        return out


class ConsensusSearchProvider(BaseSearchProvider):
    name = "consensus"

    async def search(self, query: str, limit: int = 5, **kwargs: Any) -> list[dict]:
        from knowledge_engine.config import CONSENSUS_USE_DIRECT_API

        if CONSENSUS_USE_DIRECT_API:
            try:
                from knowledge_engine.services.search.consensus_direct_client import (
                    acquire_consensus_direct_client,
                )

                client = await acquire_consensus_direct_client()
                papers = await client.search_papers(query, limit=limit)
                out: list[dict] = []
                for paper in papers:
                    url = (paper.source_url or "").strip()
                    if (
                        not url
                        and paper.paper_id
                        and str(paper.paper_id).startswith("10.")
                    ):
                        url = f"https://doi.org/{paper.paper_id}"
                    if not url:
                        continue
                    out.append(
                        {
                            "title": paper.title,
                            "url": url,
                            "snippet": (paper.abstract or paper.tldr or "")[:500],
                            "source": "consensus",
                        }
                    )
                return out[:limit]
            except Exception:
                return []

        if not SEARXNG_ENABLED:
            return []
        try:
            async with httpx.AsyncClient(
                timeout=SEARXNG_TIMEOUT_SEC, headers=SEARXNG_REQUEST_HEADERS
            ) as client:
                res = await client.get(
                    f"{SEARXNG_BASE_URL}/search",
                    params={
                        "q": f"site:consensus.app {query}",
                        "format": "json",
                        "engines": "bing",
                    },
                )
                res.raise_for_status()
                data = res.json()
            return [
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "snippet": item.get("content"),
                    "source": "consensus",
                }
                for item in data.get("results", [])[:limit]
            ]
        except Exception:
            return []


class ArxivProvider(BaseSearchProvider):
    name = "arxiv"

    async def search(self, query: str, limit: int = 5, **kwargs: Any) -> list[dict]:
        from knowledge_engine.services.search.arxiv_client import get_arxiv_client
        from knowledge_engine.services.search.arxiv_query_builder import (
            ArxivQueryBuilder,
            ArxivQueryParams,
        )

        q = (query or "").strip()
        if not q and not kwargs.get("arxiv_params"):
            return []
        start = int(kwargs.get("start") or 0)
        sort_by = kwargs.get("sort_by") or kwargs.get("sortBy")
        sort_order = kwargs.get("sort_order") or kwargs.get("sortOrder")
        raw_params = kwargs.get("arxiv_params")
        params = ArxivQueryParams.from_mapping(raw_params) if raw_params else None
        if params is not None and params.has_precision():
            built = ArxivQueryBuilder(params).build(
                free_text_fallback=q,
                start=start,
                max_results=limit,
                sort_by=str(sort_by) if sort_by else None,
                sort_order=str(sort_order) if sort_order else None,
            )
            search_query = built.search_query
            start = built.start
            sort_by = built.sort_by
            sort_order = built.sort_order
        elif q.lower().startswith(("all:", "ti:", "abs:", "cat:", "au:")):
            search_query = q
        else:
            built = ArxivQueryBuilder(params).build(
                free_text_fallback=q,
                start=start,
                max_results=limit,
                sort_by=str(sort_by) if sort_by else None,
                sort_order=str(sort_order) if sort_order else None,
            )
            search_query = built.search_query or f"all:{q}"
            start = built.start
            sort_by = built.sort_by
            sort_order = built.sort_order
        try:
            entries = await get_arxiv_client().search(
                search_query=search_query,
                start=start,
                max_results=limit,
                sort_by=str(sort_by) if sort_by else None,
                sort_order=str(sort_order) if sort_order else None,
            )
        except Exception:
            return []
        out: list[dict] = []
        for entry in entries:
            link = entry.abs_url
            if not entry.title or not link:
                continue
            out.append(
                {
                    "title": entry.title,
                    "url": link,
                    "snippet": (entry.abstract or "")[:500],
                    "source": "arxiv",
                }
            )
        return out


class CrossrefProvider(BaseSearchProvider):
    name = "crossref"

    async def search(self, query: str, limit: int = 5, **kwargs: Any) -> list[dict]:
        try:
            async with httpx.AsyncClient(timeout=12.0) as client:
                res = await client.get(
                    CROSSREF_API_URL,
                    params={"query": query, "rows": limit},
                    headers={"User-Agent": "knowledge-engine/0.2 (mailto:local@dev)"},
                )
                res.raise_for_status()
                items = res.json().get("message", {}).get("items", [])
        except Exception:
            return []
        out: list[dict] = []
        for item in items[:limit]:
            title_list = item.get("title") or []
            title = title_list[0] if title_list else "Untitled"
            doi = item.get("DOI")
            url = f"https://doi.org/{doi}" if doi else item.get("URL") or ""
            if not url:
                continue
            out.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": (item.get("abstract") or "")[:500],
                    "source": "crossref",
                }
            )
        return out


class ExaSearchProvider(BaseSearchProvider):
    """Neural search по whitelist-доменам (exa-py). contents = highlights only (no Exa AI summary)."""

    name = "exa"

    async def search(self, query: str, limit: int = 5, **kwargs: Any) -> list[dict]:
        import asyncio

        from knowledge_engine.config import EXA_API_KEY, EXA_SEARCH_ENABLED
        from knowledge_engine.services.search.exa_client import (
            ExaNotConfiguredError,
            ExaSearchClient,
        )
        from knowledge_engine.services.search.exa_transform import (
            exa_response_to_provider_dicts,
        )

        if not EXA_SEARCH_ENABLED or not EXA_API_KEY:
            return [{"error": "Exa not configured", "source": "exa"}]

        client = ExaSearchClient(api_key=EXA_API_KEY)
        num = max(1, min(limit, 25))
        try:
            response = await asyncio.to_thread(client.search, query, num_results=num)
        except ExaNotConfiguredError as exc:
            return [{"error": str(exc), "source": "exa"}]
        except Exception as exc:
            return [{"error": str(exc), "source": "exa"}]
        return exa_response_to_provider_dicts(response)[:limit]
