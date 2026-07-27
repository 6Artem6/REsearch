"""SearXNG, академические API, Habr, Consensus (dorks)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import urlencode

import httpx

from knowledge_engine.config import (
    ARXIV_API_URL,
    CROSSREF_API_URL,
    HABR_API_URL,
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
        try:
            async with httpx.AsyncClient(timeout=12.0) as client:
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
        params = urlencode(
            {"search_query": f"all:{query}", "start": 0, "max_results": limit}
        )
        url = f"{ARXIV_API_URL}?{params}"
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                text = resp.text
        except Exception:
            return []
        root = ET.fromstring(text)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        out: list[dict] = []
        for entry in root.findall("a:entry", ns):
            title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
            link = ""
            for link_el in entry.findall("a:link", ns):
                if link_el.attrib.get("type") == "html":
                    link = link_el.attrib.get("href", "")
                    break
            summary = (
                entry.findtext("a:summary", default="", namespaces=ns) or ""
            ).strip()
            if title and link:
                out.append(
                    {
                        "title": title,
                        "url": link,
                        "snippet": summary[:500],
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
