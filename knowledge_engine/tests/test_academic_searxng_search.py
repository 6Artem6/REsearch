"""Academic SearXNG must pin engines (no bing fallback)."""

from __future__ import annotations

import asyncio
from typing import Any

from knowledge_engine.src.curriculum import academic_searxng_search as mod


def test_academic_searxng_passes_arxiv_scholar_engines(monkeypatch):
    captured: dict[str, Any] = {}

    async def _fake_search(query, limit=5, *, categories=None, engines=None):
        captured["query"] = query
        captured["limit"] = limit
        captured["categories"] = categories
        captured["engines"] = engines
        return [
            {
                "title": "Paper",
                "url": "https://arxiv.org/abs/2301.00001",
                "snippet": "abs",
                "engine": "arxiv",
            }
        ]

    monkeypatch.setattr(mod, "SEARXNG_ENABLED", True)
    monkeypatch.setattr(mod, "searxng_search_json", _fake_search)
    monkeypatch.setattr(mod, "CURRICULUM_ACADEMIC_SEARXNG_ENGINES", "arxiv,google scholar")
    monkeypatch.setattr(mod, "CURRICULUM_ACADEMIC_SEARXNG_CATEGORIES", "science")

    rows = asyncio.run(
        mod.collect_searxng_academic_rows("multi-agent orchestration", limit=5)
    )
    assert captured["engines"] == "arxiv,google scholar"
    assert captured["categories"] == ["science"]
    assert len(rows) == 1
    assert rows[0]["url"].startswith("https://arxiv.org/")


def test_academic_rejects_non_academic_hosts():
    assert mod._academic_url_accept_reason("https://en.wikipedia.org/wiki/Fault")
    assert mod._academic_url_accept_reason("https://arxiv.org/abs/1") is None
