"""Zero-Waste Handover: _ingest_blog_url_precheck must reuse HTML already
downloaded by Pre-Flight Triage Stage 2 instead of calling
smart_fetch_page_html() a second time for the same URL."""

from __future__ import annotations

import asyncio

from knowledge_engine.src.curriculum import pre_flight_triage as pft
from knowledge_engine.src.curriculum import source_material_pipeline as smp
from knowledge_engine.src.curriculum.schemas import CurriculumSearchHit


def _hit(url: str) -> CurriculumSearchHit:
    return CurriculumSearchHit(
        url=url,
        title="t",
        snippet="s",
        key_extracts=["x"],
        source_tier="exa",
    )


async def _no_cache(url: str):
    return None


def test_precheck_reuses_preflight_html_and_skips_refetch(monkeypatch):
    hit = _hit("https://example.com/already-triaged")
    preflight_body = "<html><body>" + ("pre-fetched word " * 200) + "</body></html>"
    pft.stash_preflight_html(hit.url, preflight_body)

    monkeypatch.setattr(smp, "_extracts_from_lancedb_url", _no_cache)
    monkeypatch.setattr(smp, "_lancedb_has_map_windows", lambda url: False)

    calls = {"n": 0}

    def fake_fetch(url):
        calls["n"] += 1
        return "<html>freshly fetched (should not happen)</html>", "httpx"

    monkeypatch.setattr(smp, "smart_fetch_page_html", fake_fetch)
    monkeypatch.setattr(smp, "is_anti_bot_fetch_result", lambda *a, **k: False)

    early, title, html = asyncio.run(smp._ingest_blog_url_precheck(hit))

    assert (
        calls["n"] == 0
    ), "smart_fetch_page_html must NOT be called for a hit with cached preflight html"
    assert html == preflight_body
    assert (
        pft.pop_preflight_html(hit.url) is None
    ), "cache entry must be consumed exactly once"


def test_precheck_fetches_normally_when_no_preflight_html(monkeypatch):
    hit = _hit("https://example.com/not-triaged")
    assert pft.pop_preflight_html(hit.url) is None  # nothing stashed

    monkeypatch.setattr(smp, "_extracts_from_lancedb_url", _no_cache)
    monkeypatch.setattr(smp, "_lancedb_has_map_windows", lambda url: False)

    calls = {"n": 0}

    def fake_fetch(url):
        calls["n"] += 1
        return "<html>" + ("word " * 200) + "</html>", "httpx"

    monkeypatch.setattr(smp, "smart_fetch_page_html", fake_fetch)
    monkeypatch.setattr(smp, "is_anti_bot_fetch_result", lambda *a, **k: False)

    early, title, html = asyncio.run(smp._ingest_blog_url_precheck(hit))

    assert (
        calls["n"] == 1
    ), "smart_fetch_page_html must be called when nothing is cached"
    assert "word" in html
