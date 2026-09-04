"""Post-replenish must reuse LanceDB and can defer full Gemma on on-demand init."""

from __future__ import annotations

import asyncio

from knowledge_engine.src.curriculum import source_material_pipeline as smp
from knowledge_engine.src.curriculum.schemas import CurriculumSearchHit


def _hit(**kwargs) -> CurriculumSearchHit:
    base = {
        "title": "Paper",
        "url": "https://arxiv.org/pdf/2512.20660.pdf",
        "snippet": "x",
        "source_tier": "arxiv",
        "key_extracts": [],
    }
    base.update(kwargs)
    return CurriculumSearchHit(**base)


def test_mandatory_ingest_reuses_lancedb(monkeypatch) -> None:
    hit = _hit()

    async def _cached(url):
        return ["takeaway one about pipelines"], "Cached title"

    monkeypatch.setattr(smp, "_extracts_from_lancedb_url", _cached)
    called: list[str] = []

    async def _never_ingest(h, *, force_full_ingest: bool = False):
        called.append(h.url)
        return h

    monkeypatch.setattr(smp, "_ingest_academic_hit_async", _never_ingest)
    monkeypatch.setattr(smp, "_try_auto_article_diagrams", lambda h: None)
    monkeypatch.setattr(
        smp, "_spawn_mandatory_academic_ingest_daemon", lambda *a, **k: None
    )

    out = asyncio.run(smp.ingest_mandatory_academic_hits_async([hit], label="test"))
    assert called == []
    assert out[0].title == "Cached title"
    assert out[0].key_extracts == ["takeaway one about pipelines"]


def test_mandatory_ingest_defers_missing_on_demand(monkeypatch) -> None:
    hit = _hit(
        title="New paper",
        url="https://arxiv.org/pdf/9999.00000.pdf",
        snippet="abstract text",
        key_extracts=["short"],
    )
    async def _no_cache(url):
        return None

    monkeypatch.setattr(smp, "_extracts_from_lancedb_url", _no_cache)
    spawned: list[list] = []

    def _spawn(hits, *, label: str):
        spawned.append(list(hits))

    monkeypatch.setattr(smp, "_spawn_mandatory_academic_ingest_daemon", _spawn)
    monkeypatch.setattr(smp, "_try_auto_article_diagrams", lambda h: None)

    out = asyncio.run(
        smp.ingest_mandatory_academic_hits_async(
            [hit], label="test", defer_missing=True
        )
    )
    assert len(spawned) == 1
    assert spawned[0][0].url == hit.url
    assert out[0].url == hit.url
    assert out[0].key_extracts == ["short"]
