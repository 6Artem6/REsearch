"""Pre-MAP Dedup integration in source_material_pipeline.py — unit tests for
_pre_map_dedup_batch_items and its wiring into _ingest_blog_hits_batch_async."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from knowledge_engine.src.curriculum import source_material_pipeline as m
from knowledge_engine.src.curriculum.schemas import CurriculumSearchHit


def _hit(url: str, *, title: str = "") -> CurriculumSearchHit:
    return CurriculumSearchHit(url=url, title=title or url, snippet="snippet")


def _summary(title: str = "Doc", takeaways: list[str] | None = None):
    return SimpleNamespace(
        title=title,
        executive_summary="",
        key_takeaways=takeaways
        or ["A sufficiently long takeaway sentence about the topic at hand here."],
        failure_modes=[],
    )


# ---------------------------------------------------------------------------
# _pre_map_dedup_batch_items
# ---------------------------------------------------------------------------


def test_pre_map_dedup_batch_items_disabled_returns_unchanged(monkeypatch):
    monkeypatch.setattr("knowledge_engine.config.PRE_MAP_DEDUP_ENABLED", False)
    items = [
        ("t1", "https://a.example.com", "", "<html>a</html>"),
        ("t2", "https://b.example.com", "", "<html>b</html>"),
    ]
    out, alias_of = asyncio.run(m._pre_map_dedup_batch_items(items))
    assert out == items
    assert alias_of == {}


def test_pre_map_dedup_batch_items_single_item_skips(monkeypatch):
    monkeypatch.setattr("knowledge_engine.config.PRE_MAP_DEDUP_ENABLED", True)
    items = [("t1", "https://a.example.com", "", "<html>a</html>")]
    out, alias_of = asyncio.run(m._pre_map_dedup_batch_items(items))
    assert out == items
    assert alias_of == {}


def test_pre_map_dedup_batch_items_filters_aliases(monkeypatch):
    monkeypatch.setattr("knowledge_engine.config.PRE_MAP_DEDUP_ENABLED", True)

    from knowledge_engine.src.deduplication.pre_map_deduplicator import (
        PreMapDedupResult,
    )

    async def fake_dedup(candidates, **kw):
        result = PreMapDedupResult()
        result.alias_map = {"https://a.example.com": ["https://b.example.com"]}
        return result

    monkeypatch.setattr(
        "knowledge_engine.src.deduplication.pre_map_deduplicator.deduplicate_before_map_reduce",
        fake_dedup,
    )

    items = [
        ("t1", "https://a.example.com", "", "<html>a</html>"),
        ("t2", "https://b.example.com", "", "<html>b</html>"),
        ("t3", "https://c.example.com", "", "<html>c</html>"),
    ]
    out, alias_of = asyncio.run(m._pre_map_dedup_batch_items(items))
    assert sorted(item[1] for item in out) == [
        "https://a.example.com",
        "https://c.example.com",
    ]
    assert alias_of == {"https://b.example.com": "https://a.example.com"}


def test_pre_map_dedup_batch_items_fail_open_on_exception(monkeypatch):
    monkeypatch.setattr("knowledge_engine.config.PRE_MAP_DEDUP_ENABLED", True)

    async def boom(candidates, **kw):
        raise RuntimeError("dedup exploded")

    monkeypatch.setattr(
        "knowledge_engine.src.deduplication.pre_map_deduplicator.deduplicate_before_map_reduce",
        boom,
    )

    items = [
        ("t1", "https://a.example.com", "", "<html>a</html>"),
        ("t2", "https://b.example.com", "", "<html>b</html>"),
    ]
    out, alias_of = asyncio.run(m._pre_map_dedup_batch_items(items))
    assert out == items, "a dedup crash must never drop a candidate from ingest"
    assert alias_of == {}


# ---------------------------------------------------------------------------
# _ingest_blog_hits_batch_async — alias-хиты переиспользуют извлечения canonical
# ---------------------------------------------------------------------------


def test_ingest_blog_hits_batch_async_propagates_alias_extracts(monkeypatch):
    hit_a = _hit("https://a.example.com/post")
    hit_b = _hit("https://b.example.com/mirror")

    async def fake_precheck(h):
        return None, h.title, f"<html>{h.url}</html>"

    monkeypatch.setattr(m, "_ingest_blog_url_precheck", fake_precheck)

    from knowledge_engine.src.deduplication.pre_map_deduplicator import (
        PreMapDedupResult,
    )

    async def fake_dedup(candidates, **kw):
        result = PreMapDedupResult()
        result.alias_map = {hit_a.url: [hit_b.url]}
        return result

    monkeypatch.setattr(
        "knowledge_engine.src.deduplication.pre_map_deduplicator.deduplicate_before_map_reduce",
        fake_dedup,
    )
    monkeypatch.setattr("knowledge_engine.config.PRE_MAP_DEDUP_ENABLED", True)

    async def fake_ingest_batch(batch_items):
        urls = [url for _t, url, _sid, _html in batch_items]
        assert urls == [
            hit_a.url
        ], "alias must not reach ingest_blog_spatial_mapping_batch_async"
        return {hit_a.url: (None, _summary("Canonical Doc"), 1)}

    monkeypatch.setattr(
        "knowledge_engine.services.article_ingestion.blog_spatial_pipeline.ingest_blog_spatial_mapping_batch_async",
        fake_ingest_batch,
    )

    out = asyncio.run(m._ingest_blog_hits_batch_async([hit_a, hit_b]))
    by_url = {h.url: h for h in out}

    assert by_url[hit_a.url].alias_of == ""
    assert by_url[hit_a.url].key_extracts

    assert by_url[hit_b.url].alias_of == hit_a.url
    assert by_url[hit_b.url].key_extracts == by_url[hit_a.url].key_extracts
    assert by_url[hit_b.url].title == by_url[hit_a.url].title


def test_ingest_blog_hits_batch_async_no_aliases_behaves_as_before(monkeypatch):
    hit_a = _hit("https://a.example.com/post")

    async def fake_precheck(h):
        return None, h.title, f"<html>{h.url}</html>"

    monkeypatch.setattr(m, "_ingest_blog_url_precheck", fake_precheck)
    monkeypatch.setattr("knowledge_engine.config.PRE_MAP_DEDUP_ENABLED", True)

    async def fake_ingest_batch(batch_items):
        return {hit_a.url: (None, _summary("Solo Doc"), 1)}

    monkeypatch.setattr(
        "knowledge_engine.services.article_ingestion.blog_spatial_pipeline.ingest_blog_spatial_mapping_batch_async",
        fake_ingest_batch,
    )

    out = asyncio.run(m._ingest_blog_hits_batch_async([hit_a]))
    assert len(out) == 1
    assert out[0].alias_of == ""
    assert out[0].key_extracts


# ---------------------------------------------------------------------------
# _ingest_blog_hits_batch_async — backfill_margin: ALIAS дропается, добор из
# резерва вместо дублирующего контента под своим URL (DEEP_INGEST_BACKFILL_MARGIN)
# ---------------------------------------------------------------------------


def test_ingest_blog_hits_batch_async_backfill_margin_drops_aliases_and_trims_to_cap(
    monkeypatch,
):
    """Пул из desired_cap(2) + margin(2) = 4 источников, из них 2 пары
    алиасов (b->a, d->c). С backfill_margin=2 оба алиаса должны быть
    полностью исключены (не просто помечены), а на выходе — ровно 2
    уникальных канонических URL (a, c), не укороченный список 4-2=2 с
    дублирующим контентом."""
    hit_a = _hit("https://a.example.com/post")
    hit_b = _hit("https://b.example.com/mirror")
    hit_c = _hit("https://c.example.com/post")
    hit_d = _hit("https://d.example.com/mirror")

    async def fake_precheck(h):
        return None, h.title, f"<html>{h.url}</html>"

    monkeypatch.setattr(m, "_ingest_blog_url_precheck", fake_precheck)
    monkeypatch.setattr("knowledge_engine.config.PRE_MAP_DEDUP_ENABLED", True)

    from knowledge_engine.src.deduplication.pre_map_deduplicator import (
        PreMapDedupResult,
    )

    async def fake_dedup(candidates, **kw):
        result = PreMapDedupResult()
        result.alias_map = {
            hit_a.url: [hit_b.url],
            hit_c.url: [hit_d.url],
        }
        return result

    monkeypatch.setattr(
        "knowledge_engine.src.deduplication.pre_map_deduplicator.deduplicate_before_map_reduce",
        fake_dedup,
    )

    async def fake_ingest_batch(batch_items):
        urls = {url for _t, url, _sid, _html in batch_items}
        assert urls == {hit_a.url, hit_c.url}, "aliases must never reach MAP+REDUCE"
        return {
            hit_a.url: (None, _summary("Canonical A"), 1),
            hit_c.url: (None, _summary("Canonical C"), 1),
        }

    monkeypatch.setattr(
        "knowledge_engine.services.article_ingestion.blog_spatial_pipeline.ingest_blog_spatial_mapping_batch_async",
        fake_ingest_batch,
    )

    out = asyncio.run(
        m._ingest_blog_hits_batch_async([hit_a, hit_b, hit_c, hit_d], backfill_margin=2)
    )
    urls = {h.url for h in out}

    assert urls == {hit_a.url, hit_c.url}, "aliases dropped, no wasted duplicate slots"
    assert (
        len(out) == 2
    ), "result must equal desired_cap (4 - margin=2), not shrink further"
    for h in out:
        assert h.alias_of == ""
        assert h.key_extracts


def test_finish_filters_empty_extracts_and_ranks_by_relevance_before_trim(
    monkeypatch,
):
    """Regression: _finish() used to slice out[:desired_cap] by POSITION in
    the original search order — a candidate that failed the credibility gate
    (empty key_extracts, e.g. real pep-0703 case: Q=0.639, 0 chars after MAP)
    could survive the cut while an already fully MAP+REDUCE'd good candidate
    got dropped just for being later in the list. Now: drop empty-extract
    hits first, rank survivors by exa_relevance_score, THEN trim."""
    hit_a = _hit("https://a.example.com/failed-credibility-gate")
    hit_b = _hit("https://b.example.com/mid").model_copy(
        update={"exa_relevance_score": 0.5}
    )
    hit_c = _hit("https://c.example.com/best").model_copy(
        update={"exa_relevance_score": 0.9}
    )
    hit_d = _hit("https://d.example.com/worst").model_copy(
        update={"exa_relevance_score": 0.3}
    )

    async def fake_precheck(h):
        return None, h.title, f"<html>{h.url}</html>"

    monkeypatch.setattr(m, "_ingest_blog_url_precheck", fake_precheck)
    monkeypatch.setattr("knowledge_engine.config.PRE_MAP_DEDUP_ENABLED", False)

    async def fake_ingest_batch(batch_items):
        return {
            # Mirrors run_inbound_ingest_gate's credibility-gate failure: no
            # summary -> resolved to empty key_extracts downstream.
            hit_a.url: (None, None, 0),
            hit_b.url: (None, _summary("B"), 1),
            hit_c.url: (None, _summary("C"), 1),
            hit_d.url: (None, _summary("D"), 1),
        }

    monkeypatch.setattr(
        "knowledge_engine.services.article_ingestion.blog_spatial_pipeline.ingest_blog_spatial_mapping_batch_async",
        fake_ingest_batch,
    )

    out = asyncio.run(
        m._ingest_blog_hits_batch_async([hit_a, hit_b, hit_c, hit_d], backfill_margin=2)
    )
    urls = {h.url for h in out}
    assert len(out) == 2
    assert (
        hit_a.url not in urls
    ), "empty-extract (failed credibility gate) hit must never survive the trim"
    assert urls == {hit_c.url, hit_b.url}, (
        "must keep the two highest exa_relevance_score hits, "
        "not the first two by search-result position"
    )


def test_finish_drops_empty_extracts_even_when_already_under_cap(monkeypatch):
    """Old bug also hid here: when len(out) <= desired_cap, _finish() returned
    out completely unfiltered — a failed hit survived simply because the
    count happened not to exceed the cap."""
    hit_a = _hit("https://a.example.com/failed")
    hit_b = _hit("https://b.example.com/ok1")
    hit_c = _hit("https://c.example.com/ok2")

    async def fake_precheck(h):
        return None, h.title, f"<html>{h.url}</html>"

    monkeypatch.setattr(m, "_ingest_blog_url_precheck", fake_precheck)
    monkeypatch.setattr("knowledge_engine.config.PRE_MAP_DEDUP_ENABLED", False)

    async def fake_ingest_batch(batch_items):
        return {
            hit_a.url: (None, None, 0),
            hit_b.url: (None, _summary("B"), 1),
            hit_c.url: (None, _summary("C"), 1),
        }

    monkeypatch.setattr(
        "knowledge_engine.services.article_ingestion.blog_spatial_pipeline.ingest_blog_spatial_mapping_batch_async",
        fake_ingest_batch,
    )

    # pool=3, margin=1 -> desired_cap=2; only 2 valid exist, so this never
    # reaches the out[:desired_cap] branch — it must still drop hit_a.
    out = asyncio.run(
        m._ingest_blog_hits_batch_async([hit_a, hit_b, hit_c], backfill_margin=1)
    )
    urls = {h.url for h in out}
    assert hit_a.url not in urls
    assert urls == {hit_b.url, hit_c.url}


def test_finish_uses_desired_count_when_pool_has_no_margin_headroom(monkeypatch):
    """Real bug from perf_debug.log (gil_internals, 671s run): Exa only ever
    returned 4 raw candidates for a node whose quota target IS 4 — there was
    never any margin headroom to trim away, so blog_hits arrives at exactly
    the real target, not target+margin. The old len(blog_hits)-backfill_margin
    inference silently assumed the margin was always achieved and cut an
    already-correct 4 down to 2. With desired_count passed explicitly (the
    real target, as targeted_node_search.py now does), all 4 successful hits
    must survive even though backfill_margin=2 is also set."""
    hit_a = _hit("https://a.example.com/1")
    hit_b = _hit("https://b.example.com/2")
    hit_c = _hit("https://c.example.com/3")
    hit_d = _hit("https://d.example.com/4")

    async def fake_precheck(h):
        return None, h.title, f"<html>{h.url}</html>"

    monkeypatch.setattr(m, "_ingest_blog_url_precheck", fake_precheck)
    monkeypatch.setattr("knowledge_engine.config.PRE_MAP_DEDUP_ENABLED", False)

    async def fake_ingest_batch(batch_items):
        return {url: (None, _summary(url), 1) for _t, url, _sid, _h in batch_items}

    monkeypatch.setattr(
        "knowledge_engine.services.article_ingestion.blog_spatial_pipeline.ingest_blog_spatial_mapping_batch_async",
        fake_ingest_batch,
    )

    out = asyncio.run(
        m._ingest_blog_hits_batch_async(
            [hit_a, hit_b, hit_c, hit_d],
            backfill_margin=2,
            desired_count=4,
        )
    )
    assert len(out) == 4, (
        "desired_count must win over the stale len(blog_hits)-backfill_margin "
        "inference when the pool never actually had margin headroom"
    )


def test_ingest_blog_hits_batch_async_backfill_margin_zero_keeps_alias_url(
    monkeypatch,
):
    """backfill_margin=0 (default) — поведение не должно измениться: ALIAS
    остаётся под своим URL, заимствуя extracts канонического."""
    hit_a = _hit("https://a.example.com/post")
    hit_b = _hit("https://b.example.com/mirror")

    async def fake_precheck(h):
        return None, h.title, f"<html>{h.url}</html>"

    monkeypatch.setattr(m, "_ingest_blog_url_precheck", fake_precheck)
    monkeypatch.setattr("knowledge_engine.config.PRE_MAP_DEDUP_ENABLED", True)

    from knowledge_engine.src.deduplication.pre_map_deduplicator import (
        PreMapDedupResult,
    )

    async def fake_dedup(candidates, **kw):
        result = PreMapDedupResult()
        result.alias_map = {hit_a.url: [hit_b.url]}
        return result

    monkeypatch.setattr(
        "knowledge_engine.src.deduplication.pre_map_deduplicator.deduplicate_before_map_reduce",
        fake_dedup,
    )

    async def fake_ingest_batch(batch_items):
        return {hit_a.url: (None, _summary("Canonical Doc"), 1)}

    monkeypatch.setattr(
        "knowledge_engine.services.article_ingestion.blog_spatial_pipeline.ingest_blog_spatial_mapping_batch_async",
        fake_ingest_batch,
    )

    out = asyncio.run(m._ingest_blog_hits_batch_async([hit_a, hit_b]))
    urls = {h.url for h in out}
    assert urls == {hit_a.url, hit_b.url}, "backfill_margin=0 must not drop aliases"
