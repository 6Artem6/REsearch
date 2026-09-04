"""BATCH POOLING AGGREGATION task: N document hits from one node_deep_dive
grounding round must submit ONE map_reduce_jobs_pooled_async(articles=N)
call, not N separate single-article calls (perf_debug.log audit: 4
concurrent articles=1 calls per pass competed independently for the same
global GEMMA_BUDGET_MAX_TPM instead of being coordinated by one pool)."""

from __future__ import annotations

import asyncio
import types

from knowledge_engine.services.article_ingestion import blog_spatial_pipeline as bsp
from knowledge_engine.src.curriculum import source_material_pipeline as smp
from knowledge_engine.src.curriculum.schemas import CurriculumSearchHit


def _blog_hit(url: str, title: str) -> CurriculumSearchHit:
    return CurriculumSearchHit(
        title=title,
        url=url,
        snippet="x",
        source_tier="exa",
        key_extracts=[],
    )


async def _no_cache(url: str):
    return None


def test_ingest_blog_hits_batch_calls_pooled_ingest_exactly_once(monkeypatch):
    hits = [_blog_hit(f"https://example.com/doc{i}", f"Doc {i}") for i in range(4)]

    monkeypatch.setattr(smp, "_extracts_from_lancedb_url", _no_cache)
    monkeypatch.setattr(smp, "_lancedb_has_map_windows", lambda url: False)
    monkeypatch.setattr(
        smp,
        "smart_fetch_page_html",
        lambda url: ("<p>" + ("word " * 150) + "</p>", "httpx"),
    )
    monkeypatch.setattr(smp, "is_anti_bot_fetch_result", lambda *a, **k: False)
    monkeypatch.setattr(smp, "_try_blog_spatial_diagrams", lambda h: None)

    calls: list[list[tuple]] = []

    async def _fake_batch_ingest(items, *, save_lancedb=True):
        calls.append(list(items))
        return {url: (None, None, 0) for _t, url, _s, _h in items}

    monkeypatch.setattr(
        bsp, "ingest_blog_spatial_mapping_batch_async", _fake_batch_ingest
    )

    out = asyncio.run(smp._ingest_blog_hits_batch_async(hits))

    assert len(calls) == 1, "must be exactly ONE pooled call, not one per hit"
    assert len(calls[0]) == 4
    assert {u for _t, u, _s, _h in calls[0]} == {h.url for h in hits}
    assert len(out) == 4


def test_ingest_blog_spatial_mapping_batch_calls_map_reduce_exactly_once(
    monkeypatch,
):
    """The lower layer: N prepared documents must reach map_reduce_jobs_pooled_async
    as ONE articles=N call."""
    from knowledge_engine.services.article_ingestion.blog_spatial_pipeline import (
        _BlogSpatialContext,
    )

    items = [
        (f"Doc {i}", f"https://example.com/doc{i}", "", "<p>body</p>")
        for i in range(4)
    ]

    def _fake_prepare(title, url, source_id, *, raw_html=None, raw_bytes=None):
        ctx = _BlogSpatialContext(
            title=title,
            page_url=url,
            annotated=types.SimpleNamespace(annotated_markdown="x"),
            fig_ids=[],
            registry=None,
            vlm_saved=0,
            source_kind="article",
        )
        return object(), ctx

    def _fake_build_job(markdown, *, title, url, all_figure_ids, figure_registry, source_kind):
        from knowledge_engine.services.article_ingestion.blog_spatial_summarizer import (
            MapReduceArticleJob,
        )
        from knowledge_engine.services.article_ingestion.paragraph_token_splitter import (
            TokenWindowChunk,
        )

        job = MapReduceArticleJob(
            job_id=url, title=title, url=url, windows=[TokenWindowChunk(window_index=0, body="x")]
        )
        return job, job.windows

    pooled_calls: list[list] = []

    async def _fake_pooled(jobs):
        pooled_calls.append(list(jobs))
        return {}

    async def _fake_finalize(ctx, outcome, windows, *, save_lancedb):
        return (None, None, 0)

    monkeypatch.setattr(bsp, "_prepare_blog_spatial_context", _fake_prepare)
    monkeypatch.setattr(bsp, "build_article_map_reduce_job", _fake_build_job)
    monkeypatch.setattr(bsp, "map_reduce_jobs_pooled_async", _fake_pooled)
    monkeypatch.setattr(bsp, "_finalize_blog_spatial_result", _fake_finalize)

    results = asyncio.run(bsp.ingest_blog_spatial_mapping_batch_async(items))

    assert len(pooled_calls) == 1, "must be exactly ONE map_reduce_jobs_pooled_async call"
    assert len(pooled_calls[0]) == 4
    assert len(results) == 4


def test_finalize_persist_runs_concurrently_not_sequentially(monkeypatch):
    """STREAMING AUDIT (barrier #2): persisting each article (embed + Qdrant/
    LanceDB write) used to run in a plain `for job in jobs: await
    _finalize_blog_spatial_result(...)` loop — article N+1's persist
    couldn't even START until article N's fully finished, pure added tail
    latency for zero benefit (each call builds its own fresh VectorStore()).

    Deterministic proof (no wall-clock race): each fake finalize call blocks
    on an Event that only fires once ALL 4 have reached that point. Under
    the old sequential loop, call #2 never starts before call #1 returns,
    so the counter never reaches 4 and this test times out / errors. Under
    the fixed concurrent (gather-based) version, all 4 start immediately,
    the counter hits 4, and every call proceeds."""
    from knowledge_engine.services.article_ingestion.blog_spatial_pipeline import (
        _BlogSpatialContext,
    )

    items = [
        (f"Doc {i}", f"https://example.com/doc{i}", "", "<p>body</p>")
        for i in range(4)
    ]

    def _fake_prepare(title, url, source_id, *, raw_html=None, raw_bytes=None):
        ctx = _BlogSpatialContext(
            title=title,
            page_url=url,
            annotated=types.SimpleNamespace(annotated_markdown="x"),
            fig_ids=[],
            registry=None,
            vlm_saved=0,
            source_kind="article",
        )
        return object(), ctx

    def _fake_build_job(markdown, *, title, url, all_figure_ids, figure_registry, source_kind):
        from knowledge_engine.services.article_ingestion.blog_spatial_summarizer import (
            MapReduceArticleJob,
        )
        from knowledge_engine.services.article_ingestion.paragraph_token_splitter import (
            TokenWindowChunk,
        )

        job = MapReduceArticleJob(
            job_id=url, title=title, url=url, windows=[TokenWindowChunk(window_index=0, body="x")]
        )
        return job, job.windows

    async def _fake_pooled(jobs):
        return {}

    started = 0
    all_started = asyncio.Event()

    async def _fake_finalize(ctx, outcome, windows, *, save_lancedb):
        nonlocal started
        started += 1
        if started == len(items):
            all_started.set()
        await asyncio.wait_for(all_started.wait(), timeout=1.0)
        return (None, None, 0)

    monkeypatch.setattr(bsp, "_prepare_blog_spatial_context", _fake_prepare)
    monkeypatch.setattr(bsp, "build_article_map_reduce_job", _fake_build_job)
    monkeypatch.setattr(bsp, "map_reduce_jobs_pooled_async", _fake_pooled)
    monkeypatch.setattr(bsp, "_finalize_blog_spatial_result", _fake_finalize)

    results = asyncio.run(
        asyncio.wait_for(bsp.ingest_blog_spatial_mapping_batch_async(items), timeout=2.0)
    )

    assert started == len(items)
    assert len(results) == 4
