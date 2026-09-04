"""lazy_ground / post-replenish must MAP practical Exa hits even with long highlights."""

from __future__ import annotations

import asyncio

from knowledge_engine.src.curriculum.schemas import (
    CurriculumGraph,
    CurriculumNode,
    CurriculumSearchHit,
)
from knowledge_engine.src.curriculum.targeted_node_grounding import (
    lazy_ground_deep_node_on_demand,
)
from knowledge_engine.src.curriculum.targeted_node_search import (
    search_sources_for_deep_node_async,
)


def _exa_pep_hit() -> CurriculumSearchHit:
    return CurriculumSearchHit(
        url="https://peps.python.org/pep-0703/",
        title="PEP 703",
        snippet="The GIL is a major obstacle to concurrency. " * 4,
        key_extracts=["highlight " * 120],
        source_tier="exa",
        skip_ollama_summary=True,
    )


def _node(node_id: str = "gil_internals") -> CurriculumNode:
    return CurriculumNode(
        node_id=node_id,
        title="Global interpreter lock internals",
        layer="advanced",
        category="python internals",
        brief_summary="Per-interpreter GIL, ceval loop, and free-threading.",
        core_concepts=["GIL", "ceval"],
        node_risk_kind="DEEP",
        grounding_status="pending_grounding",
    )


def _graph(deep: CurriculumNode) -> CurriculumGraph:
    filler = dict(
        layer="foundation",
        category="python internals",
        brief_summary="Supporting node used only to satisfy graph min size.",
        core_concepts=["python"],
        node_risk_kind="BASE",
        grounding_status="model_only",
    )
    return CurriculumGraph(
        curriculum_id="python_internals_and_memory",
        title="Python internals and memory",
        description="Curriculum graph for lazy-ground MAP ingest tests.",
        total_nodes=3,
        nodes=[
            deep,
            CurriculumNode(node_id="memory_model", title="Memory model", **filler),
            CurriculumNode(node_id="gc_cycle", title="Garbage collector", **filler),
        ],
    )


def test_lazy_ground_does_not_re_summarize_hits_from_search(monkeypatch):
    """Regression (perf_debug.log audit, DOUBLE-PASS INGESTION fix):
    search_sources_for_deep_node_async already runs every returned hit
    through summarize_whitelist_blog_hits_async internally (its own
    "post-replenish" step) before returning — a full fetch+annotate+
    triage+MAP+REDUCE pass per hit. lazy_ground_deep_node_on_demand must
    NOT call it again on the same hits: doing so silently doubled the
    Gemma ingest cost for every deep-dive grounding call (confirmed via
    perf_debug.log: 8 MAP+REDUCE passes for 4 documents, ~805s wall time
    instead of 4 passes)."""
    hit = _exa_pep_hit()
    node = _node()
    summarized_urls: list[str] = []

    async def _search(*_a, **_k):
        return [hit]

    async def _summarize(hits, _goal=""):
        summarized_urls.extend(h.url for h in hits)
        return hits

    monkeypatch.setattr(
        "knowledge_engine.src.curriculum.targeted_node_grounding.search_sources_for_deep_node_async",
        _search,
    )
    monkeypatch.setattr(
        "knowledge_engine.src.curriculum.source_material_pipeline.summarize_whitelist_blog_hits_async",
        _summarize,
    )

    async def _enrich(hits, _goal=""):
        return hits

    async def _persist(*_a, **_k):
        return 0

    monkeypatch.setattr(
        "knowledge_engine.src.curriculum.targeted_node_grounding.enrich_search_hits_with_extracts_async",
        _enrich,
    )
    monkeypatch.setattr(
        "knowledge_engine.src.curriculum.targeted_node_grounding.persist_approved_curriculum_hits_to_lancedb_async",
        _persist,
    )

    graph, updated_node = asyncio.run(
        lazy_ground_deep_node_on_demand(
            _graph(node),
            node,
            target_goal="GIL internals",
            source_policy="practical_only",
            anchor="test",
        )
    )
    # Never re-invoked from grounding — search already did it.
    assert summarized_urls == []
    # The hit search returned (already "summarized" upstream, in the real
    # pipeline) is still used to ground the node.
    assert updated_node.grounding_status == "grounded"


def test_post_replenish_summarizes_practical_exa_hits(monkeypatch):
    hit = _exa_pep_hit()
    node = _node()
    summarized_urls: list[str] = []
    summarize_kwargs: dict = {}

    async def _stream(*_a, **_k):
        return [hit]

    async def _replenish(merged, _cap, node=None, **_kw):  # noqa: ARG001
        return merged

    async def _academic(hits, *, label="", defer_missing=False):  # noqa: ARG001
        return hits

    async def _summarize(hits, _goal="", **_kw):
        summarized_urls.extend(h.url for h in hits)
        summarize_kwargs.update(_kw)
        return hits

    monkeypatch.setattr(
        "knowledge_engine.src.curriculum.targeted_node_search._run_stream_search_pipeline",
        _stream,
    )
    monkeypatch.setattr(
        "knowledge_engine.src.curriculum.targeted_node_search.replenish_valid_hits_until_cap",
        _replenish,
    )
    monkeypatch.setattr(
        "knowledge_engine.src.curriculum.source_material_pipeline.ingest_mandatory_academic_hits_async",
        _academic,
    )
    monkeypatch.setattr(
        "knowledge_engine.src.curriculum.source_material_pipeline.summarize_whitelist_blog_hits_async",
        _summarize,
    )

    out = asyncio.run(
        search_sources_for_deep_node_async(
            node,
            "GIL internals",
            source_policy="practical_only",
            anchor="test",
            exclude_url_keys=set(),
            on_demand=True,
        )
    )
    assert [h.url for h in out] == [hit.url]
    assert summarized_urls == [hit.url]
    # Regression: the real quota cap must reach _ingest_blog_hits_batch_async
    # as an explicit target, not be re-derived from len(blog_hits) - margin.
    assert summarize_kwargs.get("desired_count") is not None


def test_blog_ingest_skips_passport_reuse_without_map_windows(monkeypatch):
    from knowledge_engine.src.curriculum import source_material_pipeline as smp

    hit = _exa_pep_hit()

    async def _cached(_u):
        return ["passport takeaway"], "PEP 703"

    monkeypatch.setattr(smp, "_extracts_from_lancedb_url", _cached)
    monkeypatch.setattr(smp, "_lancedb_has_map_windows", lambda _u: False)
    fetched: list[str] = []

    def _fetch(url: str):
        fetched.append(url)
        return ("<p>" + ("body " * 80) + "</p>", "httpx")

    monkeypatch.setattr(smp, "smart_fetch_page_html", _fetch)
    monkeypatch.setattr(smp, "is_anti_bot_fetch_result", lambda *_a, **_k: False)

    async def _spatial_map_reduce(_h, _html, tier_label=""):
        return ["map extract"], "PEP 703"

    monkeypatch.setattr(
        smp,
        "_ingest_url_with_spatial_map_reduce",
        _spatial_map_reduce,
    )
    monkeypatch.setattr(smp, "_try_blog_spatial_diagrams", lambda _h: None)

    extracts, title = asyncio.run(smp._ingest_blog_url(hit))
    assert fetched == [hit.url]
    assert extracts == ["map extract"]
    assert title == "PEP 703"
