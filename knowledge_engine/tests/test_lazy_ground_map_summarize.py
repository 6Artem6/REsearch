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


def test_lazy_ground_calls_summarizer_for_exa_long_highlights(monkeypatch):
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
        "knowledge_engine.src.curriculum.targeted_node_grounding.summarize_whitelist_blog_hits_async",
        _summarize,
    )
    monkeypatch.setattr(
        "knowledge_engine.src.curriculum.targeted_node_grounding.enrich_search_hits_with_extracts",
        lambda hits, _goal="": hits,
    )
    monkeypatch.setattr(
        "knowledge_engine.src.curriculum.targeted_node_grounding.persist_approved_curriculum_hits_to_lancedb",
        lambda *_a, **_k: 0,
    )

    asyncio.run(
        lazy_ground_deep_node_on_demand(
            _graph(node),
            node,
            target_goal="GIL internals",
            source_policy="practical_only",
            anchor="test",
        )
    )
    assert summarized_urls == [hit.url]


def test_post_replenish_summarizes_practical_exa_hits(monkeypatch):
    hit = _exa_pep_hit()
    node = _node()
    summarized_urls: list[str] = []

    async def _stream(*_a, **_k):
        return [hit]

    async def _replenish(merged, _cap, node=None):  # noqa: ARG001
        return merged

    async def _academic(hits, *, label="", defer_missing=False):  # noqa: ARG001
        return hits

    async def _summarize(hits, _goal=""):
        summarized_urls.extend(h.url for h in hits)
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


def test_blog_ingest_skips_passport_reuse_without_map_windows(monkeypatch):
    from knowledge_engine.src.curriculum import source_material_pipeline as smp

    hit = _exa_pep_hit()
    monkeypatch.setattr(
        smp, "_extracts_from_lancedb_url", lambda _u: (["passport takeaway"], "PEP 703")
    )
    monkeypatch.setattr(smp, "_lancedb_has_map_windows", lambda _u: False)
    fetched: list[str] = []

    def _fetch(url: str):
        fetched.append(url)
        return ("<p>" + ("body " * 80) + "</p>", "httpx")

    monkeypatch.setattr(smp, "smart_fetch_page_html", _fetch)
    monkeypatch.setattr(smp, "is_anti_bot_fetch_result", lambda *_a, **_k: False)
    monkeypatch.setattr(
        smp,
        "_ingest_url_with_spatial_map_reduce",
        lambda _h, _html, tier_label="": (["map extract"], "PEP 703"),
    )
    monkeypatch.setattr(smp, "_try_blog_spatial_diagrams", lambda _h: None)

    extracts, title = smp._ingest_blog_url(hit)
    assert fetched == [hit.url]
    assert extracts == ["map extract"]
    assert title == "PEP 703"
