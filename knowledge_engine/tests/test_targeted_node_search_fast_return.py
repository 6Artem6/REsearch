"""Registry cap must keep mapped_source_ids; post-practical academic wait."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from knowledge_engine.src.curriculum.schemas import (
    CurriculumGraph,
    CurriculumNode,
    CurriculumSearchHit,
    CurriculumSourceRegistryEntry,
)
from knowledge_engine.src.curriculum.source_registry import (
    cap_curriculum_sources_registry,
)
from knowledge_engine.src.curriculum.targeted_node_search import (
    _gather_practical_and_academic,
    _must_await_full_academic_gather,
    _on_demand_post_practical_academic_wait_sec,
)


def _deep_sota_node() -> CurriculumNode:
    return CurriculumNode(
        node_id="governed_agent_pipelines",
        title="Governed agent pipelines",
        layer="sota",
        category="agentic systems",
        brief_summary="Deterministic hooks and MCP rules for multi-agent pipelines.",
        core_concepts=["MCP", "agent hooks"],
        node_risk_kind="DEEP",
        mapped_source_ids=["src_21", "src_22"],
    )


def _hit(i: int, *, tier: str = "blog") -> CurriculumSearchHit:
    return CurriculumSearchHit(
        source_id=f"s{i:02d}",
        url=f"https://example.com/hit-{i}",
        title=f"Hit {i}",
        snippet="snippet",
        source_tier=tier,
    )


def _entry(sid: str) -> CurriculumSourceRegistryEntry:
    return CurriculumSourceRegistryEntry(
        source_id=sid,
        title=f"Title {sid}",
        whitelist_domain="example.com",
        source_type="Article",
        url=f"https://example.com/{sid}",
        why_read="why",
    )


def test_cap_registry_preserves_mapped_ids_over_cap():
    registry = [_entry(f"src_{i}") for i in range(1, 23)]
    base = _deep_sota_node()
    graph = CurriculumGraph(
        curriculum_id="cur1",
        title="Course title",
        description="Course description for validation.",
        nodes=[
            base,
            base.model_copy(update={"node_id": "other_a", "mapped_source_ids": []}),
            base.model_copy(update={"node_id": "other_b", "mapped_source_ids": []}),
        ],
        total_nodes=3,
    )
    capped = cap_curriculum_sources_registry(registry, graph=graph, cap=20)
    ids = {e.source_id for e in capped}
    assert "src_21" in ids
    assert "src_22" in ids
    assert len(capped) == 20


def test_must_await_full_academic_for_sota_hybrid():
    node = _deep_sota_node()
    assert _must_await_full_academic_gather(node, "hybrid", on_demand=True)


def test_post_practical_wait_at_least_40_for_sota_consensus():
    node = _deep_sota_node()
    wait = _on_demand_post_practical_academic_wait_sec(node, "hybrid")
    assert wait >= 40.0


def test_gather_keeps_academic_when_done_within_post_practical_wait():
    node = _deep_sota_node()
    practical_hits = [_hit(i, tier="blog") for i in range(5)]
    academic_hits = [_hit(i, tier="academic") for i in range(10, 12)]

    async def slow_academic(*_a, **_k):
        await asyncio.sleep(0.15)
        return academic_hits

    async def fast_practical(*_a, **_k):
        await asyncio.sleep(0.02)
        return practical_hits

    async def passthrough_summarize(hits, *_a, **_k):
        return hits

    with (
        patch(
            "knowledge_engine.src.curriculum.targeted_node_search._practical_hits_for_node",
            new=AsyncMock(side_effect=fast_practical),
        ),
        patch(
            "knowledge_engine.src.curriculum.targeted_node_search._academic_hits_for_node",
            new=AsyncMock(side_effect=slow_academic),
        ),
        patch(
            "knowledge_engine.src.curriculum.targeted_node_search._on_demand_post_practical_academic_wait_sec",
            return_value=1.0,
        ),
        patch(
            "knowledge_engine.src.curriculum.source_material_pipeline.summarize_whitelist_blog_hits_async",
            new=AsyncMock(side_effect=passthrough_summarize),
        ),
    ):
        p_out, a_out = asyncio.run(
            _gather_practical_and_academic(
                node,
                "goal",
                anchor="a",
                source_policy="hybrid",
                on_demand=True,
                registry_entries=None,
                exclude_url_keys=set(),
            )
        )
    assert len(p_out) == 5
    assert len(a_out) == 2
