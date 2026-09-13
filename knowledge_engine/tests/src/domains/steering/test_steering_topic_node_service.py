"""Mode 2 (standalone_digest) финал, итерация 2 — реальный тяжёлый

Map-Reduce по ВСЕМ Gate 2 approved URL → ОДНА сфокусированная нода
(``node_kind=graph_kind="steering_standalone"``, см.
docs/STEERING_AND_TOPIC_QNA_ROADMAP.md, "Разделение на Mode 1/Mode 2" +
prompt.log "КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ"). Заменяет более раннюю версию с
искусственным делением на ≥3 ноды-бакета — удалена по прямому запросу.

``summarize_whitelist_blog_hits_async``/``enrich_search_hits_with_extracts_
async`` мокаются (реальный ingest тестируется в test_node_grounding_gate.py
для того же движка); ``source_tier="whitelist_blog"`` в моках — чтобы не
задевать ``persist_approved_curriculum_hits_to_lancedb_async`` (реальная
LanceDB-запись, вне зоны этого файла)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from knowledge_engine.src.domains.curriculum.schemas import (
    CurriculumGraph,
    CurriculumSearchHit,
)
from knowledge_engine.src.domains.steering.services import (
    steering_topic_node_service as svc,
)
from knowledge_engine.src.domains.steering.steering_contracts import (
    SurfaceDigestItem,
    SurfaceDigestResponse,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _digest(url: str, title: str, tech: list[str] | None = None) -> SurfaceDigestItem:
    return SurfaceDigestItem(
        url=url,
        title=title,
        company_or_author="Acme",
        problem_solved=f"problem for {title}",
        main_tech_stack=tech or ["PostgreSQL"],
        two_sentence_summary=f"summary for {title}",
    )


def _hit(url: str, title: str) -> CurriculumSearchHit:
    return CurriculumSearchHit(
        url=url,
        title=title,
        snippet="snippet",
        key_extracts=[f"extract from {title} one", f"extract from {title} two"],
        source_tier="whitelist_blog",
    )


def _passthrough_ingest(hits, target_goal):
    return hits


def _fake_content(n_subtopics: int = 5) -> svc._StandaloneNodeContent:
    return svc._StandaloneNodeContent(
        title="Обзор темы",
        category="Тема",
        brief_summary="Достаточно длинное описание раздела для валидации.",
        core_concepts=["концепт один", "концепт два"],
        subtopics=[
            svc.NodeSubtopic(
                title=f"Подтема {i}",
                summary=f"Описание подтемы {i}, достаточно длинное для схемы.",
            )
            for i in range(n_subtopics)
        ],
    )


@pytest.mark.anyio
async def test_builds_single_node_graph_with_all_sources_mapped() -> None:
    """10 approved-статей — граф должен получить РОВНО 1 ноду, все 10

    источников замаплены на неё (никакого капа CURRICULUM_DEEP_NODE_MAX_HITS
    для node_kind='steering_standalone', никакого деления на бакеты)."""
    urls = [f"https://habr.com/{i}" for i in range(10)]
    digests = SurfaceDigestResponse(digests=[_digest(u, u) for u in urls])
    hits = [_hit(u, u) for u in urls]

    with (
        patch.object(
            svc, "summarize_whitelist_blog_hits_async", AsyncMock(return_value=hits)
        ),
        patch.object(
            svc,
            "enrich_search_hits_with_extracts_async",
            AsyncMock(side_effect=_passthrough_ingest),
        ),
        patch.object(svc, "_lite_structured", AsyncMock(return_value=_fake_content())),
    ):
        graph = await svc.generate_standalone_topic_graph("Тема курса", digests, urls)

    assert isinstance(graph, CurriculumGraph)
    assert graph.graph_kind == "steering_standalone"
    assert len(graph.nodes) == 1
    assert graph.total_nodes == 1
    node = graph.nodes[0]
    assert node.node_kind == "steering_standalone"
    assert node.grounding_status == "grounded"
    assert node.node_risk_kind == "DEEP"
    assert len(node.mapped_source_ids) == 10
    assert len(set(node.mapped_source_ids)) == 10


@pytest.mark.anyio
async def test_node_carries_explicit_subtopic_breakdown() -> None:
    urls = ["https://habr.com/a", "https://habr.com/b", "https://habr.com/c"]
    digests = SurfaceDigestResponse(digests=[_digest(u, u) for u in urls])
    hits = [_hit(u, u) for u in urls]

    with (
        patch.object(
            svc, "summarize_whitelist_blog_hits_async", AsyncMock(return_value=hits)
        ),
        patch.object(
            svc,
            "enrich_search_hits_with_extracts_async",
            AsyncMock(side_effect=_passthrough_ingest),
        ),
        patch.object(
            svc,
            "_lite_structured",
            AsyncMock(return_value=_fake_content(n_subtopics=7)),
        ),
    ):
        graph = await svc.generate_standalone_topic_graph("Тема курса", digests, urls)

    node = graph.nodes[0]
    breakdown = node.node_curriculum_breakdown
    assert breakdown is not None
    assert len(breakdown.subtopics) == 7
    assert all(len(s.summary) >= 10 for s in breakdown.subtopics)


@pytest.mark.anyio
async def test_source_ref_aggregates_extracts_from_all_hits() -> None:
    urls = ["https://habr.com/a", "https://habr.com/b"]
    digests = SurfaceDigestResponse(digests=[_digest(u, u) for u in urls])
    hits = [_hit(u, u) for u in urls]

    with (
        patch.object(
            svc, "summarize_whitelist_blog_hits_async", AsyncMock(return_value=hits)
        ),
        patch.object(
            svc,
            "enrich_search_hits_with_extracts_async",
            AsyncMock(side_effect=_passthrough_ingest),
        ),
        patch.object(svc, "_lite_structured", AsyncMock(return_value=_fake_content())),
    ):
        graph = await svc.generate_standalone_topic_graph("Тема курса", digests, urls)

    node = graph.nodes[0]
    assert node.source_ref is not None
    assert len(node.source_ref.relevant_extracts) <= 12
    # Экстракты пришли из ОБЕИХ статей (a и b), а не только из первой.
    extracts = node.source_ref.relevant_extracts
    assert any("https://habr.com/a" in e for e in extracts)
    assert any("https://habr.com/b" in e for e in extracts)


@pytest.mark.anyio
async def test_raises_when_no_approved_url_has_digest() -> None:
    digests = SurfaceDigestResponse(digests=[_digest("https://habr.com/a", "A")])
    with pytest.raises(ValueError):
        await svc.generate_standalone_topic_graph(
            "Тема курса", digests, ["https://habr.com/missing"]
        )


@pytest.mark.anyio
async def test_raises_when_ingest_produces_no_hits() -> None:
    urls = ["https://habr.com/a"]
    digests = SurfaceDigestResponse(digests=[_digest(u, u) for u in urls])
    with (
        patch.object(
            svc, "summarize_whitelist_blog_hits_async", AsyncMock(return_value=[])
        ),
        patch.object(
            svc, "enrich_search_hits_with_extracts_async", AsyncMock(return_value=[])
        ),
    ):
        with pytest.raises(ValueError):
            await svc.generate_standalone_topic_graph("Тема курса", digests, urls)


@pytest.mark.anyio
async def test_content_fallback_on_llm_error_still_produces_valid_node() -> None:
    urls = ["https://habr.com/a", "https://habr.com/b"]
    digests = SurfaceDigestResponse(digests=[_digest(u, u, tech=["Go"]) for u in urls])
    hits = [_hit(u, u) for u in urls]

    with (
        patch.object(
            svc, "summarize_whitelist_blog_hits_async", AsyncMock(return_value=hits)
        ),
        patch.object(
            svc,
            "enrich_search_hits_with_extracts_async",
            AsyncMock(side_effect=_passthrough_ingest),
        ),
        patch.object(
            svc, "_lite_structured", AsyncMock(side_effect=RuntimeError("boom"))
        ),
    ):
        graph = await svc.generate_standalone_topic_graph("Тема курса", digests, urls)

    node = graph.nodes[0]
    assert len(node.brief_summary) >= 10
    assert len(node.core_concepts) >= 1
    assert node.node_curriculum_breakdown is not None
    assert len(node.node_curriculum_breakdown.subtopics) >= 1


@pytest.mark.anyio
async def test_content_fallback_on_none_result() -> None:
    urls = ["https://habr.com/a"]
    digests = SurfaceDigestResponse(digests=[_digest(u, u) for u in urls])
    hits = [_hit(u, u) for u in urls]

    with (
        patch.object(
            svc, "summarize_whitelist_blog_hits_async", AsyncMock(return_value=hits)
        ),
        patch.object(
            svc,
            "enrich_search_hits_with_extracts_async",
            AsyncMock(side_effect=_passthrough_ingest),
        ),
        patch.object(svc, "_lite_structured", AsyncMock(return_value=None)),
    ):
        graph = await svc.generate_standalone_topic_graph("Тема курса", digests, urls)

    assert len(graph.nodes) == 1
