"""Штурвал — ленивый реальный ingest примапленных источников при первом

открытии ноды (``steering_lazy_ingest_service.py``). Найдено при отладке:
финальный шаг Штурвала (``WorkJobKind.STEERING_MAP_REDUCE``) фактически НЕ
делает Map-Reduce — строит курс одним Flash-вызовом прямо из лёгких Gate-2
дайджестов (``source_tier="steering_approved"`` в реестре), что
подтверждено ``scripts/log_profiler.py`` на реальном прогоне (ни одного
MAP/REDUCE события в логе). По указанию пользователя реальный Map-Reduce
делается ЛЕНИВО — при открытии конкретной ноды, а не эйджерно на
генерации курса — с дедупом между нодами по ``source_tier`` в реестре.
См. docs/STEERING_AND_TOPIC_QNA_ROADMAP.md."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from knowledge_engine.src.domains.curriculum.schemas import (
    CurriculumGraph,
    CurriculumNode,
    CurriculumSourceRegistryEntry,
)
from knowledge_engine.src.domains.steering.services import (
    steering_lazy_ingest_service as sli,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _registry_entry(source_id: str, url: str, tier: str = "steering_approved"):
    return CurriculumSourceRegistryEntry(
        source_id=source_id,
        title="Исходная статья про шардирование",
        url=url,
        snippet="Короткий дайджест с Gate 2.",
        key_extracts=["Короткий дайджест с Gate 2."],
        source_tier=tier,
    )


def _node(node_id: str, mapped_source_ids: list[str], primary_source_id: str = ""):
    return CurriculumNode(
        node_id=node_id,
        title="Тестовая нода про шардирование",
        layer="foundation",
        category="Категория",
        brief_summary="Достаточно длинное краткое описание ноды для теста.",
        core_concepts=["шардирование"],
        mapped_source_ids=mapped_source_ids,
        primary_source_id=primary_source_id
        or (mapped_source_ids[0] if mapped_source_ids else ""),
        grounding_status="model_only",
    )


def _graph(node: CurriculumNode, registry: list[CurriculumSourceRegistryEntry]):
    filler_one = _node("filler_one", [])
    filler_two = _node("filler_two", [])
    return CurriculumGraph(
        curriculum_id="test_curriculum_sli",
        title="Test Curriculum",
        description="Тестовый курс для юнит-тестов ленивого ingest Штурвала.",
        total_nodes=3,
        curriculum_sources_registry=registry,
        nodes=[node, filler_one, filler_two],
    )


@pytest.mark.anyio
async def test_ensure_ingested_is_noop_when_no_pending_sources() -> None:
    """Источник уже канонического тира (не steering_approved) — Autopilot-

    ноды и уже доингещенные Штурвал-ноды не должны вызывать ingest вообще."""
    registry = [_registry_entry("src_1", "https://habr.com/x", tier="whitelist_blog")]
    node = _node("n1", ["src_1"])
    graph = _graph(node, registry)

    with patch.object(
        sli, "summarize_whitelist_blog_hits_async", AsyncMock()
    ) as mock_ingest:
        out_graph, out_node = await sli.ensure_node_steering_sources_ingested(
            graph, node
        )
    mock_ingest.assert_not_called()
    assert out_node is node
    assert out_graph is graph


@pytest.mark.anyio
async def test_ensure_ingested_updates_registry_and_node_source_ref() -> None:
    registry = [_registry_entry("src_1", "https://habr.com/companies/x/articles/1/")]
    node = _node("n1", ["src_1"])
    graph = _graph(node, registry)

    async def fake_ingest(hits, target_goal):
        return [
            h.model_copy(
                update={
                    "key_extracts": ["Реальный извлечённый факт из полного текста."],
                    "snippet": "Реальный сниппет после Map-Reduce.",
                    "source_tier": "whitelist_blog",
                }
            )
            for h in hits
        ]

    with patch.object(sli, "summarize_whitelist_blog_hits_async", fake_ingest):
        out_graph, out_node = await sli.ensure_node_steering_sources_ingested(
            graph, node, target_goal="шардирование"
        )

    assert out_node is not node
    assert out_node.grounding_status == "grounded"
    assert out_node.source_ref is not None
    assert out_node.source_ref.source_id == "src_1"
    assert out_node.source_ref.relevant_extracts == [
        "Реальный извлечённый факт из полного текста."
    ]
    updated_entry = next(
        e for e in out_graph.curriculum_sources_registry if e.source_id == "src_1"
    )
    assert updated_entry.source_tier == "whitelist_blog"
    assert updated_entry.key_extracts == [
        "Реальный извлечённый факт из полного текста."
    ]


@pytest.mark.anyio
async def test_ensure_ingested_dedupes_across_nodes_sharing_source() -> None:
    """Один и тот же source_id примаплен на две ноды (живая находка:

    вырожденный случай, когда все ноды курса ссылались на один source_id)
    — второе открытие не должно повторно гонять Map-Reduce, раз реестр уже
    обновлён первым открытием."""
    registry = [_registry_entry("src_1", "https://habr.com/x")]
    node_a = _node("node_a", ["src_1"])
    node_b = _node("node_b", ["src_1"])
    graph = CurriculumGraph(
        curriculum_id="test_curriculum_sli_dedupe",
        title="Test Curriculum",
        description="Тестовый курс для проверки дедупа ленивого ingest.",
        total_nodes=3,
        curriculum_sources_registry=registry,
        nodes=[node_a, node_b, _node("filler", [])],
    )

    async def fake_ingest(hits, target_goal):
        return [
            h.model_copy(
                update={
                    "key_extracts": ["Факт после ingest."],
                    "source_tier": "whitelist_blog",
                }
            )
            for h in hits
        ]

    with patch.object(
        sli, "summarize_whitelist_blog_hits_async", AsyncMock(side_effect=fake_ingest)
    ) as mock_ingest:
        graph, updated_a = await sli.ensure_node_steering_sources_ingested(
            graph, node_a
        )
        assert mock_ingest.await_count == 1

        # graph уже обновлён — берём свежий node_b из него, как это делает
        # реальный эндпоинт при повторном открытии другой ноды.
        fresh_node_b = next(n for n in graph.nodes if n.node_id == "node_b")
        graph, updated_b = await sli.ensure_node_steering_sources_ingested(
            graph, fresh_node_b
        )

    # Второй вызов не должен был снова дёрнуть ingest — источник уже не
    # steering_approved.
    assert mock_ingest.await_count == 1
    assert updated_b is fresh_node_b


@pytest.mark.anyio
async def test_ensure_ingested_fail_open_on_ingest_error() -> None:
    registry = [_registry_entry("src_1", "https://habr.com/x")]
    node = _node("n1", ["src_1"])
    graph = _graph(node, registry)

    with patch.object(
        sli,
        "summarize_whitelist_blog_hits_async",
        AsyncMock(side_effect=RuntimeError("network down")),
    ):
        out_graph, out_node = await sli.ensure_node_steering_sources_ingested(
            graph, node
        )
    assert out_node is node
    assert out_graph is graph


@pytest.mark.anyio
async def test_ensure_ingested_fail_open_when_ingest_returns_nothing() -> None:
    registry = [_registry_entry("src_1", "https://habr.com/x")]
    node = _node("n1", ["src_1"])
    graph = _graph(node, registry)

    with patch.object(
        sli, "summarize_whitelist_blog_hits_async", AsyncMock(return_value=[])
    ):
        out_graph, out_node = await sli.ensure_node_steering_sources_ingested(
            graph, node
        )
    assert out_node is node
    assert out_graph is graph


@pytest.mark.anyio
async def test_endpoint_ensure_steering_sources_noop_response() -> None:
    from knowledge_engine.src.domains.grounding.schemas import NodeDataInput
    from knowledge_engine.src.entrypoints.api.routes.node_skill import (
        NodeSessionBody,
        post_node_ensure_steering_sources,
    )

    registry = [_registry_entry("src_1", "https://habr.com/x", tier="whitelist_blog")]
    node = _node("n1", ["src_1"])
    graph = _graph(node, registry)

    body = NodeSessionBody(
        curriculum_id="test_curriculum_sli",
        node_data=NodeDataInput(
            node_id="n1", title="Test node", layer="foundation", core_concepts=["c"]
        ),
    )
    with (
        patch(
            "knowledge_engine.src.shared.skill_tree_store.get_curriculum_graph",
            return_value=graph.model_dump(),
        ),
        patch(
            "knowledge_engine.src.shared.skill_tree_store.get_curriculum_meta",
            return_value={},
        ),
        patch(
            "knowledge_engine.src.shared.skill_tree_store.save_curriculum_record"
        ) as mock_save,
    ):
        result = await post_node_ensure_steering_sources(body)
    mock_save.assert_not_called()
    assert result == {"ingested": False}


@pytest.mark.anyio
async def test_endpoint_ensure_steering_sources_ingests_and_saves() -> None:
    from knowledge_engine.src.domains.grounding.schemas import NodeDataInput
    from knowledge_engine.src.entrypoints.api.routes.node_skill import (
        NodeSessionBody,
        post_node_ensure_steering_sources,
    )

    registry = [_registry_entry("src_1", "https://habr.com/companies/x/articles/1/")]
    node = _node("n1", ["src_1"])
    graph = _graph(node, registry)

    async def fake_ingest(hits, target_goal):
        return [
            h.model_copy(
                update={
                    "key_extracts": ["Реальный факт."],
                    "source_tier": "whitelist_blog",
                }
            )
            for h in hits
        ]

    body = NodeSessionBody(
        curriculum_id="test_curriculum_sli",
        node_data=NodeDataInput(
            node_id="n1", title="Test node", layer="foundation", core_concepts=["c"]
        ),
    )
    with (
        patch(
            "knowledge_engine.src.shared.skill_tree_store.get_curriculum_graph",
            return_value=graph.model_dump(),
        ),
        patch(
            "knowledge_engine.src.shared.skill_tree_store.get_curriculum_meta",
            return_value={"target_goal": "шардирование"},
        ),
        patch(
            "knowledge_engine.src.shared.skill_tree_store.save_curriculum_record"
        ) as mock_save,
        patch.object(sli, "summarize_whitelist_blog_hits_async", fake_ingest),
    ):
        result = await post_node_ensure_steering_sources(body)
    mock_save.assert_called_once()
    assert result["ingested"] is True
    assert result["grounding_status"] == "grounded"


@pytest.mark.anyio
async def test_endpoint_ensure_steering_sources_404_on_unknown_curriculum() -> None:
    from fastapi import HTTPException

    from knowledge_engine.src.domains.grounding.schemas import NodeDataInput
    from knowledge_engine.src.entrypoints.api.routes.node_skill import (
        NodeSessionBody,
        post_node_ensure_steering_sources,
    )

    body = NodeSessionBody(
        curriculum_id="does_not_exist",
        node_data=NodeDataInput(
            node_id="n1", title="Test node", layer="foundation", core_concepts=["c"]
        ),
    )
    with patch(
        "knowledge_engine.src.shared.skill_tree_store.get_curriculum_graph",
        return_value=None,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await post_node_ensure_steering_sources(body)
    assert exc_info.value.status_code == 404


def test_infer_source_tier_reused_from_node_grounding_finalize_service() -> None:
    # Явно проверяем, что это ИМПОРТ, а не копия — если оригинал изменится,
    # этот тест не должен молча разойтись с реальным поведением.
    from knowledge_engine.src.shared.node_grounding.node_grounding_finalize_service import (
        _infer_source_tier,
    )

    assert sli._infer_source_tier is _infer_source_tier
