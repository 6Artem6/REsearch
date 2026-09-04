"""enrich_node_learning_materials_from_graph: должна перечитывать
mapped_source_ids/resource_urls из графа заново, когда у переданной ноды они
пусты — на этом держится same-turn фикс в engine.py (после
persist_verified_external_sources_to_node node_for_lecture повторно
обогащается, иначе свежий registry виден лекции только следующим ходом /
после reload — см. разбор реального бага: [n] вместо [Sn] в тексте до
reload)."""

from __future__ import annotations

from knowledge_engine.services.curriculum_whitelist_prompt import (
    enrich_node_learning_materials_from_graph,
)
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput


def _node(**kwargs) -> NodeDataInput:
    base = {
        "node_id": "b_tree_indexes",
        "title": "B-Tree индексы",
        "layer": "foundation",
        "category": "indexes",
        "brief_summary": "Summary",
        "core_concepts": ["btree"],
    }
    base.update(kwargs)
    return NodeDataInput(**base)


def test_picks_up_freshly_registered_mapped_source_ids(monkeypatch):
    """Воспроизводит ровно сценарий same-turn фикса: узел передан с пустым
    mapped_source_ids/resource_urls (как захвачен ДО persist в engine.py),
    граф уже обновлён (persist только что отработал) — обогащение должно
    подтянуть новые значения, а не молча оставить узел как есть."""
    monkeypatch.setattr(
        "knowledge_engine.services.skill_tree_store.get_curriculum_graph",
        lambda _cid: {
            "nodes": [
                {
                    "node_id": "b_tree_indexes",
                    "mapped_source_ids": ["src_1", "src_2", "src_3"],
                    "resource_urls": [
                        "https://www.postgresql.org/docs/18/btree.html",
                    ],
                }
            ]
        },
    )

    stale = _node()  # как захвачен ДО persist — mapped_source_ids=[]
    fresh = enrich_node_learning_materials_from_graph(stale, "indexes_and_data_structures")

    assert fresh.mapped_source_ids == ["src_1", "src_2", "src_3"]
    assert fresh.resource_urls == ["https://www.postgresql.org/docs/18/btree.html"]


def test_does_not_refetch_when_already_has_mapped_and_whitelist(monkeypatch):
    """Early-return guard: если у узла УЖЕ есть mapped_source_ids И
    primary_whitelist_source, повторный поход в граф не нужен."""
    from knowledge_engine.src.curriculum.schemas import (
        LearningMaterials,
        PrimaryWhitelistSource,
    )

    called = {"n": 0}
    monkeypatch.setattr(
        "knowledge_engine.services.skill_tree_store.get_curriculum_graph",
        lambda _cid: called.update(n=called["n"] + 1) or {"nodes": []},
    )

    node = _node(
        mapped_source_ids=["src_1"],
        learning_materials=LearningMaterials(
            primary_whitelist_source=PrimaryWhitelistSource(
                source_name="XX", chapter_or_article="YY", core_concepts=["zz"]
            )
        ),
    )
    result = enrich_node_learning_materials_from_graph(node, "cid")

    assert result is node
    assert called["n"] == 0


def test_returns_node_unchanged_when_curriculum_id_blank():
    node = _node()
    result = enrich_node_learning_materials_from_graph(node, "")
    assert result is node
