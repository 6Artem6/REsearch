"""Regression: resolve_article_ids_for_node must not collide across curricula.

`primary_source_id` values ("src_1", "src_2", ...) are short, per-curriculum
slot labels reused by almost every curriculum. Passing one alone (without its
real URL) into `canonical_article_id` collapses to a bare "src:{sid}" key
that is identical across unrelated curricula sharing the same slot number —
diagrams/figure_registry rows saved under that key leak between nodes that
have nothing to do with each other (confirmed live on
python_internals_and_memory/gil_internals, which was picking up diagrams
from an unrelated vector-DB curriculum's own "src_1" node).
"""

from __future__ import annotations

from knowledge_engine.services.article_diagram_context import (
    canonical_article_id,
    resolve_article_ids_for_node,
)
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput


def _node(
    *, primary_source_id: str = "", mapped_source_ids: list[str] | None = None
) -> NodeDataInput:
    return NodeDataInput(
        node_id="node_x",
        title="Node Title",
        layer="advanced",
        core_concepts=["concept a"],
        primary_source_id=primary_source_id,
        mapped_source_ids=mapped_source_ids or [],
    )


def _registry_graph(entries: list[dict]) -> dict:
    return {"curriculum_sources_registry": entries}


def test_primary_source_id_resolves_real_url_not_bare_slot_key(monkeypatch):
    monkeypatch.setattr(
        "knowledge_engine.services.skill_tree_store.get_curriculum_graph",
        lambda cid: _registry_graph(
            [{"source_id": "src_1", "url": "https://peps.python.org/pep-0703/"}]
        ),
    )
    node = _node(primary_source_id="src_1")
    ids = resolve_article_ids_for_node(node, "python_internals_and_memory")
    assert ids, "must resolve at least one article_id"
    expected = canonical_article_id("src_1", "https://peps.python.org/pep-0703/")
    assert expected in ids
    # The old, collision-prone bare form must NOT appear once a real URL is known.
    assert "src:src_1" not in ids


def test_same_slot_label_different_curricula_does_not_collide(monkeypatch):
    graphs = {
        "python_internals_and_memory": _registry_graph(
            [{"source_id": "src_1", "url": "https://peps.python.org/pep-0703/"}]
        ),
        "vector_db_mechanics": _registry_graph(
            [{"source_id": "src_1", "url": "https://habr.com/ru/articles/961088/"}]
        ),
    }
    monkeypatch.setattr(
        "knowledge_engine.services.skill_tree_store.get_curriculum_graph",
        lambda cid: graphs[cid],
    )
    node = _node(primary_source_id="src_1")
    ids_gil = resolve_article_ids_for_node(node, "python_internals_and_memory")
    ids_vdb = resolve_article_ids_for_node(node, "vector_db_mechanics")
    assert set(ids_gil).isdisjoint(set(ids_vdb))


def test_primary_source_id_without_registry_entry_falls_back_to_bare_slot(monkeypatch):
    """No URL known at all (registry miss) — degrade to the old bare key
    rather than dropping the source entirely."""
    monkeypatch.setattr(
        "knowledge_engine.services.skill_tree_store.get_curriculum_graph",
        lambda cid: _registry_graph([]),
    )
    node = _node(primary_source_id="src_9")
    ids = resolve_article_ids_for_node(node, "some_curriculum")
    assert ids == ["src:src_9"]


def test_mapped_source_ids_still_resolved_alongside_primary(monkeypatch):
    monkeypatch.setattr(
        "knowledge_engine.services.skill_tree_store.get_curriculum_graph",
        lambda cid: _registry_graph(
            [
                {"source_id": "src_1", "url": "https://peps.python.org/pep-0703/"},
                {
                    "source_id": "src_2",
                    "url": "https://github.com/python/cpython/blob/main/Python/ceval_gil.c",
                },
            ]
        ),
    )
    node = _node(primary_source_id="src_1", mapped_source_ids=["src_1", "src_2"])
    ids = resolve_article_ids_for_node(node, "python_internals_and_memory")
    assert canonical_article_id("src_1", "https://peps.python.org/pep-0703/") in ids
    assert (
        canonical_article_id(
            "src_2", "https://github.com/python/cpython/blob/main/Python/ceval_gil.c"
        )
        in ids
    )
