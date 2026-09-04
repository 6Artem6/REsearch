"""persist_verified_external_sources_to_node: сохранение найденных на лекции
внешних источников как материалов ноды — ДВА слоя привязки:
document_summaries+resource_urls (Retrieval) И
curriculum_sources_registry+mapped_source_ids (citation/[Sn]/references
panel). См. разбор реального бага: curriculum=indexes_and_data_structures
node=b_tree_indexes — resource_urls сохранились, но mapped_source_ids
остался пустым, из-за чего coerce_references_to_registry отбрасывал ВСЕ
references (registry пуст), а лекция цитировала голыми [n] вместо [Sn]."""

from __future__ import annotations

import asyncio

import knowledge_engine.src.node_deep_dive.lecture_search_orchestrator as orch_mod
from knowledge_engine.src.node_deep_dive.lecture_search_orchestrator import (
    VerifiedExternalSource,
    _attach_sources_to_node_graph,
    _registry_entry_dict,
    persist_verified_external_sources_to_node,
)
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput


def _node(**kwargs) -> NodeDataInput:
    base = {
        "node_id": "hash_indexes",
        "title": "Хэш-индексы",
        "layer": "foundation",
        "category": "indexes",
        "brief_summary": "Summary",
        "core_concepts": ["hashing"],
    }
    base.update(kwargs)
    return NodeDataInput(**base)


def _source(url: str, snippet: str = "", title: str = "Title") -> VerifiedExternalSource:
    return VerifiedExternalSource(
        url=url,
        title=title,
        snippet=snippet or ("Real content snippet long enough to persist. " * 2),
    )


class _FakeVectorStore:
    def __init__(self, saved: list):
        self._saved = saved

    async def save_summary(self, summary, *, skip_rag_ingest=False):
        self._saved.append(summary.url)
        return True


def test_persist_saves_document_summaries_and_attaches_sources(monkeypatch):
    saved: list[str] = []
    monkeypatch.setattr(
        "knowledge_engine.services.vector_store.VectorStore",
        lambda: _FakeVectorStore(saved),
    )
    attached: dict = {}
    monkeypatch.setattr(
        orch_mod,
        "_attach_sources_to_node_graph",
        lambda cid, nid, sources: attached.update(
            cid=cid, node_id=nid, urls=[s.url for s in sources]
        ),
    )

    sources = [
        _source("https://github.com/postgres/postgres"),
        _source("https://en.wikipedia.org/wiki/Hash_table"),
    ]
    node = _node()

    async def _run() -> int:
        return await persist_verified_external_sources_to_node(
            "indexes_and_data_structures", node, sources
        )

    n = asyncio.run(_run())

    assert n == 2
    assert saved == [
        "https://github.com/postgres/postgres",
        "https://en.wikipedia.org/wiki/Hash_table",
    ]
    assert attached["cid"] == "indexes_and_data_structures"
    assert attached["node_id"] == "hash_indexes"
    assert set(attached["urls"]) == set(saved)


def test_persist_skips_thin_snippet_sources(monkeypatch):
    saved: list[str] = []
    monkeypatch.setattr(
        "knowledge_engine.services.vector_store.VectorStore",
        lambda: _FakeVectorStore(saved),
    )
    monkeypatch.setattr(orch_mod, "_attach_sources_to_node_graph", lambda *a: None)

    sources = [
        _source("https://example.com/thin", snippet="too short"),
        _source("https://example.com/real"),
    ]
    node = _node()

    async def _run() -> int:
        return await persist_verified_external_sources_to_node(
            "indexes_and_data_structures", node, sources
        )

    n = asyncio.run(_run())

    assert n == 1
    assert saved == ["https://example.com/real"]


def test_persist_returns_zero_without_curriculum_id_or_sources():
    node = _node()

    async def _run() -> tuple[int, int]:
        empty_cid = await persist_verified_external_sources_to_node("", node, [])
        no_valid = await persist_verified_external_sources_to_node(
            "cid", node, [_source("not-a-url", snippet="whatever, still short url")]
        )
        return empty_cid, no_valid

    empty_cid, no_valid = asyncio.run(_run())
    assert empty_cid == 0
    assert no_valid == 0


def test_registry_entry_dict_shape():
    entry = _registry_entry_dict(
        _source("https://github.com/postgres/postgres", title="Postgres")
    )
    assert entry["url"] == "https://github.com/postgres/postgres"
    assert entry["title"] == "Postgres"
    assert entry["source_type"] == "verified_external"
    assert entry["source_tier"] == "exa"
    assert "Real content snippet" in entry["snippet"]


def test_attach_sources_to_node_graph_registers_and_maps(monkeypatch):
    """Оба слоя привязки должны примениться: registry (mapped_source_ids)
    И resource_urls (Retrieval) — воспроизводит фикс реального бага."""
    registry_calls: list = []
    monkeypatch.setattr(
        "knowledge_engine.services.skill_tree_store.patch_curriculum_sources_registry",
        lambda cid, entries: registry_calls.append((cid, entries)) or ["src_1", "src_2"],
    )
    monkeypatch.setattr(
        "knowledge_engine.services.skill_tree_store.get_curriculum_graph",
        lambda _cid: {
            "nodes": [
                {
                    "node_id": "hash_indexes",
                    "mapped_source_ids": [],
                    "resource_urls": [],
                }
            ]
        },
    )
    node_patches: list = []
    monkeypatch.setattr(
        "knowledge_engine.services.skill_tree_store.patch_curriculum_graph_node",
        lambda cid, nid, updates: node_patches.append((cid, nid, updates)),
    )

    sources = [
        _source("https://github.com/postgres/postgres", title="Postgres repo"),
        _source("https://en.wikipedia.org/wiki/Hash_table", title="Hash table"),
    ]
    _attach_sources_to_node_graph("indexes_and_data_structures", "hash_indexes", sources)

    assert len(registry_calls) == 1
    assert len(registry_calls[0][1]) == 2
    assert len(node_patches) == 1
    cid, nid, updates = node_patches[0]
    assert cid == "indexes_and_data_structures"
    assert nid == "hash_indexes"
    assert updates["mapped_source_ids"] == ["src_1", "src_2"]
    assert updates["resource_urls"] == [
        "https://github.com/postgres/postgres",
        "https://en.wikipedia.org/wiki/Hash_table",
    ]


def test_attach_sources_to_node_graph_merges_with_existing_mapped_ids(monkeypatch):
    monkeypatch.setattr(
        "knowledge_engine.services.skill_tree_store.patch_curriculum_sources_registry",
        lambda cid, entries: ["src_5"],
    )
    monkeypatch.setattr(
        "knowledge_engine.services.skill_tree_store.get_curriculum_graph",
        lambda _cid: {
            "nodes": [
                {
                    "node_id": "hash_indexes",
                    "mapped_source_ids": ["src_1"],
                    "resource_urls": ["https://example.com/existing"],
                }
            ]
        },
    )
    node_patches: list = []
    monkeypatch.setattr(
        "knowledge_engine.services.skill_tree_store.patch_curriculum_graph_node",
        lambda cid, nid, updates: node_patches.append(updates),
    )

    _attach_sources_to_node_graph(
        "cid", "hash_indexes", [_source("https://example.com/new")]
    )

    assert node_patches[0]["mapped_source_ids"] == ["src_1", "src_5"]
    assert node_patches[0]["resource_urls"] == [
        "https://example.com/existing",
        "https://example.com/new",
    ]


def test_attach_sources_to_node_graph_noop_when_registry_patch_fails(monkeypatch):
    monkeypatch.setattr(
        "knowledge_engine.services.skill_tree_store.patch_curriculum_sources_registry",
        lambda cid, entries: [],
    )
    node_patches: list = []
    monkeypatch.setattr(
        "knowledge_engine.services.skill_tree_store.patch_curriculum_graph_node",
        lambda cid, nid, updates: node_patches.append(updates),
    )

    _attach_sources_to_node_graph(
        "cid", "hash_indexes", [_source("https://example.com/new")]
    )

    assert node_patches == []
