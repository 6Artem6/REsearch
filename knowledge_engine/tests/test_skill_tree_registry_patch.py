"""patch_curriculum_sources_registry: минимальный raw-dict патч
curriculum_sources_registry (не трогает target_goal/meta, в отличие от
save_curriculum_record) — источник фикса: verified_external_sources должны
регистрироваться с реальным source_id, чтобы попадать в [Sn]/references."""

from __future__ import annotations

import knowledge_engine.services.skill_tree_store as store_mod
from knowledge_engine.services.skill_tree_store import (
    patch_curriculum_sources_registry,
)


def _doc_with_registry(entries: list[dict]) -> dict:
    return {
        "curricula": [
            {
                "curriculum_id": "cid",
                "graph": {
                    "curriculum_sources_registry": entries,
                    "nodes": [],
                },
            }
        ]
    }


def test_appends_new_entries_with_sequential_ids(monkeypatch):
    doc = _doc_with_registry([])
    saved: dict = {}
    monkeypatch.setattr(store_mod, "_load_doc", lambda: doc)
    monkeypatch.setattr(store_mod, "_save_doc", lambda d: saved.update(doc=d))

    ids = patch_curriculum_sources_registry(
        "cid",
        [
            {"url": "https://github.com/postgres/postgres", "title": "Postgres"},
            {"url": "https://en.wikipedia.org/wiki/Hash_table", "title": "Wiki"},
        ],
    )

    assert ids == ["src_1", "src_2"]
    registry = saved["doc"]["curricula"][0]["graph"]["curriculum_sources_registry"]
    assert [e["source_id"] for e in registry] == ["src_1", "src_2"]
    assert [e["url"] for e in registry] == [
        "https://github.com/postgres/postgres",
        "https://en.wikipedia.org/wiki/Hash_table",
    ]


def test_reuses_existing_source_id_for_same_url(monkeypatch):
    doc = _doc_with_registry(
        [{"source_id": "src_1", "url": "https://github.com/postgres/postgres"}]
    )
    saved: dict = {}
    monkeypatch.setattr(store_mod, "_load_doc", lambda: doc)
    monkeypatch.setattr(store_mod, "_save_doc", lambda d: saved.update(doc=d))

    ids = patch_curriculum_sources_registry(
        "cid",
        [
            # trailing slash + разный регистр — должен матчиться на тот же src_1
            {"url": "https://GitHub.com/postgres/postgres/", "title": "dup"},
            {"url": "https://example.com/new", "title": "new"},
        ],
    )

    assert ids == ["src_1", "src_2"]
    registry = saved["doc"]["curricula"][0]["graph"]["curriculum_sources_registry"]
    assert len(registry) == 2  # дубль не добавлен повторно


def test_never_reassigns_id_already_used_by_another_entry(monkeypatch):
    doc = _doc_with_registry(
        [
            {"source_id": "src_1", "url": "https://a.example.com"},
            {"source_id": "src_3", "url": "https://c.example.com"},
        ]
    )
    saved: dict = {}
    monkeypatch.setattr(store_mod, "_load_doc", lambda: doc)
    monkeypatch.setattr(store_mod, "_save_doc", lambda d: saved.update(doc=d))

    ids = patch_curriculum_sources_registry(
        "cid", [{"url": "https://new.example.com", "title": "new"}]
    )

    assert ids == ["src_3"] or ids[0] not in {"src_1", "src_3"}
    registry = saved["doc"]["curricula"][0]["graph"]["curriculum_sources_registry"]
    all_ids = [e["source_id"] for e in registry]
    assert len(all_ids) == len(set(all_ids))  # нет дублирующихся source_id


def test_returns_empty_when_curriculum_not_found(monkeypatch):
    monkeypatch.setattr(store_mod, "_load_doc", lambda: {"curricula": []})
    monkeypatch.setattr(store_mod, "_save_doc", lambda d: None)
    assert patch_curriculum_sources_registry("missing_cid", [{"url": "https://x"}]) == []


def test_returns_empty_for_blank_curriculum_id_or_no_entries():
    assert patch_curriculum_sources_registry("", [{"url": "https://x"}]) == []
    assert patch_curriculum_sources_registry("cid", []) == []


def test_cap_protects_mapped_entries(monkeypatch):
    existing = [
        {"source_id": f"src_{i}", "url": f"https://example.com/{i}"} for i in range(1, 33)
    ]
    doc = {
        "curricula": [
            {
                "curriculum_id": "cid",
                "graph": {
                    "curriculum_sources_registry": existing,
                    "nodes": [
                        {"node_id": "n1", "mapped_source_ids": ["src_1", "src_2"]}
                    ],
                },
            }
        ]
    }
    saved: dict = {}
    monkeypatch.setattr(store_mod, "_load_doc", lambda: doc)
    monkeypatch.setattr(store_mod, "_save_doc", lambda d: saved.update(doc=d))

    patch_curriculum_sources_registry(
        "cid", [{"url": "https://example.com/new-entry", "title": "new"}]
    )

    registry = saved["doc"]["curricula"][0]["graph"]["curriculum_sources_registry"]
    ids = {e["source_id"] for e in registry}
    assert len(registry) == 32
    assert "src_1" in ids and "src_2" in ids  # mapped-защищённые не выброшены
