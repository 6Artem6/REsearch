"""cleanup_cloud_resources: CLI parsing + Redis logic on a fake in-memory client."""

from __future__ import annotations

import pytest

from knowledge_engine.scripts import cleanup_cloud_resources as ccr


class FakeRedis:
    def __init__(self, keys: set[str]) -> None:
        self._keys = set(keys)
        self.deleted: list[str] = []

    def exists(self, key: str) -> bool:
        return key in self._keys

    def scan_iter(self, match: str):
        import fnmatch

        for k in list(self._keys):
            if fnmatch.fnmatchcase(k, match):
                yield k

    def delete(self, key: str) -> int:
        if key in self._keys:
            self._keys.discard(key)
            self.deleted.append(key)
            return 1
        return 0


@pytest.fixture
def fake_redis(monkeypatch: pytest.MonkeyPatch):
    store = FakeRedis(
        {
            "ke:lock:node_ground:curr-1:node-a",
            "ke:lock:node_ground:curr-1:node-b",
            "ke:lock:node_ground:curr-2:node-a",
            "cache:cold:unrelated",
        }
    )
    monkeypatch.setattr(ccr, "redis_enabled", lambda: True)
    monkeypatch.setattr(ccr, "get_redis", lambda: store)
    return store


def test_parse_args_requires_at_least_one_id():
    with pytest.raises(SystemExit):
        ccr.parse_args([])


def test_parse_args_accepts_node_id_only():
    args = ccr.parse_args(["--node-id", "node-a"])
    assert args.node_id == "node-a"
    assert args.curriculum_id == ""
    assert args.dry_run is False


def test_parse_args_dry_run_flag():
    args = ccr.parse_args(["--curriculum-id", "curr-1", "--dry-run"])
    assert args.dry_run is True


def test_find_grounding_lock_keys_both_ids_exact_match(fake_redis: FakeRedis) -> None:
    keys = ccr.find_grounding_lock_keys("node-a", "curr-1")
    assert keys == ["ke:lock:node_ground:curr-1:node-a"]


def test_find_grounding_lock_keys_both_ids_no_match(fake_redis: FakeRedis) -> None:
    keys = ccr.find_grounding_lock_keys("node-zzz", "curr-1")
    assert keys == []


def test_find_grounding_lock_keys_curriculum_only(fake_redis: FakeRedis) -> None:
    keys = set(ccr.find_grounding_lock_keys("", "curr-1"))
    assert keys == {
        "ke:lock:node_ground:curr-1:node-a",
        "ke:lock:node_ground:curr-1:node-b",
    }


def test_find_grounding_lock_keys_node_only(fake_redis: FakeRedis) -> None:
    keys = set(ccr.find_grounding_lock_keys("node-a", ""))
    assert keys == {
        "ke:lock:node_ground:curr-1:node-a",
        "ke:lock:node_ground:curr-2:node-a",
    }


def test_cleanup_redis_dry_run_does_not_delete(fake_redis: FakeRedis) -> None:
    report = ccr.cleanup_redis("node-a", "curr-1", dry_run=True)
    assert report["keys_found"] == ["ke:lock:node_ground:curr-1:node-a"]
    assert report["keys_deleted"] == 0
    assert fake_redis.deleted == []
    assert fake_redis.exists("ke:lock:node_ground:curr-1:node-a")


def test_cleanup_redis_real_run_deletes(fake_redis: FakeRedis) -> None:
    report = ccr.cleanup_redis("node-a", "curr-1", dry_run=False)
    assert report["keys_deleted"] == 1
    assert fake_redis.deleted == ["ke:lock:node_ground:curr-1:node-a"]
    assert not fake_redis.exists("ke:lock:node_ground:curr-1:node-a")
    # unrelated key must survive
    assert fake_redis.exists("cache:cold:unrelated")


def test_cleanup_redis_disabled_reports_not_applicable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ccr, "redis_enabled", lambda: False)
    report = ccr.cleanup_redis("node-a", "curr-1", dry_run=True)
    assert report["applicable"] is False
    assert report["keys_found"] == []


class FakeQdrantStore:
    def __init__(self) -> None:
        self.enabled = True
        self.deleted: list[tuple[str, str, str]] = []

    async def delete_by_field(self, collection: str, field: str, value: str) -> bool:
        self.deleted.append((collection, field, value))
        return True


def test_cleanup_qdrant_node_id_without_curriculum_id_not_applicable() -> None:
    report = ccr.cleanup_qdrant("node-a", "", dry_run=True)
    assert report["applicable"] is False


def test_cleanup_qdrant_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ccr, "_get_qdrant_store", lambda: None)
    report = ccr.cleanup_qdrant("", "curr-1", dry_run=True)
    assert report["applicable"] is False


def test_cleanup_qdrant_no_urls_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ccr, "_get_qdrant_store", lambda: FakeQdrantStore())
    monkeypatch.setattr(ccr, "_urls_for_scope", lambda nid, cid: [])
    report = ccr.cleanup_qdrant("node-a", "curr-1", dry_run=True)
    assert report["applicable"] is True
    assert report["urls_found"] == []
    assert report["urls_deleted"] == 0


def test_cleanup_qdrant_dry_run_does_not_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_store = FakeQdrantStore()
    monkeypatch.setattr(ccr, "_get_qdrant_store", lambda: fake_store)
    monkeypatch.setattr(ccr, "_urls_for_scope", lambda nid, cid: ["https://example.org/a"])
    report = ccr.cleanup_qdrant("node-a", "curr-1", dry_run=True)
    assert report["urls_found"] == ["https://example.org/a"]
    assert report["urls_deleted"] == 0
    assert fake_store.deleted == []


def test_cleanup_qdrant_real_run_deletes_by_url_and_doc_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_store = FakeQdrantStore()
    monkeypatch.setattr(ccr, "_get_qdrant_store", lambda: fake_store)
    monkeypatch.setattr(ccr, "_urls_for_scope", lambda nid, cid: ["https://example.org/a"])

    from knowledge_engine.services.vector_store import VectorStore

    report = ccr.cleanup_qdrant("node-a", "curr-1", dry_run=False)
    assert report["urls_deleted"] == 1
    doc_id = VectorStore.doc_id_for_url("https://example.org/a")
    assert ("document_summaries", "url", "https://example.org/a") in fake_store.deleted
    assert ("rag_chunks", "doc_id", doc_id) in fake_store.deleted
    assert ("knowledge_atoms", "doc_id", doc_id) in fake_store.deleted


def test_urls_for_scope_node_resolves_only_its_mapped_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import knowledge_engine.services.skill_tree_store as sts

    fake_graph = {
        "nodes": [{"node_id": "node-a", "mapped_source_ids": ["src_1"]}],
        "curriculum_sources_registry": [
            {"source_id": "src_1", "url": "https://example.org/a"},
            {"source_id": "src_2", "url": "https://example.org/b"},
        ],
    }
    monkeypatch.setattr(sts, "get_curriculum_graph", lambda cid: fake_graph)
    assert ccr._urls_for_scope("node-a", "curr-1") == ["https://example.org/a"]


def test_urls_for_scope_curriculum_only_returns_full_registry_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import knowledge_engine.services.skill_tree_store as sts

    fake_graph = {
        "nodes": [],
        "curriculum_sources_registry": [
            {"source_id": "src_1", "url": "https://example.org/a"},
            {"source_id": "src_2", "url": "https://example.org/b"},
        ],
    }
    monkeypatch.setattr(sts, "get_curriculum_graph", lambda cid: fake_graph)
    assert ccr._urls_for_scope("", "curr-1") == [
        "https://example.org/a",
        "https://example.org/b",
    ]


def test_urls_for_scope_node_falls_back_to_resource_urls_when_registry_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a node's mapped_source_ids can reference ids that no longer
    exist in curriculum_sources_registry (real case found live against
    agentic_systems_architecture/subagent_architectures) — its actual URLs
    then live only on node.resource_urls/source_ref. Registry-only resolution
    silently returned [] here before the fix, making Qdrant cleanup a no-op
    even though clear_node_sources correctly found the same node's URLs."""
    import knowledge_engine.services.skill_tree_store as sts

    fake_graph = {
        "nodes": [
            {
                "node_id": "node-a",
                "mapped_source_ids": ["src_stale_1", "src_stale_2"],
                "resource_urls": ["https://example.org/direct-a"],
                "source_ref": {"url": "https://example.org/direct-b"},
            }
        ],
        # registry does NOT contain src_stale_1/src_stale_2 — stale mapping.
        "curriculum_sources_registry": [
            {"source_id": "src_unrelated", "url": "https://example.org/unrelated"},
        ],
    }
    monkeypatch.setattr(sts, "get_curriculum_graph", lambda cid: fake_graph)
    assert ccr._urls_for_scope("node-a", "curr-1") == [
        "https://example.org/direct-a",
        "https://example.org/direct-b",
    ]


def test_urls_for_scope_unknown_curriculum_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import knowledge_engine.services.skill_tree_store as sts

    monkeypatch.setattr(sts, "get_curriculum_graph", lambda cid: None)
    assert ccr._urls_for_scope("node-a", "curr-missing") == []


def test_cleanup_gemini_cache_reports_not_applicable() -> None:
    report = ccr.cleanup_gemini_cache("node-a", "curr-1", dry_run=True)
    assert report == {"backend": "gemini_cache", "applicable": False}


def test_escape_glob_neutralizes_metacharacters() -> None:
    assert ccr._escape_glob("weird*id?") == "weird\\*id\\?"
