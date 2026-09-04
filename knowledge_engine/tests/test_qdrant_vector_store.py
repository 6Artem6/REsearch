"""qdrant_vector_store: async adapter, полностью на моках qdrant_client (без сети, без пакета)."""

from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

from knowledge_engine.services.qdrant_vector_store import QdrantVectorStore, _point_id


class _FakePointStruct:
    def __init__(self, id: str, vector: list[float], payload: dict[str, Any]) -> None:
        self.id = id
        self.vector = vector
        self.payload = payload


class _FakeMatchValue:
    def __init__(self, value: Any) -> None:
        self.value = value


class _FakeMatchAny:
    def __init__(self, any: list[Any]) -> None:
        self.any = list(any)


class _FakeFieldCondition:
    def __init__(self, key: str, match: _FakeMatchValue) -> None:
        self.key = key
        self.match = match


class _FakeFilter:
    def __init__(
        self,
        must: list[_FakeFieldCondition] | None = None,
        must_not: list[_FakeFieldCondition] | None = None,
    ) -> None:
        self.must = must or []
        self.must_not = must_not or []


class _FakeAsyncQdrantClient:
    """In-memory stand-in for qdrant_client.AsyncQdrantClient."""

    def __init__(self, url: str | None = None, api_key: str | None = None) -> None:
        import asyncio

        self.url = url
        self.api_key = api_key
        self.collections: dict[str, dict[str, _FakePointStruct]] = {}
        self.missing_index_fields: set[str] = set()
        self.indexed_fields: set[str] = set()
        try:
            self._bound_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._bound_loop = None

    def _raise_if_foreign_loop(self) -> None:
        """Mimics real AsyncQdrantClient/httpx: connections don't survive their loop closing."""
        import asyncio

        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            current = None
        if self._bound_loop is not None and current is not self._bound_loop:
            raise RuntimeError("Event loop is closed")

    async def upsert(self, *, collection_name: str, points: list[_FakePointStruct]) -> Any:
        self._raise_if_foreign_loop()
        store = self.collections.setdefault(collection_name, {})
        for p in points:
            store[p.id] = p
        return SimpleNamespace(status="completed")

    async def query_points(
        self,
        *,
        collection_name: str,
        query: list[float],
        query_filter: _FakeFilter | None = None,
        limit: int = 5,
        with_payload: bool = True,
    ) -> Any:
        all_conds = list((query_filter.must if query_filter else []) or []) + list(
            (query_filter.must_not if query_filter else []) or []
        )
        for cond in all_conds:
            field = cond.key
            if field in self.missing_index_fields and field not in self.indexed_fields:
                raise Exception(
                    f'Bad request: Index required but not found for "{field}" '
                    "of one of the following types: [keyword]."
                )
        store = self.collections.get(collection_name, {})
        pts = list(store.values())
        if query_filter is not None:

            def _matches(cond: _FakeFieldCondition, payload: dict[str, Any]) -> bool:
                if isinstance(cond.match, _FakeMatchAny):
                    return payload.get(cond.key) in cond.match.any
                return payload.get(cond.key) == cond.match.value

            pts = [
                p
                for p in pts
                if all(_matches(c, p.payload) for c in query_filter.must)
                and not any(_matches(c, p.payload) for c in query_filter.must_not)
            ]
        pts = pts[:limit]
        scored = [
            SimpleNamespace(id=p.id, payload=p.payload, score=1.0 - i * 0.01)
            for i, p in enumerate(pts)
        ]
        return SimpleNamespace(points=scored)

    def _check_index(self, conds: list[_FakeFieldCondition]) -> None:
        for cond in conds:
            field = cond.key
            if field in self.missing_index_fields and field not in self.indexed_fields:
                # Reproduces qdrant_client's real UnexpectedResponse.__str__, which
                # embeds the raw HTTP body as a bytes repr (double-escaped quotes).
                body = (
                    f'{{"status":{{"error":"Bad request: Index required but not found '
                    f'for \\"{field}\\" of one of the following types: [keyword]."}}}}'
                ).encode()
                raise Exception(
                    f"Unexpected Response: 400 (Bad Request)\n"
                    f"Raw response content:\n{body!r}"
                )

    async def scroll(
        self,
        *,
        collection_name: str,
        scroll_filter: _FakeFilter | None = None,
        limit: int = 10,
        offset: Any = None,
        with_payload: bool | list[str] = True,
        with_vectors: bool = False,
    ) -> tuple[list[_FakePointStruct], Any]:
        conds = list(scroll_filter.must) if scroll_filter else []
        self._check_index(conds)
        store = self.collections.get(collection_name, {})
        matched = [
            p
            for p in store.values()
            if all(p.payload.get(c.key) == c.match.value for c in conds)
        ]
        start = int(offset) if offset is not None else 0
        page = matched[start : start + limit]
        next_offset = start + limit if start + limit < len(matched) else None
        if isinstance(with_payload, list):
            page = [
                _FakePointStruct(
                    id=p.id,
                    vector=p.vector,
                    payload={k: v for k, v in p.payload.items() if k in with_payload},
                )
                for p in page
            ]
        return page, next_offset

    async def count(
        self,
        *,
        collection_name: str,
        count_filter: _FakeFilter | None = None,
        exact: bool = True,
    ) -> Any:
        conds = list(count_filter.must) if count_filter else []
        self._check_index(conds)
        store = self.collections.get(collection_name, {})
        matched = [
            p
            for p in store.values()
            if all(p.payload.get(c.key) == c.match.value for c in conds)
        ]
        return SimpleNamespace(count=len(matched))

    async def create_payload_index(
        self, *, collection_name: str, field_name: str, field_schema: Any
    ) -> Any:
        self.indexed_fields.add(field_name)
        return SimpleNamespace(status="completed")


def _install_fake_qdrant_client(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_models = types.ModuleType("qdrant_client.models")
    fake_models.PointStruct = _FakePointStruct
    fake_models.Filter = _FakeFilter
    fake_models.FieldCondition = _FakeFieldCondition
    fake_models.MatchValue = _FakeMatchValue
    fake_models.MatchAny = _FakeMatchAny
    fake_models.PayloadSchemaType = SimpleNamespace(KEYWORD="keyword")

    fake_qdrant_client = types.ModuleType("qdrant_client")
    fake_qdrant_client.AsyncQdrantClient = _FakeAsyncQdrantClient
    fake_qdrant_client.models = fake_models

    monkeypatch.setitem(sys.modules, "qdrant_client", fake_qdrant_client)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", fake_models)


CHUNK_A = {
    "chunk_id": "c-1",
    "doc_id": "d-1",
    "url": "https://example.org/a",
    "title": "A",
    "chunk_text": "text a",
    "chunk_index": 0,
    "chunks_in_doc": 2,
    "source_type": "vendor",
    "trust_score": 1.0,
}
CHUNK_B = {
    "chunk_id": "c-2",
    "doc_id": "d-1",
    "url": "https://example.org/b",
    "title": "B",
    "chunk_text": "text b",
    "chunk_index": 1,
    "chunks_in_doc": 2,
    "source_type": "vendor",
    "trust_score": 0.8,
}


@pytest.mark.anyio
async def test_disabled_when_no_url_returns_safe_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_qdrant_client(monkeypatch)
    store = QdrantVectorStore(url="", api_key="")
    assert store.enabled is False

    assert await store.upsert_documents("chunks", [CHUNK_A], [[0.1, 0.2]]) is False
    assert await store.search_similar("chunks", [0.1, 0.2]) == []
    assert await store.fetch_by_url("chunks", "https://example.org/a") is None
    assert store._client is None


@pytest.mark.anyio
async def test_upsert_length_mismatch_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_qdrant_client(monkeypatch)
    store = QdrantVectorStore(url="http://fake-qdrant:6333", api_key="k")
    ok = await store.upsert_documents("chunks", [CHUNK_A, CHUNK_B], [[0.1, 0.2]])
    assert ok is False
    assert store._client is None  # no client work attempted


@pytest.mark.anyio
async def test_upsert_then_search_similar_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_qdrant_client(monkeypatch)
    store = QdrantVectorStore(url="http://fake-qdrant:6333", api_key="k")

    ok = await store.upsert_documents(
        "chunks", [CHUNK_A, CHUNK_B], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    )
    assert ok is True

    results = await store.search_similar("chunks", [0.1, 0.2, 0.3], limit=5)
    assert len(results) == 2
    urls = {r["url"] for r in results}
    assert urls == {"https://example.org/a", "https://example.org/b"}
    assert all("_score" in r and "_id" in r for r in results)


@pytest.mark.anyio
async def test_search_similar_doc_id_filter_restricts_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_qdrant_client(monkeypatch)
    store = QdrantVectorStore(url="http://fake-qdrant:6333", api_key="k")
    await store.upsert_documents(
        "chunks", [CHUNK_A, CHUNK_B], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    )

    # CHUNK_A and CHUNK_B share doc_id="d-1" — filter matches both.
    same_doc = await store.search_similar(
        "chunks", [0.1, 0.2, 0.3], limit=5, doc_id_filter=[CHUNK_A["doc_id"]]
    )
    assert {r["chunk_id"] for r in same_doc} == {"c-1", "c-2"}

    none_match = await store.search_similar(
        "chunks", [0.1, 0.2, 0.3], limit=5, doc_id_filter=["no-such-doc"]
    )
    assert none_match == []


@pytest.mark.anyio
async def test_search_similar_self_heals_missing_doc_id_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_qdrant_client(monkeypatch)
    store = QdrantVectorStore(url="http://fake-qdrant:6333", api_key="k")
    client = store._get_client()
    client.missing_index_fields.add("doc_id")
    await store.upsert_documents("chunks", [CHUNK_A], [[0.1, 0.2]])

    results = await store.search_similar(
        "chunks", [0.1, 0.2], doc_id_filter=[CHUNK_A["doc_id"]]
    )
    assert len(results) == 1
    assert "doc_id" in client.indexed_fields


@pytest.mark.anyio
async def test_fetch_by_url_hit_and_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_qdrant_client(monkeypatch)
    store = QdrantVectorStore(url="http://fake-qdrant:6333", api_key="k")
    await store.upsert_documents("chunks", [CHUNK_A, CHUNK_B], [[0.1, 0.2], [0.3, 0.4]])

    hit = await store.fetch_by_url("chunks", "https://example.org/b")
    assert hit is not None
    assert hit["chunk_id"] == "c-2"
    assert hit["title"] == "B"

    miss = await store.fetch_by_url("chunks", "https://example.org/does-not-exist")
    assert miss is None


@pytest.mark.anyio
async def test_fetch_by_url_empty_url_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_qdrant_client(monkeypatch)
    store = QdrantVectorStore(url="http://fake-qdrant:6333", api_key="k")
    assert await store.fetch_by_url("chunks", "") is None
    assert store._client is None


@pytest.mark.anyio
async def test_fetch_by_url_self_heals_missing_payload_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: real Qdrant Cloud 400 'Index required for "url"' must self-heal."""
    _install_fake_qdrant_client(monkeypatch)
    store = QdrantVectorStore(url="http://fake-qdrant:6333", api_key="k")
    client = store._get_client()
    client.missing_index_fields.add("url")

    await store.upsert_documents("chunks", [CHUNK_A], [[0.1, 0.2]])

    fetched = await store.fetch_by_url("chunks", CHUNK_A["url"])
    assert fetched is not None
    assert fetched["chunk_id"] == "c-1"
    assert "url" in client.indexed_fields


def test_point_id_is_deterministic_and_key_sensitive() -> None:
    a = _point_id({"chunk_id": "c-1", "url": "https://example.org/a"})
    b = _point_id({"chunk_id": "c-1", "url": "https://example.org/different"})
    c = _point_id({"chunk_id": "c-2", "url": "https://example.org/a"})
    assert a == b  # chunk_id wins over url when both present
    assert a != c
    fallback = _point_id({"url": "https://example.org/only-url"})
    assert fallback == _point_id({"url": "https://example.org/only-url"})


def test_point_id_prefers_explicit_id_over_chunk_id_and_url() -> None:
    """Regression: knowledge_atoms rows share url across many atoms of one doc —
    without 'id' taking priority, they'd all collide onto the same point id."""
    atom_1 = _point_id({"id": "atom-uuid-1", "url": "https://example.org/doc"})
    atom_2 = _point_id({"id": "atom-uuid-2", "url": "https://example.org/doc"})
    assert atom_1 != atom_2
    assert atom_1 == _point_id({"id": "atom-uuid-1", "url": "https://example.org/doc"})


@pytest.mark.anyio
async def test_search_similar_exclude_ids_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_qdrant_client(monkeypatch)
    store = QdrantVectorStore(url="http://fake-qdrant:6333", api_key="k")
    atom_a = {"id": "a-1", "doc_id": "d-1", "url": "https://x", "statement": "s1"}
    atom_b = {"id": "a-2", "doc_id": "d-1", "url": "https://x", "statement": "s2"}
    await store.upsert_documents("atoms", [atom_a, atom_b], [[0.1, 0.2], [0.3, 0.4]])

    results = await store.search_similar(
        "atoms", [0.1, 0.2], limit=10, exclude_ids_filter=["a-1"]
    )
    assert {r["id"] for r in results} == {"a-2"}


@pytest.mark.anyio
async def test_count_by_field(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_qdrant_client(monkeypatch)
    store = QdrantVectorStore(url="http://fake-qdrant:6333", api_key="k")
    await store.upsert_documents(
        "atoms",
        [
            {"id": "a-1", "doc_id": "d-1", "url": "https://x"},
            {"id": "a-2", "doc_id": "d-1", "url": "https://x"},
            {"id": "a-3", "doc_id": "d-2", "url": "https://y"},
        ],
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
    )
    assert await store.count_by_field("atoms", "doc_id", "d-1") == 2
    assert await store.count_by_field("atoms", "doc_id", "d-2") == 1
    assert await store.count_by_field("atoms", "doc_id", "no-such-doc") == 0


@pytest.mark.anyio
async def test_distinct_field_values_paginates(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_qdrant_client(monkeypatch)
    store = QdrantVectorStore(url="http://fake-qdrant:6333", api_key="k")
    docs = [
        {"id": f"a-{i}", "doc_id": f"d-{i % 3}", "url": "https://x"} for i in range(10)
    ]
    vectors = [[float(i), 0.0] for i in range(10)]
    await store.upsert_documents("atoms", docs, vectors)

    values = await store.distinct_field_values("atoms", "doc_id", page_size=3)
    assert values == {"d-0", "d-1", "d-2"}


def test_upsert_survives_separate_asyncio_run_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: reused QdrantVectorStore across two asyncio.run() calls must not hit 'Event loop is closed'."""
    _install_fake_qdrant_client(monkeypatch)
    store = QdrantVectorStore(url="http://fake-qdrant:6333", api_key="k")

    ok1 = asyncio.run(store.upsert_documents("chunks", [CHUNK_A], [[0.1, 0.2]]))
    ok2 = asyncio.run(store.upsert_documents("chunks", [CHUNK_B], [[0.3, 0.4]]))

    assert ok1 is True
    assert ok2 is True


@pytest.mark.anyio
async def test_run_resilient_reconnects_on_closed_loop_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Safety net: _run_resilient reconnects+retries once on a closed-loop error."""
    _install_fake_qdrant_client(monkeypatch)
    store = QdrantVectorStore(url="http://fake-qdrant:6333", api_key="k")

    calls = {"n": 0}

    async def _flaky(client: Any) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("Event loop is closed")
        return "ok"

    result = await store._run_resilient(
        collection_name="chunks", index_field=None, make_call=_flaky
    )
    assert result == "ok"
    assert calls["n"] == 2  # one failed attempt + one successful retry


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
