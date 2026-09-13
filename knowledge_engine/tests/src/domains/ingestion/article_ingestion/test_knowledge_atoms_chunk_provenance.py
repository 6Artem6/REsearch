"""Knowledge atoms ↔ MAP chunk provenance + LanceDB helpers."""

from __future__ import annotations

import asyncio

from knowledge_engine.src.adapters.db.rag_chunks_schema import map_window_chunk_id
from knowledge_engine.src.domains.ingestion.article_ingestion.blog_spatial_schemas import (
    DeduplicatedAtomsResponse,
    MapWindowResponse,
    normalize_map_knowledge,
)
from knowledge_engine.src.domains.ingestion.article_ingestion.blog_spatial_summarizer import (
    _REDUCE_DEDUP_SYSTEM,
    _format_atoms_json_block,
)
from knowledge_engine.src.shared.extraction import (
    KnowledgeAtom,
    ScopeType,
    attach_source_chunk_id,
    merge_source_chunk_ids,
    normalize_knowledge_atoms,
    reattach_source_chunk_ids_from_raw,
)
from knowledge_engine.src.shared.vector_store import VectorStore


def test_map_window_chunk_id_matches_lancedb_convention() -> None:
    assert map_window_chunk_id("abc123", 0) == "abc123_map_1"
    assert map_window_chunk_id("abc123", 3) == "abc123_map_4"


def test_knowledge_atom_source_chunk_ids_field() -> None:
    atom = KnowledgeAtom(
        scope=ScopeType.PRINCIPLE,
        statement="Isolation reduces blast radius across agents",
        source_chunk_ids=["doc_map_1", "doc_map_1", "doc_map_4"],
    )
    assert atom.source_chunk_ids == ["doc_map_1", "doc_map_4"]


def test_normalize_map_knowledge_attaches_chunk_id() -> None:
    mapped = MapWindowResponse(
        window_role="Intro",
        window_summary="Scaffold.",
        knowledge_atoms=[
            KnowledgeAtom(
                scope=ScopeType.MECHANIC,
                statement="Hooks run before tool dispatch always",
            )
        ],
    )
    out = normalize_map_knowledge(mapped, source_chunk_id="x_map_2")
    assert out.knowledge_atoms[0].source_chunk_ids == ["x_map_2"]


def test_normalize_knowledge_atoms_merges_source_chunk_ids() -> None:
    a = KnowledgeAtom(
        scope=ScopeType.PRINCIPLE,
        statement="Same claim text for merge testing here",
        source_chunk_ids=["chunk_1"],
    )
    b = KnowledgeAtom(
        scope=ScopeType.PRINCIPLE,
        statement="Same claim text for merge testing here",
        source_chunk_ids=["chunk_4"],
    )
    merged = normalize_knowledge_atoms([a, b])
    assert len(merged) == 1
    assert merged[0].source_chunk_ids == ["chunk_1", "chunk_4"]


def test_reattach_source_chunk_ids_from_raw_after_dedup_drop() -> None:
    raw = [
        KnowledgeAtom(
            scope=ScopeType.INSTANCE,
            statement="Latency is 8.3 ms on M1 silicon",
            source_chunk_ids=["chunk_1"],
        ),
        KnowledgeAtom(
            scope=ScopeType.INSTANCE,
            statement="Latency is 8.3 ms on M1 silicon in table 2",
            source_chunk_ids=["chunk_4"],
        ),
    ]
    clean = [
        KnowledgeAtom(
            scope=ScopeType.INSTANCE,
            statement="Latency is 8.3 ms on M1 silicon in table 2",
            source_chunk_ids=[],
        )
    ]
    fixed = reattach_source_chunk_ids_from_raw(clean, raw)
    assert set(fixed[0].source_chunk_ids) == {"chunk_1", "chunk_4"}


def test_dedup_prompt_requires_source_chunk_ids_union() -> None:
    assert "source_chunk_ids" in _REDUCE_DEDUP_SYSTEM
    assert (
        "UNION" in _REDUCE_DEDUP_SYSTEM or "ALL source chunks" in _REDUCE_DEDUP_SYSTEM
    )


def test_format_atoms_json_includes_source_chunk_ids() -> None:
    block = _format_atoms_json_block(
        [
            KnowledgeAtom(
                scope=ScopeType.PRINCIPLE,
                statement="Governed perimeter before execution path",
                source_chunk_ids=["a_map_1", "a_map_3"],
            )
        ]
    )
    assert "a_map_1" in block
    assert "source_chunk_ids" in block


def test_deduplicated_schema_keeps_source_chunk_ids() -> None:
    out = DeduplicatedAtomsResponse.model_validate(
        {
            "knowledge_atoms": [
                {
                    "scope": "PRINCIPLE",
                    "statement": "Governed hooks before tool calls",
                    "source_chunk_ids": ["chunk_1", "chunk_4"],
                }
            ]
        }
    )
    assert out.knowledge_atoms[0].source_chunk_ids == ["chunk_1", "chunk_4"]


def test_attach_and_merge_helpers() -> None:
    atom = KnowledgeAtom(
        scope=ScopeType.MECHANIC,
        statement="Pipeline stage validates schema before call",
    )
    attached = attach_source_chunk_id([atom], "c1")[0]
    assert attached.source_chunk_ids == ["c1"]
    assert merge_source_chunk_ids(["c1"], ["c1", "c2"]) == ["c1", "c2"]


class _FakeActiveBackend:
    """Мини-стенд под _get_active_vector_store(): rag_chunks/knowledge_atoms
    больше не пишутся в локальный LanceDB (см. аудит Postgres-миграции,
    _add_rag_chunk_rows/upsert_knowledge_atoms), а идут через этот бэкенд."""

    def __init__(self) -> None:
        self.upserted: dict[str, list[dict]] = {}

    async def ensure_collection(self, table: str, dim: int) -> None:
        return None

    async def upsert_documents(self, table: str, docs, vectors) -> bool:
        self.upserted.setdefault(table, []).extend(docs)
        return True

    async def delete_by_field(self, table: str, field: str, value: str) -> int:
        return 0


def test_upsert_knowledge_atoms_and_window_summary(monkeypatch) -> None:
    """upsert_knowledge_atoms/upsert_rag_academic_map_windows пишут через
    _get_active_vector_store() (Postgres/Qdrant по флагу), а не в LanceDB."""
    from unittest.mock import MagicMock

    import knowledge_engine.src.shared.vector_store as vs_mod
    from knowledge_engine.src.shared.schemas import DocumentSummary

    store = vs_mod.VectorStore.__new__(vs_mod.VectorStore)
    store._embeddings = MagicMock()
    store._embeddings.embed_query = MagicMock(return_value=[0.05] * 8)

    backend = _FakeActiveBackend()

    async def _fake_get_active_vector_store():
        return backend

    monkeypatch.setattr(
        vs_mod, "_get_active_vector_store", _fake_get_active_vector_store
    )

    url = "https://example.com/persist-atoms"
    atoms = [
        KnowledgeAtom(
            scope=ScopeType.INSTANCE,
            statement="Measured end-to-end latency is 8.3 ms on M1",
            source_chunk_ids=["chunk_1", "chunk_4"],
        )
    ]
    assert asyncio.run(store.upsert_knowledge_atoms(url, atoms)) == 1
    atom_rows = backend.upserted[vs_mod.KNOWLEDGE_ATOMS_TABLE]
    assert len(atom_rows) == 1
    assert atom_rows[0][vs_mod.KA_COL_SOURCE_CHUNK_IDS] == ["chunk_1", "chunk_4"]

    summary = DocumentSummary(
        title="Persist",
        url=url,
        cs_concepts=[],
        key_takeaways=[],
        failure_modes=[],
        diagram_descriptions=[],
    )
    n = asyncio.run(
        store.upsert_rag_academic_map_windows(
            url,
            "Persist",
            ["body window 1", "body window 2"],
            summary,
            window_summaries=["ws1", "ws2"],
        )
    )
    assert n == 2
    chunk_rows = backend.upserted[vs_mod.RAG_CHUNKS_TABLE]
    assert [r[vs_mod.COL_WINDOW_SUMMARY] for r in chunk_rows] == ["ws1", "ws2"]


def test_persist_spatial_lancedb_upserts_atoms() -> None:
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from knowledge_engine.src.domains.ingestion.article_ingestion.blog_spatial_pipeline import (
        _persist_spatial_lancedb,
    )
    from knowledge_engine.src.shared.schemas import DocumentSummary

    store = MagicMock()
    store.save_summary = AsyncMock(return_value=True)
    store.upsert_knowledge_atoms = AsyncMock(return_value=1)
    store.upsert_rag_academic_map_windows = AsyncMock(return_value=1)
    summary = DocumentSummary(
        title="t",
        url="https://example.com/a",
        cs_concepts=[],
        key_takeaways=[],
        failure_modes=[],
        diagram_descriptions=[],
    )
    atom = KnowledgeAtom(
        scope=ScopeType.PRINCIPLE,
        statement="Isolation is not a security boundary",
    )
    n = asyncio.run(
        _persist_spatial_lancedb(
            store,
            url="https://example.com/a",
            title="t",
            summary=summary,
            window_texts=["body"],
            map_results=[],
            knowledge_atoms=[atom],
        )
    )
    assert n == 1
    store.save_summary.assert_called_once()
    store.upsert_knowledge_atoms.assert_called_once()
    args, _kwargs = store.upsert_knowledge_atoms.call_args
    assert args[0] == "https://example.com/a"
    assert args[1] == [atom]


def test_doc_id_for_url_stable() -> None:
    a = VectorStore.doc_id_for_url("https://Example.com/paper/")
    b = VectorStore.doc_id_for_url("https://example.com/paper")
    assert a == b
    assert len(a) == 24
