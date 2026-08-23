"""Knowledge atoms ↔ MAP chunk provenance + LanceDB helpers."""

from __future__ import annotations

from knowledge_engine.db.rag_chunks_schema import map_window_chunk_id
from knowledge_engine.schemas.extraction import (
    KnowledgeAtom,
    ScopeType,
    attach_source_chunk_id,
    merge_source_chunk_ids,
    normalize_knowledge_atoms,
    reattach_source_chunk_ids_from_raw,
)
from knowledge_engine.services.article_ingestion.blog_spatial_schemas import (
    DeduplicatedAtomsResponse,
    MapWindowResponse,
    normalize_map_knowledge,
)
from knowledge_engine.services.article_ingestion.blog_spatial_summarizer import (
    _REDUCE_DEDUP_SYSTEM,
    _format_atoms_json_block,
)
from knowledge_engine.services.vector_store import VectorStore


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


def test_upsert_knowledge_atoms_and_window_summary(tmp_path, monkeypatch) -> None:
    """LanceDB persist path without Ollama (mocked embeddings)."""
    from unittest.mock import MagicMock

    import lancedb

    import knowledge_engine.config as cfg
    import knowledge_engine.services.vector_store as vs_mod
    from knowledge_engine.schemas import DocumentSummary

    monkeypatch.setattr(cfg, "LANCE_DB_PATH", tmp_path)
    monkeypatch.setattr(vs_mod, "LANCE_DB_PATH", tmp_path)

    store = vs_mod.VectorStore.__new__(vs_mod.VectorStore)
    store._embeddings = MagicMock()
    store._embeddings.embed_query = MagicMock(return_value=[0.05] * 8)
    store._db = lancedb.connect(str(tmp_path))

    url = "https://example.com/persist-atoms"
    atoms = [
        KnowledgeAtom(
            scope=ScopeType.INSTANCE,
            statement="Measured end-to-end latency is 8.3 ms on M1",
            source_chunk_ids=["chunk_1", "chunk_4"],
        )
    ]
    assert store.upsert_knowledge_atoms(url, atoms) == 1
    arrow = store._db.open_table("knowledge_atoms").to_arrow()
    assert arrow.num_rows == 1
    assert list(arrow.column("source_chunk_ids")[0].as_py()) == [
        "chunk_1",
        "chunk_4",
    ]

    summary = DocumentSummary(
        title="Persist",
        url=url,
        cs_concepts=[],
        key_takeaways=[],
        failure_modes=[],
        diagram_descriptions=[],
    )
    n = store.upsert_rag_academic_map_windows(
        url,
        "Persist",
        ["body window 1", "body window 2"],
        summary,
        window_summaries=["ws1", "ws2"],
    )
    assert n == 2
    chunks = store._db.open_table("rag_chunks").to_arrow()
    assert [
        chunks.column("window_summary")[i].as_py() for i in range(chunks.num_rows)
    ] == ["ws1", "ws2"]


def test_rag_chunks_migrates_missing_window_summary(tmp_path, monkeypatch) -> None:
    """Pre-window_summary Lance table is rebuilt instead of dropping the column."""
    from unittest.mock import MagicMock

    import lancedb

    import knowledge_engine.config as cfg
    import knowledge_engine.services.vector_store as vs_mod
    from knowledge_engine.db.embed_model_guard import expected_embed_model
    from knowledge_engine.schemas import DocumentSummary

    monkeypatch.setattr(cfg, "LANCE_DB_PATH", tmp_path)
    monkeypatch.setattr(vs_mod, "LANCE_DB_PATH", tmp_path)

    store = vs_mod.VectorStore.__new__(vs_mod.VectorStore)
    store._embeddings = MagicMock()
    store._embeddings.embed_query = MagicMock(return_value=[0.05] * 8)
    store._db = lancedb.connect(str(tmp_path))

    store._db.create_table(
        "rag_chunks",
        data=[
            {
                "chunk_id": "legacy_1",
                "doc_id": "legacy-doc",
                "url": "https://example.com/legacy",
                "title": "Legacy",
                "chunk_text": "legacy body",
                "chunk_vector": [0.01] * 8,
                "doc_summary_text": "legacy sum",
                "doc_meta_vector": [0.02] * 8,
                "chunk_index": 0,
                "chunks_in_doc": 1,
                "trust_score": 1.0,
                "embed_model": expected_embed_model(),
            }
        ],
    )

    url = "https://example.com/new-map"
    summary = DocumentSummary(
        title="New",
        url=url,
        cs_concepts=[],
        key_takeaways=[],
        failure_modes=[],
        diagram_descriptions=[],
    )
    n = store.upsert_rag_academic_map_windows(
        url,
        "New",
        ["window body"],
        summary,
        window_summaries=["gemma window digest"],
    )
    assert n == 1
    chunks = store._db.open_table("rag_chunks").to_arrow()
    names = set(chunks.schema.names)
    assert "window_summary" in names
    by_id = {
        chunks.column("chunk_id")[i].as_py(): chunks.column("window_summary")[i].as_py()
        for i in range(chunks.num_rows)
    }
    assert by_id["legacy_1"] in ("", None)
    new_summaries = [v for k, v in by_id.items() if k != "legacy_1"]
    assert new_summaries == ["gemma window digest"]


def test_persist_spatial_lancedb_upserts_atoms() -> None:
    from unittest.mock import MagicMock

    from knowledge_engine.schemas import DocumentSummary
    from knowledge_engine.services.article_ingestion.blog_spatial_pipeline import (
        _persist_spatial_lancedb,
    )

    store = MagicMock()
    store.upsert_knowledge_atoms.return_value = 1
    store.upsert_rag_academic_map_windows.return_value = 1
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
    n = _persist_spatial_lancedb(
        store,
        url="https://example.com/a",
        title="t",
        summary=summary,
        window_texts=["body"],
        map_results=[],
        knowledge_atoms=[atom],
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
