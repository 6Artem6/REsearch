"""LanceDB embed_model guard: refuse mixed nomic / bge-m3 spaces."""

from __future__ import annotations

import pytest

from knowledge_engine.db.embed_model_guard import (
    drop_if_embed_space_mismatch,
    expected_embed_model,
    row_matches_embed_model,
    stamp_embed_model,
)
from knowledge_engine.db.lancedb_pool import reset_lancedb_pool_for_tests
from knowledge_engine.services.search.bge_m3_embed import _assert_bi_encoder_name


def test_expected_embed_model_is_bge_m3():
    assert expected_embed_model() == "BAAI/bge-m3"


def test_stamp_and_match():
    row = stamp_embed_model({"domain": "docs.python.org"})
    assert row_matches_embed_model(row)
    assert not row_matches_embed_model({"domain": "x"})
    assert not row_matches_embed_model({"embed_model": "nomic-embed-text"})


def test_bi_encoder_rejects_nomic_and_reranker():
    with pytest.raises(ValueError, match="nomic"):
        _assert_bi_encoder_name("nomic-embed-text")
    with pytest.raises(ValueError, match="cross-encoder"):
        _assert_bi_encoder_name("BAAI/bge-reranker-v2-m3")
    assert _assert_bi_encoder_name("BAAI/bge-m3") == "BAAI/bge-m3"


def test_drop_legacy_embed_space(tmp_path):
    import lancedb

    from knowledge_engine.db.embed_model_guard import _table_names

    reset_lancedb_pool_for_tests()
    db = lancedb.connect(str(tmp_path / "guard.lancedb"))
    db.create_table(
        "document_summaries",
        data=[
            {"title": "old", "vector": [0.1, 0.2], "embed_model": "nomic-embed-text"}
        ],
    )
    assert drop_if_embed_space_mismatch(db, "document_summaries") is True
    assert "document_summaries" not in _table_names(db)
    reset_lancedb_pool_for_tests()


def test_keep_matching_bge_space(tmp_path):
    import lancedb

    from knowledge_engine.db.embed_model_guard import _table_names

    reset_lancedb_pool_for_tests()
    db = lancedb.connect(str(tmp_path / "guard_ok.lancedb"))
    db.create_table(
        "rag_chunks",
        data=[
            {
                "chunk_id": "c1",
                "vector": [0.1, 0.2],
                "embed_model": "BAAI/bge-m3",
            }
        ],
    )
    assert drop_if_embed_space_mismatch(db, "rag_chunks") is False
    assert "rag_chunks" in _table_names(db)
    reset_lancedb_pool_for_tests()
