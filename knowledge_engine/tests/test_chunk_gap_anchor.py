"""Knee cutoff, dedup, and positional reorder for chunk selection."""

from __future__ import annotations

import numpy as np

from knowledge_engine.services.chunk_cross_attention_mmr import (
    ChunkCandidate,
    _greedy_mmr_select,
    _PoolEntry,
    filter_pool_by_knee_cutoff,
    positional_reorder_entries,
    select_diverse_chunks_with_cross_attention,
    semantic_dedup_entries,
)


def _entry(rel: float, tag: str = "x") -> _PoolEntry:
    return _PoolEntry(
        0,
        ChunkCandidate("chunk body text " * 3, tag),
        np.array([1.0, 0.0]),
        rel,
    )


def test_knee_cutoff_on_sharp_drop():
    entries = [
        _entry(0.82, "a"),
        _entry(0.80, "b"),
        _entry(0.79, "c"),
        _entry(0.52, "d"),
    ]
    kept = filter_pool_by_knee_cutoff(entries, min_floor=0.30, knee_drop_ratio=0.12)
    assert len(kept) == 3
    assert {e.relevance for e in kept} == {0.82, 0.80, 0.79}


def test_knee_cutoff_dense_cluster_keeps_all():
    entries = [_entry(r) for r in (0.82, 0.81, 0.80, 0.79)]
    kept = filter_pool_by_knee_cutoff(entries, min_floor=0.30, knee_drop_ratio=0.12)
    assert len(kept) == 4


def test_knee_cutoff_below_floor_rejects_all_no_keep_anyway():
    """Раньше при top_score < min_floor всё равно возвращался лучший
    кандидат ("keep best only") — именно так cos=0.000 чанки попадали в
    RAG-инспектор (см. баг: 3 identичных R1/R2/R3 из названия темы). Порог
    жёсткий: ничего не проходит — пустой список, а не "хоть что-нибудь"."""
    entries = [_entry(0.0, "a"), _entry(0.0, "b"), _entry(0.0, "c")]
    kept = filter_pool_by_knee_cutoff(entries, min_floor=0.30, knee_drop_ratio=0.12)
    assert kept == []


def test_effective_gamma_rises_on_dense_scores():
    entries = [_entry(r) for r in (0.82, 0.81, 0.80, 0.79)]
    v0 = np.array([1.0, 0.0])
    v1 = np.array([0.0, 1.0])
    entries[0].chunk_vec = v0
    entries[1].chunk_vec = v1
    entries[2].chunk_vec = v0 * 0.99 + v1 * 0.01
    entries[3].chunk_vec = v1 * 0.99 + v0 * 0.01
    base_gamma = 0.55
    picked = _greedy_mmr_select(entries, top_k=2, gamma=base_gamma, max_per_source=2)
    assert len(picked) == 2
    assert picked[0].relevance == 0.82


def test_semantic_dedup_removes_near_duplicate():
    v = np.array([1.0, 0.0])
    entries = [
        _PoolEntry(
            0,
            ChunkCandidate("first chunk text " * 3, "a"),
            v,
            0.9,
        ),
        _PoolEntry(
            1,
            ChunkCandidate("second near dup " * 3, "a"),
            np.array([0.99, 0.01]),
            0.85,
        ),
    ]
    out = semantic_dedup_entries(entries, threshold=0.85)
    assert len(out) == 1
    assert out[0].relevance == 0.9


def test_positional_reorder_puts_top_two_at_ends():
    entries = [
        _PoolEntry(
            i, ChunkCandidate(f"chunk {i} " * 5, "s"), np.eye(4)[i % 4], float(i)
        )
        for i in range(5)
    ]
    for e in entries:
        e.relevance = float(int(e.ch.text.split()[1]))
    ordered = positional_reorder_entries(entries)
    assert ordered[0].relevance == 4.0
    assert ordered[-1].relevance == 3.0
    mids = [e.relevance for e in ordered[1:-1]]
    assert mids == [2.0, 1.0, 0.0]


def test_multi_source_mode_when_top_below_anchor():
    topic = np.array([1.0, 0.0])
    candidates = [
        ChunkCandidate(
            text="Moderate relevance chunk for topic here.",
            source_id="d1",
            source_title="A",
            chunk_vector=np.array([0.7, 0.71]),
            doc_meta_vector=np.array([0.7, 0.71]),
            meta={"origin_i": "0", "doc_id": "d1"},
        ),
        ChunkCandidate(
            text="Another moderate chunk from other source.",
            source_id="d2",
            source_title="B",
            chunk_vector=np.array([0.68, 0.73]),
            doc_meta_vector=np.array([0.68, 0.73]),
            meta={"origin_i": "1", "doc_id": "d2"},
        ),
    ]
    picked = select_diverse_chunks_with_cross_attention(
        topic,
        candidates,
        top_k=2,
        alpha=1.0,
        beta=0.0,
        gamma=0.0,
        doc_gate_threshold=0.0,
        anchor_threshold=1.01,
        min_floor=0.30,
        knee_drop_ratio=0.12,
    )
    assert len(picked) >= 1
    assert picked[0].source_index == 1


def test_select_diverse_chunks_returns_empty_when_all_below_floor():
    """Все кандидаты ортогональны теме (rel≈0) — должен вернуться пустой
    список без падения на max() пустой последовательности (см. фикс
    filter_pool_by_knee_cutoff: 'keep best only' убран)."""
    topic = np.array([1.0, 0.0])
    candidates = [
        ChunkCandidate(
            text="Completely unrelated chunk about the topic name only.",
            source_id="d1",
            source_title="A",
            chunk_vector=np.array([0.0, 1.0]),
            doc_meta_vector=np.array([0.0, 1.0]),
            meta={"origin_i": "0", "doc_id": "d1"},
        ),
        ChunkCandidate(
            text="Another unrelated duplicate chunk about the topic name only.",
            source_id="d2",
            source_title="B",
            chunk_vector=np.array([0.0, 1.0]),
            doc_meta_vector=np.array([0.0, 1.0]),
            meta={"origin_i": "1", "doc_id": "d2"},
        ),
    ]
    picked = select_diverse_chunks_with_cross_attention(
        topic,
        candidates,
        top_k=2,
        alpha=1.0,
        beta=0.0,
        gamma=0.0,
        doc_gate_threshold=0.0,
        anchor_threshold=1.01,
        min_floor=0.30,
        knee_drop_ratio=0.12,
    )
    assert picked == []
