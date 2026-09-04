"""Score threshold floor / dedup / skip-retrieval для lecture RAG (см. баг:
3 identичных R1/R2/R3 чанка, cos=0.000, просто название темы в RAG-инспекторе).
"""

from __future__ import annotations

import knowledge_engine.services.lecture_context_rerank as rerank_mod
from knowledge_engine.services.lecture_context_rerank import (
    LectureContextCandidate,
    diversify_lecture_candidates_sync,
)
from knowledge_engine.services.lecture_rag_context import (
    _dedupe_candidates_by_text_hash,
    _lecture_node_needs_retrieval,
)
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput


def _candidate(plain: str, label: str = "hybrid_semantic") -> LectureContextCandidate:
    return LectureContextCandidate(
        label=label,
        formatted=plain,
        plain=plain,
        url_key=f"https://example.com/{hash(plain) & 0xFFFF}",
    )


def test_diversify_returns_empty_when_all_below_ce_floor(monkeypatch):
    """Раньше при all(sc < floor) всё равно брались top-N по CE score
    ('keep top by CE score') — источник cos≈0 дублей в инспекторе. Теперь
    ничего не проходит порог → пустой список, без keep-anyway fallback."""
    monkeypatch.setattr(rerank_mod, "LECTURE_RAG_CE_MIN_SCORE", 0.50)
    monkeypatch.setattr(
        rerank_mod,
        "score_relevance_pairs",
        lambda query, texts: [0.01 for _ in texts],
    )
    candidates = [
        _candidate("Index name only, no real content whatsoever here."),
        _candidate("Index name only, no real content whatsoever here again."),
        _candidate("Index name only, no real content whatsoever here thrice."),
    ]
    selected = diversify_lecture_candidates_sync("index topic", candidates)
    assert selected == []


def test_diversify_keeps_candidates_above_floor(monkeypatch):
    monkeypatch.setattr(rerank_mod, "LECTURE_RAG_CE_MIN_SCORE", 0.50)
    monkeypatch.setattr(
        rerank_mod,
        "score_relevance_pairs",
        lambda query, texts: [0.9, 0.1],
    )
    monkeypatch.setattr(
        rerank_mod,
        "_embed_texts_sync",
        lambda texts: [__import__("numpy").array([1.0, 0.0]) for _ in texts],
    )
    candidates = [
        _candidate("Relevant real content chunk about the actual topic."),
        _candidate("Irrelevant low-score chunk about something else entirely."),
    ]
    selected = diversify_lecture_candidates_sync("index topic", candidates)
    assert len(selected) == 1
    assert "Relevant real content" in selected[0].plain


def test_dedupe_candidates_by_text_hash_removes_exact_duplicates():
    dup_text = "Индексы и структура данных"
    candidates = [
        _candidate(dup_text),
        _candidate(dup_text),
        _candidate(dup_text),
        _candidate("Другой, уникальный текст чанка"),
    ]
    out = _dedupe_candidates_by_text_hash(candidates)
    assert len(out) == 2
    assert out[0].plain == dup_text


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


def test_lecture_node_needs_retrieval_false_for_model_first(monkeypatch):
    monkeypatch.setattr(
        "knowledge_engine.services.lecture_rag_context.get_curriculum_graph",
        lambda _cid: {
            "nodes": [{"node_id": "hash_indexes", "node_risk_kind": "BASE"}]
        },
    )
    node = _node()
    assert _lecture_node_needs_retrieval("some_curriculum", node) is False


def test_lecture_node_needs_retrieval_true_for_base_with_persisted_sources(
    monkeypatch,
):
    """BASE-нода, у которой ранее (например, через
    persist_verified_external_sources_to_node после лекции) появился
    resource_urls — retrieval должен запуститься и переиспользовать
    сохранённый материал, а не игнорировать его из-за исходной BASE-
    классификации (иначе каждая лекция заново гоняет Exa/Gemini waterfall)."""
    monkeypatch.setattr(
        "knowledge_engine.services.lecture_rag_context.get_curriculum_graph",
        lambda _cid: {
            "nodes": [{"node_id": "hash_indexes", "node_risk_kind": "BASE"}]
        },
    )
    node = _node(resource_urls=["https://github.com/postgres/postgres"])
    assert _lecture_node_needs_retrieval("some_curriculum", node) is True


def test_lecture_node_needs_retrieval_false_when_no_attached_sources(monkeypatch):
    monkeypatch.setattr(
        "knowledge_engine.services.lecture_rag_context.get_curriculum_graph",
        lambda _cid: {
            "nodes": [{"node_id": "hash_indexes", "node_risk_kind": "DEEP"}]
        },
    )
    monkeypatch.setattr(
        "knowledge_engine.services.lecture_rag_source_scope.mapped_doc_ids_for_node",
        lambda _cid, _node: [],
    )
    node = _node()
    assert _lecture_node_needs_retrieval("some_curriculum", node) is False


def test_lecture_node_needs_retrieval_true_when_sources_attached(monkeypatch):
    monkeypatch.setattr(
        "knowledge_engine.services.lecture_rag_context.get_curriculum_graph",
        lambda _cid: {
            "nodes": [{"node_id": "hash_indexes", "node_risk_kind": "DEEP"}]
        },
    )
    node = _node(resource_urls=["https://example.com/pep-0703"])
    assert _lecture_node_needs_retrieval("some_curriculum", node) is True
