"""Two-Stage Fact Relevance & Anchor Filtering — fast unit tests (no real
Cross-Encoder / LLM calls; see tests/benchmarks/test_fact_relevance_pipeline.py
for the slow, real-API A/B harness this graduated from)."""

from __future__ import annotations

import asyncio

from knowledge_engine.src.domains.ingestion.article_ingestion.anchor_relevance import (
    AtomAnchorFilterService,
    get_synthetic_anchor,
)
from knowledge_engine.src.shared.extraction import KnowledgeAtom, ScopeType


def _mock_score_relevance_pairs(monkeypatch, scores: list[float]) -> None:
    def fake(criterion: str, texts: list[str]) -> list[float]:
        assert len(texts) == len(scores)
        return list(scores)

    monkeypatch.setattr(
        "knowledge_engine.src.domains.ingestion.article_ingestion.anchor_relevance.score_relevance_pairs",
        fake,
    )


def test_get_synthetic_anchor_combines_title_and_lead() -> None:
    anchor = get_synthetic_anchor("Секционирование в Postgres", "Разделение таблицы...")
    assert anchor == "Секционирование в Postgres\n\nРазделение таблицы..."


def test_get_synthetic_anchor_empty_inputs() -> None:
    assert get_synthetic_anchor("", "") == ""


def test_filter_atoms_by_anchor_stamps_score_and_cuts_below_threshold(
    monkeypatch,
) -> None:
    atoms = [
        KnowledgeAtom(scope=ScopeType.CONCEPT, statement="High relevance fact"),
        KnowledgeAtom(scope=ScopeType.PRACTICE, statement="Low relevance fact"),
    ]
    _mock_score_relevance_pairs(monkeypatch, [0.8, 0.1])

    service = AtomAnchorFilterService()
    kept = asyncio.run(
        service.filter_atoms_by_anchor(atoms, "some anchor", threshold=0.35)
    )

    assert [a.statement for a in kept] == ["High relevance fact"]
    # Both atoms are stamped, including the one dropped by the caller.
    assert atoms[0].core_relevance_score == 0.8
    assert atoms[1].core_relevance_score == 0.1


def test_filter_atoms_by_anchor_empty_anchor_passes_through_unscored(
    monkeypatch,
) -> None:
    atoms = [KnowledgeAtom(scope=ScopeType.CONCEPT, statement="Some fact")]

    def fail_if_called(criterion: str, texts: list[str]) -> list[float]:
        raise AssertionError("Cross-Encoder must not be called for an empty anchor")

    monkeypatch.setattr(
        "knowledge_engine.src.domains.ingestion.article_ingestion.anchor_relevance.score_relevance_pairs",
        fail_if_called,
    )

    service = AtomAnchorFilterService()
    kept = asyncio.run(service.filter_atoms_by_anchor(atoms, "  ", threshold=0.35))
    assert kept == atoms
    assert atoms[0].core_relevance_score is None


def test_calculate_hybrid_fact_score_formula() -> None:
    score = AtomAnchorFilterService.calculate_hybrid_fact_score(
        query_score=0.8, core_relevance_score=0.2, alpha=0.7
    )
    assert round(score, 6) == round(0.7 * 0.8 + 0.3 * 0.2, 6)


def test_rerank_atoms_for_query_uses_hybrid_when_scored(monkeypatch) -> None:
    atom_high_core = KnowledgeAtom(
        scope=ScopeType.CONCEPT, statement="High core", core_relevance_score=0.9
    )
    atom_low_core = KnowledgeAtom(
        scope=ScopeType.PRACTICE, statement="Low core", core_relevance_score=0.1
    )
    _mock_score_relevance_pairs(monkeypatch, [0.5, 0.5])  # equal query relevance

    service = AtomAnchorFilterService()
    ranked = asyncio.run(
        service.rerank_atoms_for_query(
            "query", [atom_low_core, atom_high_core], alpha=0.7
        )
    )
    assert [a.statement for a, _ in ranked] == ["High core", "Low core"]


def test_rerank_atoms_for_query_falls_back_to_pure_query_score_when_unscored(
    monkeypatch,
) -> None:
    """Unlike the production dialog_atoms_rag.py caller (neutral 0.5
    fallback), this generic service method falls back to pure query_score
    for atoms never scored by filter_atoms_by_anchor — see docstring."""
    unscored = KnowledgeAtom(scope=ScopeType.MECHANIC, statement="Unscored")
    _mock_score_relevance_pairs(monkeypatch, [0.42])

    service = AtomAnchorFilterService()
    ranked = asyncio.run(service.rerank_atoms_for_query("query", [unscored], alpha=0.7))
    assert ranked == [(unscored, 0.42)]
