"""Unit tests for dialog knowledge_atoms RAG (scope filter + format)."""

from __future__ import annotations

import pytest

from knowledge_engine.src.domains.grounding.dialog_atoms_rag import (
    _hybrid_rerank_pairs_async,
    _rows_to_atoms,
    detect_code_intent,
    filter_atoms_for_dialog,
    format_dialog_atoms_block,
)
from knowledge_engine.src.shared.extraction import KnowledgeAtom, ScopeType


def test_detect_code_intent_ru_en() -> None:
    assert detect_code_intent("Покажи код lifecycle hooks")
    assert detect_code_intent("How to implement the function")
    assert detect_code_intent("пример кода для Worker")
    assert not detect_code_intent("Что такое lifecycle hooks?")
    assert not detect_code_intent("Объясни принцип изоляции агентов")


def test_filter_drops_instance_without_code_intent() -> None:
    atoms = [
        KnowledgeAtom(
            scope=ScopeType.CONCEPT,
            statement="Isolation reduces blast radius across agents",
        ),
        KnowledgeAtom(
            scope=ScopeType.MECHANIC,
            statement="Hooks run before tool dispatch in the pipeline",
        ),
        KnowledgeAtom(
            scope=ScopeType.PRACTICE,
            statement="Latency measured at 8.3 ms on Apple Silicon M1",
        ),
    ]
    filtered = filter_atoms_for_dialog(atoms, allow_instance=False, limit=6)
    assert len(filtered) == 2
    assert all(a.scope != ScopeType.PRACTICE for a in filtered)


def test_filter_keeps_instance_with_code_intent() -> None:
    atoms = [
        KnowledgeAtom(
            scope=ScopeType.CONCEPT,
            statement="Isolation reduces blast radius across agents",
        ),
        KnowledgeAtom(
            scope=ScopeType.PRACTICE,
            statement="Use onRequest hook before fetch in Workers runtime",
        ),
    ]
    filtered = filter_atoms_for_dialog(atoms, allow_instance=True, limit=6)
    assert len(filtered) == 2
    assert filtered[0].scope is ScopeType.CONCEPT
    assert filtered[1].scope is ScopeType.PRACTICE


def test_format_dialog_atoms_block() -> None:
    block = format_dialog_atoms_block(
        [
            KnowledgeAtom(
                scope=ScopeType.MECHANIC,
                statement="Validate schema before tool dispatch always",
            )
        ]
    )
    assert "### dialog_knowledge_atoms" in block
    assert "[ФАКТ (MECHANIC)]: Validate schema before tool dispatch always" in block


def test_format_dialog_atoms_block_r_index() -> None:
    block = format_dialog_atoms_block(
        [
            KnowledgeAtom(
                scope=ScopeType.CONCEPT,
                statement="Isolation reduces blast radius",
            ),
            KnowledgeAtom(
                scope=ScopeType.MECHANIC,
                statement="Hooks run before tool dispatch",
            ),
        ],
        cite_r_index=True,
    )
    assert "### RAG MATERIAL" in block
    assert "[R1] (CONCEPT): Isolation reduces blast radius" in block
    assert "[R2] (MECHANIC): Hooks run before tool dispatch" in block
    assert "[ФАКТ" not in block


def test_format_empty() -> None:
    assert format_dialog_atoms_block([]) == ""


def test_rows_to_atoms_passes_through_core_relevance_score() -> None:
    rows = [
        {
            "statement": "Partition pruning excludes irrelevant sections",
            "scope": "MECHANIC",
            "source_chunk_ids": [],
            "core_relevance_score": 0.42,
        },
        {
            "statement": "No score on this legacy row (pre-feature ingest)",
            "scope": "PRINCIPLE",
            "source_chunk_ids": [],
        },
    ]
    atoms = _rows_to_atoms(rows)
    assert len(atoms) == 2
    assert atoms[0].core_relevance_score == 0.42
    # Missing field on an older row must stay None, never silently become 0.0.
    assert atoms[1].core_relevance_score is None


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_hybrid_rerank_combines_query_and_core_score(monkeypatch) -> None:
    atom_high_core = KnowledgeAtom(
        scope=ScopeType.CONCEPT,
        statement="Fundamental principle atom",
        core_relevance_score=0.9,
    )
    atom_low_core = KnowledgeAtom(
        scope=ScopeType.PRACTICE,
        statement="Narrow instance atom",
        core_relevance_score=0.1,
    )
    pairs = [({"_score": 0.5}, atom_low_core), ({"_score": 0.5}, atom_high_core)]

    def fake_score_relevance_pairs(query: str, texts: list[str]) -> list[float]:
        # Equal query relevance for both — only core_relevance_score should
        # decide the final order.
        return [0.5 for _ in texts]

    monkeypatch.setattr(
        "knowledge_engine.src.rag_gateway.cross_encoder.score_relevance_pairs",
        fake_score_relevance_pairs,
    )

    reranked = await _hybrid_rerank_pairs_async("query", pairs, alpha=0.7)
    assert [atom.statement for _, atom in reranked] == [
        "Fundamental principle atom",
        "Narrow instance atom",
    ]


@pytest.mark.anyio
async def test_hybrid_rerank_falls_back_to_neutral_score_when_unscored(
    monkeypatch,
) -> None:
    """Production caller cannot distinguish "unscored" from "average" —
    unlike the benchmark harness, missing core_relevance_score uses 0.5,
    not pure query_score (see anchor_relevance.py / dialog_atoms_rag.py)."""
    unscored_atom = KnowledgeAtom(scope=ScopeType.MECHANIC, statement="Unscored atom")
    high_core_atom = KnowledgeAtom(
        scope=ScopeType.CONCEPT, statement="High core atom", core_relevance_score=0.9
    )
    low_core_atom = KnowledgeAtom(
        scope=ScopeType.PRACTICE, statement="Low core atom", core_relevance_score=0.1
    )
    pairs = [({}, low_core_atom), ({}, unscored_atom), ({}, high_core_atom)]

    def fake_score_relevance_pairs(query: str, texts: list[str]) -> list[float]:
        return [1.0 for _ in texts]  # equal query relevance for all three

    monkeypatch.setattr(
        "knowledge_engine.src.rag_gateway.cross_encoder.score_relevance_pairs",
        fake_score_relevance_pairs,
    )

    reranked = await _hybrid_rerank_pairs_async("query", pairs, alpha=0.7)
    # alpha=0.7: final = 0.7 + 0.3*core -> high_core(0.9) > unscored(0.5 fallback) > low_core(0.1)
    assert [atom.statement for _, atom in reranked] == [
        "High core atom",
        "Unscored atom",
        "Low core atom",
    ]
