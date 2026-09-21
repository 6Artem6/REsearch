"""Mastery-Guided Concept Affinity — regression test for the sql_cte_group_by
-> sql_cte cross-node repetition scenario that motivated this feature.

No real network/LLM/Postgres calls: the BGE-M3 singleton
(``socratic_poles.py::_embeddings``) is monkeypatched with a deterministic
fake embedder so cosine outcomes are exact and reproducible — same "no real
external calls" tier as ``tests/domains/grounding/test_pedagogical_reranker.py``.

Covers the three Step-4 scenarios from the Concept Affinity task:
1. Mastery Gate: a MASTERED (status="verified") concept's matching atoms get
   penalized and lose to fresh material; an UNASSIMILATED ("gap") concept
   must never be penalized even though it was discussed.
2. Multiperspective Shift: response_depth_signal="high" (stuck on nuance)
   boosts MECHANIC/EDGE_CASE/ANTI_PATTERN over CONCEPT/PRACTICE.
3. Relevance Fallback: when concept-affinity manipulation would starve every
   candidate below the relevance floor, the turn reverts to the bare hybrid
   base instead of shipping a confidently-wrong top-k.
"""

from __future__ import annotations

import knowledge_engine.src.domains.grounding.socratic_poles as socratic_poles
from knowledge_engine.src.config import settings as ke_settings
from knowledge_engine.src.domains.grounding.pedagogical_reranker import (
    REPULSION_FLOOR_RATIO,
    AtomAnchorFilterService,
    apply_pedagogical_weights,
    calculate_pedagogy_weight,
)
from knowledge_engine.src.domains.grounding.schemas import RetrievalPedagogicalContext
from knowledge_engine.src.shared.extraction import KnowledgeAtom, ScopeType


class _OneHotFakeEmbedder:
    """Deterministic embedder: exact text match -> identical vector (cosine
    1.0); anything else -> orthogonal (cosine 0.0). No real model load."""

    def __init__(self, buckets: dict[str, int], dim: int = 8) -> None:
        self._buckets = buckets
        self._dim = dim

    def _vector(self, text: str) -> list[float]:
        v = [0.0] * self._dim
        v[self._buckets.get(text, self._dim - 1)] = 1.0
        return v

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def _patch_embedder(monkeypatch, buckets: dict[str, int]) -> None:
    monkeypatch.setattr(
        socratic_poles, "_embeddings", lambda: _OneHotFakeEmbedder(buckets)
    )
    monkeypatch.setattr(ke_settings, "DIALOG_ATOMS_CONCEPT_AFFINITY_ENABLED", True)


def test_mastery_gate_penalizes_verified_concept_atoms(monkeypatch):
    """sql_cte_group_by -> sql_cte regression: a concept the learner already
    passed (status="verified") must be penalized so fresh material wins."""
    mastered_claim = "Recursive CTE anchor plus recursive part until empty"
    mastered_atom = KnowledgeAtom(
        scope=ScopeType.MECHANIC, statement=mastered_claim, core_relevance_score=0.6
    )
    fresh_atom = KnowledgeAtom(
        scope=ScopeType.MECHANIC,
        statement="GROUPING SETS combine multiple GROUP BY groupings in one pass",
        core_relevance_score=0.6,
    )
    _patch_embedder(
        monkeypatch,
        {mastered_claim: 0, mastered_atom.statement: 0, fresh_atom.statement: 1},
    )

    context = RetrievalPedagogicalContext(
        depth_level="deep_dive",
        repulsion_claims=[mastered_claim],
        attraction_claims=[],
    )
    ranked = apply_pedagogical_weights(
        [(mastered_atom, 0.9), (fresh_atom, 0.9)], context, alpha=0.7
    )
    assert ranked[0][0] is fresh_atom, "fresh material must outrank a mastered repeat"
    assert ranked[1][0] is mastered_atom


def test_mastery_gate_never_penalizes_unassimilated_gap_concept(monkeypatch):
    """A concept merely discussed but NOT passed (status="gap") must not be
    treated as repulsion — Mastery Gate only fires on real mastery."""
    from knowledge_engine.src.domains.grounding.memory_schemas import (
        SessionMemory,
        SubConceptRecord,
    )
    from knowledge_engine.src.domains.grounding.socratic_poles import (
        _local_repulsion_facts,
        mastery_gate_filter,
    )

    gap_concept = SubConceptRecord(
        id="union_vs_union_all", label="UNION vs UNION ALL", status="gap"
    )
    verified_concept = SubConceptRecord(
        id="cte_anchor", label="CTE anchor + recursive part", status="verified"
    )
    memory = SessionMemory(sub_concepts=[gap_concept, verified_concept])

    gated = mastery_gate_filter(memory.sub_concepts)
    assert [sc.id for sc in gated] == [
        "cte_anchor"
    ], "gap concept must never pass the Mastery Gate, verified concept must"

    repulsion = _local_repulsion_facts(memory, node_id="sql_cte")
    concept_ids = {r["concept_id"] for r in repulsion}
    assert "union_vs_union_all" not in concept_ids
    assert "cte_anchor" in concept_ids


def test_multiperspective_shift_high_depth_boosts_mechanic_and_edge_case(monkeypatch):
    """response_depth_signal="high" (WHY+HOW passed, stuck on MECHANIC) must
    boost MECHANIC/EDGE_CASE/ANTI_PATTERN over CONCEPT for the same base score."""
    concept_atom = KnowledgeAtom(
        scope=ScopeType.CONCEPT,
        statement="Why recursive CTEs exist",
        core_relevance_score=0.7,
    )
    edge_case_atom = KnowledgeAtom(
        scope=ScopeType.EDGE_CASE,
        statement="Infinite recursion when the stop condition is missing",
        core_relevance_score=0.7,
    )
    _patch_embedder(monkeypatch, {})  # no repulsion/attraction claims in this test

    context = RetrievalPedagogicalContext(
        depth_level="deep_dive",
        response_depth_signal="high",
    )
    ranked = apply_pedagogical_weights(
        [(concept_atom, 0.8), (edge_case_atom, 0.8)], context, alpha=0.7
    )
    assert (
        ranked[0][0] is edge_case_atom
    ), "high-depth signal must favor EDGE_CASE over CONCEPT"


def test_multiperspective_shift_low_depth_boosts_practice_and_concept(monkeypatch):
    """response_depth_signal="low" (WHY not even passed) must boost
    PRACTICE/CONCEPT over EDGE_CASE/ANTI_PATTERN for the same base score."""
    practice_atom = KnowledgeAtom(
        scope=ScopeType.PRACTICE,
        statement="Basic example: generate a date sequence",
        core_relevance_score=0.7,
    )
    anti_pattern_atom = KnowledgeAtom(
        scope=ScopeType.ANTI_PATTERN,
        statement="Do not forget the stop condition in recursive CTEs",
        core_relevance_score=0.7,
    )
    _patch_embedder(monkeypatch, {})

    context = RetrievalPedagogicalContext(
        depth_level="deep_dive", response_depth_signal="low"
    )
    ranked = apply_pedagogical_weights(
        [(practice_atom, 0.8), (anti_pattern_atom, 0.8)], context, alpha=0.7
    )
    assert (
        ranked[0][0] is practice_atom
    ), "low-depth signal must favor PRACTICE over ANTI_PATTERN"


def test_repulsion_floor_never_drops_penalized_atom_below_ratio_of_base(monkeypatch):
    claim = "mastered mechanic detail"
    atom = KnowledgeAtom(
        scope=ScopeType.EDGE_CASE, statement=claim, core_relevance_score=0.9
    )
    _patch_embedder(monkeypatch, {claim: 0})

    context = RetrievalPedagogicalContext(
        depth_level="intro",  # EDGE_CASE weight 0.4 at intro -> deep penalty stack
        repulsion_claims=[claim],
    )
    ranked = apply_pedagogical_weights([(atom, 0.9)], context, alpha=0.7)
    base = AtomAnchorFilterService.calculate_hybrid_fact_score(0.9, 0.9, 0.7)
    assert ranked[0][1] >= REPULSION_FLOOR_RATIO * base - 1e-9


def test_relevance_fallback_reverts_to_bare_hybrid_base_when_all_scores_collapse(
    monkeypatch,
):
    claim = "shared repulsion claim"
    atom_a = KnowledgeAtom(
        scope=ScopeType.CONCEPT, statement="atom a", core_relevance_score=0.3
    )
    atom_b = KnowledgeAtom(
        scope=ScopeType.CONCEPT, statement="atom b", core_relevance_score=0.3
    )
    _patch_embedder(monkeypatch, {claim: 0, "atom a": 0, "atom b": 0})

    context = RetrievalPedagogicalContext(
        depth_level="advanced",  # CONCEPT weight 0.5 at advanced
        repulsion_claims=[claim],
    )
    ranked = apply_pedagogical_weights(
        [(atom_a, 0.35), (atom_b, 0.35)], context, alpha=0.7
    )
    expected_base = AtomAnchorFilterService.calculate_hybrid_fact_score(0.35, 0.3, 0.7)
    for _, score in ranked:
        assert abs(score - expected_base) < 1e-9, (
            "Relevance Fallback must drop w_pedagogy/w_novelty/w_concept_affinity/"
            "w_multiperspective_shift entirely, not just the concept-affinity terms"
        )


def test_concept_affinity_disabled_flag_is_a_pure_noop(monkeypatch):
    """DIALOG_ATOMS_CONCEPT_AFFINITY_ENABLED=False (default) must behave
    identically to the pre-existing hybrid+pedagogy+novelty formula, even
    when repulsion/attraction claims are (incorrectly) populated by a caller."""
    monkeypatch.setattr(ke_settings, "DIALOG_ATOMS_CONCEPT_AFFINITY_ENABLED", False)
    claim = "would-be repulsion claim"
    atom = KnowledgeAtom(
        scope=ScopeType.MECHANIC, statement=claim, core_relevance_score=0.6
    )
    context = RetrievalPedagogicalContext(
        depth_level="deep_dive", repulsion_claims=[claim]
    )

    ranked = apply_pedagogical_weights([(atom, 0.9)], context, alpha=0.7)
    base = AtomAnchorFilterService.calculate_hybrid_fact_score(0.9, 0.6, 0.7)
    expected = base * calculate_pedagogy_weight(ScopeType.MECHANIC, "deep_dive")
    assert abs(ranked[0][1] - expected) < 1e-9
