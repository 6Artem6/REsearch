"""Pedagogical-Aware RAG Reranking — fast unit tests (no network/LLM calls).

See tests/benchmarks/test_pedagogical_rag_session.py for the 4-turn A/B
session benchmark (mocked Cross-Encoder, still no real external calls)."""

from __future__ import annotations

from knowledge_engine.src.domains.grounding.pedagogical_reranker import (
    NOVEL_ARTICLE_BOOST,
    REPEAT_ATOM_PENALTY,
    apply_pedagogical_weights,
    calculate_novelty_weight,
    calculate_pedagogy_weight,
    depth_level_from_overlay_type,
)
from knowledge_engine.src.domains.grounding.schemas import RetrievalPedagogicalContext
from knowledge_engine.src.shared.extraction import KnowledgeAtom, ScopeType


def test_pedagogy_weight_matrix_matches_spec() -> None:
    expected = {
        "intro": {
            ScopeType.CONCEPT: 1.3,
            ScopeType.MECHANIC: 1.0,
            ScopeType.PRACTICE: 0.7,
            ScopeType.EDGE_CASE: 0.4,
            ScopeType.ANTI_PATTERN: 0.5,
        },
        "deep_dive": {
            ScopeType.MECHANIC: 1.35,
            ScopeType.CONCEPT: 0.85,
            ScopeType.PRACTICE: 1.1,
            ScopeType.EDGE_CASE: 0.7,
            ScopeType.ANTI_PATTERN: 0.8,
        },
        "practice": {
            ScopeType.PRACTICE: 1.4,
            ScopeType.MECHANIC: 1.2,
            ScopeType.CONCEPT: 0.7,
            ScopeType.EDGE_CASE: 1.0,
            ScopeType.ANTI_PATTERN: 1.1,
        },
        "advanced": {
            ScopeType.PRACTICE: 1.5,
            ScopeType.MECHANIC: 1.3,
            ScopeType.EDGE_CASE: 1.4,
            ScopeType.ANTI_PATTERN: 1.3,
            ScopeType.CONCEPT: 0.5,
        },
        "expert": {
            ScopeType.EDGE_CASE: 1.7,
            ScopeType.ANTI_PATTERN: 1.6,
            ScopeType.PRACTICE: 1.4,
            ScopeType.MECHANIC: 1.1,
            ScopeType.CONCEPT: 0.3,
        },
    }
    for depth_level, per_scope in expected.items():
        for scope, weight in per_scope.items():
            assert calculate_pedagogy_weight(scope, depth_level) == weight


def test_depth_level_from_overlay_type_mapping() -> None:
    assert depth_level_from_overlay_type("ADVANCED_ASTERISK") == "advanced"
    assert depth_level_from_overlay_type("DEEP_ASTERISK") == "expert"


def test_apply_pedagogical_weights_expert_favors_instance_hardest() -> None:
    """expert gives PRACTICE a solid boost (1.4) and the harshest CONCEPT
    penalty (0.3) of all 5 levels — the two must not swap ranking."""
    principle = KnowledgeAtom(
        scope=ScopeType.CONCEPT,
        statement="Foundational claim",
        core_relevance_score=0.6,
    )
    instance = KnowledgeAtom(
        scope=ScopeType.PRACTICE, statement="Edge-case detail", core_relevance_score=0.6
    )
    context = RetrievalPedagogicalContext(depth_level="expert")
    ranked = apply_pedagogical_weights(
        [(principle, 0.6), (instance, 0.6)], context, alpha=0.7
    )
    assert [a.statement for a, _ in ranked] == [
        "Edge-case detail",
        "Foundational claim",
    ]


def test_apply_pedagogical_weights_advanced_penalizes_repeat_atom() -> None:
    seen = KnowledgeAtom(
        scope=ScopeType.PRACTICE,
        statement="Already shown",
        core_relevance_score=0.6,
        id="atom-seen",
    )
    fresh = KnowledgeAtom(
        scope=ScopeType.PRACTICE,
        statement="Never shown",
        core_relevance_score=0.6,
        id="atom-fresh",
    )
    context = RetrievalPedagogicalContext(
        depth_level="advanced", recently_used_atom_ids={"atom-seen"}
    )
    ranked = apply_pedagogical_weights([(seen, 0.6), (fresh, 0.6)], context, alpha=0.7)
    assert [a.statement for a, _ in ranked] == ["Never shown", "Already shown"]


def test_novelty_weight_repeat_atom_penalty() -> None:
    context = RetrievalPedagogicalContext(recently_used_atom_ids={"atom-1"})
    atom = KnowledgeAtom(scope=ScopeType.CONCEPT, statement="Some fact", id="atom-1")
    assert calculate_novelty_weight(atom, context) == REPEAT_ATOM_PENALTY


def test_novelty_weight_novel_article_boost() -> None:
    context = RetrievalPedagogicalContext(recently_used_article_ids={"article-seen"})
    atom = KnowledgeAtom(
        scope=ScopeType.CONCEPT, statement="Some fact", article_id="article-new"
    )
    assert calculate_novelty_weight(atom, context) == NOVEL_ARTICLE_BOOST


def test_novelty_weight_combines_penalty_and_boost() -> None:
    context = RetrievalPedagogicalContext(
        recently_used_atom_ids={"atom-1"},
        recently_used_article_ids={"article-seen"},
    )
    atom = KnowledgeAtom(
        scope=ScopeType.CONCEPT,
        statement="Some fact",
        id="atom-1",
        article_id="article-new",
    )
    assert calculate_novelty_weight(atom, context) == (
        REPEAT_ATOM_PENALTY * NOVEL_ARTICLE_BOOST
    )


def test_novelty_weight_neutral_when_no_id_or_article_id() -> None:
    context = RetrievalPedagogicalContext(
        recently_used_atom_ids={"atom-1"}, recently_used_article_ids={"article-1"}
    )
    atom = KnowledgeAtom(scope=ScopeType.CONCEPT, statement="Some fact")
    assert calculate_novelty_weight(atom, context) == 1.0


def test_apply_pedagogical_weights_intro_favors_principle_over_instance() -> None:
    principle = KnowledgeAtom(
        scope=ScopeType.CONCEPT, statement="Why it matters", core_relevance_score=0.6
    )
    instance = KnowledgeAtom(
        scope=ScopeType.PRACTICE,
        statement="A specific number",
        core_relevance_score=0.6,
    )
    context = RetrievalPedagogicalContext(depth_level="intro")
    ranked = apply_pedagogical_weights(
        [(instance, 0.6), (principle, 0.6)], context, alpha=0.7
    )
    assert [a.statement for a, _ in ranked] == ["Why it matters", "A specific number"]


def test_apply_pedagogical_weights_practice_favors_instance_over_principle() -> None:
    principle = KnowledgeAtom(
        scope=ScopeType.CONCEPT, statement="Why it matters", core_relevance_score=0.6
    )
    instance = KnowledgeAtom(
        scope=ScopeType.PRACTICE, statement="Edge case detail", core_relevance_score=0.6
    )
    context = RetrievalPedagogicalContext(depth_level="practice")
    ranked = apply_pedagogical_weights(
        [(principle, 0.6), (instance, 0.6)], context, alpha=0.7
    )
    assert [a.statement for a, _ in ranked] == ["Edge case detail", "Why it matters"]


def test_apply_pedagogical_weights_penalizes_recently_used_atom() -> None:
    seen = KnowledgeAtom(
        scope=ScopeType.MECHANIC,
        statement="Already shown",
        core_relevance_score=0.6,
        id="atom-seen",
    )
    fresh = KnowledgeAtom(
        scope=ScopeType.MECHANIC,
        statement="Never shown",
        core_relevance_score=0.6,
        id="atom-fresh",
    )
    context = RetrievalPedagogicalContext(
        depth_level="deep_dive", recently_used_atom_ids={"atom-seen"}
    )
    # Equal query_score and core_relevance_score/scope — only novelty must decide.
    ranked = apply_pedagogical_weights([(seen, 0.6), (fresh, 0.6)], context, alpha=0.7)
    assert [a.statement for a, _ in ranked] == ["Never shown", "Already shown"]


def test_apply_pedagogical_weights_missing_core_score_falls_back_to_neutral() -> None:
    unscored = KnowledgeAtom(scope=ScopeType.MECHANIC, statement="Unscored")
    context = RetrievalPedagogicalContext(depth_level="deep_dive")
    ranked = apply_pedagogical_weights([(unscored, 1.0)], context, alpha=0.7)
    # base = 0.7*1.0 + 0.3*0.5 = 0.85; W_pedagogy(MECHANIC, deep_dive) = 1.35;
    # W_novelty = 1.0 (no id/article_id) -> 0.85 * 1.35 = 1.1475
    assert round(ranked[0][1], 4) == round(0.85 * 1.35, 4)
