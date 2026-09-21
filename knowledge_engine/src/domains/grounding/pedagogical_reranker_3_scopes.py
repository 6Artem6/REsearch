"""Pedagogical-Aware RAG Reranking.

Problem: plain semantic-similarity retrieval stalls on the same 1-2 popular
articles across a multi-turn dialog session and never advances the learner
from WHY (PRINCIPLE) to HOW (MECHANIC) to edge-cases/details (INSTANCE).

Isolated, add-only: gated by ``DIALOG_ATOMS_PEDAGOGICAL_BOOST_ENABLED``
(default off) in ``dialog_atoms_rag.py``. Reuses the existing hybrid
(query + core_relevance_score) formula from ``anchor_relevance.py`` as its
base term — this module only adds the two extra multipliers on top. See
``tests/benchmarks/test_pedagogical_rag_session.py`` for the 4-turn A/B
session benchmark this was validated against.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from knowledge_engine.src.domains.grounding.memory_schemas import OverlayType
from knowledge_engine.src.domains.grounding.schemas import (
    DepthLevel,
    RetrievalPedagogicalContext,
)
from knowledge_engine.src.domains.ingestion.article_ingestion.anchor_relevance import (
    AtomAnchorFilterService,
)
from knowledge_engine.src.shared.extraction_3_scopes import KnowledgeAtom, ScopeType

# W_pedagogy — depth-of-understanding boost/penalty per scope.
_PEDAGOGY_WEIGHTS: Dict[DepthLevel, Dict[ScopeType, float]] = {
    "intro": {
        ScopeType.PRINCIPLE: 1.3,
        ScopeType.MECHANIC: 1.0,
        ScopeType.INSTANCE: 0.7,
    },
    "deep_dive": {
        ScopeType.MECHANIC: 1.35,
        ScopeType.PRINCIPLE: 0.85,
        ScopeType.INSTANCE: 1.1,
    },
    "practice": {
        ScopeType.INSTANCE: 1.4,
        ScopeType.MECHANIC: 1.2,
        ScopeType.PRINCIPLE: 0.7,
    },
    # Star Task overlay levels (ADVANCED_ASTERISK / DEEP_ASTERISK) — same
    # INSTANCE-leaning shape as "practice" but pushed further: PRINCIPLE is
    # actively penalized (the learner already has the fundamentals; restating
    # them is noise), MECHANIC stays a solid secondary pick.
    "advanced": {
        ScopeType.INSTANCE: 1.5,
        ScopeType.MECHANIC: 1.3,
        ScopeType.PRINCIPLE: 0.5,
    },
    "expert": {
        ScopeType.INSTANCE: 1.7,
        ScopeType.MECHANIC: 1.1,
        ScopeType.PRINCIPLE: 0.3,
    },
}

# W_novelty — anti-stagnation: hard rollback of already-shown atoms, boost
# for facts from articles not yet surfaced this session.
REPEAT_ATOM_PENALTY = 0.4
NOVEL_ARTICLE_BOOST = 1.25


def calculate_pedagogy_weight(scope: ScopeType, depth_level: DepthLevel) -> float:
    return _PEDAGOGY_WEIGHTS.get(depth_level, {}).get(scope, 1.0)


_OVERLAY_TYPE_TO_DEPTH_LEVEL: Dict[OverlayType, DepthLevel] = {
    "ADVANCED_ASTERISK": "advanced",
    "DEEP_ASTERISK": "expert",
}


def depth_level_from_overlay_type(overlay_type: OverlayType) -> DepthLevel:
    """Tutor stage mapping: Star Task overlay (``star_task_fsm.py``,
    ``memory_schemas.OverlayType``) → Pedagogical Reranker ``depth_level``.
    ``ADVANCED_ASTERISK`` (Bloom L4, targeted weakness) → ``"advanced"``;
    ``DEEP_ASTERISK`` (L5/L6, clean-core deep design) → ``"expert"``."""
    return _OVERLAY_TYPE_TO_DEPTH_LEVEL[overlay_type]


def calculate_novelty_weight(
    atom: KnowledgeAtom, context: RetrievalPedagogicalContext
) -> float:
    weight = 1.0
    if atom.id is not None and atom.id in context.recently_used_atom_ids:
        weight *= REPEAT_ATOM_PENALTY
    if (
        atom.article_id is not None
        and atom.article_id not in context.recently_used_article_ids
    ):
        weight *= NOVEL_ARTICLE_BOOST
    return weight


def apply_pedagogical_weights(
    scored_atoms: List[Tuple[KnowledgeAtom, float]],
    context: RetrievalPedagogicalContext,
    *,
    alpha: float = 0.7,
) -> List[Tuple[KnowledgeAtom, float]]:
    """
    final_score = (alpha*query_score + (1-alpha)*core_relevance_score)
                  * W_pedagogy(scope, depth_level) * W_novelty(atom, context)

    ``scored_atoms`` is ``(atom, query_score)`` — the dynamic Cross-Encoder
    score against the current turn's query, computed by the caller.
    Missing ``core_relevance_score`` falls back to a neutral 0.5 — a
    production caller cannot tell "unscored" from "genuinely average"
    (same convention as ``dialog_atoms_rag.py``'s plain hybrid rerank).

    Returns atoms sorted by ``final_score`` descending.
    """
    ranked: List[Tuple[KnowledgeAtom, float]] = []
    for atom, query_score in scored_atoms:
        core = (
            atom.core_relevance_score if atom.core_relevance_score is not None else 0.5
        )
        base = AtomAnchorFilterService.calculate_hybrid_fact_score(
            query_score, core, alpha
        )
        scope = (
            atom.scope if isinstance(atom.scope, ScopeType) else ScopeType(atom.scope)
        )
        w_pedagogy = calculate_pedagogy_weight(scope, context.depth_level)
        w_novelty = calculate_novelty_weight(atom, context)
        ranked.append((atom, base * w_pedagogy * w_novelty))
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked
