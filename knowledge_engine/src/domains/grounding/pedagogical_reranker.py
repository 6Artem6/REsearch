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

from typing import Dict, List, Literal, Tuple

from knowledge_engine.src.domains.grounding.memory_schemas import OverlayType
from knowledge_engine.src.domains.grounding.schemas import (
    DepthLevel,
    RetrievalPedagogicalContext,
)
from knowledge_engine.src.domains.ingestion.article_ingestion.anchor_relevance import (
    AtomAnchorFilterService,
)
from knowledge_engine.src.shared.extraction import KnowledgeAtom, ScopeType

# W_pedagogy — depth-of-understanding boost/penalty per scope (5 categories).
# intro/deep_dive prioritize CONCEPT/MECHANIC (learner is still building
# fundamentals); practice/advanced/expert progressively raise EDGE_CASE and
# ANTI_PATTERN (a learner ready for star-task-level depth benefits more from
# "here's the trap" / "here's what breaks" than from restated basics).
_PEDAGOGY_WEIGHTS: Dict[DepthLevel, Dict[ScopeType, float]] = {
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
    # Star Task overlay levels (ADVANCED_ASTERISK / DEEP_ASTERISK) — CONCEPT
    # is actively penalized (the learner already has the fundamentals;
    # restating them is noise), EDGE_CASE/ANTI_PATTERN rise the most since
    # boundary behavior and known traps are exactly the "star task" content.
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

# W_novelty — anti-stagnation: hard rollback of already-shown atoms, boost
# for facts from articles not yet surfaced this session.
REPEAT_ATOM_PENALTY = 0.4
NOVEL_ARTICLE_BOOST = 1.25

# Mastery-Guided Concept Affinity (gated by DIALOG_ATOMS_CONCEPT_AFFINITY_ENABLED).
# Repulsion = atom semantically duplicates an already-MASTERED concept (Mastery
# Gate in socratic_poles.py::mastery_gate_filter already ensures repulsion_claims
# only contain verified/passed concepts — never unassimilated ones).
CONCEPT_AFFINITY_REPULSION_THRESHOLD = 0.82
CONCEPT_AFFINITY_REPULSION_PENALTY = 0.45
# Attraction = atom addresses a known gap/bottleneck for this learner.
CONCEPT_AFFINITY_ATTRACTION_THRESHOLD = 0.75
CONCEPT_AFFINITY_ATTRACTION_BOOST = 1.35
# Repulsion Floor: a concept-affinity penalty alone can never sink an atom
# below this fraction of its pre-affinity hybrid base score.
REPULSION_FLOOR_RATIO = 0.35
# Relevance Fallback: if concept-affinity manipulation starves every
# candidate below this floor, the turn reverts to the bare hybrid base
# (w_pedagogy / w_novelty / w_concept_affinity / w_multiperspective_shift
# all dropped) rather than surface a confidently-wrong top pick.
RELEVANCE_FALLBACK_THRESHOLD = 0.30

# W_multiperspective_shift — response_depth_signal-driven angle change on a
# repeat visit to the same/adjacent subtopic (see derive_response_depth_signal).
_MULTIPERSPECTIVE_LOW_DEPTH_BOOST: Dict[ScopeType, float] = {
    ScopeType.PRACTICE: 1.25,
    ScopeType.CONCEPT: 1.15,
}
_MULTIPERSPECTIVE_HIGH_DEPTH_BOOST: Dict[ScopeType, float] = {
    ScopeType.MECHANIC: 1.25,
    ScopeType.EDGE_CASE: 1.35,
    ScopeType.ANTI_PATTERN: 1.30,
}


def calculate_pedagogy_weight(scope: ScopeType, depth_level: DepthLevel) -> float:
    return _PEDAGOGY_WEIGHTS.get(depth_level, {}).get(scope, 1.0)


def derive_response_depth_signal(
    why_passed: bool, how_passed: bool, mechanic_passed: bool
) -> Literal["low", "high"] | None:
    """Multiperspective Shift signal from the last-evaluated SubConceptRecord's
    persisted layer flags (no new evaluator call needed — see
    sub_concept_evaluator.py's PROBE_NEXT_LAYER directive derivation, which
    this mirrors).

    "low": WHY itself is not yet passed — learner is still at the surface.
    "high": WHY+HOW passed but MECHANIC is not — learner engages deeply but
    is stuck on the nuance/mechanics layer.
    None: any other combination (e.g. all three passed, or no signal yet) —
    neutral, no shift applied.
    """
    if not why_passed:
        return "low"
    if why_passed and how_passed and not mechanic_passed:
        return "high"
    return None


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    import math

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def calculate_concept_affinity_weight(
    atom_embedding: List[float] | None,
    repulsion_embeddings: List[List[float]],
    attraction_embeddings: List[List[float]],
) -> float:
    """Cosine-similarity concept affinity multiplier.

    Takes precomputed embedding VECTORS (not text) so the caller can batch-embed
    once per turn via the BGE-M3 singleton (``socratic_poles.py::_embeddings()``)
    — embedding per-atom-per-claim inside a per-atom loop would be O(n_atoms *
    n_claims) separate model calls, unusable at per-turn latency. See
    ``apply_pedagogical_weights`` for the batching call site.

    Repulsion and attraction are independent checks — an atom that clears both
    thresholds gets both effects multiplied (unlikely in practice, since a
    repulsion claim and an attraction claim about the same statement would
    themselves be near-duplicates, but not excluded by design).

    Missing ``atom_embedding`` (embed call failed for this turn) is neutral
    (1.0) — a failed embedding must never silently penalize an atom.
    """
    if atom_embedding is None:
        return 1.0
    weight = 1.0
    if repulsion_embeddings:
        best = max(_cosine_similarity(atom_embedding, e) for e in repulsion_embeddings)
        if best >= CONCEPT_AFFINITY_REPULSION_THRESHOLD:
            weight *= CONCEPT_AFFINITY_REPULSION_PENALTY
    if attraction_embeddings:
        best = max(_cosine_similarity(atom_embedding, e) for e in attraction_embeddings)
        if best >= CONCEPT_AFFINITY_ATTRACTION_THRESHOLD:
            weight *= CONCEPT_AFFINITY_ATTRACTION_BOOST
    return weight


def calculate_multiperspective_shift_weight(
    scope: ScopeType, response_depth_signal: Literal["low", "high"] | None
) -> float:
    """Boost the scope classes matching the learner's current angle of
    attack (see derive_response_depth_signal). Neutral (1.0) when signal is
    None or the scope has no boost defined at this signal."""
    if response_depth_signal == "low":
        return _MULTIPERSPECTIVE_LOW_DEPTH_BOOST.get(scope, 1.0)
    if response_depth_signal == "high":
        return _MULTIPERSPECTIVE_HIGH_DEPTH_BOOST.get(scope, 1.0)
    return 1.0


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


def _batch_embed_for_affinity(
    scored_atoms: List[Tuple[KnowledgeAtom, float]],
    context: RetrievalPedagogicalContext,
) -> tuple[List[List[float] | None], List[List[float]], List[List[float]]]:
    """One batch embed call each for atoms / repulsion_claims / attraction_claims
    via the BGE-M3 singleton — never per-atom. Any embed failure degrades to
    neutral (all None / empty), never raises into the retrieval hot path."""
    n = len(scored_atoms)
    if not context.repulsion_claims and not context.attraction_claims:
        return [None] * n, [], []
    try:
        from knowledge_engine.src.domains.grounding.socratic_poles import _embeddings

        embedder = _embeddings()
        atom_embeddings: List[List[float] | None] = (
            list(embedder.embed_documents([atom.statement for atom, _ in scored_atoms]))
            if n
            else []
        )
        repulsion_embeddings = (
            embedder.embed_documents(context.repulsion_claims)
            if context.repulsion_claims
            else []
        )
        attraction_embeddings = (
            embedder.embed_documents(context.attraction_claims)
            if context.attraction_claims
            else []
        )
        return atom_embeddings, repulsion_embeddings, attraction_embeddings
    except Exception:
        return [None] * n, [], []


def apply_pedagogical_weights(
    scored_atoms: List[Tuple[KnowledgeAtom, float]],
    context: RetrievalPedagogicalContext,
    *,
    alpha: float = 0.7,
) -> List[Tuple[KnowledgeAtom, float]]:
    """
    final_score = (alpha*query_score + (1-alpha)*core_relevance_score)
                  * W_pedagogy(scope, depth_level) * W_novelty(atom, context)
                  [* W_concept_affinity * W_multiperspective_shift, when
                   DIALOG_ATOMS_CONCEPT_AFFINITY_ENABLED]

    ``scored_atoms`` is ``(atom, query_score)`` — the dynamic Cross-Encoder
    score against the current turn's query, computed by the caller.
    Missing ``core_relevance_score`` falls back to a neutral 0.5 — a
    production caller cannot tell "unscored" from "genuinely average"
    (same convention as ``dialog_atoms_rag.py``'s plain hybrid rerank).

    Mastery-Guided Concept Affinity (add-only, gated by
    ``DIALOG_ATOMS_CONCEPT_AFFINITY_ENABLED``): applies W_concept_affinity
    (repulsion penalty for atoms duplicating an already-mastered concept,
    attraction boost for atoms addressing an open gap — see
    ``calculate_concept_affinity_weight``) and W_multiperspective_shift
    (angle-of-attack boost from ``context.response_depth_signal`` — see
    ``calculate_multiperspective_shift_weight``) on top of the existing
    formula. Two safety mechanics guard this:
    - Repulsion Floor: a repulsion penalty alone can never sink an atom
      below ``REPULSION_FLOOR_RATIO`` of its pre-affinity hybrid ``base``.
    - Relevance Fallback: if every candidate's final score still falls
      below ``RELEVANCE_FALLBACK_THRESHOLD`` after all weights, the whole
      turn reverts to the bare hybrid ``base`` (w_pedagogy / w_novelty /
      w_concept_affinity / w_multiperspective_shift all dropped) — a
      confidently-reordered-but-irrelevant top-k is worse than a plain
      relevance-ranked one.

    Returns atoms sorted by ``final_score`` descending.
    """
    from knowledge_engine.src.config.settings import (
        DIALOG_ATOMS_CONCEPT_AFFINITY_ENABLED,
    )

    concept_affinity_on = DIALOG_ATOMS_CONCEPT_AFFINITY_ENABLED and (
        bool(context.repulsion_claims)
        or bool(context.attraction_claims)
        or context.response_depth_signal is not None
    )

    atom_embeddings: List[List[float] | None] = [None] * len(scored_atoms)
    repulsion_embeddings: List[List[float]] = []
    attraction_embeddings: List[List[float]] = []
    if concept_affinity_on:
        atom_embeddings, repulsion_embeddings, attraction_embeddings = (
            _batch_embed_for_affinity(scored_atoms, context)
        )

    ranked: List[Tuple[KnowledgeAtom, float]] = []
    bare_bases: List[Tuple[KnowledgeAtom, float]] = []
    for idx, (atom, query_score) in enumerate(scored_atoms):
        core = (
            atom.core_relevance_score if atom.core_relevance_score is not None else 0.5
        )
        base = AtomAnchorFilterService.calculate_hybrid_fact_score(
            query_score, core, alpha
        )
        bare_bases.append((atom, base))
        scope = (
            atom.scope if isinstance(atom.scope, ScopeType) else ScopeType(atom.scope)
        )
        w_pedagogy = calculate_pedagogy_weight(scope, context.depth_level)
        w_novelty = calculate_novelty_weight(atom, context)
        final = base * w_pedagogy * w_novelty

        if concept_affinity_on:
            w_affinity = calculate_concept_affinity_weight(
                atom_embeddings[idx], repulsion_embeddings, attraction_embeddings
            )
            w_shift = calculate_multiperspective_shift_weight(
                scope, context.response_depth_signal
            )
            final = final * w_affinity * w_shift
            if w_affinity < 1.0:
                final = max(final, REPULSION_FLOOR_RATIO * base)

        ranked.append((atom, final))

    if (
        concept_affinity_on
        and ranked
        and max(score for _, score in ranked) < RELEVANCE_FALLBACK_THRESHOLD
    ):
        ranked = bare_bases

    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked
