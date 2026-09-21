"""Pedagogical-Aware RAG Reranking — 4-turn dialog session A/B benchmark.

No network/LLM/Cross-Encoder calls — per the task's explicit instruction to
mock/fix whatever can be mocked, this benchmark exercises the reranking
algorithm directly (``AtomAnchorFilterService.calculate_hybrid_fact_score``
for Baseline, ``apply_pedagogical_weights`` for Pedagogical) against a
synthetic, hand-scored candidate pool, instead of the full DB-backed
``retrieve_dialog_knowledge_atoms_detailed`` (which would need a mocked
VectorStore/Postgres layer for no added signal on the algorithm itself).
Query relevance (``query_score``) is fixed at 0.6 for every atom on every
turn — the topic stays the same across turns, only ``depth_level`` changes,
so the benchmark isolates the effect of the pedagogical/novelty weights
rather than mixing in Cross-Encoder noise.

Fast, deterministic, runs as a normal pytest file (no opt-in env var needed
— contrast with tests/benchmarks/test_fact_relevance_pipeline.py, which
does make real cloud calls).

Synthetic candidate pool (21 atoms, 7 "articles", 3 atoms each —
CONCEPT/MECHANIC/PRACTICE):
- ``pop-1``/``pop-2`` — core_relevance_score=0.8 ("popular" articles that
  dominate plain similarity search).
- ``alt-1``/``alt-2``/``alt-3`` — core_relevance_score=0.5 (ordinary
  alternative coverage of the same topic).
- ``analog-1``/``analog-2`` — core_relevance_score=0.3 (a different-angle
  "compare X to analogs" source, only worth surfacing once the popular/
  alternative pool has already been shown — Turn 4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from knowledge_engine.src.domains.grounding.pedagogical_reranker import (
    apply_pedagogical_weights,
)
from knowledge_engine.src.domains.grounding.schemas import RetrievalPedagogicalContext
from knowledge_engine.src.domains.ingestion.article_ingestion.anchor_relevance import (
    AtomAnchorFilterService,
)
from knowledge_engine.src.shared.extraction import KnowledgeAtom, ScopeType

_QUERY_SCORE = 0.6  # fixed — same topic every turn, isolates pedagogy/novelty
_ALPHA = 0.7
_TOP_N = 5

_ARTICLE_CORE_SCORE: Dict[str, float] = {
    "pop-1": 0.8,
    "pop-2": 0.8,
    "alt-1": 0.5,
    "alt-2": 0.5,
    "alt-3": 0.5,
    "analog-1": 0.3,
    "analog-2": 0.3,
}


def _build_synthetic_pool() -> List[KnowledgeAtom]:
    atoms: List[KnowledgeAtom] = []
    for article_id, core_score in _ARTICLE_CORE_SCORE.items():
        for scope in (ScopeType.CONCEPT, ScopeType.MECHANIC, ScopeType.PRACTICE):
            atoms.append(
                KnowledgeAtom(
                    scope=scope,
                    statement=f"{article_id} {scope.value} fact about X",
                    id=f"{article_id}-{scope.value}",
                    article_id=article_id,
                    core_relevance_score=core_score,
                )
            )
    return atoms


@dataclass
class TurnQuery:
    label: str
    depth_level: str
    include_analogs: bool = False


TURNS = [
    TurnQuery("Turn 1 (Intro / WHY)", "intro"),
    TurnQuery("Turn 2 (Deep Dive / HOW)", "deep_dive"),
    TurnQuery("Turn 3 (Practice / edge-cases)", "practice"),
    # "Сравни X с аналогами в других статьях" — differently-worded query,
    # realistically pulls a DIFFERENT candidate set from vector search
    # (including analog-* sources the intro/deep_dive/practice queries
    # never surfaced) — not just a depth_level change over the same pool.
    TurnQuery("Turn 4 (Alternative perspective)", "deep_dive", include_analogs=True),
]


def _scope_distribution(atoms: List[KnowledgeAtom]) -> Dict[str, int]:
    dist = {"CONCEPT": 0, "MECHANIC": 0, "PRACTICE": 0}
    for a in atoms:
        dist[a.scope.value] += 1
    return dist


def _overlap_ratio(current: set, previous: set) -> float:
    if not current:
        return 0.0
    return 100.0 * len(current & previous) / len(current)


# --- Independent-ish relevance rubric for Precision@5 / NDCG@5 -------------
# NOT a fully independent ground truth (no human relevance judgments exist
# for a synthetic pool) — but graded by a rubric distinct from the ranking
# formula itself: pure scope-match to the turn's pedagogical target, zeroed
# out for atoms already shown earlier in the SAME session (so an atom that
# scores well on scope but adds no new information gets no credit). This is
# an honest proxy for "did this turn's retrieval move the lesson forward",
# not a claim of ground-truth user relevance.
_TARGET_SCOPE_FOR_DEPTH = {
    "intro": "CONCEPT",
    "deep_dive": "MECHANIC",
    "practice": "PRACTICE",
}


def _relevance_grade(
    atom: KnowledgeAtom, depth_level: str, previously_shown_ids: set
) -> float:
    if atom.id and atom.id in previously_shown_ids:
        return 0.0  # already shown this session — no new pedagogical value
    target = _TARGET_SCOPE_FOR_DEPTH[depth_level]
    if atom.scope.value == target:
        return 2.0
    if atom.scope.value == "MECHANIC":
        return 1.0  # generically useful "how it works" middle ground
    return 0.0


def _dcg(grades: List[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(grades))


def _ndcg_at_5(
    candidates: List[KnowledgeAtom],
    top5: List[KnowledgeAtom],
    depth_level: str,
    previously_shown_ids: set,
) -> float:
    actual = [_relevance_grade(a, depth_level, previously_shown_ids) for a in top5]
    ideal = sorted(
        (_relevance_grade(a, depth_level, previously_shown_ids) for a in candidates),
        reverse=True,
    )[:_TOP_N]
    idcg = _dcg(ideal)
    return (_dcg(actual) / idcg) if idcg > 0 else 1.0


def _precision_at_5(top5: List[KnowledgeAtom], depth_level: str) -> float:
    target = _TARGET_SCOPE_FOR_DEPTH[depth_level]
    return sum(1 for a in top5 if a.scope.value == target) / len(top5)


def _candidates_for_turn(
    pool: List[KnowledgeAtom], turn: TurnQuery
) -> List[KnowledgeAtom]:
    if turn.include_analogs:
        return pool
    return [a for a in pool if not (a.article_id or "").startswith("analog-")]


def _run_baseline_session(
    pool: List[KnowledgeAtom],
) -> List[Tuple[TurnQuery, List[KnowledgeAtom]]]:
    """Baseline RAG: plain hybrid rerank (existing prod behavior) over
    whatever candidates the turn's query pulled — depth_level and session
    history are never consulted, so Turns 1-3 (same candidate pool) collapse
    to the identical selection regardless of the user's actual intent."""
    results: List[Tuple[TurnQuery, List[KnowledgeAtom]]] = []
    for turn in TURNS:
        candidates = _candidates_for_turn(pool, turn)
        scored = [
            (
                a,
                AtomAnchorFilterService.calculate_hybrid_fact_score(
                    _QUERY_SCORE, a.core_relevance_score, _ALPHA
                ),
            )
            for a in candidates
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        results.append((turn, [a for a, _ in scored[:_TOP_N]]))
    return results


def _run_pedagogical_session(
    pool: List[KnowledgeAtom],
) -> List[Tuple[TurnQuery, List[KnowledgeAtom]]]:
    """Pedagogical RAG: depth-of-understanding boost + novelty rotation,
    accumulating recently_used_* across turns (as a real dialog session
    would via SessionMemory)."""
    recently_used_atom_ids: set = set()
    recently_used_article_ids: set = set()
    results: List[Tuple[TurnQuery, List[KnowledgeAtom]]] = []
    for turn in TURNS:
        candidates = _candidates_for_turn(pool, turn)
        context = RetrievalPedagogicalContext(
            depth_level=turn.depth_level,
            recently_used_atom_ids=set(recently_used_atom_ids),
            recently_used_article_ids=set(recently_used_article_ids),
        )
        scored_atoms = [(a, _QUERY_SCORE) for a in candidates]
        ranked = apply_pedagogical_weights(scored_atoms, context, alpha=_ALPHA)
        top5 = [a for a, _ in ranked[:_TOP_N]]
        results.append((turn, top5))
        recently_used_atom_ids.update(a.id for a in top5 if a.id)
        recently_used_article_ids.update(a.article_id for a in top5 if a.article_id)
    return results


def _print_session_report(
    label: str,
    pool: List[KnowledgeAtom],
    session: List[Tuple[TurnQuery, List[KnowledgeAtom]]],
) -> Dict[str, float]:
    print(f"\n### {label}\n")
    print(
        "| turn | CONCEPT | MECHANIC | PRACTICE | articles | overlap_vs_prev% "
        "| Precision@5 | NDCG@5 |"
    )
    print("|---|---|---|---|---|---|---|---|")
    seen_atom_ids: set = set()
    all_article_ids: set = set()
    precisions: List[float] = []
    ndcgs: List[float] = []
    for turn, top5 in session:
        dist = _scope_distribution(top5)
        current_ids = {a.id for a in top5 if a.id}
        overlap = _overlap_ratio(current_ids, seen_atom_ids)
        article_ids = {a.article_id for a in top5 if a.article_id}
        all_article_ids |= article_ids
        candidates = _candidates_for_turn(pool, turn)
        precision = _precision_at_5(top5, turn.depth_level)
        ndcg = _ndcg_at_5(candidates, top5, turn.depth_level, seen_atom_ids)
        precisions.append(precision)
        ndcgs.append(ndcg)
        print(
            f"| {turn.label} | {dist['CONCEPT']} | {dist['MECHANIC']} | "
            f"{dist['PRACTICE']} | {sorted(article_ids)} | {overlap:.0f} "
            f"| {precision:.2f} | {ndcg:.2f} |"
        )
        seen_atom_ids |= current_ids
    avg_precision = sum(precisions) / len(precisions)
    avg_ndcg = sum(ndcgs) / len(ndcgs)
    print(f"\nArticle Diversity (unique across 4 turns): {len(all_article_ids)}")
    print(f"Average Precision@5 (scope-match): {avg_precision:.3f}")
    print(f"Average NDCG@5 (scope-match + novelty-graded): {avg_ndcg:.3f}")
    return {
        "article_diversity": float(len(all_article_ids)),
        "avg_precision_at_5": avg_precision,
        "avg_ndcg_at_5": avg_ndcg,
    }


def test_pedagogical_rag_session_report() -> None:
    pool = _build_synthetic_pool()
    baseline = _run_baseline_session(pool)
    pedagogical = _run_pedagogical_session(pool)

    baseline_metrics = _print_session_report(
        "Baseline RAG (plain hybrid, no pedagogy/novelty)", pool, baseline
    )
    pedagogical_metrics = _print_session_report(
        "Pedagogical RAG (depth + novelty)", pool, pedagogical
    )
    print(
        "\n### Verdict\n"
        f"Precision@5: baseline={baseline_metrics['avg_precision_at_5']:.3f} "
        f"vs pedagogical={pedagogical_metrics['avg_precision_at_5']:.3f}\n"
        f"NDCG@5:      baseline={baseline_metrics['avg_ndcg_at_5']:.3f} "
        f"vs pedagogical={pedagogical_metrics['avg_ndcg_at_5']:.3f}\n"
        "Article Diversity: baseline="
        f"{baseline_metrics['article_diversity']:.0f} vs pedagogical="
        f"{pedagogical_metrics['article_diversity']:.0f}"
    )

    # --- Baseline demonstrates the reported problem: lock-in on 1-2 popular
    # articles, identical selection turn after turn regardless of depth. ---
    baseline_article_ids = {a.article_id for _, top5 in baseline for a in top5}
    assert baseline_article_ids == {"pop-1", "pop-2"}
    for _, top5 in baseline[1:]:
        assert {a.id for a in top5} == {a.id for a in baseline[0][1]}

    # --- Pedagogical: Turn 1 (intro) must skew CONCEPT. ---
    turn1_dist = _scope_distribution(pedagogical[0][1])
    assert turn1_dist["CONCEPT"] == 5

    # --- Turn 2 (deep_dive) must skew MECHANIC and NOT repeat Turn 1's atoms. ---
    turn2_dist = _scope_distribution(pedagogical[1][1])
    assert turn2_dist["MECHANIC"] == 5
    turn1_ids = {a.id for a in pedagogical[0][1]}
    turn2_ids = {a.id for a in pedagogical[1][1]}
    assert turn1_ids.isdisjoint(turn2_ids)

    # --- Turn 3 (practice) must skew PRACTICE and not repeat Turns 1-2. ---
    turn3_dist = _scope_distribution(pedagogical[2][1])
    assert turn3_dist["PRACTICE"] == 5
    turn3_ids = {a.id for a in pedagogical[2][1]}
    assert turn3_ids.isdisjoint(turn1_ids | turn2_ids)

    # --- Turn 4 ("alternative perspective"): by now every pop/alt atom has
    # been shown once, so novelty rotation must surface the analog articles
    # that never won a slot in Turns 1-3 (0% overlap with all prior turns,
    # brand-new article_ids). ---
    turn4_article_ids = {a.article_id for a in pedagogical[3][1]}
    assert turn4_article_ids == {"analog-1", "analog-2"}
    turn4_ids = {a.id for a in pedagogical[3][1]}
    assert turn4_ids.isdisjoint(turn1_ids | turn2_ids | turn3_ids)

    # --- Pedagogical must beat Baseline on the two headline metrics. ---
    pedagogical_article_ids = {a.article_id for _, top5 in pedagogical for a in top5}
    assert len(pedagogical_article_ids) > len(baseline_article_ids)

    def _avg_overlap(session: List[Tuple[TurnQuery, List[KnowledgeAtom]]]) -> float:
        seen: set = set()
        ratios: List[float] = []
        for _, top5 in session:
            current = {a.id for a in top5 if a.id}
            ratios.append(_overlap_ratio(current, seen))
            seen |= current
        return sum(ratios) / len(ratios)

    assert _avg_overlap(pedagogical) < _avg_overlap(baseline)

    # --- Relevance metrics (Precision@5 / NDCG@5, see rubric docstring
    # above) must also favor Pedagogical — not just diversity/overlap. ---
    assert (
        pedagogical_metrics["avg_precision_at_5"]
        > baseline_metrics["avg_precision_at_5"]
    )
    assert pedagogical_metrics["avg_ndcg_at_5"] > baseline_metrics["avg_ndcg_at_5"]
