"""Pedagogical-Aware RAG Reranking — large-dataset 5-turn session benchmark.

Scale-up of tests/benchmarks/test_pedagogical_rag_session.py (4 turns, 21
atoms / 7 articles) to 22 articles / 66 atoms and the full 5-level depth
ladder (intro -> deep_dive -> practice -> advanced -> expert), covering the
two Star Task overlay levels (ADVANCED_ASTERISK / DEEP_ASTERISK) added to
pedagogical_reranker.py.

No network/LLM/Cross-Encoder calls — same rationale as the smaller
benchmark: query relevance is fixed (0.6) for every atom on every turn, so
the benchmark isolates the pedagogy/novelty weights rather than mixing in
Cross-Encoder noise. Fast, deterministic, runs as a normal pytest file.

Relevance rubric for Precision@5/NDCG@5 — generalized to 5 levels by
ranking each depth_level's own row in the PRODUCTION weight table
(``pedagogical_reranker._PEDAGOGY_WEIGHTS``, not a second hand-copied
table): the highest-weighted scope for that depth_level grades 2, the
middle one 1, the lowest 0, and any atom already shown earlier in the
session grades 0 regardless of scope. This is graded by a rubric distinct
from the ranking formula itself, but still derived from the same intended
weight ordering — an honest proxy for "did this turn advance the lesson",
not an independent human-judged ground truth (see the 4-turn benchmark's
verdict section for the same caveat, which applies here unchanged).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from knowledge_engine.src.domains.grounding.pedagogical_reranker import (
    _PEDAGOGY_WEIGHTS,
    apply_pedagogical_weights,
)
from knowledge_engine.src.domains.grounding.schemas import RetrievalPedagogicalContext
from knowledge_engine.src.domains.ingestion.article_ingestion.anchor_relevance import (
    AtomAnchorFilterService,
)
from knowledge_engine.src.shared.extraction import KnowledgeAtom, ScopeType

_QUERY_SCORE = 0.6
_ALPHA = 0.7
_TOP_N = 5
_NUM_ARTICLES = 22  # >= 20 required; x3 scopes = 66 atoms >= 60 required


def _build_large_pool() -> List[KnowledgeAtom]:
    atoms: List[KnowledgeAtom] = []
    for i in range(_NUM_ARTICLES):
        article_id = f"article-{i + 1:02d}"
        # Deterministic spread 0.90 -> ~0.48 — a handful of "popular"
        # articles dominate plain hybrid ranking, just like real
        # core_relevance_score distributions skew toward a few sources.
        core_score = round(0.9 - 0.02 * i, 3)
        for scope in (ScopeType.CONCEPT, ScopeType.MECHANIC, ScopeType.PRACTICE):
            atoms.append(
                KnowledgeAtom(
                    scope=scope,
                    statement=(
                        f"{article_id} {scope.value} fact about DB partitioning"
                    ),
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
    query_text: str


TURNS = [
    TurnQuery("Turn 1 (intro)", "intro", "Зачем нужно партиционирование баз данных?"),
    TurnQuery(
        "Turn 2 (deep_dive)",
        "deep_dive",
        "Как устроены секционированные индексы под капотом?",
    ),
    TurnQuery(
        "Turn 3 (practice)",
        "practice",
        "Примеры применения и конфигурации секций в Postgres",
    ),
    TurnQuery(
        "Turn 4 (advanced)",
        "advanced",
        "Подводные камни и блокировки при DDL на секциях",
    ),
    TurnQuery(
        "Turn 5 (expert)",
        "expert",
        "Edge-cases с деградацией производительности planner и усечением секций",
    ),
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


def _scope_rank_grades(depth_level: str) -> Dict[str, float]:
    """Rank the PRODUCTION weight table's own row for this depth_level:
    highest weight -> 2, middle -> 1, lowest -> 0. Reused from
    pedagogical_reranker._PEDAGOGY_WEIGHTS so the rubric can never silently
    drift from the real coefficients."""
    weights = _PEDAGOGY_WEIGHTS[depth_level]
    ordered = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    return {scope.value: float(2 - i) for i, (scope, _) in enumerate(ordered)}


def _relevance_grade(
    atom: KnowledgeAtom, depth_level: str, previously_shown_ids: set
) -> float:
    if atom.id and atom.id in previously_shown_ids:
        return 0.0
    return _scope_rank_grades(depth_level).get(atom.scope.value, 0.0)


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
    rank = _scope_rank_grades(depth_level)
    return sum(1 for a in top5 if rank.get(a.scope.value) == 2.0) / len(top5)


def _run_baseline_session(
    pool: List[KnowledgeAtom],
) -> List[Tuple[TurnQuery, List[KnowledgeAtom]]]:
    """Plain hybrid rerank (existing prod behavior) — identical every turn,
    the full 66-atom pool never changes and depth_level is never consulted."""
    scored = [
        (
            a,
            AtomAnchorFilterService.calculate_hybrid_fact_score(
                _QUERY_SCORE, a.core_relevance_score, _ALPHA
            ),
        )
        for a in pool
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    top5 = [a for a, _ in scored[:_TOP_N]]
    return [(turn, top5) for turn in TURNS]


def _run_pedagogical_session(
    pool: List[KnowledgeAtom],
) -> List[Tuple[TurnQuery, List[KnowledgeAtom]]]:
    recently_used_atom_ids: set = set()
    recently_used_article_ids: set = set()
    results: List[Tuple[TurnQuery, List[KnowledgeAtom]]] = []
    for turn in TURNS:
        context = RetrievalPedagogicalContext(
            depth_level=turn.depth_level,
            recently_used_atom_ids=set(recently_used_atom_ids),
            recently_used_article_ids=set(recently_used_article_ids),
        )
        scored_atoms = [(a, _QUERY_SCORE) for a in pool]
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
        "| turn | CONCEPT | MECHANIC | PRACTICE | n_articles | overlap_vs_prev% "
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
        precision = _precision_at_5(top5, turn.depth_level)
        ndcg = _ndcg_at_5(pool, top5, turn.depth_level, seen_atom_ids)
        precisions.append(precision)
        ndcgs.append(ndcg)
        print(
            f"| {turn.label} | {dist['CONCEPT']} | {dist['MECHANIC']} | "
            f"{dist['PRACTICE']} | {len(article_ids)} | {overlap:.0f} "
            f"| {precision:.2f} | {ndcg:.2f} |"
        )
        seen_atom_ids |= current_ids
    avg_precision = sum(precisions) / len(precisions)
    avg_ndcg = sum(ndcgs) / len(ndcgs)
    diversity_pct = 100.0 * len(all_article_ids) / _NUM_ARTICLES
    print(
        f"\nArticle Diversity: {len(all_article_ids)}/{_NUM_ARTICLES} articles "
        f"used across 5 turns ({diversity_pct:.1f}%)"
    )
    print(f"Average Precision@5: {avg_precision:.3f}")
    print(f"Average NDCG@5: {avg_ndcg:.3f}")
    return {
        "article_diversity_pct": diversity_pct,
        "avg_precision_at_5": avg_precision,
        "avg_ndcg_at_5": avg_ndcg,
    }


def test_pedagogical_large_dataset_session_report() -> None:
    pool = _build_large_pool()
    assert len(pool) >= 60
    assert len({a.article_id for a in pool}) >= 20

    baseline = _run_baseline_session(pool)
    pedagogical = _run_pedagogical_session(pool)

    baseline_metrics = _print_session_report(
        "Baseline RAG (plain hybrid, 22 articles / 66 atoms)", pool, baseline
    )
    pedagogical_metrics = _print_session_report(
        "Pedagogical RAG (5-level depth ladder + novelty)", pool, pedagogical
    )
    print(
        "\n### Verdict\n"
        f"Precision@5: baseline={baseline_metrics['avg_precision_at_5']:.3f} "
        f"vs pedagogical={pedagogical_metrics['avg_precision_at_5']:.3f}\n"
        f"NDCG@5:      baseline={baseline_metrics['avg_ndcg_at_5']:.3f} "
        f"vs pedagogical={pedagogical_metrics['avg_ndcg_at_5']:.3f}\n"
        "Article Diversity: baseline="
        f"{baseline_metrics['article_diversity_pct']:.1f}% vs pedagogical="
        f"{pedagogical_metrics['article_diversity_pct']:.1f}%"
    )

    # --- Baseline: lock-in on a small handful of high-core-score articles,
    # identical selection on every one of the 5 turns. ---
    baseline_article_ids = {a.article_id for _, top5 in baseline for a in top5}
    assert len(baseline_article_ids) <= 2
    for _, top5 in baseline[1:]:
        assert {a.id for a in top5} == {a.id for a in baseline[0][1]}

    # --- Pedagogical: scope gradient across the 5-level ladder. Target
    # scope (grade-2 in the rubric) must dominate each turn's Top-5. ---
    expected_target_scope = {
        "intro": "CONCEPT",
        "deep_dive": "MECHANIC",
        "practice": "PRACTICE",
        "advanced": "PRACTICE",
        "expert": "PRACTICE",
    }
    prior_ids: set = set()
    for turn, top5 in pedagogical:
        dist = _scope_distribution(top5)
        target = expected_target_scope[turn.depth_level]
        assert dist[target] == max(dist.values())
        current_ids = {a.id for a in top5}
        # 0% overlap with everything shown in a strictly earlier turn.
        assert current_ids.isdisjoint(prior_ids)
        prior_ids |= current_ids

    # --- Pedagogical must beat Baseline on diversity and both relevance
    # metrics at this larger scale too, not just on the small 7-article demo.
    assert (
        pedagogical_metrics["article_diversity_pct"]
        > baseline_metrics["article_diversity_pct"]
    )
    assert (
        pedagogical_metrics["avg_precision_at_5"]
        > baseline_metrics["avg_precision_at_5"]
    )
    assert pedagogical_metrics["avg_ndcg_at_5"] > baseline_metrics["avg_ndcg_at_5"]
