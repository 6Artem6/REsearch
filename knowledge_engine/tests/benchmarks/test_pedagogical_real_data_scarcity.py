"""Pedagogical-Aware RAG Reranking — REAL-DATA scarcity stress test.

Built from an actual query against the production Postgres ``knowledge_atoms``
table (see ``tests/fixtures/articles/real_scarcity_pool.json``, pulled via a
one-off read-only query — not fabricated). Two real, unrequested findings
drove the design of this scenario instead of the originally-requested
"PRACTICE deficit":

1. Across the WHOLE table (703 rows) scope is 695 PRACTICE / 8 CONCEPT /
   **0 MECHANIC**. PRACTICE is the opposite of scarce in production — it is
   the overwhelming majority. The real deficit is CONCEPT (~1.1%) and
   MECHANIC (literally absent). No article in the corpus has an
   PRACTICE-scarce profile, so that scenario cannot be built from real data
   without inventing atoms — which would defeat the point of using
   Postgres. This benchmark stress-tests the REAL deficit instead.
2. 0/703 rows carry ``core_relevance_score`` — the Two-Stage Fact Relevance
   field has never been applied to production data. Every real atom here is
   genuinely unscored (``None``), so the hybrid base term is IDENTICAL
   (0.7*0.6 + 0.3*0.5 = 0.57) for every candidate — Baseline's ranking is a
   pure tie broken by fetch order, which is itself an honest finding: today,
   nothing in production differentiates retrieval order beyond that tie.

Fixture pool (20 real atoms, 8 real articles, sourced from real
statements/scopes/ids — nothing invented):
- 8 real CONCEPT atoms across 4 cpython/postgres source-code articles
  (the entire CONCEPT supply in the whole table).
- 12 real PRACTICE atoms, capped at 3 per article across 4 real articles
  (Habr partitioning, kariernik.ru recursive CTE, PostgreSQL GIN docs,
  Yandex sharding) — a deliberately narrow "what this turn's vector search
  actually surfaced" slice of the 695 real PRACTICE atoms available, not
  the whole corpus. practice+advanced+expert need 15 total; only 12 exist
  in this slice — genuine exhaustion by Turn 5, unlike the synthetic
  large-dataset benchmark where PRACTICE supply always had slack.
- 0 MECHANIC atoms — because there are none in production, full stop.

No network/LLM/Cross-Encoder calls (query_score fixed at 0.6, same
rationale as the other pedagogical benchmarks)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from knowledge_engine.src.domains.grounding.pedagogical_reranker import (
    _PEDAGOGY_WEIGHTS,
    apply_pedagogical_weights,
)
from knowledge_engine.src.domains.grounding.schemas import RetrievalPedagogicalContext
from knowledge_engine.src.domains.ingestion.article_ingestion.anchor_relevance import (
    AtomAnchorFilterService,
)
from knowledge_engine.src.shared.extraction import KnowledgeAtom, ScopeType

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "articles"
    / "real_scarcity_pool.json"
)
_QUERY_SCORE = 0.6
_ALPHA = 0.7
_TOP_N = 5


def _load_real_pool() -> List[KnowledgeAtom]:
    with open(_FIXTURE_PATH, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = json.load(f)
    atoms: List[KnowledgeAtom] = []
    for row in raw["principle"]:
        atoms.append(
            KnowledgeAtom(
                scope=ScopeType.CONCEPT,
                statement=row["statement"],
                id=row["id"],
                article_id=row["url"],
                core_relevance_score=None,  # real state: never scored
            )
        )
    for article in raw["instance"]:
        for row in article["atoms"]:
            atoms.append(
                KnowledgeAtom(
                    scope=ScopeType.PRACTICE,
                    statement=row["statement"],
                    id=row["id"],
                    article_id=article["url"],
                    core_relevance_score=None,
                )
            )
    return atoms


@dataclass
class TurnQuery:
    label: str
    depth_level: str


TURNS = [
    TurnQuery("Turn 1 (intro)", "intro"),
    TurnQuery("Turn 2 (deep_dive)", "deep_dive"),
    TurnQuery("Turn 3 (practice)", "practice"),
    TurnQuery("Turn 4 (advanced)", "advanced"),
    TurnQuery("Turn 5 (expert)", "expert"),
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
    """Real prod behavior today: every atom is unscored, so the hybrid
    formula ties for all of them — ranking collapses to fetch order.
    ``calculate_hybrid_fact_score`` itself takes no None-fallback — the
    caller applies it (same convention as dialog_atoms_rag.py's plain
    hybrid branch: missing core_relevance_score -> neutral 0.5)."""
    scored = [
        (
            a,
            AtomAnchorFilterService.calculate_hybrid_fact_score(
                _QUERY_SCORE,
                a.core_relevance_score if a.core_relevance_score is not None else 0.5,
                _ALPHA,
            ),
        )
        for a in pool
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)  # stable: ties keep fetch order
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
        "| turn | CONCEPT | MECHANIC | PRACTICE | overlap_vs_prev% "
        "| Precision@5 | NDCG@5 |"
    )
    print("|---|---|---|---|---|---|---|")
    seen_atom_ids: set = set()
    all_article_ids: set = set()
    precisions: List[float] = []
    ndcgs: List[float] = []
    for turn, top5 in session:
        dist = _scope_distribution(top5)
        current_ids = {a.id for a in top5 if a.id}
        overlap = _overlap_ratio(current_ids, seen_atom_ids)
        all_article_ids |= {a.article_id for a in top5 if a.article_id}
        precision = _precision_at_5(top5, turn.depth_level)
        ndcg = _ndcg_at_5(pool, top5, turn.depth_level, seen_atom_ids)
        precisions.append(precision)
        ndcgs.append(ndcg)
        print(
            f"| {turn.label} | {dist['CONCEPT']} | {dist['MECHANIC']} | "
            f"{dist['PRACTICE']} | {overlap:.0f} | {precision:.2f} | {ndcg:.2f} |"
        )
        seen_atom_ids |= current_ids
    avg_precision = sum(precisions) / len(precisions)
    avg_ndcg = sum(ndcgs) / len(ndcgs)
    print(f"\nArticle Diversity: {len(all_article_ids)}/8 real articles used")
    print(f"Average Precision@5: {avg_precision:.3f}")
    print(f"Average NDCG@5: {avg_ndcg:.3f}")
    return {
        "article_diversity": float(len(all_article_ids)),
        "avg_precision_at_5": avg_precision,
        "avg_ndcg_at_5": avg_ndcg,
    }


def test_pedagogical_real_data_scarcity_report() -> None:
    pool = _load_real_pool()
    assert len(pool) == 20
    assert sum(1 for a in pool if a.scope == ScopeType.CONCEPT) == 8
    assert sum(1 for a in pool if a.scope == ScopeType.PRACTICE) == 12
    assert sum(1 for a in pool if a.scope == ScopeType.MECHANIC) == 0
    assert all(a.core_relevance_score is None for a in pool)

    baseline = _run_baseline_session(pool)
    pedagogical = _run_pedagogical_session(pool)

    baseline_metrics = _print_session_report(
        "Baseline RAG (real unscored prod data — pure tie, fetch order)",
        pool,
        baseline,
    )
    pedagogical_metrics = _print_session_report(
        "Pedagogical RAG (real data, genuine PRACTICE/MECHANIC scarcity)",
        pool,
        pedagogical,
    )
    print(
        "\n### Verdict\n"
        f"Precision@5: baseline={baseline_metrics['avg_precision_at_5']:.3f} "
        f"vs pedagogical={pedagogical_metrics['avg_precision_at_5']:.3f}\n"
        f"NDCG@5:      baseline={baseline_metrics['avg_ndcg_at_5']:.3f} "
        f"vs pedagogical={pedagogical_metrics['avg_ndcg_at_5']:.3f}\n"
        "Article Diversity (/8): baseline="
        f"{baseline_metrics['article_diversity']:.0f} vs pedagogical="
        f"{pedagogical_metrics['article_diversity']:.0f}"
    )

    # --- Turn 2 (deep_dive): target scope MECHANIC has ZERO real supply.
    # The algorithm must not silently fail or return an empty/irrelevant
    # slate — it must gracefully substitute the next-best-weighted scope
    # (PRACTICE, 1.1 > CONCEPT's 0.85 at deep_dive). ---
    turn2_dist = _scope_distribution(pedagogical[1][1])
    assert turn2_dist["MECHANIC"] == 0
    assert turn2_dist["PRACTICE"] >= turn2_dist["CONCEPT"]

    # --- Turn 5 (expert): by now the 12-atom PRACTICE slice is exhausted
    # by practice+advanced (10 consumed), forcing a real fallback choice
    # between recycling used PRACTICE (harsh 0.4x penalty, but expert's
    # PRACTICE weight is the highest of all 5 levels, 1.7) and fresh
    # CONCEPT (novelty-boosted, but expert's CONCEPT weight is the
    # harshest penalty of all 5 levels, 0.3). This is the actual
    # stress-test the small/large synthetic benchmarks could not exercise
    # (they never ran out of PRACTICE facts). We assert the mechanism
    # engaged (some repeat AND/OR fallback to CONCEPT), not a fixed
    # split — that split depends on tie-break order and is reported above,
    # not hard-asserted, since it is the finding itself, not the mechanism.
    turn5_ids = {a.id for a in pedagogical[4][1]}
    turn4_and_earlier_ids = {a.id for _, top5 in pedagogical[:4] for a in top5}
    assert not turn5_ids.isdisjoint(
        turn4_and_earlier_ids
    ), "Turn 5 must be forced to reuse or fall back once the real PRACTICE slice is exhausted"
