"""Hybrid academic candidate scoring + query/constraint relaxation cascade.

score = α·relevance_sim + β·trust_score + γ·normalized_citations + δ·recency_factor
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Callable, Sequence, TypeVar

from knowledge_engine.config import (
    ACADEMIC_RELAX_L0_MIN_CITATIONS,
    ACADEMIC_RELAX_L0_MIN_TRUST,
    ACADEMIC_RELAX_L1_MIN_CITATIONS,
    ACADEMIC_RELAX_L1_MIN_TRUST,
    ACADEMIC_RELAX_L1_YEAR_PAD,
    ACADEMIC_RELAXATION_ENABLED,
    ACADEMIC_RELAXATION_MIN_HITS,
    ACADEMIC_RERANK_C_SAT,
    ACADEMIC_RERANK_ENABLED,
    ACADEMIC_RERANK_RECENCY_HALF_LIFE_YEARS,
    ACADEMIC_RERANK_WEIGHTS,
)
from knowledge_engine.services.search.arxiv_query_builder import ArxivQueryParams
from knowledge_engine.ui.run_log import trace

T = TypeVar("T")


class RelaxationLevel(IntEnum):
    """Cascading constraint softening when candidate pool is thin."""

    STRICT = 0
    SOFT_DATE_CITATIONS = 1
    BROAD_RELEVANCE = 2


@dataclass(frozen=True)
class AcademicRerankWeights:
    alpha: float = 0.45  # relevance
    beta: float = 0.25  # trust
    gamma: float = 0.20  # citations
    delta: float = 0.10  # recency

    def normalized(self) -> AcademicRerankWeights:
        total = self.alpha + self.beta + self.gamma + self.delta
        if total <= 0:
            return AcademicRerankWeights()
        return AcademicRerankWeights(
            alpha=self.alpha / total,
            beta=self.beta / total,
            gamma=self.gamma / total,
            delta=self.delta / total,
        )


@dataclass(frozen=True)
class RerankSignals:
    relevance_sim: float = 0.0
    trust_score: float = 1.0
    citation_count: int = 0
    year: int | None = None


@dataclass(frozen=True)
class RelaxationPolicy:
    level: RelaxationLevel
    min_trust: float
    min_citations: int
    drop_date_filter: bool
    drop_categories: bool
    year_pad: int
    relevance_priority: bool


def parse_academic_rerank_weights(raw: str | None = None) -> AcademicRerankWeights:
    text = (raw if raw is not None else ACADEMIC_RERANK_WEIGHTS) or ""
    parts = [p.strip() for p in re.split(r"[,;\s]+", text) if p.strip()]
    nums: list[float] = []
    for p in parts[:4]:
        try:
            nums.append(float(p))
        except ValueError:
            nums.append(0.0)
    while len(nums) < 4:
        nums.append(0.0)
    return AcademicRerankWeights(*nums[:4]).normalized()


def normalize_citations(
    citation_count: int,
    *,
    c_sat: float | None = None,
) -> float:
    cites = max(0, int(citation_count or 0))
    sat = float(ACADEMIC_RERANK_C_SAT if c_sat is None else c_sat)
    sat = max(1.0, sat)
    return min(1.0, math.log1p(cites) / math.log1p(sat))


def recency_factor(
    year: int | None,
    *,
    now_year: int | None = None,
    half_life_years: float | None = None,
) -> float:
    """
    Exponential decay by age. Missing year → neutral 0.5.
    half_life ≈ years for factor to drop to ~0.5.
    """
    if year is None:
        return 0.5
    try:
        y = int(year)
    except (TypeError, ValueError):
        return 0.5
    now = int(now_year or datetime.now(timezone.utc).year)
    if y < 1990 or y > now + 1:
        return 0.5
    age = max(0.0, float(now - y))
    half = float(
        ACADEMIC_RERANK_RECENCY_HALF_LIFE_YEARS
        if half_life_years is None
        else half_life_years
    )
    half = max(0.5, half)
    return float(math.exp(-math.log(2.0) * age / half))


def hybrid_academic_score(
    signals: RerankSignals,
    *,
    weights: AcademicRerankWeights | None = None,
    c_sat: float | None = None,
    now_year: int | None = None,
) -> float:
    w = (weights or parse_academic_rerank_weights()).normalized()
    rel = max(0.0, min(1.0, float(signals.relevance_sim)))
    trust = max(0.0, min(1.0, float(signals.trust_score)))
    cites = normalize_citations(signals.citation_count, c_sat=c_sat)
    rec = recency_factor(signals.year, now_year=now_year)
    return round(
        w.alpha * rel + w.beta * trust + w.gamma * cites + w.delta * rec,
        6,
    )


def policy_for_level(level: RelaxationLevel) -> RelaxationPolicy:
    if level <= RelaxationLevel.STRICT:
        return RelaxationPolicy(
            level=RelaxationLevel.STRICT,
            min_trust=float(ACADEMIC_RELAX_L0_MIN_TRUST),
            min_citations=int(ACADEMIC_RELAX_L0_MIN_CITATIONS),
            drop_date_filter=False,
            drop_categories=False,
            year_pad=0,
            relevance_priority=False,
        )
    if level == RelaxationLevel.SOFT_DATE_CITATIONS:
        return RelaxationPolicy(
            level=RelaxationLevel.SOFT_DATE_CITATIONS,
            min_trust=float(ACADEMIC_RELAX_L1_MIN_TRUST),
            min_citations=int(ACADEMIC_RELAX_L1_MIN_CITATIONS),
            drop_date_filter=True,
            drop_categories=False,
            year_pad=max(0, int(ACADEMIC_RELAX_L1_YEAR_PAD)),
            relevance_priority=False,
        )
    return RelaxationPolicy(
        level=RelaxationLevel.BROAD_RELEVANCE,
        min_trust=0.0,
        min_citations=0,
        drop_date_filter=True,
        drop_categories=True,
        year_pad=0,
        relevance_priority=True,
    )


def relax_arxiv_params(
    params: ArxivQueryParams | None,
    level: RelaxationLevel,
) -> ArxivQueryParams:
    """Return a copy of arXiv params softened for the given relaxation level."""
    base = params or ArxivQueryParams()
    policy = policy_for_level(level)
    if level == RelaxationLevel.STRICT:
        return replace(base)

    start_year = base.start_year
    end_year = base.end_year
    categories = list(base.categories)
    exclude = list(base.exclude_terms)

    if policy.year_pad and (start_year is not None or end_year is not None):
        if start_year is not None:
            start_year = max(1991, int(start_year) - policy.year_pad)
        if end_year is not None:
            end_year = min(2100, int(end_year) + policy.year_pad)

    if policy.drop_date_filter:
        start_year = None
        end_year = None

    if policy.drop_categories:
        categories = []
        # Keep mild excludes only at L1; drop noisy excludes at L2
        exclude = []

    sort_by = base.sort_by
    if policy.relevance_priority:
        sort_by = "relevance"

    return replace(
        base,
        start_year=start_year,
        end_year=end_year,
        categories=categories,
        exclude_terms=exclude,
        sort_by=sort_by,
        sort_order=base.sort_order or "descending",
    )


def passes_relaxation_gate(
    signals: RerankSignals,
    level: RelaxationLevel,
) -> bool:
    policy = policy_for_level(level)
    if policy.relevance_priority:
        # L2: keep anything with non-trivial relevance; trust/cites optional
        return float(signals.relevance_sim) >= 0.0
    if float(signals.trust_score) < policy.min_trust:
        return False
    if int(signals.citation_count or 0) < policy.min_citations:
        return False
    return True


def sort_by_hybrid_score(
    items: Sequence[T],
    *,
    signals_of: Callable[[T], RerankSignals],
    enabled: bool | None = None,
    weights: AcademicRerankWeights | None = None,
    level: RelaxationLevel = RelaxationLevel.STRICT,
) -> list[T]:
    """
    Pre-sort candidates by hybrid score (and optionally gate by relaxation level).

    When ACADEMIC_RERANK_ENABLED is false, returns items unchanged (still gated
    only if caller asks — by default no gate when disabled).
    """
    use = ACADEMIC_RERANK_ENABLED if enabled is None else bool(enabled)
    if not items:
        return []
    if not use:
        return list(items)

    w = weights or parse_academic_rerank_weights()
    if level >= RelaxationLevel.BROAD_RELEVANCE:
        # Emphasize relevance at L2
        w = AcademicRerankWeights(
            alpha=max(w.alpha, 0.55),
            beta=w.beta * 0.7,
            gamma=w.gamma * 0.5,
            delta=w.delta,
        ).normalized()

    scored: list[tuple[float, int, T]] = []
    for idx, item in enumerate(items):
        sig = signals_of(item)
        if not passes_relaxation_gate(sig, level):
            continue
        score = hybrid_academic_score(sig, weights=w)
        scored.append((score, idx, item))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [item for _, _, item in scored]


def relaxation_levels() -> list[RelaxationLevel]:
    return [
        RelaxationLevel.STRICT,
        RelaxationLevel.SOFT_DATE_CITATIONS,
        RelaxationLevel.BROAD_RELEVANCE,
    ]


def should_relax(*, hit_count: int, min_hits: int | None = None) -> bool:
    if not ACADEMIC_RELAXATION_ENABLED:
        return False
    threshold = (
        ACADEMIC_RELAXATION_MIN_HITS if min_hits is None else max(1, int(min_hits))
    )
    return int(hit_count) < threshold


def next_relaxation_level(current: RelaxationLevel) -> RelaxationLevel | None:
    nxt = int(current) + 1
    if nxt > int(RelaxationLevel.BROAD_RELEVANCE):
        return None
    return RelaxationLevel(nxt)


def signals_from_scholar_paper(
    paper: Any,
    *,
    relevance_sim: float = 0.75,
    trust_score: float = 1.0,
) -> RerankSignals:
    return RerankSignals(
        relevance_sim=float(relevance_sim),
        trust_score=float(trust_score),
        citation_count=int(getattr(paper, "citation_count", 0) or 0),
        year=getattr(paper, "year", None),
    )


def signals_from_lecture_candidate(cand: Any) -> RerankSignals:
    rel = getattr(cand, "vector_similarity", None)
    if rel is None or float(rel) <= 0:
        rel = getattr(cand, "retrieval_score", 0.0) or 0.0
    trust = getattr(cand, "trust_score", 1.0)
    return RerankSignals(
        relevance_sim=float(rel or 0.0),
        trust_score=float(1.0 if trust is None else trust),
        citation_count=int(getattr(cand, "citation_count", 0) or 0),
        year=getattr(cand, "year", None),
    )


def presort_lecture_candidates(
    candidates: Sequence[T],
    *,
    enabled: bool | None = None,
) -> list[T]:
    """Pre-sort lecture RAG pool before CE/MMR when flag is on."""
    use = ACADEMIC_RERANK_ENABLED if enabled is None else bool(enabled)
    if not use or not candidates:
        return list(candidates)
    out = sort_by_hybrid_score(
        candidates,
        signals_of=signals_from_lecture_candidate,
        enabled=True,
        level=RelaxationLevel.BROAD_RELEVANCE,
    )
    trace(f"LECTURE_RAG academic_rerank ▶ | candidates={len(out)}/{len(candidates)}")
    return out if out else list(candidates)
