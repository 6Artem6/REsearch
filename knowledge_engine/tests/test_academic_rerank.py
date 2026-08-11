"""Hybrid academic rerank formula + relaxation cascade."""

from __future__ import annotations

from knowledge_engine.services.search.arxiv_query_builder import ArxivQueryParams
from knowledge_engine.src.retrieval import academic_rerank as ar
from knowledge_engine.src.retrieval.academic_rerank import (
    AcademicRerankWeights,
    RelaxationLevel,
    RerankSignals,
    hybrid_academic_score,
    normalize_citations,
    parse_academic_rerank_weights,
    passes_relaxation_gate,
    policy_for_level,
    presort_lecture_candidates,
    recency_factor,
    relax_arxiv_params,
    should_relax,
    sort_by_hybrid_score,
)
from knowledge_engine.src.retrieval.semantic_scholar import ScholarPaper


def test_normalize_citations_log_scale():
    assert normalize_citations(0, c_sat=40) == 0.0
    assert 0.0 < normalize_citations(5, c_sat=40) < 1.0
    assert normalize_citations(40, c_sat=40) == 1.0
    assert normalize_citations(400, c_sat=40) == 1.0


def test_recency_factor_decays_with_age():
    fresh = recency_factor(2025, now_year=2026, half_life_years=6)
    old = recency_factor(2010, now_year=2026, half_life_years=6)
    assert fresh > old
    assert recency_factor(None, now_year=2026) == 0.5


def test_hybrid_score_respects_weights():
    w = AcademicRerankWeights(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0).normalized()
    high_rel = hybrid_academic_score(
        RerankSignals(relevance_sim=0.9, trust_score=0.1, citation_count=0, year=2010),
        weights=w,
        now_year=2026,
    )
    low_rel = hybrid_academic_score(
        RerankSignals(
            relevance_sim=0.1, trust_score=1.0, citation_count=100, year=2025
        ),
        weights=w,
        now_year=2026,
    )
    assert high_rel > low_rel


def test_parse_weights_normalizes():
    w = parse_academic_rerank_weights("1,1,1,1")
    assert abs(w.alpha + w.beta + w.gamma + w.delta - 1.0) < 1e-9
    assert abs(w.alpha - 0.25) < 1e-9


def test_flag_off_keeps_original_order(monkeypatch):
    monkeypatch.setattr(ar, "ACADEMIC_RERANK_ENABLED", False)
    papers = [
        ScholarPaper(title="a", citation_count=1, year=2020),
        ScholarPaper(title="b", citation_count=100, year=2024),
    ]
    out = sort_by_hybrid_score(
        papers,
        signals_of=lambda p: RerankSignals(
            relevance_sim=0.5,
            trust_score=1.0,
            citation_count=p.citation_count,
            year=p.year,
        ),
        enabled=False,
    )
    assert [p.title for p in out] == ["a", "b"]


def test_flag_on_sorts_by_hybrid(monkeypatch):
    monkeypatch.setattr(ar, "ACADEMIC_RERANK_ENABLED", True)
    papers = [
        ScholarPaper(title="low", citation_count=0, year=2010),
        ScholarPaper(title="high", citation_count=80, year=2024),
    ]
    out = sort_by_hybrid_score(
        papers,
        signals_of=lambda p: RerankSignals(
            relevance_sim=0.6,
            trust_score=0.8,
            citation_count=p.citation_count,
            year=p.year,
        ),
        enabled=True,
        level=RelaxationLevel.BROAD_RELEVANCE,
        weights=AcademicRerankWeights(0.2, 0.2, 0.4, 0.2),
    )
    assert out[0].title == "high"


def test_relaxation_levels_widen_params():
    strict = ArxivQueryParams(
        title_keywords=["transformers"],
        categories=["cs.CL"],
        exclude_terms=["survey"],
        start_year=2023,
        end_year=2024,
    )
    l1 = relax_arxiv_params(strict, RelaxationLevel.SOFT_DATE_CITATIONS)
    assert l1.start_year is None and l1.end_year is None
    assert l1.categories == ["cs.CL"]

    l2 = relax_arxiv_params(strict, RelaxationLevel.BROAD_RELEVANCE)
    assert l2.categories == []
    assert l2.exclude_terms == []
    assert l2.sort_by == "relevance"


def test_l0_gate_filters_low_citations(monkeypatch):
    monkeypatch.setattr(ar, "ACADEMIC_RELAX_L0_MIN_CITATIONS", 5)
    monkeypatch.setattr(ar, "ACADEMIC_RELAX_L0_MIN_TRUST", 0.35)
    weak = RerankSignals(relevance_sim=0.9, trust_score=0.9, citation_count=1)
    strong = RerankSignals(relevance_sim=0.5, trust_score=0.9, citation_count=10)
    assert passes_relaxation_gate(weak, RelaxationLevel.STRICT) is False
    assert passes_relaxation_gate(strong, RelaxationLevel.STRICT) is True
    assert passes_relaxation_gate(weak, RelaxationLevel.SOFT_DATE_CITATIONS) is True


def test_should_relax_respects_min_hits(monkeypatch):
    monkeypatch.setattr(ar, "ACADEMIC_RELAXATION_ENABLED", True)
    monkeypatch.setattr(ar, "ACADEMIC_RELAXATION_MIN_HITS", 3)
    assert should_relax(hit_count=2) is True
    assert should_relax(hit_count=3) is False
    monkeypatch.setattr(ar, "ACADEMIC_RELAXATION_ENABLED", False)
    assert should_relax(hit_count=0) is False


def test_presort_lecture_candidates_noop_when_disabled(monkeypatch):
    from knowledge_engine.services.lecture_context_rerank import LectureContextCandidate

    monkeypatch.setattr(ar, "ACADEMIC_RERANK_ENABLED", False)
    cands = [
        LectureContextCandidate(
            label="a",
            formatted="",
            plain="a",
            vector_similarity=0.2,
            trust_score=1.0,
        ),
        LectureContextCandidate(
            label="b",
            formatted="",
            plain="b",
            vector_similarity=0.9,
            trust_score=0.5,
        ),
    ]
    out = presort_lecture_candidates(cands, enabled=False)
    assert [c.label for c in out] == ["a", "b"]


def test_presort_lecture_candidates_orders_when_enabled(monkeypatch):
    from knowledge_engine.services.lecture_context_rerank import LectureContextCandidate

    monkeypatch.setattr(ar, "ACADEMIC_RERANK_ENABLED", True)
    cands = [
        LectureContextCandidate(
            label="a",
            formatted="",
            plain="a",
            vector_similarity=0.2,
            trust_score=0.3,
        ),
        LectureContextCandidate(
            label="b",
            formatted="",
            plain="b",
            vector_similarity=0.9,
            trust_score=0.9,
        ),
    ]
    out = presort_lecture_candidates(cands, enabled=True)
    assert out[0].label == "b"


def test_l1_year_pad_policy():
    p = policy_for_level(RelaxationLevel.SOFT_DATE_CITATIONS)
    assert p.drop_date_filter is True
    assert p.year_pad >= 0
    strict = ArxivQueryParams(start_year=2022, end_year=2023, categories=["cs.AI"])
    # year_pad applied before drop in relax_arxiv_params — then dates cleared
    softened = relax_arxiv_params(strict, RelaxationLevel.SOFT_DATE_CITATIONS)
    assert softened.start_year is None
