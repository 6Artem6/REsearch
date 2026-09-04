"""Source quota matrix (layer × risk)."""

from __future__ import annotations

from knowledge_engine.src.curriculum.schemas import CurriculumSearchHit
from knowledge_engine.src.curriculum.source_quota_policy import get_source_quota
from knowledge_engine.src.curriculum.source_quota_selection import (
    classify_hit_content_bucket,
    select_hits_by_quota,
)


def test_foundation_base_zero_academic():
    q = get_source_quota("foundation", "BASE")
    assert q.academic_max == 0
    assert q.practical_max == 2
    assert q.total_max == 2


def test_advanced_balanced():
    q = get_source_quota("advanced", "DEEP")
    assert q.academic_max == 2
    assert q.practical_max == 2
    assert q.total_max == 4


def test_sota_deep_academic_heavy():
    q = get_source_quota("sota", "DEEP")
    assert q.academic_max == 3
    assert q.practical_max == 1


def _hit(url: str, tier: str, score: float | None = None) -> CurriculumSearchHit:
    return CurriculumSearchHit(
        url=url,
        title=url,
        snippet="s",
        key_extracts=["x"],
        source_tier=tier,
        exa_relevance_score=score,
    )


def test_select_advanced_two_each():
    q = get_source_quota("advanced", "DEEP")
    candidates = [
        _hit("https://arxiv.org/abs/1", "arxiv", 0.9),
        _hit("https://arxiv.org/abs/2", "arxiv", 0.8),
        _hit("https://arxiv.org/abs/3", "arxiv", 0.7),
        _hit("https://developers.example.com/a", "exa", 0.95),
        _hit("https://developers.example.com/b", "exa", 0.85),
        _hit("https://habr.com/a", "exa", 0.75),
    ]
    picked = select_hits_by_quota(candidates, q)
    assert len(picked) == 4
    academic_n = sum(1 for h in picked if classify_hit_content_bucket(h) == "academic")
    practical_n = len(picked) - academic_n
    assert academic_n == 2
    assert practical_n == 2


def test_select_hits_by_quota_limit_widens_beyond_total_max():
    """Regression: order_candidates_for_node used to hard-cap at
    quota.total_max BEFORE replenish_valid_hits_until_cap's backfill_margin
    loop ever got a chance to use the extra headroom — margin candidates were
    physically unreachable even when the raw pool had plenty. limit= lets the
    fallback spill (and final slice) grow past total_max while per-bucket
    TOP-N/TOP-M picks stay exactly as quota dictates."""
    q = get_source_quota(
        "advanced", "DEEP"
    )  # academic_max=2 practical_max=2 total_max=4
    candidates = [
        _hit(f"https://developers.example.com/{i}", "exa", 0.9 - i * 0.01)
        for i in range(8)
    ]
    picked_default = select_hits_by_quota(candidates, q)
    assert len(picked_default) == 4  # unchanged default behaviour

    picked_widened = select_hits_by_quota(candidates, q, limit=6)
    assert len(picked_widened) == 6
    assert [h.url for h in picked_widened[:4]] == [h.url for h in picked_default]


def test_sota_selection_academic_majority():
    q = get_source_quota("sota", "DEEP")
    candidates = [
        _hit("https://arxiv.org/abs/1", "arxiv", 0.9),
        _hit("https://arxiv.org/abs/2", "arxiv", 0.85),
        _hit("https://arxiv.org/abs/3", "arxiv", 0.8),
        _hit("https://developers.example.com/doc", "exa", 0.99),
    ]
    picked = select_hits_by_quota(candidates, q)
    assert len(picked) == 4
    academic_n = sum(1 for h in picked if classify_hit_content_bucket(h) == "academic")
    assert academic_n == 3
