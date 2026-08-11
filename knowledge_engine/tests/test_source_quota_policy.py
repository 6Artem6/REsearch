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
