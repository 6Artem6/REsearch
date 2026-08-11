"""Unit tests for Exa transform ranking, merge, and domain round-robin."""

from __future__ import annotations

from knowledge_engine.config import EXA_RECALL_MAX_PER_DOMAIN
from knowledge_engine.services.search.exa_client import ExaSearchHit
from knowledge_engine.services.search.exa_transform import (
    _combined_exa_rank_score,
    _exa_url_quality_score,
    _normalize_exa_score,
    _normalize_url_quality_score,
    fair_domain_round_robin,
    filter_and_rank_exa_curriculum_hits,
    merge_multi_vector_exa_hits,
)
from knowledge_engine.src.curriculum.schemas import CurriculumSearchHit


def _curriculum_hit(
    url: str,
    *,
    exa_score: float | None = None,
    title: str = "t",
) -> CurriculumSearchHit:
    return CurriculumSearchHit(
        url=url,
        title=title,
        exa_relevance_score=exa_score,
    )


def test_filter_rejects_swagger_and_openapi_urls():
    swagger = _curriculum_hit(
        "https://api.vendor.com/swagger/ui/index.html",
        exa_score=0.99,
    )
    openapi = _curriculum_hit(
        "https://api.vendor.com/openapi/v3/spec",
        exa_score=0.95,
    )
    good = _curriculum_hit(
        "https://engineering.vendor.com/blog/deep-dive-architecture",
        exa_score=0.4,
    )
    ranked = filter_and_rank_exa_curriculum_hits([swagger, openapi, good])
    urls = {h.url for h in ranked}
    assert good.url in urls
    assert swagger.url not in urls
    assert openapi.url not in urls
    assert _exa_url_quality_score(swagger.url) <= -5


def test_composite_rank_weighted_formula_and_ordering():
    blog_a = "https://example.com/blog/post-a"
    blog_b = "https://example.com/blog/post-b"

    low_exa_high_url = _curriculum_hit(blog_a, exa_score=0.2)
    high_exa_same_url_tier = _curriculum_hit(blog_b, exa_score=0.95)

    exa_n = _normalize_exa_score(0.95)
    url_n = _normalize_url_quality_score(_exa_url_quality_score(blog_b))
    expected_high = 0.65 * exa_n + 0.35 * url_n

    exa_n_low = _normalize_exa_score(0.2)
    url_n_a = _normalize_url_quality_score(_exa_url_quality_score(blog_a))
    expected_low = 0.65 * exa_n_low + 0.35 * url_n_a

    assert expected_high > expected_low
    assert _combined_exa_rank_score(high_exa_same_url_tier) == expected_high
    assert _combined_exa_rank_score(low_exa_high_url) == expected_low

    ranked = filter_and_rank_exa_curriculum_hits(
        [low_exa_high_url, high_exa_same_url_tier],
    )
    assert ranked[0].url == blog_b
    assert ranked[1].url == blog_a


def test_merge_multi_vector_keeps_highest_score_per_url():
    shared = "https://habr.com/ru/articles/12345/"
    batch_en = [
        ExaSearchHit(url=shared, title="a", score=0.35),
        ExaSearchHit(url="https://eng.example.com/posts/x", title="b", score=0.5),
    ]
    batch_ru = [
        ExaSearchHit(url=shared, title="a2", score=0.88),
    ]
    merged = merge_multi_vector_exa_hits([batch_en, batch_ru], cap=50)
    by_url = {h.url: h for h in merged}
    assert len(merged) == 2
    assert by_url[shared].score == 0.88
    assert merged[0].score == 0.88


def test_fair_domain_round_robin_respects_recall_max_per_domain():
    per_domain = max(1, EXA_RECALL_MAX_PER_DOMAIN)
    hits = [
        ExaSearchHit(url="https://blog.alpha.com/post-1", title="", score=0.9),
        ExaSearchHit(url="https://blog.alpha.com/post-2", title="", score=0.85),
        ExaSearchHit(url="https://blog.alpha.com/post-3", title="", score=0.8),
        ExaSearchHit(url="https://blog.beta.com/post-1", title="", score=0.75),
        ExaSearchHit(url="https://blog.beta.com/post-2", title="", score=0.7),
    ]
    cap = 10
    out = fair_domain_round_robin(
        hits,
        cap,
        max_per_domain=per_domain,
        get_url=lambda h: h.url,
    )
    alpha_count = sum(1 for h in out if "alpha.com" in h.url)
    beta_count = sum(1 for h in out if "beta.com" in h.url)
    assert alpha_count <= per_domain
    assert beta_count <= per_domain
    assert len(out) == min(cap, per_domain * 2)
