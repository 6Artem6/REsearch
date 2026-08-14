"""Replenishment loop (production module targeted_hit_replenishment)."""

from __future__ import annotations

from knowledge_engine.db.domain_blocklist import (
    add_blocked_domain,
    load_blocked_domain_set,
)
from knowledge_engine.services.parsers.html_attr import coerce_html_attr
from knowledge_engine.src.curriculum.schemas import CurriculumSearchHit
from knowledge_engine.src.curriculum.targeted_hit_replenishment import (
    precheck_candidate_hit,
    precheck_candidate_url,
    replenish_valid_hits_until_cap,
)


def test_coerce_html_attr_list_like():
    class FakeList(list):
        pass

    raw = FakeList(["  https://example.com/a.png  ", "alt text"])
    assert "example.com" in coerce_html_attr(raw)


def test_precheck_rejects_swagger_and_blocklist():
    blocked = load_blocked_domain_set()
    swagger = "https://vendor.example.com/swagger/ui/index"
    reason = precheck_candidate_url(swagger, blocked_domains=blocked)
    assert reason is not None and "swagger" in reason

    add_blocked_domain("https://blocked-host.example.com/page", "test")
    blocked = load_blocked_domain_set()
    reason_bl = precheck_candidate_url(
        "https://blocked-host.example.com/other",
        blocked_domains=blocked,
    )
    assert reason_bl is not None and reason_bl.startswith("domain_blocklist:")


def test_precheck_allows_academic_doi_tier():
    from knowledge_engine.src.curriculum.academic_url_canonicalizer import (
        canonicalize_academic_url_pure,
    )

    blocked = load_blocked_domain_set()
    canon = canonicalize_academic_url_pure("https://doi.org/10.48550/arxiv.2512.08290")
    assert canon is not None
    hit = CurriculumSearchHit(
        url=canon,
        title="paper",
        snippet="s",
        key_extracts=["x"],
        source_tier="arxiv",
    )
    assert precheck_candidate_hit(hit, blocked_domains=blocked) is None
    reason = precheck_candidate_url(
        "https://doi.org/10.48550/arxiv.2512.08290",
        blocked_domains=blocked,
        skip_practical_filter=True,
    )
    assert reason is None


def test_replenish_skips_bad_and_fills_cap():
    good_a = CurriculumSearchHit(
        url="https://example.com/blog/post-a",
        title="a",
        snippet="s",
        key_extracts=["x"],
        source_tier="exa",
    )
    good_b = CurriculumSearchHit(
        url="https://example.com/blog/post-b",
        title="b",
        snippet="s",
        key_extracts=["y"],
        source_tier="exa",
    )
    good_c = CurriculumSearchHit(
        url="https://example.com/blog/post-c",
        title="c",
        snippet="s",
        key_extracts=["w"],
        source_tier="exa",
    )
    api_doc = CurriculumSearchHit(
        url="https://vendor.example.com/openapi/v3/spec",
        title="api",
        snippet="s",
        key_extracts=["z"],
        source_tier="searxng",
    )

    import asyncio

    from knowledge_engine.src.curriculum import targeted_hit_replenishment as mod

    async def fake_check(url: str, timeout: float = 10.0):
        if "post-b" in url:
            return False, "soft_404"
        return True, "ok"

    mod.check_url_live = fake_check  # type: ignore[method-assign]

    out = asyncio.run(
        replenish_valid_hits_until_cap([api_doc, good_a, good_b, good_c], 2)
    )
    assert len(out) == 2
    urls = {h.url for h in out}
    assert "https://example.com/blog/post-a" in urls
    assert "https://example.com/blog/post-c" in urls
    assert all("/openapi/" not in u for u in urls)
