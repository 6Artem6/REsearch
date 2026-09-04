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
    # source_tier="searxng" (not "exa") — this test exercises replenish's own
    # check_url_live() filtering; exa-tier hits take a different path, see
    # test_replenish_skips_check_url_live_for_preflight_triaged_exa_hits below.
    good_a = CurriculumSearchHit(
        url="https://example.com/blog/post-a",
        title="a",
        snippet="s",
        key_extracts=["x"],
        source_tier="searxng",
    )
    good_b = CurriculumSearchHit(
        url="https://example.com/blog/post-b",
        title="b",
        snippet="s",
        key_extracts=["y"],
        source_tier="searxng",
    )
    good_c = CurriculumSearchHit(
        url="https://example.com/blog/post-c",
        title="c",
        snippet="s",
        key_extracts=["w"],
        source_tier="searxng",
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


def test_replenish_skips_check_url_live_for_preflight_triaged_exa_hits():
    """Exa hits already passed Pre-Flight Triage Stage 2 liveness check —
    replenish must NOT re-validate them with a second check_url_live() call
    (double HTTP), it should accept them as-is."""
    import asyncio

    from knowledge_engine.src.curriculum import targeted_hit_replenishment as mod

    exa_hit = CurriculumSearchHit(
        url="https://example.com/blog/already-triaged",
        title="a",
        snippet="s",
        key_extracts=["x"],
        source_tier="exa",
    )

    called = {"n": 0}

    async def fake_check(url: str, timeout: float = 10.0):
        called["n"] += 1
        return False, "soft_404"  # if called, this hit would be dropped

    mod.check_url_live = fake_check  # type: ignore[method-assign]

    out = asyncio.run(replenish_valid_hits_until_cap([exa_hit], 5))
    assert called["n"] == 0, "check_url_live must not be called for exa-tier hits"
    assert len(out) == 1
    assert out[0].url == exa_hit.url


def test_replenish_backfill_margin_keeps_extra_valid_hits():
    """DEEP_INGEST_BACKFILL_MARGIN: replenish should stop at cap+margin (not
    cap), drawing the extra hits from the same already-fetched candidates —
    downstream _ingest_blog_hits_batch_async needs this reserve to backfill
    dropped ALIAS hits instead of shrinking the final set."""
    import asyncio

    from knowledge_engine.src.curriculum import targeted_hit_replenishment as mod

    hits = [
        CurriculumSearchHit(
            url=f"https://example.com/blog/post-{i}",
            title=str(i),
            snippet="s",
            key_extracts=["x"],
            source_tier="exa",
        )
        for i in range(6)
    ]

    async def fake_check(url: str, timeout: float = 10.0):
        return True, "ok"

    mod.check_url_live = fake_check  # type: ignore[method-assign]

    out_no_margin = asyncio.run(replenish_valid_hits_until_cap(hits, 2))
    assert len(out_no_margin) == 2

    out_with_margin = asyncio.run(
        replenish_valid_hits_until_cap(hits, 2, backfill_margin=2)
    )
    assert len(out_with_margin) == 4


def test_replenish_backfill_margin_reaches_effective_cap_with_node_quota():
    """Real bug from perf_debug.log (gil_internals): with node= set,
    order_candidates_for_node used to hard-cap the candidate list to
    quota.total_max BEFORE this function's cap+margin loop ran — margin was
    then unreachable no matter how large the raw candidates pool was. Here
    the raw pool has 8 valid practical hits and quota.total_max=4 (advanced/
    DEEP); with backfill_margin=2 the function must be able to reach 6, not
    get silently stuck at 4."""
    import asyncio

    from knowledge_engine.src.curriculum import targeted_hit_replenishment as mod
    from knowledge_engine.src.curriculum.schemas import CurriculumNode

    node = CurriculumNode(
        node_id="gil_internals",
        title="Global interpreter lock internals",
        layer="advanced",
        category="python internals",
        brief_summary="Per-interpreter GIL, ceval loop, and free-threading.",
        core_concepts=["GIL", "ceval"],
        node_risk_kind="DEEP",
        grounding_status="pending_grounding",
    )
    hits = [
        CurriculumSearchHit(
            url=f"https://example.com/blog/post-{i}",
            title=str(i),
            snippet="s",
            key_extracts=["x"],
            source_tier="exa",
            exa_relevance_score=0.9 - i * 0.01,
        )
        for i in range(8)
    ]

    async def fake_check(url: str, timeout: float = 10.0):
        return True, "ok"

    mod.check_url_live = fake_check  # type: ignore[method-assign]

    out_no_margin = asyncio.run(replenish_valid_hits_until_cap(hits, 4, node=node))
    assert len(out_no_margin) == 4

    out_with_margin = asyncio.run(
        replenish_valid_hits_until_cap(hits, 4, node=node, backfill_margin=2)
    )
    assert len(out_with_margin) == 6, (
        "backfill_margin must actually widen past quota.total_max when the "
        "raw candidate pool has enough hits — not get stuck at total_max"
    )
