"""Exa kwargs, official-docs ranking, Lite expansion, and two-pass search."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from knowledge_engine.schemas.llm_contracts.exa_search import (
    BatchDomainAuthorityResponse,
    DomainAuthorityItem,
    DomainAuthorityVerdict,
    ExaSearchContextExpansion,
)
from knowledge_engine.services.search.exa_client import (
    ExaSearchClient,
    ExaSearchHit,
    ExaSearchResponse,
    build_exa_search_kwargs,
)
from knowledge_engine.services.search.exa_domain_validate import (
    validate_exa_domains_blocking,
)
from knowledge_engine.services.search.exa_domains import (
    add_dynamic_exa_domain,
    clean_domain_for_exa,
    is_official_docs_host,
)
from knowledge_engine.services.search.exa_source_expand import (
    _BATCH_AUTHORITY_SYSTEM,
    _EXPAND_SYSTEM,
    classify_exa_domains_batch_with_flash_lite,
    expand_search_context_with_flash_lite,
    fallback_exa_search_context,
    filter_pass1_official_hosts,
)
from knowledge_engine.services.search.exa_transform import _exa_url_quality_score
from knowledge_engine.src.curriculum.practical_url_filters import (
    practical_url_reject_reason,
)
from knowledge_engine.src.source_evaluator.whitelist import APPROVED_SOURCES_WHITELIST


@pytest.fixture(autouse=True)
def _isolate_domain_registry(monkeypatch):
    """Do not hit production LanceDB / BGE-M3 from expand or batch classify."""

    class _EmptyRegistry:
        def search_official_docs(self, topic_text: str) -> list[str]:
            return []

        def upsert_keep_items(self, items) -> int:
            return 0

    monkeypatch.setattr(
        "knowledge_engine.services.search.domain_registry.get_domain_registry",
        lambda: _EmptyRegistry(),
    )


def test_clean_domain_strips_habr_company_path():
    assert clean_domain_for_exa("habr.com/ru/companies/yandex") == "habr.com"


def test_build_kwargs_default_include_is_static_whitelist_hosts():
    kwargs = build_exa_search_kwargs("cpython gil internals")
    inc = kwargs.get("include_domains") or []
    assert "habr.com" in inc
    assert "docs.python.org" in inc
    assert kwargs["type"] == "auto"
    assert "category" not in kwargs
    assert "exclude_text" in kwargs


def test_build_kwargs_primary_domains_and_keyword_docs_lane():
    kwargs = build_exa_search_kwargs(
        "PEP 703",
        include_domains=["docs.python.org", "peps.python.org"],
        search_type="keyword",
        category="research paper",
        exclude_text=[],
    )
    assert kwargs["include_domains"] == ["docs.python.org", "peps.python.org"]
    assert kwargs["type"] == "keyword"
    assert kwargs["category"] == "research paper"
    assert "exclude_text" not in kwargs


def test_build_kwargs_empty_include_omits_include_domains():
    kwargs = build_exa_search_kwargs(
        "gil",
        include_domains=[],
        search_type="auto",
        exclude_text=[],
    )
    assert "include_domains" not in kwargs


def test_official_docs_hosts_and_practical_filter():
    assert is_official_docs_host("https://docs.python.org/3/c-api/")
    assert is_official_docs_host("https://developer.mozilla.org/en-US/docs/Web")
    assert not is_official_docs_host("https://habr.com/ru/articles/123/")
    add_dynamic_exa_domain("peps.python.org", "OFFICIAL_DOCS")
    assert is_official_docs_host("peps.python.org")
    pep = "https://peps.python.org/pep-0703/"
    mdn = "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference"
    swagger = "https://api.vendor.com/swagger/ui/index.html"
    habr = "https://habr.com/ru/articles/12345/"
    assert practical_url_reject_reason(pep) is None
    assert practical_url_reject_reason(mdn) is None
    assert practical_url_reject_reason(swagger) is not None
    assert practical_url_reject_reason(habr) is None
    assert _exa_url_quality_score(pep) > 0
    assert _exa_url_quality_score(mdn) > 0
    assert _exa_url_quality_score(swagger) <= -5


def test_docs_prefix_is_not_enough_for_official():
    assert not is_official_docs_host("docs.madeup-stack.example")
    assert not is_official_docs_host("peps.madeup-stack.example")
    assert not is_official_docs_host("ietf.org")


def test_fallback_expansion_skips_pass1_domains():
    ctx = fallback_exa_search_context("CPython GIL ceval")
    assert ctx.include_official_docs is True
    assert ctx.use_broader_search is True
    assert ctx.search_type == "auto"
    assert ctx.primary_domains == []
    assert "github" in ctx.allowed_categories


def test_expand_short_query_does_not_call_network():
    ctx = expand_search_context_with_flash_lite("ab")
    assert ctx.primary_domains == []
    assert ctx.allowed_categories


def test_expand_and_authority_prompts_share_taxonomy():
    assert "docs.*" not in _EXPAND_SYSTEM
    assert "CANONICAL_SPEC" in _EXPAND_SYSTEM
    assert "OFFICIAL_DOCS" in _EXPAND_SYSTEM
    assert "SOURCE_TREE" in _EXPAND_SYSTEM
    assert "AGGREGATOR_BLOG" in _EXPAND_SYSTEM
    assert "QNA_FORUM" in _EXPAND_SYSTEM
    assert "ACADEMY_SEO" in _EXPAND_SYSTEM
    assert "habr.com" in _EXPAND_SYSTEM
    assert "topic_vector_query" in _EXPAND_SYSTEM
    assert "FORBIDDEN: listing narrow subtopics" in _EXPAND_SYSTEM
    assert "ACADEMIC_OR_PAPER" in _BATCH_AUTHORITY_SYSTEM
    assert "COMMUNITY_BLOG → REJECT" in _BATCH_AUTHORITY_SYSTEM
    assert "OFFICIAL_DOCS → KEEP" in _BATCH_AUTHORITY_SYSTEM
    assert "VENDOR_BLOG → KEEP" in _BATCH_AUTHORITY_SYSTEM
    assert "general_summary" in _BATCH_AUTHORITY_SYSTEM
    assert "FORBIDDEN: enumerating narrow lecture subtopics" in _BATCH_AUTHORITY_SYSTEM


def test_dynamic_domain_marks_official():
    host = add_dynamic_exa_domain("docs.pytorch.org", "OFFICIAL_DOCS")
    assert host == "docs.pytorch.org"
    assert is_official_docs_host("https://docs.pytorch.org/docs/stable/")
    assert "foundational_docs" in APPROVED_SOURCES_WHITELIST


def test_http_validate_keeps_2xx_drops_4xx_timeout(monkeypatch):
    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def head(self, url, **kwargs):
            host = url.split("://", 1)[-1]
            if host == "docs.python.org":
                return SimpleNamespace(status_code=200)
            if host == "dead.example":
                return SimpleNamespace(status_code=404)
            if host == "timeout.example":
                raise httpx.ConnectTimeout("timeout")
            if host == "dns.example":
                raise httpx.ConnectError("Name or service not known")
            return SimpleNamespace(status_code=500)

        async def get(self, url, **kwargs):
            return await self.head(url)

    monkeypatch.setattr(
        "knowledge_engine.services.search.exa_domain_validate.httpx.AsyncClient",
        _Client,
    )
    live = validate_exa_domains_blocking(
        ["docs.python.org", "dead.example", "timeout.example", "dns.example"]
    )
    assert live == ["docs.python.org"]


def test_http_validate_retries_get_after_head_403(monkeypatch):
    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def head(self, url, **kwargs):
            return SimpleNamespace(status_code=403)

        async def get(self, url, **kwargs):
            return SimpleNamespace(status_code=200)

    monkeypatch.setattr(
        "knowledge_engine.services.search.exa_domain_validate.httpx.AsyncClient",
        _Client,
    )
    live = validate_exa_domains_blocking(["peps.python.org"])
    assert live == ["peps.python.org"]


def _gil_expansion() -> ExaSearchContextExpansion:
    return ExaSearchContextExpansion(
        intent="language_api",
        primary_domains=["peps.python.org", "docs.python.org"],
        allowed_categories=["github"],
        search_type="keyword",
        use_broader_search=True,
        include_official_docs=True,
    )


def test_search_expanded_gil_pass1_uses_validated_hosts(monkeypatch):
    monkeypatch.setattr(
        "knowledge_engine.services.search.exa_source_expand.expand_search_context_with_flash_lite",
        lambda q: _gil_expansion(),
    )
    monkeypatch.setattr(
        "knowledge_engine.services.search.exa_domain_validate.prepare_exa_pass1_domains_blocking",
        lambda domains: list(domains),
    )
    monkeypatch.setattr(
        "knowledge_engine.services.search.exa_source_expand.filter_pass1_official_hosts",
        lambda domains: list(domains),
    )
    monkeypatch.setattr(
        "knowledge_engine.services.search.exa_source_expand.absorb_new_exa_hosts",
        lambda urls, max_new=3: None,
    )
    calls: list[dict] = []

    def fake_search(self, query, **kwargs):
        calls.append(kwargs)
        return ExaSearchResponse(
            query=query,
            hits=[
                ExaSearchHit(
                    url="https://peps.python.org/pep-0703/",
                    title="PEP 703",
                ),
                ExaSearchHit(
                    url="https://docs.python.org/3/c-api/init.html",
                    title="Initialization",
                ),
            ],
            include_domains=list(kwargs.get("include_domains") or []),
            exclude_domains=[],
            category=str(kwargs.get("category") or ""),
        )

    monkeypatch.setattr(ExaSearchClient, "search", fake_search)
    resp = ExaSearchClient(api_key="test").search_expanded("CPython GIL internals")
    assert len(calls) == 1
    assert calls[0]["include_domains"] == ["peps.python.org", "docs.python.org"]
    assert calls[0]["category"] is None
    urls = {h.url for h in resp.hits}
    assert any("peps.python.org" in u for u in urls)
    assert any("docs.python.org" in u for u in urls)


def test_search_expanded_empty_validated_goes_pass2(monkeypatch):
    monkeypatch.setattr(
        "knowledge_engine.services.search.exa_source_expand.expand_search_context_with_flash_lite",
        lambda q: _gil_expansion(),
    )
    monkeypatch.setattr(
        "knowledge_engine.services.search.exa_domain_validate.prepare_exa_pass1_domains_blocking",
        lambda domains: [],
    )
    monkeypatch.setattr(
        "knowledge_engine.services.search.exa_source_expand.filter_pass1_official_hosts",
        lambda domains: list(domains),
    )
    monkeypatch.setattr(
        "knowledge_engine.services.search.exa_source_expand.absorb_new_exa_hosts",
        lambda urls, max_new=3: None,
    )
    calls: list[dict] = []

    def fake_search(self, query, **kwargs):
        calls.append(kwargs)
        return ExaSearchResponse(
            query=query,
            hits=[
                ExaSearchHit(url="https://github.com/python/cpython", title="CPython")
            ],
            include_domains=list(kwargs.get("include_domains") or []),
            exclude_domains=[],
            category=str(kwargs.get("category") or ""),
        )

    monkeypatch.setattr(ExaSearchClient, "search", fake_search)
    ExaSearchClient(api_key="test").search_expanded("CPython GIL internals")
    assert len(calls) == 1
    assert "include_domains" not in calls[0] or calls[0]["include_domains"] == []
    assert calls[0]["category"] == "github"


def test_search_expanded_pass1_zero_hits_goes_pass2(monkeypatch):
    monkeypatch.setattr(
        "knowledge_engine.services.search.exa_source_expand.expand_search_context_with_flash_lite",
        lambda q: _gil_expansion(),
    )
    monkeypatch.setattr(
        "knowledge_engine.services.search.exa_domain_validate.prepare_exa_pass1_domains_blocking",
        lambda domains: list(domains),
    )
    monkeypatch.setattr(
        "knowledge_engine.services.search.exa_source_expand.filter_pass1_official_hosts",
        lambda domains: list(domains),
    )
    monkeypatch.setattr(
        "knowledge_engine.services.search.exa_source_expand.absorb_new_exa_hosts",
        lambda urls, max_new=3: None,
    )
    calls: list[dict] = []

    def fake_search(self, query, **kwargs):
        calls.append(kwargs)
        inc = kwargs.get("include_domains") or []
        if inc:
            return ExaSearchResponse(
                query=query,
                hits=[],
                include_domains=list(inc),
                exclude_domains=[],
                category="",
            )
        return ExaSearchResponse(
            query=query,
            hits=[
                ExaSearchHit(url="https://github.com/python/cpython", title="CPython")
            ],
            include_domains=[],
            exclude_domains=[],
            category=str(kwargs.get("category") or ""),
        )

    monkeypatch.setattr(ExaSearchClient, "search", fake_search)
    ExaSearchClient(api_key="test").search_expanded("CPython GIL internals")
    assert len(calls) == 2
    assert calls[0]["include_domains"] == ["peps.python.org", "docs.python.org"]
    assert calls[0]["category"] is None
    assert calls[1]["include_domains"] == []
    assert calls[1]["category"] == "github"


def test_community_blog_status_is_reject_by_contract():
    verdict = DomainAuthorityVerdict(
        domain="habr.com",
        classification="COMMUNITY_BLOG",
        status="KEEP",
        reason="community writing",
    )
    assert verdict.status == "REJECT"


def test_filter_pass1_drops_habr_keeps_foundational_docs(monkeypatch):
    batch_calls: list[list[str]] = []

    def fake_batch(hosts: list[str]) -> list[DomainAuthorityItem]:
        batch_calls.append(list(hosts))
        out: list[DomainAuthorityItem] = []
        for domain in hosts:
            host = clean_domain_for_exa(domain)
            cls = "VENDOR_BLOG" if host == "engineering.fb.com" else "COMMUNITY_BLOG"
            out.append(
                DomainAuthorityItem(
                    domain=host,
                    classification=cls,
                    general_summary=(
                        "Vendor engineering blog" if cls == "VENDOR_BLOG" else ""
                    ),
                    reason="not pass1 official",
                )
            )
        return out

    monkeypatch.setattr(
        "knowledge_engine.services.search.exa_source_expand.classify_exa_domains_batch_with_flash_lite",
        fake_batch,
    )
    out = filter_pass1_official_hosts(
        ["habr.com", "engineering.fb.com", "docs.python.org"]
    )
    assert out == ["docs.python.org"]
    assert not is_official_docs_host("habr.com")
    assert len(batch_calls) == 1
    assert set(batch_calls[0]) == {"habr.com", "engineering.fb.com"}


def test_search_expanded_habr_live_not_in_pass1(monkeypatch):
    monkeypatch.setattr(
        "knowledge_engine.services.search.exa_source_expand.expand_search_context_with_flash_lite",
        lambda q: ExaSearchContextExpansion(
            intent="mixed",
            primary_domains=["habr.com", "docs.python.org"],
            allowed_categories=["github"],
            search_type="auto",
            use_broader_search=True,
            include_official_docs=True,
        ),
    )
    monkeypatch.setattr(
        "knowledge_engine.services.search.exa_domain_validate.prepare_exa_pass1_domains_blocking",
        lambda domains: list(domains),
    )
    monkeypatch.setattr(
        "knowledge_engine.services.search.exa_source_expand.classify_exa_domains_batch_with_flash_lite",
        lambda hosts: [
            DomainAuthorityItem(
                domain=clean_domain_for_exa(h),
                classification="COMMUNITY_BLOG",
                general_summary="",
                reason="habr community blog",
            )
            for h in hosts
        ],
    )
    monkeypatch.setattr(
        "knowledge_engine.services.search.exa_source_expand.absorb_new_exa_hosts",
        lambda urls, max_new=3: None,
    )
    calls: list[dict] = []

    def fake_search(self, query, **kwargs):
        calls.append(kwargs)
        return ExaSearchResponse(
            query=query,
            hits=[
                ExaSearchHit(
                    url="https://docs.python.org/3/c-api/init.html",
                    title="Init",
                )
            ],
            include_domains=list(kwargs.get("include_domains") or []),
            exclude_domains=[],
            category=str(kwargs.get("category") or ""),
        )

    monkeypatch.setattr(ExaSearchClient, "search", fake_search)
    ExaSearchClient(api_key="test").search_expanded("GIL")
    assert calls
    assert calls[0]["include_domains"] == ["docs.python.org"]
    assert "habr.com" not in (calls[0]["include_domains"] or [])


def test_batch_classify_one_lite_call_for_unknown_hosts(monkeypatch):
    lite_calls: list[tuple[object, str]] = []

    def fake_lite(system, user, anchor, schema, label, **kwargs):
        lite_calls.append((schema, label))
        return BatchDomainAuthorityResponse(
            items=[
                DomainAuthorityItem(
                    domain="habr.com",
                    classification="COMMUNITY_BLOG",
                    general_summary="Community engineering articles and tutorials",
                    reason="aggregator blog",
                ),
                DomainAuthorityItem(
                    domain="engineering.fb.com",
                    classification="VENDOR_BLOG",
                    general_summary="Meta production engineering blog",
                    reason="vendor blog",
                ),
                DomainAuthorityItem(
                    domain="docs.pytorch.org",
                    classification="OFFICIAL_DOCS",
                    general_summary=(
                        "PyTorch core internals, official API specifications, "
                        "and framework documentation"
                    ),
                    reason="official docs",
                ),
            ]
        )

    monkeypatch.setattr(
        "knowledge_engine.src.analytics.gemini_v07.run_gemini_lite_structured",
        fake_lite,
    )
    items = classify_exa_domains_batch_with_flash_lite(
        ["habr.com", "engineering.fb.com", "docs.pytorch.org"]
    )
    assert len(lite_calls) == 1
    assert lite_calls[0][0] is BatchDomainAuthorityResponse
    assert lite_calls[0][1] == "exa / domain_authority_batch"
    by_host = {it.domain: it.classification for it in items}
    assert by_host["habr.com"] == "COMMUNITY_BLOG"
    assert by_host["engineering.fb.com"] == "VENDOR_BLOG"
    assert by_host["docs.pytorch.org"] == "OFFICIAL_DOCS"


def test_batch_classify_skips_lite_for_foundational_docs(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("Lite must not run for foundational_docs only")

    monkeypatch.setattr(
        "knowledge_engine.src.analytics.gemini_v07.run_gemini_lite_structured",
        boom,
    )
    items = classify_exa_domains_batch_with_flash_lite(["docs.python.org"])
    assert items[0].domain == "docs.python.org"
    assert items[0].classification == "OFFICIAL_DOCS"


def test_expand_merges_registry_official_docs(monkeypatch):
    monkeypatch.setattr(
        "knowledge_engine.services.search.exa_source_expand._lookup_registry_official_docs",
        lambda topic: ["docs.python.org"],
    )

    def fake_lite(system, user, anchor, schema, label, **kwargs):
        return ExaSearchContextExpansion(
            intent="language_api",
            primary_domains=["peps.python.org"],
            allowed_categories=["github"],
            search_type="keyword",
            use_broader_search=True,
            include_official_docs=True,
            topic_vector_query=(
                "CPython core internals, official PEP specifications, "
                "and standard library documentation"
            ),
        )

    monkeypatch.setattr(
        "knowledge_engine.src.analytics.gemini_v07.run_gemini_lite_structured",
        fake_lite,
    )
    ctx = expand_search_context_with_flash_lite("CPython GIL internals")
    assert ctx.topic_vector_query.startswith("CPython core internals")
    assert "peps.python.org" in ctx.primary_domains
    assert "docs.python.org" in ctx.primary_domains
