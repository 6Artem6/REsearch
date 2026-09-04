"""LanceDB domain_registry: BGE-M3 bi-encoder + hard cosine threshold."""

from __future__ import annotations

import inspect
import math

import pytest

from knowledge_engine.db.lancedb_pool import reset_lancedb_pool_for_tests
from knowledge_engine.schemas.llm_contracts.exa_search import DomainAuthorityItem
from knowledge_engine.services.search.bge_m3_embed import (
    _assert_bi_encoder_name,
    set_domain_registry_embed_fn_for_tests,
)
from knowledge_engine.services.search.domain_registry import (
    DomainRegistry,
    reset_domain_registry_for_tests,
)

_CPYTHON = (
    "CPython core internals, official PEP specifications, "
    "and standard library documentation"
)
_LINUX = "Linux kernel architecture, syscall specs, and subsystem documentation"
_VEC_PY = [1.0, 0.0, 0.0]
_VEC_LINUX = [0.0, 1.0, 0.0]
_VEC_OTHER = [0.0, 0.0, 1.0]


def _embed(texts):
    out: list[list[float]] = []
    for raw in texts:
        t = (raw or "").lower()
        if "cpython" in t or "pep" in t:
            out.append(list(_VEC_PY))
        elif "linux" in t or "kernel" in t:
            out.append(list(_VEC_LINUX))
        else:
            out.append(list(_VEC_OTHER))
    return out


@pytest.fixture
def registry(tmp_path):
    set_domain_registry_embed_fn_for_tests(_embed)
    reset_lancedb_pool_for_tests()
    reset_domain_registry_for_tests()
    reg = DomainRegistry(db_path=tmp_path / "domain_registry.lancedb", cosine_min=0.82)
    yield reg
    set_domain_registry_embed_fn_for_tests(None)
    reset_lancedb_pool_for_tests()
    reset_domain_registry_for_tests()


def test_bi_encoder_rejects_cross_encoder_name():
    with pytest.raises(ValueError, match="bi-encoder"):
        _assert_bi_encoder_name("BAAI/bge-reranker-v2-m3")
    with pytest.raises(ValueError, match="nomic"):
        _assert_bi_encoder_name("nomic-embed-text")
    assert _assert_bi_encoder_name("BAAI/bge-m3") == "BAAI/bge-m3"


def test_embed_module_does_not_import_cross_encoder():
    import knowledge_engine.services.search.bge_m3_embed as embed_mod

    src = inspect.getsource(embed_mod)
    assert "CrossEncoder" not in src
    assert "bge-reranker-v2-m3" in src  # mentioned only as forbidden


def test_upsert_keep_and_search_above_threshold(registry: DomainRegistry):
    n = registry.upsert_keep_items(
        [
            DomainAuthorityItem(
                domain="docs.python.org",
                classification="OFFICIAL_DOCS",
                general_summary=_CPYTHON,
                reason="official",
            ),
            DomainAuthorityItem(
                domain="docs.kernel.org",
                classification="OFFICIAL_DOCS",
                general_summary=_LINUX,
                reason="official",
            ),
        ]
    )
    assert n == 2
    hits = registry.search_official_docs(_CPYTHON)
    assert hits == ["docs.python.org"]
    assert "docs.kernel.org" not in hits


def test_search_drops_below_hard_cosine(registry: DomainRegistry):
    registry.upsert_keep_items(
        [
            DomainAuthorityItem(
                domain="docs.python.org",
                classification="OFFICIAL_DOCS",
                general_summary=_CPYTHON,
                reason="official",
            )
        ]
    )
    # Orthogonal topic → cosine 0.0 < 0.82
    assert registry.search_official_docs(_LINUX) == []


def test_search_boundary_cosine_min(registry: DomainRegistry):
    """Cosine exactly 0.82 is a hit; 0.819 is not."""
    hi = 0.82
    rest = math.sqrt(max(0.0, 1.0 - hi * hi))

    def embed_boundary(texts):
        out = []
        for raw in texts:
            t = (raw or "").lower()
            if "query-hi" in t:
                out.append([hi, rest, 0.0])
            elif "query-lo" in t:
                out.append([0.819, math.sqrt(max(0.0, 1.0 - 0.819**2)), 0.0])
            else:
                out.append([1.0, 0.0, 0.0])
        return out

    set_domain_registry_embed_fn_for_tests(embed_boundary)
    registry.upsert_keep_items(
        [
            DomainAuthorityItem(
                domain="docs.python.org",
                classification="OFFICIAL_DOCS",
                general_summary="canonical gist stored as axis-x",
                reason="official",
            )
        ]
    )
    assert registry.search_official_docs("query-hi") == ["docs.python.org"]
    assert registry.search_official_docs("query-lo") == []


def test_reject_and_vendor_blog_not_in_pass1_lookup(registry: DomainRegistry):
    n = registry.upsert_keep_items(
        [
            DomainAuthorityItem(
                domain="habr.com",
                classification="COMMUNITY_BLOG",
                general_summary=_CPYTHON,
                reason="reject",
            ),
            DomainAuthorityItem(
                domain="engineering.fb.com",
                classification="VENDOR_BLOG",
                general_summary=_CPYTHON,
                reason="keep but not pass1",
            ),
        ]
    )
    assert n == 1  # vendor KEEP upserted; community skipped
    assert registry.search_official_docs(_CPYTHON) == []


def test_empty_general_summary_not_upserted(registry: DomainRegistry):
    n = registry.upsert_keep_items(
        [
            DomainAuthorityItem(
                domain="docs.python.org",
                classification="OFFICIAL_DOCS",
                general_summary="   ",
                reason="empty gist",
            )
        ]
    )
    assert n == 0
    assert registry.search_official_docs(_CPYTHON) == []
