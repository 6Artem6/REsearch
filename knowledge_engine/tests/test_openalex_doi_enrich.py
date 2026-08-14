"""OpenAlex DOI enrichment path (non-arXiv) + vendor short-circuit."""

from __future__ import annotations

from knowledge_engine.src.services.openalex_evaluator import (
    OpenAlexEvaluator,
    looks_like_vendor_doc,
)


def test_extract_doi_from_resolver_and_bare():
    ev = OpenAlexEvaluator(enabled=False)
    assert (
        ev.extract_doi("https://doi.org/10.1038/s41586-023-12345-6")
        == "10.1038/s41586-023-12345-6"
    )
    assert ev.extract_doi("doi:10.1145/1234567.7654321") == "10.1145/1234567.7654321"
    assert ev.extract_doi("https://arxiv.org/abs/2301.07041") is None


def test_vendor_docs_still_short_circuit_to_one():
    ev = OpenAlexEvaluator(enabled=True)
    assert looks_like_vendor_doc(
        "https://docs.python.org/3/library/asyncio.html",
        "documentation",
    )
    assert (
        ev.fetch_trust_score_sync(
            "https://docs.python.org/3/library/asyncio.html",
            is_doc=True,
        )
        == 1.0
    )


def test_non_arxiv_without_doi_soft_fallback():
    """After removing arXiv-only gate, plain blogs are soft-fallback (not 1.0)."""
    ev = OpenAlexEvaluator(enabled=True)
    score = ev.fetch_trust_score_sync("https://blog.example.com/post", is_doc=False)
    assert score == 0.3


def test_doi_lookup_calls_openalex_works(monkeypatch):
    seen: list[str] = []

    class _Resp:
        status_code = 200

        def json(self):
            return {
                "cited_by_count": 120,
                "open_access": {"is_oa": True},
                "primary_location": {
                    "version": "publishedVersion",
                    "source": {"type": "journal", "display_name": "Nature"},
                    "is_oa": True,
                },
                "locations": [],
            }

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, headers=None):
            seen.append(url)
            return _Resp()

    monkeypatch.setattr(
        "knowledge_engine.src.services.openalex_evaluator.httpx.Client",
        _Client,
    )
    monkeypatch.setattr(
        "knowledge_engine.src.services.openalex_evaluator._quota_try_consume",
        lambda n=1: True,
    )

    ev = OpenAlexEvaluator(enabled=True, email="test@example.com")
    score = ev.fetch_trust_score_sync(
        "https://www.semanticscholar.org/paper/abc",
        doi="10.1038/s41586-023-05371-5",
    )
    assert score >= 0.7
    assert len(seen) == 1
    assert "doi.org" in seen[0]
    assert "10.1038" in seen[0]


def test_doi_in_url_without_arxiv(monkeypatch):
    seen: list[str] = []

    class _Resp:
        status_code = 200

        def json(self):
            return {
                "cited_by_count": 10,
                "open_access": {"is_oa": False},
                "primary_location": {
                    "source": {"type": "repository"},
                },
                "locations": [],
            }

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, headers=None):
            seen.append(url)
            return _Resp()

    monkeypatch.setattr(
        "knowledge_engine.src.services.openalex_evaluator.httpx.Client",
        _Client,
    )
    monkeypatch.setattr(
        "knowledge_engine.src.services.openalex_evaluator._quota_try_consume",
        lambda n=1: True,
    )

    ev = OpenAlexEvaluator(enabled=True, email="test@example.com")
    score = ev.fetch_trust_score_sync("https://doi.org/10.1145/3292500.3330701")
    assert 0.0 < score <= 1.0
    assert any("10.1145" in u for u in seen)


def test_is_oa_boosts_unpublished_score():
    preprint = OpenAlexEvaluator.score_from_openalex_payload(
        {
            "cited_by_count": 0,
            "open_access": {"is_oa": False},
            "primary_location": {"source": {"type": "repository"}},
            "locations": [],
        }
    )
    oa_preprint = OpenAlexEvaluator.score_from_openalex_payload(
        {
            "cited_by_count": 0,
            "open_access": {"is_oa": True},
            "primary_location": {"source": {"type": "repository"}, "is_oa": True},
            "locations": [],
        }
    )
    assert oa_preprint > preprint
