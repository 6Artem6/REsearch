"""OpenAlex trust_score enrichment + RAG hard-cutoff / ranking integrity."""

from __future__ import annotations

import asyncio
import json

from knowledge_engine.src.services import openalex_evaluator as oa
from knowledge_engine.src.services.openalex_evaluator import (
    OpenAlexEvaluator,
    coerce_trust_score,
    final_retrieval_score,
    looks_like_vendor_doc,
    openalex_quota_snapshot,
    passes_trust_hard_cutoff,
    prefetch_trust_scores_async,
)


def test_extract_arxiv_id_from_urls():
    ev = OpenAlexEvaluator(enabled=False)
    assert ev.extract_arxiv_id("https://arxiv.org/abs/2301.07041") == "2301.07041"
    assert ev.extract_arxiv_id("https://arxiv.org/pdf/1706.03762v7.pdf") == "1706.03762"
    assert ev.extract_arxiv_id("not-a-paper") is None


def test_zero_citation_preprint_score_in_expected_band():
    score = OpenAlexEvaluator.score_from_openalex_payload(
        {
            "cited_by_count": 0,
            "primary_location": {
                "source": {"type": "repository", "display_name": "arXiv"},
            },
            "locations": [],
        }
    )
    assert 0.15 <= score <= 0.25
    assert score == 0.15


def test_high_citation_published_score_high():
    score = OpenAlexEvaluator.score_from_openalex_payload(
        {
            "cited_by_count": 500,
            "primary_location": {
                "version": "publishedVersion",
                "source": {"type": "journal", "display_name": "Nature"},
            },
            "locations": [
                {"source": {"type": "journal", "display_name": "Nature"}},
            ],
        }
    )
    assert 0.7 <= score <= 1.0


def test_vendor_docs_are_full_trust():
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


def test_non_academic_url_soft_fallback_without_doi():
    ev = OpenAlexEvaluator(enabled=True)
    assert (
        ev.fetch_trust_score_sync("https://blog.example.com/post", is_doc=False) == 0.3
    )


def test_missing_trust_defaults_to_one():
    assert coerce_trust_score(None) == 1.0
    assert coerce_trust_score("bad") == 1.0


def test_final_score_trust_beats_low_trust_higher_similarity():
    weak_preprint = final_retrieval_score(0.80, 0.15)
    trusted_doc = final_retrieval_score(0.75, 1.0)
    assert trusted_doc > weak_preprint


def test_hard_cutoff_drops_weak_preprint_unless_very_similar():
    assert passes_trust_hard_cutoff(0.70, 0.15) is False
    assert passes_trust_hard_cutoff(0.90, 0.15) is True
    assert passes_trust_hard_cutoff(0.50, 0.85) is True
    assert passes_trust_hard_cutoff(0.50, None) is True  # legacy default trust=1


def test_candidate_early_exit_before_heavy_rerank():
    """Filter weak rag hits before CE/MMR; leave unscored stubs alone."""
    from knowledge_engine.services.lecture_context_rerank import LectureContextCandidate
    from knowledge_engine.src.services.openalex_evaluator import (
        filter_candidates_trust_hard_cutoff,
    )

    pool = [
        LectureContextCandidate(
            label="rag_fine_chunk",
            formatted="",
            plain="weak preprint",
            trust_score=0.15,
            vector_similarity=0.70,
        ),
        LectureContextCandidate(
            label="rag_fine_chunk",
            formatted="",
            plain="strong docs",
            trust_score=1.0,
            vector_similarity=0.60,
        ),
        LectureContextCandidate(
            label="registry_stub",
            formatted="",
            plain="no vector signal yet",
            trust_score=0.15,
            vector_similarity=0.0,
        ),
    ]
    kept, dropped = filter_candidates_trust_hard_cutoff(pool)
    assert dropped == 1
    assert len(kept) == 2
    assert kept[0].plain == "strong docs"
    assert kept[1].label == "registry_stub"


def test_rank_candidates_by_final_score():
    rows = [
        {"id": "arxiv_weak", "sim": 0.82, "trust": 0.15},
        {"id": "docs", "sim": 0.77, "trust": 1.0},
        {"id": "old_missing_trust", "sim": 0.70, "trust": None},
    ]
    ranked = sorted(
        rows,
        key=lambda r: final_retrieval_score(
            r["sim"], coerce_trust_score(r["trust"], default=1.0)
        ),
        reverse=True,
    )
    assert [r["id"] for r in ranked] == ["docs", "old_missing_trust", "arxiv_weak"]


def test_openalex_http_fallback_on_404(monkeypatch):
    class _Resp:
        status_code = 404

        def json(self):
            return {}

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, *args, **kwargs):
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
    score = ev.fetch_trust_score_sync("https://arxiv.org/abs/9999.99999")
    assert score == 0.25


def test_daily_quota_fallback(tmp_path, monkeypatch):
    path = tmp_path / "openalex_quota_state.json"
    path.write_text(
        json.dumps(
            {
                "day_utc": oa._utc_day(),
                "requests_today": 100000,
                "daily_limit": 100000,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(oa, "_QUOTA_PATH", path)
    monkeypatch.setattr(oa, "OPENALEX_DAILY_LIMIT", 100000)

    called = {"n": 0}

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, *args, **kwargs):
            called["n"] += 1
            raise AssertionError("HTTP must not run when quota exhausted")

    monkeypatch.setattr(oa.httpx, "Client", _Client)
    ev = OpenAlexEvaluator(enabled=True, email="test@example.com")
    score = ev.fetch_trust_score_sync("https://arxiv.org/abs/1234.56789")
    assert score == 0.3
    assert called["n"] == 0
    snap = openalex_quota_snapshot()
    assert int(snap.get("requests_today") or 0) >= 100000


def test_batch_prefetch_uses_gather(monkeypatch):
    calls: list[str] = []

    async def _fake_fetch(self, url, is_doc=False, *, client=None):
        calls.append(url)
        await asyncio.sleep(0.01)
        return 0.42 if "arxiv" in url else 1.0

    monkeypatch.setattr(OpenAlexEvaluator, "fetch_trust_score", _fake_fetch)

    async def _run():
        ev = OpenAlexEvaluator(enabled=True, concurrency=5)
        return await prefetch_trust_scores_async(
            [
                "https://arxiv.org/abs/1111.22222",
                "https://docs.example.com/x",
                "https://arxiv.org/abs/3333.44444",
            ],
            evaluator=ev,
            concurrency=5,
        )

    out = asyncio.run(_run())
    assert len(out) == 3
    assert len(calls) == 3
    assert out["https://docs.example.com/x"] == 1.0


def test_lecture_finalize_orders_by_trust():
    from knowledge_engine.services.lecture_context_rerank import LectureContextCandidate
    from knowledge_engine.services.lecture_rag_context import (
        _finalize_lecture_citation_candidates,
    )

    selected = [
        LectureContextCandidate(
            label="a",
            formatted="",
            plain="weak preprint chunk text enough length",
            source_title="Weak arXiv",
            trust_score=0.15,
            retrieval_score=0.9,
        ),
        LectureContextCandidate(
            label="b",
            formatted="",
            plain="vendor docs chunk text enough length here",
            source_title="Cloudflare Docs",
            trust_score=1.0,
            retrieval_score=0.7,
        ),
    ]
    out = _finalize_lecture_citation_candidates(selected)
    assert out[0].source_title == "Cloudflare Docs"
    assert out[0].source_index == 1
    assert "trust=1.00" in out[0].formatted
    assert out[1].source_title == "Weak arXiv"
    assert "trust=0.15" in out[1].formatted
