"""Static Anchor Filtering wired into the ingest pipeline (pre-REDUCE cut).

Fast unit tests, no real Cross-Encoder call — see
tests/benchmarks/test_fact_relevance_pipeline.py for the real-API A/B
harness this graduated from. BLOG_SPATIAL_ANCHOR_FILTER_ENABLED itself
(default off) is a settings flag checked at the call site in
_reduce_final_from_maps — these tests exercise _apply_static_anchor_filter
directly, since a settings-flag branch needs no test of its own."""

from __future__ import annotations

import asyncio

from knowledge_engine.src.domains.ingestion.article_ingestion.blog_spatial_schemas import (
    MapWindowResponse,
)
from knowledge_engine.src.domains.ingestion.article_ingestion.blog_spatial_summarizer import (
    MapReduceArticleJob,
    _apply_static_anchor_filter,
)
from knowledge_engine.src.domains.ingestion.article_ingestion.paragraph_token_splitter import (
    TokenWindowChunk,
)
from knowledge_engine.src.shared.extraction import KnowledgeAtom, ScopeType


def _mock_score_relevance_pairs(monkeypatch, scores: list[float]) -> None:
    def fake(criterion: str, texts: list[str]) -> list[float]:
        assert len(texts) == len(scores)
        return list(scores)

    monkeypatch.setattr(
        "knowledge_engine.src.domains.ingestion.article_ingestion.anchor_relevance.score_relevance_pairs",
        fake,
    )


def _job(title: str, body: str) -> MapReduceArticleJob:
    return MapReduceArticleJob(
        job_id="https://example.com/a",
        title=title,
        url="https://example.com/a",
        windows=[TokenWindowChunk(window_index=0, body=body)],
    )


def test_apply_static_anchor_filter_cuts_below_threshold(monkeypatch) -> None:
    job = _job("Partitioning in Postgres", "Article: Partitioning in Postgres\n\nLead.")
    map_result = MapWindowResponse(
        window_summary="summary",
        knowledge_atoms=[
            KnowledgeAtom(scope=ScopeType.CONCEPT, statement="Relevant fact"),
            KnowledgeAtom(scope=ScopeType.PRACTICE, statement="Noise fact"),
        ],
    )
    _mock_score_relevance_pairs(monkeypatch, [0.8, 0.1])

    out = asyncio.run(_apply_static_anchor_filter(job, [map_result]))

    assert len(out) == 1
    assert [a.statement for a in out[0].knowledge_atoms] == ["Relevant fact"]
    # window_summary and other MapWindowResponse fields must survive the cut.
    assert out[0].window_summary == "summary"


def test_apply_static_anchor_filter_no_lead_no_title_keeps_atoms_unfiltered(
    monkeypatch,
) -> None:
    job = _job("", "")
    map_result = MapWindowResponse(
        window_summary="summary",
        knowledge_atoms=[KnowledgeAtom(scope=ScopeType.CONCEPT, statement="Fact")],
    )

    def fail_if_called(criterion: str, texts: list[str]) -> list[float]:
        raise AssertionError("Cross-Encoder must not be called without an anchor")

    monkeypatch.setattr(
        "knowledge_engine.src.domains.ingestion.article_ingestion.anchor_relevance.score_relevance_pairs",
        fail_if_called,
    )

    out = asyncio.run(_apply_static_anchor_filter(job, [map_result]))
    assert out == [map_result]


def test_apply_static_anchor_filter_preserves_none_windows(monkeypatch) -> None:
    job = _job("Title", "Article: Title\n\nLead text here.")
    _mock_score_relevance_pairs(monkeypatch, [])
    out = asyncio.run(_apply_static_anchor_filter(job, [None]))
    assert out == [None]
