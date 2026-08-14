"""Lecture external-search gate + waterfall helpers."""

from __future__ import annotations

import asyncio
import logging

import pytest

from knowledge_engine.services.lecture_context_rerank import LectureContextCandidate
from knowledge_engine.services.lecture_pipeline import (
    LectureRagStats,
    build_lecture_rag_stats,
    log_external_search_bypass,
    needs_primary_external_search,
    should_bypass_primary_external_search,
)
from knowledge_engine.src.node_deep_dive.lecture_search_orchestrator import (
    VerifiedExternalSource,
    _merge_sources,
    query_needs_en_translation,
    translate_to_en_query,
)
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput


def _cand(plain: str, score: float) -> LectureContextCandidate:
    return LectureContextCandidate(
        label="t",
        formatted=plain,
        plain=plain,
        retrieval_score=score,
    )


def test_build_stats_avg_score() -> None:
    stats = build_lecture_rag_stats(
        [],
        [
            _cand("x" * 40, 0.8),
            _cand("y" * 40, 0.4),
        ],
        [],
    )
    assert stats.local_sources_count == 2
    assert stats.local_avg_score == pytest.approx(0.6)
    assert stats.local_sum_score == pytest.approx(1.2)


def test_gate_requires_both_low_count_and_low_quality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "knowledge_engine.services.lecture_pipeline.LECTURE_MIN_LOCAL_SOURCES",
        3,
    )
    monkeypatch.setattr(
        "knowledge_engine.services.lecture_pipeline.LOCAL_QUALITY_THRESHOLD",
        0.5,
    )
    monkeypatch.setattr(
        "knowledge_engine.services.lecture_pipeline.LECTURE_LOCAL_QUALITY_THRESHOLD",
        0.5,
    )
    # Enough sources → bypass even if scores weak
    many_weak = LectureRagStats(
        local_sources_count=3,
        mmr_selected=3,
        pinned_count=0,
        route_url_count=0,
        has_quality_pinned=False,
        local_avg_score=0.2,
    )
    assert should_bypass_primary_external_search(many_weak)
    assert not needs_primary_external_search(many_weak)

    # Few sources but strong avg → bypass
    few_strong = LectureRagStats(
        local_sources_count=1,
        mmr_selected=1,
        pinned_count=0,
        route_url_count=0,
        has_quality_pinned=False,
        local_avg_score=0.7,
    )
    assert should_bypass_primary_external_search(few_strong)

    # Few + weak → need external
    few_weak = LectureRagStats(
        local_sources_count=1,
        mmr_selected=1,
        pinned_count=0,
        route_url_count=0,
        has_quality_pinned=False,
        local_avg_score=0.2,
    )
    assert needs_primary_external_search(few_weak)


def test_bypass_log_message(caplog: pytest.LogCaptureFixture) -> None:
    stats = LectureRagStats(
        local_sources_count=4,
        mmr_selected=4,
        pinned_count=0,
        route_url_count=0,
        has_quality_pinned=False,
        local_avg_score=0.6,
    )
    with caplog.at_level(logging.INFO):
        log_external_search_bypass(stats)
    assert any(
        "Local RAG context is sufficient (4 sources). Skipping external search."
        in r.message
        for r in caplog.records
    )


def test_merge_ranks_by_score_and_caps() -> None:
    a = VerifiedExternalSource(
        url="https://a.example/1",
        title="A",
        snippet="a",
        provider="exa",
        score=0.4,
    )
    b = VerifiedExternalSource(
        url="https://b.example/2",
        title="B",
        snippet="b",
        provider="consensus",
        score=0.9,
    )
    c = VerifiedExternalSource(
        url="https://c.example/3",
        title="C",
        snippet="c",
        provider="ss",
        score=0.7,
    )
    d = VerifiedExternalSource(
        url="https://d.example/4",
        title="D",
        snippet="d",
        provider="exa",
        score=0.2,
    )
    merged = _merge_sources([[a, d], [b], [c]], 3)
    assert [m.url for m in merged] == [
        "https://b.example/2",
        "https://c.example/3",
        "https://a.example/1",
    ]


def test_merge_keeps_higher_score_on_dedupe() -> None:
    low = VerifiedExternalSource(
        url="https://same.example/x",
        title="low",
        snippet="",
        provider="ss",
        score=0.2,
    )
    high = VerifiedExternalSource(
        url="https://same.example/x",
        title="high",
        snippet="",
        provider="exa",
        score=0.95,
    )
    merged = _merge_sources([[low], [high]], 3)
    assert len(merged) == 1
    assert merged[0].title == "high"


def test_query_needs_en_translation() -> None:
    assert query_needs_en_translation("Lifecycle hooks для агентов Cloudflare")
    assert not query_needs_en_translation("Agent lifecycle hooks Cloudflare Workers")
    assert translate_to_en_query("Agent lifecycle hooks") == "Agent lifecycle hooks"


def test_waterfall_exa_early_exit(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from knowledge_engine.src.node_deep_dive import lecture_search_orchestrator as orch

    exa = [
        VerifiedExternalSource(
            url=f"https://exa.example/{i}",
            title=f"t{i}",
            snippet="s",
            provider="exa",
            score=0.9 - 0.1 * i,
        )
        for i in range(3)
    ]

    async def _fake_exa(query: str, per_provider: int):
        return list(exa)

    called: list[str] = []

    async def _fake_provider(name: str, query: str, limit: int):
        called.append(name)
        return []

    monkeypatch.setattr(orch, "_exa_batch", _fake_exa)
    monkeypatch.setattr(orch, "_provider_sources", _fake_provider)
    monkeypatch.setattr(orch, "LECTURE_EXTERNAL_SEARCH_ENABLED", True)

    node = NodeDataInput(
        node_id="n1",
        title="Hooks",
        layer="foundation",
        core_concepts=["hooks"],
    )
    with caplog.at_level(logging.INFO):
        out = asyncio.run(
            orch.fetch_verified_external_sources(node, "lifecycle", top_k=3)
        )
    assert len(out) == 3
    assert called == []
    assert any("Skipping Consensus/SS" in r.message for r in caplog.records)


def test_waterfall_academic_when_exa_short(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from knowledge_engine.src.node_deep_dive import lecture_search_orchestrator as orch

    async def _fake_exa(query: str, per_provider: int):
        return [
            VerifiedExternalSource(
                url="https://exa.example/only",
                title="one",
                snippet="s",
                provider="exa",
                score=0.8,
            )
        ]

    async def _fake_provider(name: str, query: str, limit: int):
        # academic path must receive English-ish query (no Cyrillic focus)
        assert "агент" not in query.lower()
        return [
            VerifiedExternalSource(
                url=f"https://{name}.example/p",
                title=name,
                snippet="s",
                provider=name,
                score=0.6,
            )
        ]

    monkeypatch.setattr(orch, "_exa_batch", _fake_exa)
    monkeypatch.setattr(orch, "_provider_sources", _fake_provider)
    monkeypatch.setattr(orch, "LECTURE_EXTERNAL_SEARCH_ENABLED", True)
    monkeypatch.setattr(
        orch,
        "translate_to_en_query",
        lambda focus: "agent lifecycle hooks architecture",
    )

    node = NodeDataInput(
        node_id="n1",
        title="Хуки агентов",
        layer="foundation",
        core_concepts=["hooks"],
    )
    out = asyncio.run(
        orch.fetch_verified_external_sources(
            node,
            "lifecycle хуки агентов",
            top_k=3,
        )
    )
    assert len(out) == 3
    providers = {s.provider for s in out}
    assert "exa" in providers
