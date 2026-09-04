"""Vector edge-case lexicon for deep_analysis digests (no stem regex)."""

from __future__ import annotations

import pytest

from knowledge_engine.src.node_deep_dive.deep_analysis_coverage import (
    make_technical_digest,
)
from knowledge_engine.src.node_deep_dive.edge_case_lexicon import (
    VectorEdgeCaseLexicon,
    set_edge_case_lexicon_for_tests,
)
from knowledge_engine.tests.edge_case_embed_probe import edge_case_probe_embed


@pytest.fixture()
def edge_lexicon() -> VectorEdgeCaseLexicon:
    lex = VectorEdgeCaseLexicon(
        embed_fn=edge_case_probe_embed,
        persist=False,
        auto_sync=True,
        threshold=0.35,
        enabled=True,
    )
    set_edge_case_lexicon_for_tests(lex)
    yield lex
    set_edge_case_lexicon_for_tests(None)


def test_edge_lexicon_classifies_timeout_thesis(
    edge_lexicon: VectorEdgeCaseLexicon,
) -> None:
    label, score = edge_lexicon.classify(
        "Edge: таймаут одного воркера при gather раздувает latency каскадом"
    )
    assert label in ("edge_case", "bottleneck", "trade_off")
    assert score >= edge_lexicon.threshold
    assert edge_lexicon.is_edge_related(
        "Cancel vs wait — архитектурный компромисс при частичном сбое"
    )


def test_edge_lexicon_rejects_generic_overview(
    edge_lexicon: VectorEdgeCaseLexicon,
) -> None:
    label, _score = edge_lexicon.classify(
        "Субагенты делегируют задачи главному агенту в иерархии"
    )
    assert label == ""


def test_exhausted_digest_uses_vector_lexicon(
    edge_lexicon: VectorEdgeCaseLexicon,
) -> None:
    text = (
        "## 1. Обзор оркестратора\n"
        "Субагенты делегируют задачи главному агенту.\n"
        "## 2. Edge: таймаут одного воркера при gather\n"
        "Если один субагент зависает, asyncio.gather блокирует агрегацию и "
        "раздувает latency каскадом.\n"
        "## 3. Trade-off: cancel vs wait\n"
        "Cancel быстрее, но теряет частичный прогресс — явный компромисс.\n"
    )
    digest = make_technical_digest(text, rag_exhausted=True)
    assert digest.startswith("EDGE_CASES_COVERED:")
    assert "таймаут" in digest.lower() or "gather" in digest.lower()
