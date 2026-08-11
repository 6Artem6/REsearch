"""Two-phase REDUCE strategy, ScopeType coercion, parse error logging."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock

import pytest

from knowledge_engine.schemas.extraction import (
    KnowledgeAtom,
    ScopeType,
    coerce_scope_type,
)
from knowledge_engine.services.article_ingestion.blog_spatial_schemas import (
    DeduplicatedAtomsResponse,
    FinalArticleSummaryResponse,
    MapWindowResponse,
)
from knowledge_engine.services.article_ingestion.blog_spatial_summarizer import (
    _REDUCE_DEDUP_SYSTEM,
    _REDUCE_SYNTHESIS_SYSTEM,
    _REDUCE_SYSTEM,
    MapReduceArticleJob,
    _run_two_phase_reduce,
    run_reduce,
)
from knowledge_engine.services.article_ingestion.paragraph_token_splitter import (
    TokenWindowChunk,
)
from knowledge_engine.services.llm import gemma_client as gc


def test_coerce_scope_type_aliases() -> None:
    assert coerce_scope_type("principle") == ScopeType.PRINCIPLE
    assert coerce_scope_type("MECHANICS") == ScopeType.MECHANIC
    assert coerce_scope_type("instance") == ScopeType.INSTANCE
    assert coerce_scope_type("Evidence") == ScopeType.INSTANCE
    assert coerce_scope_type("nope") == ScopeType.PRINCIPLE
    assert KnowledgeAtom(
        scope="global", statement="Agent isolation is required"
    ).scope == (ScopeType.PRINCIPLE)


def test_final_response_soft_takeaways() -> None:
    final = FinalArticleSummaryResponse.model_validate(
        {
            "executive_summary": "Digest.",
            "key_takeaways": [
                {
                    "scope": "mechanic",
                    "statement": "Hook pipeline runs before tool call",
                },
            ],
            "knowledge_atoms": [],
            "target_diagrams_for_vlm": None,
        }
    )
    assert final.key_takeaways[0].startswith("[SCOPE: MECHANIC]")
    assert final.target_diagrams_for_vlm == []


def test_deduplicated_atoms_schema() -> None:
    out = DeduplicatedAtomsResponse.model_validate(
        {
            "knowledge_atoms": [
                {
                    "scope": "INSTANCE",
                    "statement": "Latency is 8.3 ms on M1",
                    "context_quote": "table 2",
                }
            ]
        }
    )
    assert len(out.knowledge_atoms) == 1
    assert out.knowledge_atoms[0].scope == ScopeType.INSTANCE


def test_reduce_prompt_constants_are_module_level() -> None:
    assert "DeduplicatedAtomsResponse" in _REDUCE_DEDUP_SYSTEM
    assert "Do NOT extract new facts" in _REDUCE_SYNTHESIS_SYSTEM
    assert "FinalArticleSummaryResponse" in _REDUCE_SYSTEM
    assert "DO NOT output <thought>" in _REDUCE_DEDUP_SYSTEM


def test_parse_structured_logs_validation_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from pydantic import BaseModel

    class Tiny(BaseModel):
        must_be_int: int

    with caplog.at_level(logging.ERROR, logger=gc.__name__):
        parsed = gc._parse_structured('{"must_be_int": "nope"}', Tiny)
    assert parsed is None
    msgs = [r.getMessage() for r in caplog.records]
    assert any("validation_errors" in m for m in msgs)
    assert any("nope" in m for m in msgs)


def test_run_reduce_dispatches_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def _legacy(*a, **k):
        calls.append("legacy")
        return FinalArticleSummaryResponse(
            executive_summary="x",
            key_takeaways=["[SCOPE: PRINCIPLE] claim here ok"],
            knowledge_atoms=[],
        )

    async def _two(*a, **k):
        calls.append("two")
        return None

    monkeypatch.setattr(
        "knowledge_engine.services.article_ingestion.blog_spatial_summarizer.REDUCE_STRATEGY",
        "legacy",
    )
    monkeypatch.setattr(
        "knowledge_engine.services.article_ingestion.blog_spatial_summarizer._run_legacy_reduce",
        _legacy,
    )
    monkeypatch.setattr(
        "knowledge_engine.services.article_ingestion.blog_spatial_summarizer._run_two_phase_reduce",
        _two,
    )
    job = MapReduceArticleJob(
        job_id="https://example.com/a",
        title="T",
        url="https://example.com/a",
        windows=[],
    )
    out = asyncio.run(run_reduce(job, [], http_client=AsyncMock()))
    assert out is not None
    assert calls == ["legacy"]


def test_two_phase_pins_atoms_from_phase1(monkeypatch: pytest.MonkeyPatch) -> None:
    atom_a = KnowledgeAtom(
        scope=ScopeType.PRINCIPLE,
        statement="Governed hooks must run before tool dispatch",
        source_chunk_ids=["paper_map_1"],
    )
    atom_b = KnowledgeAtom(
        scope=ScopeType.INSTANCE,
        statement="Latency measured at 8.3 ms on Apple Silicon",
        source_chunk_ids=["paper_map_1"],
    )

    async def _call(system, prompt, schema, **kwargs):
        if schema is DeduplicatedAtomsResponse:
            # Simulate Gemma forgetting source_chunk_ids — reattach must restore.
            return DeduplicatedAtomsResponse(
                knowledge_atoms=[
                    KnowledgeAtom(
                        scope=atom_a.scope,
                        statement=atom_a.statement,
                        source_chunk_ids=[],
                    ),
                    KnowledgeAtom(
                        scope=atom_b.scope,
                        statement=atom_b.statement,
                        source_chunk_ids=[],
                    ),
                ]
            )
        if schema is FinalArticleSummaryResponse:
            # Model "forgets" atoms — dispatcher must pin phase-1 list.
            return FinalArticleSummaryResponse(
                executive_summary="Summary about hooks.",
                key_takeaways=["[SCOPE: PRINCIPLE] hooks first"],
                knowledge_atoms=[],
            )
        return None

    monkeypatch.setattr(
        "knowledge_engine.services.article_ingestion.blog_spatial_summarizer._structured_reduce_call",
        _call,
    )
    job = MapReduceArticleJob(
        job_id="https://example.com/paper",
        title="Paper",
        url="https://example.com/paper",
        windows=[TokenWindowChunk(window_index=0, body="x")],
    )
    maps = [
        MapWindowResponse(
            window_role="Intro",
            window_summary="Window scaffold text.",
            knowledge_atoms=[atom_a, atom_b],
        )
    ]
    final = asyncio.run(
        _run_two_phase_reduce(job, maps, http_client=AsyncMock(), gemma_rl=None)
    )
    assert final is not None
    assert len(final.knowledge_atoms) == 2
    assert final.knowledge_atoms[0].statement == atom_a.statement
    assert "paper_map_1" in final.knowledge_atoms[0].source_chunk_ids
    assert "paper_map_1" in final.knowledge_atoms[1].source_chunk_ids
