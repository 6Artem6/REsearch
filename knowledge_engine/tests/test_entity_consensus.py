"""Entity-guided consensus: grouping, local vector merge, anti-bloat, opt-in gate."""

from __future__ import annotations

import asyncio

from knowledge_engine.models.consensus import ConsensusNode, RawFact
from knowledge_engine.schemas.extraction import KnowledgeAtom, ScopeType
from knowledge_engine.services.deduplication.consensus_synthesizer import (
    consensus_aggregator_system_prompt,
    consensus_batch_limits,
    max_nodes_per_consensus_batch,
)
from knowledge_engine.services.deduplication.entity_consensus_engine import (
    LocalFactDeduplicator,
    apply_anti_bloat_anchors,
    apply_entity_consensus_to_atoms,
    build_token_bounded_batches,
    claim_dedup_is_enabled,
    collapse_facts_locally,
    consensus_batch_token_counts,
    group_facts_by_entity,
)


def _fact(
    fact_id: str,
    subject: str,
    *,
    entity_type: str = "python_library",
    predicate: str = "provides",
    obj: str = "HTTP APIs",
    anchor: str = "A1",
) -> RawFact:
    return RawFact(
        fact_id=fact_id,
        subject=subject,
        entity_type=entity_type,
        predicate=predicate,
        obj=obj,
        anchor=anchor,
        all_anchors=[anchor],
    )


def test_group_facts_by_entity_splits_fastapi_and_pydantic():
    facts = [
        _fact("f1", "FastAPI", obj="ASGI routing", anchor="A1"),
        _fact("f2", "FastAPI", obj="OpenAPI schema", anchor="A2"),
        _fact("f3", "Pydantic", predicate="validates", obj="JSON models", anchor="A3"),
        _fact(
            "f4",
            "unknown",
            entity_type="general",
            predicate="states",
            obj="misc",
            anchor="A4",
        ),
    ]
    grouped = group_facts_by_entity(facts)
    assert "python_library|fastapi" in grouped
    assert "python_library|pydantic" in grouped
    assert grouped["python_library|fastapi"][0].subject == "FastAPI"
    assert grouped["python_library|pydantic"][0].subject == "Pydantic"
    assert "general" in grouped
    fastapi_ids = {f.fact_id for f in grouped["python_library|fastapi"]}
    pydantic_ids = {f.fact_id for f in grouped["python_library|pydantic"]}
    assert fastapi_ids.isdisjoint(pydantic_ids)


def test_local_deduplicator_merges_paraphrases_via_embed_and_rerank():
    facts = [
        _fact("f1", "FastAPI", obj="high-performance web APIs", anchor="A1"),
        _fact("f2", "FastAPI", obj="fast web API framework", anchor="A2"),
        _fact("f3", "FastAPI", predicate="uses", obj="Starlette underneath", anchor="A3"),
    ]

    def embed(texts):
        vecs = []
        for text in texts:
            low = text.lower()
            if "web api" in low:
                vecs.append([1.0, 0.0])
            else:
                vecs.append([0.0, 1.0])
        return vecs

    def rerank(_query: str, docs: list[str]) -> list[float]:
        return [0.95 for _ in docs]

    engine = LocalFactDeduplicator(
        embed_fn=embed,
        rerank_fn=rerank,
        cluster_threshold=0.85,
        rerank_threshold=0.88,
    )
    collapsed = engine.deduplicate_entity_group(facts)
    assert len(collapsed) == 2
    merged = next(f for f in collapsed if set(f.merged_anchors()) >= {"A1", "A2"})
    assert "A1" in merged.merged_anchors() and "A2" in merged.merged_anchors()
    starlette = next(f for f in collapsed if "A3" in f.merged_anchors())
    assert starlette.merged_anchors() == ["A3"]


def test_anti_bloat_primary_anchors_cap_at_three():
    anchors = [f"A{i}" for i in range(1, 7)]
    primary = apply_anti_bloat_anchors(anchors, limit=3)
    assert primary == ["A1", "A2", "A3"]
    node = ConsensusNode(
        node_id="n1",
        entity="FastAPI",
        summary_text="FastAPI serves HTTP APIs.",
        primary_anchors=anchors,
        all_anchors=anchors,
        status="consensus",
    )
    assert len(node.primary_anchors) <= 3
    assert node.primary_anchors == ["A1", "A2", "A3"]
    assert node.all_anchors == anchors


def test_claim_dedup_mode_none_skips_pipeline(monkeypatch):
    import knowledge_engine.config as ke_config

    monkeypatch.setattr(ke_config, "CLAIM_DEDUP_MODE", "none")
    assert claim_dedup_is_enabled() is False

    calls: list[str] = []

    class _Spy(LocalFactDeduplicator):
        def deduplicate_entity_group(self, facts):
            calls.append("dedup")
            return list(facts)

    facts = [_fact("f1", "FastAPI"), _fact("f2", "Pydantic")]
    grouped = collapse_facts_locally(facts, mode="none", deduplicator=_Spy())
    assert len(grouped["python_library|fastapi"]) == 1
    assert len(grouped["python_library|pydantic"]) == 1
    assert calls == []

    atom = KnowledgeAtom(
        scope=ScopeType.INSTANCE,
        statement="FastAPI provides routing.",
        source_chunk_ids=["c1"],
    )
    result = asyncio.run(
        apply_entity_consensus_to_atoms([atom], deduplicator=_Spy())
    )
    assert result is None
    assert calls == []


def test_legacy_llm_and_claim_mmr_modes_skip_consensus(monkeypatch):
    import knowledge_engine.config as ke_config

    for mode in ("llm", "claim_mmr"):
        monkeypatch.setattr(ke_config, "CLAIM_DEDUP_MODE", mode)
        assert claim_dedup_is_enabled() is False


def test_batches_stay_under_token_cap():
    facts = [
        _fact(f"f{i}", "FastAPI", obj=f"detail {i}", anchor=f"A{i}")
        for i in range(1, 8)
    ]
    grouped = group_facts_by_entity(facts)
    system = consensus_aggregator_system_prompt()
    assert "Consensus Aggregator" in system
    assert "THREE" in system or "three" in system.lower()
    batches = build_token_bounded_batches(grouped, max_tokens=200, system_prompt=system)
    assert batches
    assert sum(len(b) for b in batches) == len(facts)


def test_batches_use_model_token_budget_without_dropping_facts():
    facts = [
        _fact(f"f{i}", f"Entity {i}", obj=f"detail {i}", anchor=f"A{i}")
        for i in range(1, 18)
    ]
    grouped = group_facts_by_entity(facts)
    batches = build_token_bounded_batches(
        grouped,
        max_tokens=4096,
        max_output_tokens=300,
        output_utilization=0.75,
    )

    assert len(batches) > 1
    assert [fact.fact_id for batch in batches for fact in batch] == [
        fact.fact_id for fact in facts
    ]
    for batch in batches:
        input_tokens, output_tokens = consensus_batch_token_counts(batch)
        assert input_tokens <= 4096
        assert output_tokens <= 225 or len(batch) == 1


def test_default_limits_are_ten_facts_and_3072_tokens(monkeypatch):
    import knowledge_engine.config as ke_config

    monkeypatch.setattr(ke_config, "MAX_CONSENSUS_BATCH_TOKENS", 3072)
    monkeypatch.setattr(ke_config, "MAX_CONSENSUS_NODES_PER_BATCH", 10)
    limits = consensus_batch_limits()
    assert limits.max_nodes == 10
    assert limits.max_input_tokens == 3072
    assert max_nodes_per_consensus_batch() == 10
    from pathlib import Path

    src = Path(ke_config.__file__).read_text(encoding="utf-8")
    assert 'MAX_CONSENSUS_BATCH_TOKENS", "3072"' in src
    assert 'MAX_CONSENSUS_NODES_PER_BATCH", "10"' in src


def test_node_cap_is_not_clamped_to_eight(monkeypatch):
    import knowledge_engine.config as ke_config

    monkeypatch.setattr(ke_config, "MAX_CONSENSUS_NODES_PER_BATCH", 12)
    assert max_nodes_per_consensus_batch() == 12
    facts = [
        _fact(f"f{i}", "FastAPI", obj=f"detail {i}", anchor=f"A{i}")
        for i in range(1, 13)
    ]
    batches = build_token_bounded_batches(group_facts_by_entity(facts))
    assert len(batches) == 1
    assert len(batches[0]) == 12


def test_ten_short_facts_fit_one_batch_under_3072(monkeypatch):
    import knowledge_engine.config as ke_config

    monkeypatch.setattr(ke_config, "MAX_CONSENSUS_BATCH_TOKENS", 3072)
    monkeypatch.setattr(ke_config, "MAX_CONSENSUS_NODES_PER_BATCH", 10)
    facts = [
        _fact(f"f{i}", "FastAPI", obj=f"detail {i}", anchor=f"A{i}")
        for i in range(1, 11)
    ]
    batches = build_token_bounded_batches(group_facts_by_entity(facts))
    assert len(batches) == 1
    assert len(batches[0]) == 10
    input_tokens, _output_tokens = consensus_batch_token_counts(batches[0])
    assert input_tokens <= 3072


def test_batches_cap_nodes_at_ten_per_call(monkeypatch):
    import knowledge_engine.config as ke_config

    monkeypatch.setattr(ke_config, "MAX_CONSENSUS_NODES_PER_BATCH", 10)
    facts = [
        _fact(f"f{i}", "FastAPI", obj=f"detail {i} extra context", anchor=f"A{i}")
        for i in range(1, 14)
    ]
    grouped = group_facts_by_entity(facts)
    batches = build_token_bounded_batches(grouped)
    assert [fact.fact_id for batch in batches for fact in batch] == [
        fact.fact_id for fact in facts
    ]
    assert all(len(batch) <= 10 for batch in batches)
    assert len(batches) >= 2
    for batch in batches:
        input_tokens, _output = consensus_batch_token_counts(batch)
        assert input_tokens <= 3072


def test_consensus_node_validates_without_status():
    node = ConsensusNode.model_validate(
        {
            "node_id": "n1",
            "entity": "FastAPI",
            "summary_text": "FastAPI отдаёт OpenAPI схему.",
            "primary_anchors": ["A1"],
            "all_anchors": ["A1", "A2"],
        }
    )
    assert node.status == "consensus"


def test_consensus_node_coerces_bad_status_and_alias_summary():
    from knowledge_engine.models.consensus import ConsensusBatchResponse

    node = ConsensusNode.model_validate(
        {
            "entity": "Pydantic",
            "summary": "Модели валидируют JSON.",
            "status": None,
        }
    )
    assert node.status == "consensus"
    assert node.entity == "Pydantic"
    assert "JSON" in node.summary_text
    assert node.node_id == "n-unknown"

    node2 = ConsensusNode.model_validate({"status": "nope", "entity": 12})
    assert node2.status == "consensus"
    assert node2.entity == "12"

    batch = ConsensusBatchResponse.model_validate(
        {
            "nodes": [
                {"summary_text": "ok text here", "entity": "X"},
                {"status": True, "entity": "Y", "summary_text": "still ok"},
            ]
        }
    )
    assert len(batch.nodes) == 2
    assert all(n.status == "consensus" for n in batch.nodes)


def test_raw_fact_text_fallbacks():
    fact = RawFact.model_validate({"fact_id": "f1"})
    assert fact.subject == "unknown"
    assert fact.predicate == "states"
    assert fact.obj == "unspecified"


def test_micro_batches_split_oversized_entity_group(monkeypatch):
    import knowledge_engine.config as ke_config
    from knowledge_engine.services.deduplication.consensus_synthesizer import (
        _micro_batches,
    )

    monkeypatch.setattr(ke_config, "MAX_CONSENSUS_NODES_PER_BATCH", 10)
    facts = [
        _fact(f"f{i}", "FastAPI", obj=f"detail {i}", anchor=f"A{i}")
        for i in range(1, 15)
    ]
    parts = _micro_batches(facts)
    assert sum(len(p) for p in parts) == 14
    assert all(len(p) <= 10 for p in parts)
    assert len(parts) == 2
    for part in parts:
        input_tokens, _output = consensus_batch_token_counts(part)
        assert input_tokens <= 3072


def test_synthesize_counts_tokens_without_nameerror():
    from unittest.mock import AsyncMock

    from knowledge_engine.services.deduplication.consensus_synthesizer import (
        synthesize_consensus_batches,
    )

    class _Rl:
        async def post_structured(self, *_a, **_k):
            return None

    facts = [_fact("f1", "FastAPI", obj="ASGI routing", anchor="A1")]
    nodes = asyncio.run(
        synthesize_consensus_batches(
            [facts],
            http_client=AsyncMock(),
            gemma_rl=_Rl(),
            allow_cloud=True,
        )
    )
    assert len(nodes) == 1
    assert nodes[0].entity == "FastAPI"


def test_synthesize_runs_batches_in_parallel():
    from unittest.mock import AsyncMock

    from knowledge_engine.services.deduplication.consensus_synthesizer import (
        synthesize_consensus_batches,
    )

    in_flight = 0
    peak = 0

    class _Rl:
        async def post_structured(self, *_a, **_k):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.05)
            in_flight -= 1
            return None

    batch_a = [_fact("f1", "FastAPI", obj="ASGI routing", anchor="A1")]
    batch_b = [_fact("f2", "Pydantic", obj="JSON models", anchor="A2")]
    nodes = asyncio.run(
        synthesize_consensus_batches(
            [batch_a, batch_b],
            http_client=AsyncMock(),
            gemma_rl=_Rl(),
            allow_cloud=True,
        )
    )
    assert len(nodes) == 2
    assert peak >= 2


def test_two_phase_reduce_skips_consensus_when_mode_none(monkeypatch):
    import knowledge_engine.config as ke_config
    from unittest.mock import AsyncMock

    from knowledge_engine.services.article_ingestion.blog_spatial_schemas import (
        DeduplicatedAtomsResponse,
        FinalArticleSummaryResponse,
        MapWindowResponse,
    )
    from knowledge_engine.services.article_ingestion.blog_spatial_summarizer import (
        MapReduceArticleJob,
        _run_two_phase_reduce,
    )
    from knowledge_engine.services.article_ingestion.paragraph_token_splitter import (
        TokenWindowChunk,
    )

    monkeypatch.setattr(ke_config, "CLAIM_DEDUP_MODE", "none")
    consensus_calls: list[int] = []

    async def _consensus_boom(*_a, **_k):
        consensus_calls.append(1)
        raise AssertionError("entity consensus must not run when CLAIM_DEDUP_MODE=none")

    monkeypatch.setattr(
        "knowledge_engine.services.deduplication.entity_consensus_engine.apply_entity_consensus_to_atoms",
        _consensus_boom,
    )

    atom = KnowledgeAtom(
        scope=ScopeType.PRINCIPLE,
        statement="Governed hooks must run before tool dispatch",
        source_chunk_ids=["paper_map_1"],
    )

    async def _call(system, prompt, schema, **kwargs):
        if schema is DeduplicatedAtomsResponse:
            return DeduplicatedAtomsResponse(knowledge_atoms=[atom])
        if schema is FinalArticleSummaryResponse:
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
            knowledge_atoms=[atom],
        )
    ]
    final = asyncio.run(
        _run_two_phase_reduce(job, maps, http_client=AsyncMock(), gemma_rl=None)
    )
    assert final is not None
    assert consensus_calls == []
    assert job.consensus_nodes == []


def test_two_phase_reduce_uses_consensus_when_enabled(monkeypatch):
    import knowledge_engine.config as ke_config
    from unittest.mock import AsyncMock

    from knowledge_engine.services.article_ingestion.blog_spatial_schemas import (
        DeduplicatedAtomsResponse,
        FinalArticleSummaryResponse,
        MapWindowResponse,
    )
    from knowledge_engine.services.article_ingestion.blog_spatial_summarizer import (
        MapReduceArticleJob,
        _run_two_phase_reduce,
    )
    from knowledge_engine.services.article_ingestion.paragraph_token_splitter import (
        TokenWindowChunk,
    )

    monkeypatch.setattr(ke_config, "CLAIM_DEDUP_MODE", "entity_consensus")
    gemma_dedup: list[str] = []
    consensus_calls: list[int] = []

    atom = KnowledgeAtom(
        scope=ScopeType.PRINCIPLE,
        statement="Governed hooks must run before tool dispatch",
        source_chunk_ids=["paper_map_1"],
    )
    node = ConsensusNode(
        node_id="n1",
        entity="hooks",
        summary_text="Хуки выполняются до dispatch.",
        primary_anchors=["A1"],
        all_anchors=["A1"],
        status="unique",
    )

    async def _consensus(*_a, **_k):
        consensus_calls.append(1)
        return [atom], [node]

    monkeypatch.setattr(
        "knowledge_engine.services.deduplication.entity_consensus_engine.apply_entity_consensus_to_atoms",
        _consensus,
    )

    async def _call(system, prompt, schema, **kwargs):
        if schema is DeduplicatedAtomsResponse:
            gemma_dedup.append("dedup")
            return DeduplicatedAtomsResponse(knowledge_atoms=[atom])
        if schema is FinalArticleSummaryResponse:
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
            knowledge_atoms=[atom],
        )
    ]
    final = asyncio.run(
        _run_two_phase_reduce(job, maps, http_client=AsyncMock(), gemma_rl=None)
    )
    assert final is not None
    assert consensus_calls == [1]
    assert gemma_dedup == []
    assert job.consensus_nodes == [node]
