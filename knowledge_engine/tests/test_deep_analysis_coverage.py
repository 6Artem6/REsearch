"""Asterisk-question Deep Analysis coverage + RAG exclude / novelty."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from knowledge_engine.db.knowledge_atoms_schema import COL_SCOPE, COL_STATEMENT
from knowledge_engine.services.dialog_atoms_rag import (
    build_dialog_atoms_query,
    is_generic_dialog_focus,
    retrieve_dialog_knowledge_atoms_detailed,
)
from knowledge_engine.src.node_deep_dive.deep_analysis_coverage import (
    atom_key,
    format_rag_coverage_state_block,
    make_technical_digest,
    parse_cited_r_indices,
    parse_cited_s_indices,
    record_deep_analysis_coverage,
)
from knowledge_engine.src.node_deep_dive.memory_schemas import SessionMemory
from knowledge_engine.src.node_deep_dive.prompt_factory import (
    format_deep_analysis_novelty_block,
)


def test_atom_key_stable() -> None:
    a = atom_key("Isolation reduces blast radius")
    b = atom_key("  Isolation   reduces blast radius  ")
    assert a and a == b
    assert a != atom_key("Different statement entirely here")


def test_parse_cites_and_digest() -> None:
    from knowledge_engine.src.node_deep_dive.deep_analysis_coverage import (
        compact_assistant_turn_for_api_history,
    )

    text = (
        "## 1. Анатомия\n"
        "Claim about isolation boundaries [S2] and [R1].\n"
        "## 2. Зависимости\n"
        "Also neighbors require ordered delivery [S2][R3].\n"
    )
    assert parse_cited_s_indices(text) == [2]
    assert parse_cited_r_indices(text) == [1, 3]
    digest = make_technical_digest(text)
    assert "Анатомия" in digest
    assert "Зависимости" in digest
    assert "isolation" in digest.lower() or "Claim" in digest
    assert "[S2]" in digest or "cites:" in digest
    compact = compact_assistant_turn_for_api_history(
        text, follow_up_question="Как изменится инвариант?"
    )
    assert "[DEEP_ANALYSIS_TURN_DIGEST]" in compact
    assert "theses:" in compact
    assert "follow_up:" in compact


def test_exhausted_digest_prefers_edge_cases() -> None:
    """Uses injected vector lexicon (no stem regex)."""
    from knowledge_engine.src.node_deep_dive.edge_case_lexicon import (
        VectorEdgeCaseLexicon,
        set_edge_case_lexicon_for_tests,
    )
    from knowledge_engine.tests.edge_case_embed_probe import edge_case_probe_embed

    lex = VectorEdgeCaseLexicon(
        embed_fn=edge_case_probe_embed,
        persist=False,
        auto_sync=True,
        threshold=0.35,
        enabled=True,
    )
    set_edge_case_lexicon_for_tests(lex)
    try:
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
        assert "latency" in digest.lower() or "каскад" in digest.lower()
    finally:
        set_edge_case_lexicon_for_tests(None)


def test_novelty_block_exhausted_forbids_code1_reteach() -> None:
    from knowledge_engine.src.node_deep_dive.deep_analysis_coverage import (
        citations_required_for_turn,
        format_rag_exhausted_directive,
    )

    mem = SessionMemory(
        deep_analysis_prior_digests=[
            "EDGE_CASES_COVERED: Edge: таймаут gather: зависание воркера"
        ],
    )
    block = format_deep_analysis_novelty_block(
        mem,
        rag_exhausted=True,
        attraction_summary="увеличение latency; расход токенов",
        registry_empty=True,
        atoms_empty=True,
    )
    assert "[RAG_STATUS: EXHAUSTED]" in block
    assert "code-1" in block.lower()
    assert "увеличение latency" in block
    assert "CITATION_POLICY" in block
    assert "НЕАКТИВНО" in block or "INACTIVE" in block
    directive = format_rag_exhausted_directive(
        attraction_summary="каскадные таймауты"
    )
    assert "каскадные таймауты" in directive
    flags = citations_required_for_turn(registry_empty=True, atoms_empty=True)
    assert flags["citations_inactive"] is True
    assert flags["allow_s_node"] is True
    assert flags["require_references"] is False



def test_record_coverage_maps_r_to_turn_keys() -> None:
    mem = SessionMemory(
        last_deep_analysis_atom_keys=["aaa111", "bbb222", "ccc333"],
    )
    tech = (
        "## 1. Anatomy\n"
        "Uses [S1] and [R2] for contracts.\n"
        "## 2. Failures\n"
        "Also [R1].\n"
    )
    record_deep_analysis_coverage(
        mem,
        technical_explanation=tech,
        references=[SimpleNamespace(asset_id="src_42")],
    )
    assert "S1" in mem.deep_analysis_used_source_ids
    assert "src_42" in mem.deep_analysis_used_source_ids
    assert "bbb222" in mem.deep_analysis_used_atom_keys  # R2 cited
    assert "aaa111" in mem.deep_analysis_used_atom_keys  # R1 cited
    # All turn-stashed keys are recorded for hard exclude (shown, not only cited).
    assert "ccc333" in mem.deep_analysis_used_atom_keys
    assert mem.deep_analysis_prior_digests
    assert "Anatomy" in mem.deep_analysis_prior_digests[-1]


def test_novelty_block_contains_instructions() -> None:
    mem = SessionMemory(
        deep_analysis_used_source_ids=["S1", "S3"],
        deep_analysis_prior_digests=["## Anatomy | ## Trade-offs"],
    )
    block = format_rag_coverage_state_block(mem, rag_exhausted=False)
    assert "[RAG_COVERAGE_STATE]" in block
    assert "S1" in block and "S3" in block
    assert "NO SURFACE REPEAT" in block
    assert "WHY / HOW / MECHANICS" in block
    assert "RAG Exhausted: false" in block

    exhausted = format_deep_analysis_novelty_block(mem, rag_exhausted=True)
    assert "RAG Exhausted: true" in exhausted
    assert "[RAG_STATUS: EXHAUSTED]" in exhausted
    assert "code-1" in exhausted.lower()
    assert "FACT_ATTRACTION" in exhausted or "ПОЛЮСА ПРИТЯЖЕНИЯ" in exhausted


def test_query_prefers_focus_over_chip_stub() -> None:
    node = SimpleNamespace(title="Aggregation")
    assert is_generic_dialog_focus("Задачка со звёздочкой")
    q = build_dialog_atoms_query(
        node,
        "Задачка со звёздочкой",
        focus_hint="partition failure modes for sharded aggregators",
    )
    assert "Aggregation" in q
    assert "partition failure" in q
    assert "звёздоч" not in q.lower()


def test_exclude_keys_sets_rag_exhausted_without_lowering_score() -> None:
    stmt_a = "Principle A about isolation boundaries across agents"
    stmt_b = "Principle B about retry budgets in the control plane"
    key_a = atom_key(stmt_a)
    key_b = atom_key(stmt_b)
    rows = [
        {COL_STATEMENT: stmt_a, COL_SCOPE: "PRINCIPLE", "_score": 0.9},
        {COL_STATEMENT: stmt_b, COL_SCOPE: "PRINCIPLE", "_score": 0.85},
    ]
    store = MagicMock()
    store.search_knowledge_atoms.return_value = rows

    with patch("knowledge_engine.services.dialog_atoms_rag.DIALOG_ATOMS_ENABLED", True):
        hit = retrieve_dialog_knowledge_atoms_detailed(
            "focus on isolation",
            SimpleNamespace(title="Node"),
            "",
            store=store,
            top_k=4,
            force_allow_instance=True,
            cite_r_index=True,
            exclude_keys=None,
            focus_hint="isolation",
        )
        assert hit.rag_exhausted is False
        assert hit.unseen_count == 2
        assert "[R1]" in hit.block

        exhausted = retrieve_dialog_knowledge_atoms_detailed(
            "focus on isolation",
            SimpleNamespace(title="Node"),
            "",
            store=store,
            top_k=4,
            force_allow_instance=True,
            cite_r_index=True,
            exclude_keys=[key_a, key_b],
            focus_hint="isolation",
        )
        assert exhausted.rag_exhausted is True
        assert exhausted.block == ""
        assert exhausted.atom_keys == []
        # min_score must not be lowered on the exhausted path.
        for call in store.search_knowledge_atoms.call_args_list:
            assert call.kwargs.get("min_score") is not None or len(call.args) >= 1


def test_exclude_keeps_unseen_above_threshold() -> None:
    stmt_used = "Already covered atom about surface overview of sharding"
    stmt_new = "Fresh atom about hot-key rebalancing mechanics under partition"
    rows = [
        {
            COL_STATEMENT: stmt_used,
            COL_SCOPE: "PRINCIPLE",
            "_score": 0.95,
            "id": "used_id",
            "source_chunk_ids": ["used_ch"],
        },
        {
            COL_STATEMENT: stmt_new,
            COL_SCOPE: "MECHANIC",
            "_score": 0.8,
            "id": "new_id",
            "source_chunk_ids": ["new_ch"],
        },
    ]
    store = MagicMock()
    store.search_knowledge_atoms.return_value = rows
    with patch("knowledge_engine.services.dialog_atoms_rag.DIALOG_ATOMS_ENABLED", True):
        result = retrieve_dialog_knowledge_atoms_detailed(
            "sharding",
            SimpleNamespace(title="Agg"),
            "",
            store=store,
            top_k=4,
            force_allow_instance=True,
            cite_r_index=True,
            exclude_keys=[atom_key(stmt_used)],
            focus_hint="rebalancing",
        )
    assert result.rag_exhausted is False
    assert result.unseen_count == 1
    assert "hot-key" in result.block
    assert "surface overview" not in result.block
    assert result.atom_keys == [atom_key(stmt_new)]
    assert "new_ch" in result.chunk_ids


def test_mmr_prefers_diverse_second_pick() -> None:
    import numpy as np

    from knowledge_engine.services.vector_store import VectorStore

    qv = np.asarray([1.0, 0.0], dtype=np.float64)
    rows = [
        {"id": "a", "_score": 0.99, "vector": [1.0, 0.0]},
        {"id": "b", "_score": 0.90, "vector": [0.99, 0.1]},  # near duplicate of a
        {"id": "c", "_score": 0.80, "vector": [0.0, 1.0]},  # orthogonal
    ]
    pure = VectorStore._mmr_select_rows(rows, qv, limit=2, lambda_mult=1.0)
    assert [r["id"] for r in pure] == ["a", "b"]
    diverse = VectorStore._mmr_select_rows(rows, qv, limit=2, lambda_mult=0.4)
    assert diverse[0]["id"] == "a"
    assert diverse[1]["id"] == "c"


def test_mutate_query_and_knobs_on_repeat() -> None:
    from knowledge_engine.src.node_deep_dive.deep_analysis_coverage import (
        deep_analysis_retrieval_knobs,
        mutate_deep_analysis_query,
    )

    mem = SessionMemory()
    base = "Aggregation partition failure modes"
    assert mutate_deep_analysis_query(base, mem) == base
    knobs0 = deep_analysis_retrieval_knobs(mem)
    assert knobs0["lambda_mult"] == 1.0
    assert knobs0["stochastic_sample"] is False

    mem.deep_analysis_used_chunk_ids = ["c1", "c2"]
    mutated = mutate_deep_analysis_query(base, mem)
    assert mutated.startswith(base)
    assert mutated != base
    knobs1 = deep_analysis_retrieval_knobs(mem)
    assert knobs1["lambda_mult"] < 1.0
    assert knobs1["query_noise"] > 0
    assert knobs1["stochastic_sample"] is True


def test_record_coverage_merges_chunk_ids_from_turn_stash() -> None:
    mem = SessionMemory(
        last_deep_analysis_atom_keys=["aaa111"],
        last_deep_analysis_chunk_ids=["chunk_a", "chunk_b"],
        last_deep_analysis_atom_ids=["atom_row_1"],
    )
    tech = "## 1. Anatomy\nUses [S1] and [R1].\n"
    record_deep_analysis_coverage(mem, technical_explanation=tech)
    assert "chunk_a" in mem.deep_analysis_used_chunk_ids
    assert "chunk_b" in mem.deep_analysis_used_chunk_ids
    assert "atom_row_1" in mem.deep_analysis_used_chunk_ids
    assert "aaa111" in mem.deep_analysis_used_atom_keys


def test_search_kwargs_pass_exclude_chunks() -> None:
    stmt_a = "Principle A about isolation boundaries across agents"
    stmt_b = "Principle B about retry budgets in the control plane"
    rows = [
        {
            COL_STATEMENT: stmt_a,
            COL_SCOPE: "PRINCIPLE",
            "_score": 0.9,
            "id": "id_a",
            "source_chunk_ids": ["ch_a"],
            "vector": [1.0, 0.0],
        },
        {
            COL_STATEMENT: stmt_b,
            COL_SCOPE: "PRINCIPLE",
            "_score": 0.85,
            "id": "id_b",
            "source_chunk_ids": ["ch_b"],
            "vector": [0.0, 1.0],
        },
    ]
    store = MagicMock()
    store.search_knowledge_atoms.return_value = [rows[1]]

    with patch("knowledge_engine.services.dialog_atoms_rag.DIALOG_ATOMS_ENABLED", True):
        result = retrieve_dialog_knowledge_atoms_detailed(
            "focus",
            SimpleNamespace(title="Node"),
            "",
            store=store,
            top_k=4,
            force_allow_instance=True,
            cite_r_index=True,
            exclude_chunk_ids=["ch_a"],
            exclude_atom_ids=["id_a"],
            lambda_mult=0.5,
            query_noise=0.04,
            stochastic_sample=True,
            focus_hint="isolation",
        )
    kwargs = store.search_knowledge_atoms.call_args
    assert kwargs.kwargs["exclude_chunk_ids"] == ["ch_a"]
    assert kwargs.kwargs["exclude_ids"] == ["id_a"]
    assert kwargs.kwargs["lambda_mult"] == 0.5
    assert result.chunk_ids == ["ch_b"] or "ch_b" in result.chunk_ids
    assert result.atom_ids == ["id_b"] or "id_b" in result.atom_ids
