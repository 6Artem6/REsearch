"""Selection Explainer context bundle (anchor / invariants / causal)."""

from __future__ import annotations

from knowledge_engine.schemas.extraction import KnowledgeAtom, ScopeType
from knowledge_engine.services.explain_context_bundle import (
    ExplainContextBundle,
    build_explain_context_bundle,
    format_causal_facts_block,
    format_explain_invariants_block,
    retrieve_causal_facts,
    retrieve_explain_invariants,
)
from knowledge_engine.services.node_selection_explain import _build_node_explain_payload
from knowledge_engine.src.node_deep_dive.interaction_prompt_layout import BLOCK_RAG_TAG
from knowledge_engine.src.node_deep_dive.memory_schemas import SessionMemory
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput


def test_format_explain_invariants_block_limits_and_header():
    atoms = [
        KnowledgeAtom(scope=ScopeType.PRINCIPLE, statement="Invariant A holds under X"),
        KnowledgeAtom(scope=ScopeType.MECHANIC, statement="Mechanic B causes Y"),
        KnowledgeAtom(scope=ScopeType.INSTANCE, statement="code-only detail"),
    ]
    # INSTANCE excluded by filter_atoms_for_dialog upstream; format still prints what given.
    block = format_explain_invariants_block(atoms[:2])
    assert "### fundamental_invariants" in block
    assert "[ФАКТ (PRINCIPLE)]:" in block
    assert "[ФАКТ (MECHANIC)]:" in block
    assert "code-only" not in block


def test_format_causal_facts_block():
    block = format_causal_facts_block(
        ["If fsync skipped → durability gap", "Group commit batches flushes"]
    )
    assert "### causal_facts" in block
    assert "durability gap" in block


def test_profile_overlap_causal_without_light_rag():
    profile = (
        "Write-ahead log must fsync before acknowledge commit.\n"
        "Unrelated latency tip about caches and CDN edges.\n"
        "Skipping fsync after WAL write breaks crash recovery guarantees."
    )
    block = retrieve_causal_facts(
        "why fsync after WAL",
        rag_profile=profile,
        use_light_rag=False,
        top_k=3,
    )
    assert "### causal_facts" in block
    assert "fsync" in block.lower()


def test_retrieve_invariants_skips_without_curriculum():
    assert retrieve_explain_invariants("fsync WAL durability", curriculum_id="") == ""


def test_retrieve_invariants_uses_mock_store(monkeypatch):
    class _FakeStore:
        def search_knowledge_atoms(
            self, query, *, limit=8, allowed_doc_ids=None, min_score=0.0
        ):
            return [
                {
                    "statement": "WAL durability requires fsync before ack",
                    "scope": "PRINCIPLE",
                    "doc_id": "doc_a",
                    "_score": 0.9,
                },
                {
                    "statement": "Group commit amortizes fsync cost",
                    "scope": "MECHANIC",
                    "doc_id": "doc_b",
                    "_score": 0.8,
                },
                {
                    "statement": "print(fsync())",
                    "scope": "INSTANCE",
                    "doc_id": "doc_a",
                    "_score": 0.95,
                },
            ]

    monkeypatch.setattr(
        "knowledge_engine.services.explain_context_bundle._mapped_doc_ids",
        lambda curriculum_id, node: ["doc_a", "doc_b"],
    )
    block = retrieve_explain_invariants(
        "fsync",
        curriculum_id="cur-1",
        node=NodeDataInput(
            node_id="n1",
            title="WAL",
            layer="foundation",
            category="storage",
            brief_summary="s",
            core_concepts=["WAL"],
            learning_goal="g",
        ),
        prefer_doc_ids=["doc_a"],
        store=_FakeStore(),
        top_k=3,
    )
    assert "### fundamental_invariants" in block
    assert "PRINCIPLE" in block
    assert "MECHANIC" in block
    assert "print(fsync" not in block


def test_build_bundle_formats_resolved_anchor(monkeypatch):
    monkeypatch.setattr(
        "knowledge_engine.services.explain_context_bundle.retrieve_explain_invariants",
        lambda *a, **k: "### fundamental_invariants\n[ФАКТ (PRINCIPLE)]: P",
    )
    monkeypatch.setattr(
        "knowledge_engine.services.explain_context_bundle.retrieve_causal_facts",
        lambda *a, **k: "### causal_facts\n- cause → effect",
    )
    bundle = build_explain_context_bundle(
        selected_text="checkpointer",
        user_question="что это?",
        resolved_r_chunks=[
            {
                "rag_id": "R6",
                "title": "t",
                "url": "https://x",
                "chunk_text": "checkpointer mode text",
                "doc_id": "d1",
            }
        ],
        curriculum_id="cur",
        node_title="Subagents",
    )
    assert isinstance(bundle, ExplainContextBundle)
    assert "### target_anchor" in bundle.anchor_block
    assert "[R6]" in bundle.anchor_block
    assert "fundamental_invariants" in bundle.invariants_block
    assert "causal_facts" in bundle.causal_block


def test_explain_payload_includes_triad_blocks(monkeypatch):
    mem = SessionMemory()
    mem.lecture_rag_inspector = [
        {
            "rag_id": "R2",
            "title": "WAL",
            "url": "https://example.com/wal",
            "chunk_text": "fsync before commit ack",
            "doc_id": "doc_wal",
        }
    ]

    def _fake_bundle(**kwargs):
        return ExplainContextBundle(
            resolved_r_chunks=kwargs.get("resolved_r_chunks") or [],
            anchor_block=(
                "### target_anchor\n"
                "--- ТОЧНЫЕ ИСХОДНЫЕ ЧАНКИ ЛЕКЦИИ ДЛЯ ВЫДЕЛЕННОГО ФРАГМЕНТА ---\n"
                "- [R2]: fsync before commit ack"
            ),
            invariants_block=(
                "### fundamental_invariants\n"
                "[ФАКТ (PRINCIPLE)]: Durability needs durable WAL"
            ),
            causal_block="### causal_facts\n- skip fsync → lost commits",
            prefer_doc_ids=["doc_wal"],
        )

    monkeypatch.setattr(
        "knowledge_engine.services.explain_context_bundle.build_explain_context_bundle",
        _fake_bundle,
    )

    from knowledge_engine.services.node_selection_explain import _NODE_EXPLAIN_SYSTEM

    payload, _, resolved = _build_node_explain_payload(
        "WAL",
        "fsync",
        "почему?",
        "durable write [R2]",
        "summary",
        "profile",
        [{"id": "S1", "title": "Doc", "url": "https://x", "snippet": "wal"}],
        memory=mem,
        curriculum_id="cur-1",
        node=NodeDataInput(
            node_id="n1",
            title="WAL",
            layer="foundation",
            category="storage",
            brief_summary="s",
            core_concepts=["WAL"],
            learning_goal="g",
        ),
    )
    assert BLOCK_RAG_TAG in payload
    assert "### fundamental_invariants" in payload
    assert "### causal_facts" in payload
    assert "### target_anchor" in payload
    assert len(resolved) == 1
    assert "Почему так" in _NODE_EXPLAIN_SYSTEM
    assert "fundamental_invariants" in _NODE_EXPLAIN_SYSTEM
