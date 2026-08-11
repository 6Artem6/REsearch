"""Unified context blocks for Tutor, Dense lecture, and selection explain."""

from __future__ import annotations

from knowledge_engine.schemas.llm_contracts.tutor import NodeExplainContract
from knowledge_engine.services.lecture_rag_context import (
    LECTURE_RAG_FALLBACK,
    build_lecture_generation_payload,
)
from knowledge_engine.services.node_selection_explain import (
    _NODE_EXPLAIN_SYSTEM,
    _build_node_explain_payload,
    _resolve_explain_source_ref,
    run_node_selection_explain,
)
from knowledge_engine.src.node_deep_dive.dialog_context import (
    SHARED_SESSION_CONTEXT_TAG,
    build_shared_session_context_block,
)
from knowledge_engine.src.node_deep_dive.interaction_prompt_layout import (
    BLOCK_DYNAMIC_TAG,
    BLOCK_RAG_TAG,
    BLOCK_SEMI_STATIC_TAG,
    BLOCK_USER_QUERY_TAG,
    LAYOUT_AND_TYPOGRAPHY_RULES,
    PINNED_REGISTRY_TAG,
    PROMPT_CITATION_ID_RULES,
    semi_static_user_prefix,
)
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    DialogueFactManifest,
    SessionMemory,
)
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput
from knowledge_engine.src.node_deep_dive.tutor_prompt_builder import (
    LECTURE_DENSE_RULES,
    build_dense_system,
)


def _sample_node() -> NodeDataInput:
    return NodeDataInput(
        node_id="ctx_node",
        title="WAL и durability",
        layer="foundation",
        category="storage",
        brief_summary="Write-ahead log",
        core_concepts=["WAL", "fsync"],
        learning_goal="Понимание WAL",
    )


def test_shared_session_context_includes_fact_manifest():
    mem = SessionMemory()
    mem.fact_manifest = DialogueFactManifest(
        agreed_concepts=["group commit"],
        current_subtopic="WAL recovery",
    )
    block = build_shared_session_context_block(mem)
    assert SHARED_SESSION_CONTEXT_TAG in block
    assert "fact_manifest" in block
    assert "group commit" in block
    assert "WAL recovery" in block


def test_dense_payload_includes_rag_manifest_and_registry():
    mem = SessionMemory()
    mem.fact_manifest = DialogueFactManifest(agreed_concepts=["checkpoint"])
    node = _sample_node()
    rag_body = "### Chunk [R1]\nWAL fsync policy trade-offs."
    payload = build_lecture_generation_payload(
        node,
        "rag profile hint",
        "объясни WAL",
        rag_body,
        "(matrix empty)",
        "",
        curriculum_id="cur-test",
        memory=mem,
        rag_citation_registry="### RAG CHUNK SOURCE INDEX\n[R1] wal.md",
    )
    assert "НАЧАЛО МАТЕРИАЛА" in payload
    assert "WAL fsync" in payload
    assert SHARED_SESSION_CONTEXT_TAG in payload
    assert "checkpoint" in payload
    assert "RAG CHUNK SOURCE INDEX" in payload
    assert PINNED_REGISTRY_TAG in payload


def test_dense_payload_fallback_when_rag_empty():
    node = _sample_node()
    payload = build_lecture_generation_payload(
        node,
        "",
        "лекция",
        "",
        "",
        "",
        memory=None,
    )
    assert "НАЧАЛО МАТЕРИАЛА" in payload
    assert LECTURE_RAG_FALLBACK in payload


def test_explain_payload_includes_shared_session_context():
    mem = SessionMemory()
    mem.fact_manifest = DialogueFactManifest(
        stack_mentions=["PostgreSQL"],
        current_subtopic="WAL",
    )
    payload, _, _ = _build_node_explain_payload(
        "WAL",
        "fsync",
        "почему?",
        "context around fsync",
        "summary excerpt",
        "rag profile",
        [{"id": "S1", "title": "Doc", "url": "https://x", "snippet": "wal"}],
        memory=mem,
    )
    assert SHARED_SESSION_CONTEXT_TAG in payload
    assert "PostgreSQL" in payload
    assert PINNED_REGISTRY_TAG in payload
    assert BLOCK_USER_QUERY_TAG in payload


def test_dense_prompt_caching_structure():
    mem = SessionMemory()
    mem.fact_manifest = DialogueFactManifest(agreed_concepts=["wal"])
    node = _sample_node()
    system = build_dense_system(memory=mem)

    assert (
        "Mode lecture_dense" in system or LECTURE_DENSE_RULES.split("\n")[0] in system
    )
    assert PROMPT_CITATION_ID_RULES.split("\n")[0] in system
    assert LAYOUT_AND_TYPOGRAPHY_RULES.split("\n")[0] in system
    assert system.index(LECTURE_DENSE_RULES) < system.index("StructuredLectureResponse")

    rag = "### [R1] chunk text"
    payload = build_lecture_generation_payload(
        node,
        "profile",
        "вопрос один",
        rag,
        "matrix",
        "rolling",
        curriculum_id="cur-1",
        memory=mem,
        rag_citation_registry="[R1] index",
    )

    i_registry = payload.index(PINNED_REGISTRY_TAG)
    i_rag = payload.index(BLOCK_RAG_TAG)
    i_query = payload.index(BLOCK_USER_QUERY_TAG)
    assert payload.index(BLOCK_SEMI_STATIC_TAG) < payload.index(BLOCK_DYNAMIC_TAG)
    assert i_registry < i_rag < i_query
    assert "STRICT GROUNDEDNESS" not in payload
    assert "ИНСТРУКЦИЯ ПО ГЕНЕРАЦИИ ЛЕКЦИИ" not in payload


def test_dense_semi_static_prefix_identical_across_user_queries():
    mem = SessionMemory()
    node = _sample_node()
    base_kwargs = {
        "node": node,
        "memory_rag_profile": "profile",
        "rag_context": "[R1] same",
        "concepts_matrix": "m",
        "rolling_summary": "r",
        "curriculum_id": "cur-1",
        "memory": mem,
    }
    p1 = build_lecture_generation_payload(
        **base_kwargs,
        user_query="первый вопрос",
    )
    p2 = build_lecture_generation_payload(
        **base_kwargs,
        user_query="совершенно другой вопрос",
    )
    assert semi_static_user_prefix(p1) == semi_static_user_prefix(p2)
    assert "первый вопрос" in p1
    assert "совершенно другой вопрос" in p2


def test_explain_prompt_caching_structure():
    mem = SessionMemory()
    payload, _, _ = _build_node_explain_payload(
        "WAL",
        "fsync",
        "почему?",
        "ctx",
        "summary",
        "rag",
        [{"id": "S1", "title": "Doc", "url": "https://x", "snippet": "wal"}],
        memory=mem,
    )
    assert PROMPT_CITATION_ID_RULES.split("\n")[0] in _NODE_EXPLAIN_SYSTEM
    assert LAYOUT_AND_TYPOGRAPHY_RULES.split("\n")[0] in _NODE_EXPLAIN_SYSTEM
    assert payload.index(BLOCK_SEMI_STATIC_TAG) < payload.index(BLOCK_DYNAMIC_TAG)
    assert payload.index(PINNED_REGISTRY_TAG) < payload.index(BLOCK_RAG_TAG)
    assert payload.index(BLOCK_RAG_TAG) < payload.index(BLOCK_USER_QUERY_TAG)


def test_explain_prompt_mentions_triad_and_dense_output():
    assert "fundamental_invariants" in _NODE_EXPLAIN_SYSTEM
    assert "causal_facts" in _NODE_EXPLAIN_SYSTEM
    assert "Почему так" in _NODE_EXPLAIN_SYSTEM
    assert "В двух словах" in _NODE_EXPLAIN_SYSTEM
    assert "no filler" in _NODE_EXPLAIN_SYSTEM


def test_explain_uses_r_chunks_from_surrounding_paragraph(monkeypatch):
    mem = SessionMemory()
    mem.lecture_rag_inspector = [
        {
            "rag_id": "R6",
            "title": "Checkpointer mode",
            "url": "https://example.com/langgraph",
            "chunk_text": "Текст чанка R6 про checkpointer и fresh state",
        },
    ]
    surrounding = (
        "По умолчанию субагенты используют унаследованный режим контрольных точек [R6]."
    )
    payload, source_ref, resolved = _build_node_explain_payload(
        "Subagents",
        "режим контрольных точек",
        "что это?",
        surrounding,
        "summary",
        "",
        [{"id": "S1", "title": "Doc", "url": "https://x", "snippet": "generic"}],
        memory=mem,
    )
    assert "ТОЧНЫЕ ИСХОДНЫЕ ЧАНКИ ЛЕКЦИИ" in payload
    assert "checkpointer" in payload.lower()
    assert "[R6]" in payload
    assert len(resolved) == 1
    assert source_ref.source_id == "R6"

    def _fake_invoke(user_payload, anchor, stream_callback=None):
        return NodeExplainContract(
            explanation="Разбор checkpointer [R6].",
            cited_source_ids=["R6"],
        )

    monkeypatch.setattr(
        "knowledge_engine.services.node_selection_explain._invoke_node_explain_gemini",
        _fake_invoke,
    )
    result = run_node_selection_explain(
        "Subagents",
        "режим контрольных точек",
        "что это?",
        surrounding,
        "summary",
        "",
        [{"id": "S1", "title": "Doc", "url": "https://x", "snippet": "generic"}],
        "anchor",
        memory=mem,
    )
    assert result.source_ref.source_id == "R6"
    assert result.source_ref.title == "Checkpointer mode"

    ref = _resolve_explain_source_ref(["R6"], resolved, source_ref)
    assert ref.source_id == "R6"
