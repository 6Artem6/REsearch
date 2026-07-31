"""Плотный материал ноды (Lite / tutor chain, не Reasoner)."""

from __future__ import annotations

from knowledge_engine.config import GEMINI_RPM_PAUSE_SEC, GEMINI_TUTOR_MODEL, GEMINI_TUTOR_TIMEOUT_SEC
from knowledge_engine.services.chat_session_manager import ChatSessionManager
from knowledge_engine.services.gemini_stateless import (
    gemini_tutor_model_chain,
    run_gemini_structured_with_chain,
)
from knowledge_engine.src.node_deep_dive.memory_schemas import SessionMemory
from knowledge_engine.src.node_deep_dive.schemas import (
    DenseMaterialOutput,
    NodeDataInput,
)
from knowledge_engine.src.node_deep_dive.tiered_memory import (
    build_handoff_summary,
    format_matrix_for_llm,
)
from knowledge_engine.src.node_deep_dive.tutor_prompt_builder import build_dense_system

from knowledge_engine.services.lecture_rag_context import build_lecture_generation_payload


def _dense_system_instruction(lecture_scope: str) -> str:
    return build_dense_system(targeted=lecture_scope == "targeted_lecture")


def generate_dense_material(
    node: NodeDataInput,
    memory: SessionMemory,
    rag_profile: str,
    anchor: str,
    chat_manager: ChatSessionManager | None = None,
    user_query: str = "",
    rag_context: str = "",
    curriculum_id: str = "",
    lecture_scope: str = "full_node_lecture",
    focus_text: str = "",
) -> DenseMaterialOutput:
    matrix = format_matrix_for_llm(memory.concepts_matrix)
    scope = (lecture_scope or "full_node_lecture").strip()
    focus = (focus_text or "").strip()
    rag_query = focus if scope == "targeted_lecture" and focus else user_query
    payload = build_lecture_generation_payload(
        node,
        rag_profile,
        rag_query,
        rag_context,
        matrix,
        memory.rolling_dialogue_summary or "",
        curriculum_id,
    )
    if scope == "targeted_lecture" and focus:
        payload += (
            f"\n\n### lecture_scope\n{scope}\n"
            f"### user_focus\n{focus}\n"
            f"### node_title\n{node.title}\n"
            "ИНСТРУКЦИЯ: лекция только про user_focus; не обзор всей ноды.\n"
        )
    mgr = chat_manager or ChatSessionManager.from_memory_blob(
        anchor, memory.chat_sessions
    )
    handoff = build_handoff_summary(memory)
    from knowledge_engine.ui.run_log import trace

    trace(
        f"NODE_DIVE dense_material ▶ Gemini | scope={scope} "
        f"focus_len={len(focus)} RAG_CONTEXT len={len(rag_context or '')}"
    )
    result = run_gemini_structured_with_chain(
        GEMINI_TUTOR_MODEL,
        _dense_system_instruction(scope),
        payload,
        anchor,
        DenseMaterialOutput,
        "node_deep_dive / dense_material",
        rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
        chat_manager=None,
        chat_label="node_deep_dive/dense_material",
        handoff_summary=handoff,
        session_registry=mgr,
        models=gemini_tutor_model_chain(),
        http_timeout_sec=GEMINI_TUTOR_TIMEOUT_SEC,
    )
    memory.chat_sessions = mgr.to_memory_blob()
    trace("NODE_DIVE dense_material ✓ | лекция в чат + панель (summary/diagram/refs)")
    return result
