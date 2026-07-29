"""Плотный материал ноды (Lite / tutor chain, не Reasoner)."""

from __future__ import annotations

from knowledge_engine.config import GEMINI_RPM_PAUSE_SEC, GEMINI_TUTOR_MODEL, GEMINI_TUTOR_TIMEOUT_SEC
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
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

from knowledge_engine.services.lecture_rag_context import build_lecture_generation_payload
from knowledge_engine.src.processors.source_anchors import REASONER_SOURCE_ATTRIBUTION_PROMPT
from knowledge_engine.src.source_evaluator.evaluator import format_whitelist_for_reasoner_prompt

_DENSE_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "Ты — генератор плотного учебного материала для одной ноды графа.\n"
    "Не задавай Сократовских вопросов.\n\n"
    "lecture_body (ОБЯЗАТЕЛЬНО): полноценная лекция для чата — от 400 слов. "
    "Главный базис — ФУНДАМЕНТАЛЬНЫЙ ИСТОЧНИК из user payload и LanceDB. "
    "summary: структурированная выжимка «суть механики» для панели UI — дополняет lecture_body, "
    "не заменяет его. Edge cases, короткие примеры кода.\n"
    "В summary после ключевых фактов ставь [S1], [S2]… — порядок = порядок в references (1-й ref = [S1]).\n"
    "diagram: Mermaid в fence ```mermaid с переводами строк (НЕ одна строка). "
    "После sequenceDiagram/graph TD/flowchart — новая строка. "
    "Каждый participant, rect, loop, end, Note и каждая стрелка (->>, -->>) — отдельная строка. "
    "Алиасы participant с / или () — в кавычках. loop с пробелами — в кавычках.\n"
    "references: 2–4 RichReference — НЕ сухие URL. Для каждой:\n"
    "  title, source_name, url, why_read (зачем читать), key_focus (на что смотреть), "
    "read_time_minutes (целое, минуты).\n"
    "code_snippets: 0–3 коротких блоков с разбором подводных камней.\n"
    "bridge_to_next: логический мост к смежной теме в графе.\n"
    "checkpoint_prompt: ОДИН короткий вопрос самопроверки (не допрос).\n\n"
    f"{REASONER_SOURCE_ATTRIBUTION_PROMPT}\n\n"
    f"{format_whitelist_for_reasoner_prompt()}\n"
    "references и primary_whitelist_source маршрута — только из Whitelist.\n"
)

_TARGETED_LECTURE_SYSTEM_APPEND = (
    "\n\nРежим **targeted_lecture** (ФОКУСНАЯ лекция):\n"
    "Пользователь запросил ПОДРОБНУЮ ФОКУСНУЮ ЛЕКЦИЮ.\n"
    "1. lecture_body — не менее 400 слов, ПОЛНОСТЬЮ про СПЕЦИФИЧЕСКИЙ ФОКУС из payload "
    "(user_focus / user_query), не про всю ноду.\n"
    "2. ЗАПРЕЩЕНО вводный пересказ всей темы ноды — сразу углубленный разбор механизма/паттерна.\n"
    "3. Архитектура: схемы работы, узкие места, failure modes, примеры кода/конфигураций.\n"
    "4. checkpoint_prompt — ОДИН глубокий вопрос строго по прочитанному фокусному материалу.\n"
    "summary/diagram для панели — тоже вокруг фокуса, не общий обзор ноды.\n"
)


def _dense_system_instruction(lecture_scope: str) -> str:
    if lecture_scope == "targeted_lecture":
        return _DENSE_SYSTEM + _TARGETED_LECTURE_SYSTEM_APPEND
    return _DENSE_SYSTEM


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
