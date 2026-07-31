"""Node Deep-Dive: tiered memory, step pipeline, тьютор."""

from __future__ import annotations

import asyncio

from knowledge_engine.config import (
    GEMINI_PROBE_BEFORE_USE,
    GEMINI_RPM_PAUSE_SEC,
    GEMINI_TUTOR_MODEL,
    GEMINI_TUTOR_TIMEOUT_SEC,
    KE_RAG_TIMEOUT_SEC,
    RAG_DEFAULT_MAX_FACTS,
    RAG_DEFAULT_MIN_RELEVANCE,
)
from knowledge_engine.services.gemini_stateless import (
    GeminiUnavailableError,
    gemini_tutor_model_chain,
    is_gemini_available,
    run_gemini_structured_with_chain,
)
from knowledge_engine.services.chat_session_manager import ChatSessionManager
from knowledge_engine.services.llm_markdown_service import enrich_node_deep_dive_response
from knowledge_engine.src.curriculum.schemas import CurriculumNode
from knowledge_engine.src.node_deep_dive.memory_schemas import SessionMemory, UserIntent
from knowledge_engine.src.node_deep_dive.schemas import (
    DeepDiveLLMOutput,
    DenseMaterialOutput,
    IntroAssessmentOutput,
    NodeContentBlock,
    NodeDataInput,
    NodeDeepDiveRequest,
    NodeDeepDiveResponse,
    NodeStatus,
)
from knowledge_engine.services.node_content_generator import generate_dense_material
from knowledge_engine.services.lecture_rag_context import retrieve_lecture_rag_context
from knowledge_engine.services.node_source_registry import (
    build_session_source_registry,
)
from knowledge_engine.src.node_deep_dive.lecture_scope import (
    dialogue_focus_text,
    resolve_lecture_scope,
)
from knowledge_engine.src.node_deep_dive.learning_loop import (
    advance_phase_after_chat,
    build_mastery_dashboard,
    set_learning_mode,
)
from knowledge_engine.src.node_deep_dive.session_store import (
    get_all_sessions_for_curriculum,
    get_session,
    repair_history_with_memory,
    save_session,
)
from knowledge_engine.src.node_deep_dive.dialog_context import (
    build_anchor_and_manifest,
    build_dynamic_suffix,
    build_static_prefix,
    set_dialogue_anchor,
)
from knowledge_engine.src.node_deep_dive.dialog_ids import sync_session_history_turns
from knowledge_engine.src.node_deep_dive.step_pipeline import (
    build_tutor_behavior_state_block,
    process_user_message_pipeline,
    rotate_window_after_message,
)
from knowledge_engine.src.node_deep_dive.tiered_memory import (
    append_to_active_window,
    build_handoff_summary,
    build_tiered_context_payload,
    derive_node_status,
    init_session_memory,
)
from knowledge_engine.src.node_deep_dive.tutor_prompt_builder import (
    INTRO_ASSESSMENT_SYSTEM,
    build_dialogue_system,
    build_lecture_chat_system,
)
from knowledge_engine.src.rag_gateway.gateway import (
    query_directional_rag,
    save_user_fact,
)
from knowledge_engine.src.rag_gateway.schemas import (
    DirectionalRAGQuery,
    SearchDirection,
)
from knowledge_engine.ui.run_log import trace

from knowledge_engine.services.curriculum_whitelist_prompt import (
    enrich_node_learning_materials_from_graph,
    format_node_curriculum_context_for_tutor,
)

_LECTURE_STUB_MARKERS = (
    "материал в панели",
    "материал перед вами",
    "материал справа",
    "смотрите справа",
    "смотрите материал",
    "в панели справа",
)


def _is_explicit_lecture_request(user_msg: str) -> bool:
    t = (user_msg or "").lower()
    return any(
        k in t
        for k in (
            "дай лекцию",
            "дай плотн",
            "плотный материал",
            "dense material",
            "лекцию по",
            "развернут",
            "подробн",
            "объясни подроб",
        )
    )


def _lecture_request(
    intent: UserIntent,
    learning_mode: str,
    user_msg: str,
) -> bool:
    """Явный запрос лекции (не любой INTENT_EXPLAIN от step_analysis)."""
    if learning_mode == "lecture" and _is_explicit_lecture_request(user_msg):
        return True
    if intent == "INTENT_EXPLAIN" and _is_explicit_lecture_request(user_msg):
        return True
    return False


def _needs_dense_material(
    memory: SessionMemory,
    intent: UserIntent,
    user_msg: str,
    lecture_button_pressed: bool,
) -> bool:
    """Плотная лекция в чат — только кнопка / явный запрос, не ответ на вопрос тьютора."""
    if lecture_button_pressed:
        return True
    if memory.learning_phase == "intro_assessment":
        return False
    if intent == "INTENT_FINALIZE":
        return False
    return _lecture_request(intent, memory.learning_mode, user_msg)


def _is_dialogue_feedback_mode(
    intent: UserIntent,
    learning_mode: str,
    user_msg: str,
) -> bool:
    """Текстовый ответ пользователя → рецензия + дискуссия, не lecture_dense."""
    if intent in ("INTENT_EXPLAIN", "INTENT_FINALIZE"):
        return False
    if _lecture_request(intent, learning_mode, user_msg):
        return False
    return True


def _tutor_system_instruction(
    intent: UserIntent,
    learning_mode: str,
    user_msg: str,
) -> str:
    if _is_dialogue_feedback_mode(intent, learning_mode, user_msg):
        return build_dialogue_system()
    return build_lecture_chat_system()


def _tutor_message_is_lecture_stub(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 280:
        return True
    low = t.lower()
    return any(m in low for m in _LECTURE_STUB_MARKERS) and len(t) < 900


def _compose_dense_chat_message(dense: DenseMaterialOutput) -> str:
    body = (dense.lecture_body or "").strip() or (dense.summary or "").strip()
    checkpoint = (dense.checkpoint_prompt or "").strip()
    parts: list[str] = []
    if body:
        parts.append(body)
    if checkpoint:
        parts.append(f"\n\n**Самопроверка:** {checkpoint}")
    return "\n".join(parts).strip()[:12_000]


def _ensure_lecture_in_tutor_message(
    tutor: str,
    content_summary: str,
    lecture_body: str = "",
) -> str:
    merged = (tutor or "").strip()
    fallback = (lecture_body or "").strip() or (content_summary or "").strip()
    if fallback and _tutor_message_is_lecture_stub(merged):
        if merged and len(merged) >= 80 and not any(
            m in merged.lower() for m in _LECTURE_STUB_MARKERS
        ):
            return merged[:12_000]
        checkpoint_tail = ""
        if merged and "**Самопроверка:**" in merged:
            checkpoint_tail = merged.split("**Самопроверка:**", 1)[-1].strip()
        elif merged and len(merged) < 400:
            checkpoint_tail = merged
        out = fallback
        if checkpoint_tail and checkpoint_tail not in out:
            out = f"{out}\n\n**Самопроверка:** {checkpoint_tail}"
        return out[:12_000]
    return (merged or fallback)[:12_000]


def _merge_node_data_from_graph(
    req: NodeDeepDiveRequest,
    node: CurriculumNode,
) -> NodeDeepDiveRequest:
    nd = req.node_data
    return req.model_copy(
        update={
            "node_data": nd.model_copy(
                update={
                    "mapped_source_ids": list(node.mapped_source_ids or []),
                    "primary_source_id": (node.primary_source_id or "")[:16],
                    "source_ref": node.source_ref,
                    "node_curriculum_breakdown": node.node_curriculum_breakdown,
                    "learning_goal": (node.learning_goal or nd.learning_goal or "")[
                        :600
                    ],
                }
            )
        }
    )


async def _apply_lazy_grounding_for_init(req: NodeDeepDiveRequest) -> NodeDeepDiveRequest:
    from knowledge_engine.config import CURRICULUM_TARGETED_NODE_GROUNDING_ENABLED
    if not CURRICULUM_TARGETED_NODE_GROUNDING_ENABLED:
        return req

    from knowledge_engine.services.skill_tree_store import (
        get_curriculum_graph,
        get_curriculum_meta,
        save_curriculum_record,
    )
    from knowledge_engine.src.curriculum.schemas import CurriculumGraph
    from knowledge_engine.src.curriculum.source_policy import resolve_source_policy
    from knowledge_engine.src.curriculum.targeted_node_grounding import (
        lazy_ground_deep_node_on_demand,
    )

    raw = get_curriculum_graph(req.curriculum_id)
    if not raw:
        return req
    graph = CurriculumGraph.model_validate(raw)
    node = next((n for n in graph.nodes if n.node_id == req.node_data.node_id), None)
    if not node:
        return req

    if node.node_risk_kind != "DEEP":
        return _merge_node_data_from_graph(req, node)

    status = (node.grounding_status or "").strip()
    if status == "grounded" and node.source_ref:
        return _merge_node_data_from_graph(req, node)
    if status != "pending_grounding":
        return _merge_node_data_from_graph(req, node)

    meta = get_curriculum_meta(req.curriculum_id) or {}
    target_goal = str(meta.get("target_goal") or graph.description or "").strip()
    source_policy = resolve_source_policy(
        meta.get("source_policy"),
        str(meta.get("generation_mode") or "fast"),
        default="hybrid",
    )
    anchor = f"lazy:{req.curriculum_id}:{node.node_id}"
    graph, node = await lazy_ground_deep_node_on_demand(
        graph,
        node,
        target_goal=target_goal,
        source_policy=source_policy,
        anchor=anchor,
    )
    save_curriculum_record(
        graph,
        target_goal=target_goal,
        generation_mode=str(meta.get("generation_mode") or "fast"),
        depth_level=str(meta.get("depth_level") or "Standard"),
        user_level=str(meta.get("user_level") or "Intermediate/Advanced"),
        source_policy=source_policy,
    )
    return _merge_node_data_from_graph(req, node)


def _anchor(curriculum_id: str, node_id: str) -> str:
    return f"node_deep_dive:{curriculum_id}:{node_id}"


def _build_rag_request(node: NodeDataInput) -> DirectionalRAGQuery:
    concepts = ", ".join(node.core_concepts[:5])
    summary = (node.brief_summary or "").strip()
    return DirectionalRAGQuery(
        target_node=node.node_id,
        search_directions=[
            SearchDirection(
                direction_label="Опыт и стек",
                vector_query=f"Практический опыт пользователя в теме {node.title}",
                weight=1.0,
            ),
            SearchDirection(
                direction_label="Слепые зоны",
                vector_query=f"Пробелы и слабые места: {concepts}",
                weight=1.2,
            ),
            SearchDirection(
                direction_label=f"Слой {node.layer}",
                vector_query=(
                    f"{node.category or node.title} — "
                    f"фундамент и практика ({node.layer})"
                ),
                weight=0.9,
            ),
        ],
        relevance_criteria=(
            f"Смысловое совпадение с узлом «{node.title}». "
            f"Концепции: {concepts}. "
            + (f"Суть: {summary}. " if summary else "")
            + "Только факты про опыт, стек или пробелы по этой теме."
        ),
        max_facts=RAG_DEFAULT_MAX_FACTS,
        min_relevance_threshold=RAG_DEFAULT_MIN_RELEVANCE,
    )


def _rag_fact_preview(text: str, max_len: int = 72) -> str:
    clean = " ".join((text or "").split())
    if not clean:
        return ""
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 1].rstrip() + "…"


def _format_rag_facts(facts: list) -> str:
    if not facts:
        return "(RAG Gateway: факты выше порога релевантности не найдены)"
    lines = []
    for i, f in enumerate(facts, 1):
        direction = getattr(f, "direction", "") or ""
        text = getattr(f, "fact", getattr(f, "text", str(f)))
        score = getattr(f, "relevance_score", None)
        suffix = f" [{direction}]" if direction else ""
        if score is not None:
            suffix += f" score={score:.3f}"
        lines.append(f"{i}. {text}{suffix}")
    return "\n".join(lines)


def _invoke_intro_assessment(
    user_payload: str,
    anchor: str,
    chat_mgr: ChatSessionManager,
) -> IntroAssessmentOutput:
    system = INTRO_ASSESSMENT_SYSTEM
    trace(
        f"NODE_DIVE этап 2/2 intro ▶ экспресс-вопрос | chain: "
        f"{' → '.join(gemini_tutor_model_chain()[:4])} | "
        f"локальный RPD, probe={'on' if GEMINI_PROBE_BEFORE_USE else 'off'} | "
        f"HTTP лимит={GEMINI_TUTOR_TIMEOUT_SEC:.0f}s"
    )
    out = run_gemini_structured_with_chain(
        GEMINI_TUTOR_MODEL,
        system,
        user_payload,
        anchor,
        IntroAssessmentOutput,
        "node_deep_dive / intro_assessment",
        rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
        chat_manager=None,
        chat_label="node_deep_dive/intro_assessment",
        handoff_summary="",
        session_registry=chat_mgr,
        models=gemini_tutor_model_chain(),
        http_timeout_sec=GEMINI_TUTOR_TIMEOUT_SEC,
    )
    tutor = (out.tutor_message or "").strip()
    if tutor:
        chat_mgr.record_turn(
            "node_deep_dive/intro_assessment",
            user_payload[:8000],
            tutor,
        )
    trace("NODE_DIVE этап 2/2 intro ✓ | learning_phase=intro_assessment | в чате первый вопрос")
    return out


def _invoke_tutor(
    memory: SessionMemory,
    node: NodeDataInput,
    intent: str,
    action: str,
    user_msg: str,
    anchor: str,
    label: str,
    chat_mgr: ChatSessionManager,
    handoff: str,
    curriculum_id: str = "",
) -> DeepDiveLLMOutput:
    node_for_tutor = enrich_node_learning_materials_from_graph(node, curriculum_id)
    node_ctx = format_node_curriculum_context_for_tutor(node_for_tutor, curriculum_id)
    dlg_focus = dialogue_focus_text(user_msg, memory)
    behavior_block = build_tutor_behavior_state_block(
        intent,
        action,
        memory.learning_mode,
        memory.learning_phase,
        user_msg,
        has_user_focus=bool(dlg_focus),
    )
    static_prefix = build_static_prefix(
        memory, node_for_tutor, intent, action, node_ctx
    )
    if dlg_focus:
        static_prefix += f"\n### user_focus_topic\n{dlg_focus}\n"
    static_body = "\n\n".join(
        [
            static_prefix,
            build_anchor_and_manifest(memory),
            behavior_block,
        ]
    )
    dynamic = build_dynamic_suffix(memory, user_msg)
    system = _tutor_system_instruction(
        intent,
        memory.learning_mode,
        user_msg,
    )
    tutor_label = "node_deep_dive/tutor"
    return run_gemini_structured_with_chain(
        GEMINI_TUTOR_MODEL,
        system,
        static_body,
        anchor,
        DeepDiveLLMOutput,
        label,
        rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
        chat_manager=chat_mgr,
        chat_label=tutor_label,
        handoff_summary=handoff,
        delta_user_message=dynamic,
        models=gemini_tutor_model_chain(),
        http_timeout_sec=GEMINI_TUTOR_TIMEOUT_SEC,
    )


from knowledge_engine.web.llm_text_repair import repair_diagram_markdown


def _merge_content(
    prev: NodeContentBlock,
    llm: DeepDiveLLMOutput,
    is_init: bool,
) -> NodeContentBlock:
    if is_init:
        return NodeContentBlock()
    summary = (llm.summary or "").strip() or prev.summary
    diagram = repair_diagram_markdown((llm.diagram or "").strip() or prev.diagram)
    refs = llm.references if llm.references else prev.references
    return NodeContentBlock(
        summary=summary,
        diagram=diagram,
        references=refs[:6],
        code_snippets=prev.code_snippets,
    )


def _apply_dense_material(content: NodeContentBlock, dense) -> NodeContentBlock:
    return NodeContentBlock(
        summary=(dense.summary or "").strip() or content.summary,
        diagram=repair_diagram_markdown(
            (dense.diagram or "").strip() or content.diagram
        ),
        references=dense.references[:6] if dense.references else content.references,
        code_snippets=list(dense.code_snippets or [])[:4],
    )


def _coerce_llm_status(raw: str | None) -> NodeStatus | None:
    s = (raw or "").strip()
    if s in ("in_progress", "deep_understanding", "mastered", "gap"):
        return s
    return None


async def _record_gap_if_needed(gap: str | None, node_id: str) -> None:
    text = (gap or "").strip()
    if len(text) < 16:
        return
    await save_user_fact(
        f"Пробел в учебной ноде: {text}",
        category="learning_gap",
        node_id=node_id,
    )


def _ensure_memory(session, node: NodeDataInput, rag_text: str):
    from knowledge_engine.src.node_deep_dive.memory_schemas import SessionMemory

    if session.memory is not None:
        return session.memory
    mem = init_session_memory(node, rag_text)
    session.memory = mem
    return mem


async def fetch_node_init_rag_facts(
    req: NodeDeepDiveRequest,
) -> tuple[str, int, list[str]]:
    """Directional RAG только (без intro payload) — можно overlap с lazy grounding."""
    from knowledge_engine.src.memory.light_rag import sync_profile_from_markdown_if_needed

    node = req.node_data
    await sync_profile_from_markdown_if_needed()

    threshold = RAG_DEFAULT_MIN_RELEVANCE
    node_title = (node.title or node.node_id).strip()
    trace(
        f"[PERSONAL_RAG] Querying facts for node '{node_title}'… "
        f"threshold={threshold:.2f}"
    )
    trace(
        f"NODE_DIVE этап 1/2 RAG ▶ | {req.curriculum_id}/{node.node_id} "
        "(векторный поиск + cross-encoder, без LLM)"
    )
    rag_req = _build_rag_request(node)
    rag_resp = await asyncio.wait_for(
        query_directional_rag(rag_req),
        timeout=KE_RAG_TIMEOUT_SEC,
    )
    rag_facts_text = _format_rag_facts(rag_resp.facts)
    rag_facts_count = len(rag_resp.facts)
    rag_fact_labels = [
        _rag_fact_preview(f.fact) for f in rag_resp.facts if _rag_fact_preview(f.fact)
    ][:8]
    dir_tags = sorted({(f.direction or "").strip() for f in rag_resp.facts if f.direction})
    trace(
        f"[PERSONAL_RAG] Querying facts for node '{node_title}'… "
        f"Found {rag_facts_count} facts above threshold {threshold:.2f}"
        + (f" | axes={', '.join(dir_tags)}" if dir_tags else "")
    )
    trace(
        f"NODE_DIVE этап 1/2 RAG ✓ | facts={rag_facts_count} "
        f"(кандидатов в gateway см. RAG_GATEWAY) | latency={rag_resp.latency_ms:.0f}ms"
    )
    return rag_facts_text, rag_facts_count, rag_fact_labels


async def finalize_node_init_after_grounding(
    req: NodeDeepDiveRequest,
    rag_facts_text: str,
    rag_facts_count: int,
    rag_fact_labels: list[str],
) -> tuple[str, str, ChatSessionManager, int, list[str]]:
    """Intro payload после merge grounding (source_ref в node_data)."""
    node = req.node_data
    anchor = _anchor(req.curriculum_id, node.node_id)
    session = get_session(req.curriculum_id, node.node_id)
    session.history = []
    session.memory = init_session_memory(node, rag_facts_text)
    chat_mgr = ChatSessionManager.from_memory_blob(anchor, session.memory.chat_sessions)
    chat_mgr.clear_all("init")
    intro_payload = build_tiered_context_payload(
        session.memory,
        node,
        "ANSWER",
        "init",
        build_tutor_behavior_state_block(
            "ANSWER",
            "init",
            session.memory.learning_mode,
            session.memory.learning_phase,
            "",
        ),
        "",
    )
    session.memory.chat_sessions = chat_mgr.to_memory_blob()
    init_registry = build_session_source_registry(
        req.curriculum_id,
        list(node.mapped_source_ids or []),
        [],
    )
    save_session(
        req.curriculum_id,
        node.node_id,
        "in_progress",
        NodeContentBlock(),
        [],
        rag_fact_labels=rag_fact_labels,
        memory=session.memory,
        source_registry=init_registry,
    )
    return intro_payload, anchor, chat_mgr, rag_facts_count, rag_fact_labels


async def prepare_node_init_rag(
    req: NodeDeepDiveRequest,
) -> tuple[str, str, ChatSessionManager, int, list[str]]:
    """Этап 1 init: RAG + memory (async). Gemini — после asyncio.run, без Lance loop."""
    rag_facts_text, rag_facts_count, rag_fact_labels = await fetch_node_init_rag_facts(req)
    return await finalize_node_init_after_grounding(
        req,
        rag_facts_text,
        rag_facts_count,
        rag_fact_labels,
    )


def complete_node_init_gemini(
    req: NodeDeepDiveRequest,
    intro_payload: str,
    anchor: str,
    chat_mgr: ChatSessionManager,
    rag_facts_count: int,
    rag_fact_labels: list[str],
) -> NodeDeepDiveResponse:
    """Этап 2 init: Gemini без активного asyncio loop (после prepare_node_init_rag)."""
    trace(
        f"NODE_DIVE этап 2/2 intro ▶ | chain tutor | "
        f"локальный RPD, probe={'on' if GEMINI_PROBE_BEFORE_USE else 'off'} | "
        f"HTTP лимит={GEMINI_TUTOR_TIMEOUT_SEC:.0f}s"
    )
    intro_out = _invoke_intro_assessment(intro_payload, anchor, chat_mgr)
    return asyncio.run(
        _finish_init_after_intro(
            req,
            intro_out,
            chat_mgr,
            anchor,
            rag_facts_count,
            rag_fact_labels,
        )
    )


async def _finish_init_after_intro(
    req: NodeDeepDiveRequest,
    intro_out: IntroAssessmentOutput,
    chat_mgr: ChatSessionManager,
    anchor: str,
    rag_facts_count: int,
    rag_fact_labels: list[str],
) -> NodeDeepDiveResponse:
    node = req.node_data
    session = get_session(req.curriculum_id, node.node_id)
    session.memory.chat_sessions = chat_mgr.to_memory_blob()
    tutor = (intro_out.tutor_message or "").strip()
    if not tutor:
        tutor = (
            f"Один практический кейс по «{node.title}»: "
            "опишите главный риск или механику в 3–5 предложений."
        )
    if session.memory:
        set_dialogue_anchor(session.memory, node, tutor)
    content = NodeContentBlock()
    llm_out = DeepDiveLLMOutput(
        node_status="in_progress",
        tutor_message=tutor,
    )
    pipeline_gap: str | None = None
    return await _finalize_node_deep_dive(
        req,
        session,
        anchor,
        content,
        llm_out,
        tutor,
        rag_facts_count,
        rag_fact_labels,
        pipeline_gap,
        "init",
    )


async def _finalize_node_deep_dive(
    req: NodeDeepDiveRequest,
    session,
    anchor: str,
    content: NodeContentBlock,
    llm_out: DeepDiveLLMOutput,
    tutor: str,
    rag_facts_count: int,
    rag_fact_labels: list[str],
    pipeline_gap: str | None,
    action: str,
) -> NodeDeepDiveResponse:
    node = req.node_data
    trace(f"NODE_DIVE ▶ {action} | {req.curriculum_id}/{node.node_id}")
    session.content = content

    memory = session.memory
    if memory is not None and (tutor or "").strip():
        append_to_active_window(memory, "tutor", tutor)
        rotate_window_after_message(memory, anchor)

    if action in ("chat", "verify"):
        session.history = sync_session_history_turns(
            session.history,
            memory,
            user_message=(req.user_message or "").strip(),
            tutor_message=tutor,
        )
    else:
        session.history = sync_session_history_turns(
            session.history,
            memory,
            tutor_message=tutor,
        )

    session.history = repair_history_with_memory(session.history, memory)

    gap = (llm_out.new_gap_to_record or "").strip() or pipeline_gap or None
    if gap:
        await _record_gap_if_needed(gap, node.node_id)

    memory = session.memory
    status: NodeStatus
    if memory is not None:
        status = derive_node_status(memory, gap)
        llm_status = _coerce_llm_status(llm_out.node_status)
        if llm_status == "mastered" and status != "gap":
            status = "mastered"
        elif llm_status == "gap":
            status = "gap"
    else:
        status = _coerce_llm_status(llm_out.node_status) or "in_progress"

    labels_for_store = list(rag_fact_labels)
    if not labels_for_store:
        prev_blob = get_all_sessions_for_curriculum(req.curriculum_id).get(
            node.node_id, {}
        )
        labels_for_store = list(prev_blob.get("rag_fact_labels") or [])

    source_registry = build_session_source_registry(
        req.curriculum_id,
        list(node.mapped_source_ids or []),
        content.references,
    )

    key = save_session(
        req.curriculum_id,
        node.node_id,
        status,
        content,
        session.history,
        rag_fact_labels=labels_for_store,
        memory=memory,
        source_registry=source_registry,
    )

    trace(
        f"NODE_DIVE ✓ {action} | {req.curriculum_id}/{node.node_id} | "
        f"status={status} phase={memory.learning_phase if memory else '—'} "
        f"mastery={memory.topic_mastery_score if memory else 0}%"
    )
    stored = get_session(req.curriculum_id, node.node_id)
    display_history = repair_history_with_memory(list(stored.history), stored.memory)
    mem = stored.memory
    dash = build_mastery_dashboard(mem, status)
    base = NodeDeepDiveResponse(
        node_id=node.node_id,
        node_status=status,
        content=content,
        tutor_message=tutor,
        history=display_history,
        new_gap_to_record=gap,
        session_key=key,
        rag_facts_count=rag_facts_count,
        rag_fact_labels=rag_fact_labels,
        topic_mastery_score=mem.topic_mastery_score if mem else 0,
        concepts_matrix=list(mem.concepts_matrix) if mem else [],
        mastery_dashboard=dash,
        learning_phase=mem.learning_phase if mem else "intro_assessment",
        learning_mode=mem.learning_mode if mem else "lecture",
        source_registry=source_registry,
    )
    return enrich_node_deep_dive_response(base, source_registry)


async def run_node_deep_dive(req: NodeDeepDiveRequest) -> NodeDeepDiveResponse:
    if not is_gemini_available():
        raise GeminiUnavailableError("Gemini недоступен для Node Deep-Dive")

    action = req.user_action
    node = req.node_data
    if action in ("chat", "verify") and not (req.user_message or "").strip():
        raise ValueError("user_message обязателен для chat и verify")

    session = get_session(req.curriculum_id, node.node_id)
    anchor = _anchor(req.curriculum_id, node.node_id)
    rag_facts_text = "(не запрашивался — продолжение сессии)"
    rag_facts_count = 0
    rag_fact_labels: list[str] = []

    intent: UserIntent = "ANSWER"
    pipeline_gap: str | None = None

    if action == "init":
        trace("NODE_DIVE init parallel ▶ | lazy_grounding ∥ directional RAG")
        grounded_req, rag_facts = await asyncio.gather(
            _apply_lazy_grounding_for_init(req),
            asyncio.wait_for(fetch_node_init_rag_facts(req), timeout=KE_RAG_TIMEOUT_SEC),
        )
        intro_payload, anchor, chat_mgr, rag_facts_count, rag_fact_labels = (
            await finalize_node_init_after_grounding(grounded_req, *rag_facts)
        )
        intro_out = await asyncio.to_thread(
            _invoke_intro_assessment,
            intro_payload,
            anchor,
            chat_mgr,
        )
        return await _finish_init_after_intro(
            grounded_req,
            intro_out,
            chat_mgr,
            anchor,
            rag_facts_count,
            rag_fact_labels,
        )

    memory = _ensure_memory(session, node, rag_facts_text)
    chat_mgr = ChatSessionManager.from_memory_blob(anchor, memory.chat_sessions)
    handoff = build_handoff_summary(memory)
    raw_user = (req.user_message or "").strip()
    lecture_button_pressed = raw_user.startswith("[mode:lecture]")
    user_msg = raw_user
    if user_msg.startswith("[mode:lecture]"):
        set_learning_mode(memory, "lecture")
        user_msg = user_msg.replace("[mode:lecture]", "").strip()
    elif user_msg.startswith("[mode:blitz]"):
        set_learning_mode(memory, "express_blitz")
        user_msg = user_msg.replace("[mode:blitz]", "").strip()
    elif user_msg.startswith("[mode:socratic]"):
        set_learning_mode(memory, "socratic_point")
        user_msg = user_msg.replace("[mode:socratic]", "").strip()

    intent, pipeline_gap = await asyncio.to_thread(
        process_user_message_pipeline,
        user_msg,
        memory,
        node,
        anchor,
        action,
    )
    trace(
        f"NODE_DIVE step_analysis done | intent={intent} "
        f"phase={memory.learning_phase} mode={memory.learning_mode}"
    )

    if (
        intent == "INTENT_EXPLAIN"
        and not lecture_button_pressed
        and not _is_explicit_lecture_request(user_msg)
    ):
        intent = "ANSWER"
        trace(
            "NODE_DIVE intent coerce | INTENT_EXPLAIN→ANSWER "
            "(развёрнутый ответ, не запрос лекции)"
        )

    content = session.content
    llm_out: DeepDiveLLMOutput
    wants_lecture = _needs_dense_material(
        memory, intent, user_msg, lecture_button_pressed
    )
    needs_dense = wants_lecture
    if needs_dense:
        lecture_scope, focus_text = resolve_lecture_scope(
            user_msg,
            memory,
            lecture_button_pressed=lecture_button_pressed,
        )
        trace(
            f"NODE_DIVE dense_material ▶ | phase={memory.learning_phase} | "
            f"wants_lecture={wants_lecture} | scope={lecture_scope} | "
            f"focus={focus_text[:80]!r}… | модель={GEMINI_TUTOR_MODEL} "
            "(LanceDB → лекция в чат + панель)"
        )
        node_for_lecture = enrich_node_learning_materials_from_graph(
            node, req.curriculum_id
        )
        rag_query = (
            focus_text
            if lecture_scope == "targeted_lecture" and focus_text
            else user_msg
        )
        rag_context = await retrieve_lecture_rag_context(
            node_for_lecture, rag_query, req.curriculum_id
        )
        dense = await asyncio.to_thread(
            generate_dense_material,
            node_for_lecture,
            memory,
            memory.rag_profile_compressed,
            anchor,
            chat_mgr,
            user_msg,
            rag_context,
            req.curriculum_id,
            lecture_scope,
            focus_text,
        )
        content = _apply_dense_material(content, dense)
        memory.learning_phase = "dense_material"
        memory.pathway_bridge = (dense.bridge_to_next or "").strip()
        tutor = _compose_dense_chat_message(dense)
        if _tutor_message_is_lecture_stub(tutor):
            tutor = _ensure_lecture_in_tutor_message(
                tutor,
                content.summary,
                (dense.lecture_body or "").strip(),
            )
        llm_out = DeepDiveLLMOutput(
            node_status="in_progress",
            tutor_message=tutor,
        )
    else:
        if memory.learning_phase == "intro_assessment" and intent == "ANSWER":
            trace(
                "NODE_DIVE ▶ dialogue_feedback | intro_assessment + ANSWER "
                "(не dense_material)"
            )
        advance_phase_after_chat(memory, intent, action)
        trace(
            f"NODE_DIVE ▶ {action} intent={intent} "
            f"phase={memory.learning_phase} mode={memory.learning_mode}"
        )
        llm_out = await asyncio.to_thread(
            _invoke_tutor,
            memory,
            node,
            intent,
            action,
            user_msg,
            anchor,
            f"node_deep_dive / {action}",
            chat_mgr,
            handoff,
            req.curriculum_id,
        )
        content = _merge_content(session.content, llm_out, False)
        tutor = (llm_out.tutor_message or "").strip()
        if wants_lecture:
            tutor = _ensure_lecture_in_tutor_message(
                tutor,
                content.summary,
            )
        elif not tutor:
            tutor = "Продолжим по теме — один конкретный вопрос или уточнение."

    memory.chat_sessions = chat_mgr.to_memory_blob()

    return await _finalize_node_deep_dive(
        req,
        session,
        anchor,
        content,
        llm_out,
        tutor,
        rag_facts_count,
        rag_fact_labels,
        pipeline_gap,
        action,
    )
