"""Node Deep-Dive: tiered memory, step pipeline, тьютор."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from knowledge_engine.config import (
    GEMINI_INTRO_MAX_OUTPUT_TOKENS,
    GEMINI_PROBE_BEFORE_USE,
    GEMINI_RPM_PAUSE_SEC,
    GEMINI_TUTOR_MAX_OUTPUT_TOKENS,
    GEMINI_TUTOR_MODEL,
    GEMINI_TUTOR_TIMEOUT_SEC,
    KE_RAG_TIMEOUT_SEC,
    LECTURE_EXTERNAL_SEARCH_ENABLED,
    RAG_DEFAULT_MAX_FACTS,
    RAG_DEFAULT_MIN_RELEVANCE,
)
from knowledge_engine.schemas.llm_contracts.tutor import (
    DeepDiveTutorContract,
    IntroAssessmentContract,
)
from knowledge_engine.services.blocking_pools import (
    pool_llm_lecture,
    pool_llm_sync,
    run_blocking,
)
from knowledge_engine.services.chat_session_manager import ChatSessionManager
from knowledge_engine.services.curriculum_whitelist_prompt import (
    enrich_node_learning_materials_from_graph,
    format_node_curriculum_context_for_tutor,
)
from knowledge_engine.services.gemini_stateless import (
    GeminiUnavailableError,
    gemini_tutor_model_chain,
    is_gemini_available,
    run_gemini_structured_with_chain,
)
from knowledge_engine.services.lecture_pipeline import (
    log_external_search_bypass,
    log_external_search_run,
    should_bypass_primary_external_search,
)
from knowledge_engine.services.lecture_rag_context import retrieve_lecture_rag_context
from knowledge_engine.services.llm_markdown_service import (
    enrich_node_deep_dive_response,
)
from knowledge_engine.services.node_content_generator import (
    generate_dense_material,
    merge_dense_material_delta,
)
from knowledge_engine.services.node_source_registry import build_session_source_registry
from knowledge_engine.services.session_prompt_trace import (
    PromptTraceContext,
    PromptTraceMetrics,
)
from knowledge_engine.src.curriculum.schemas import CurriculumNode
from knowledge_engine.src.node_deep_dive.competency_extraction import (
    schedule_competency_extraction,
    wrap_stream_callback_for_competency_extraction,
)
from knowledge_engine.src.node_deep_dive.concept_map import (
    ensure_sub_concept_map,
    find_sub_concept,
    format_concept_map_for_tutor,
    list_verified_sub_concept_ids,
    orchestrate_tutor_llm_output,
    select_next_sub_concept,
    set_pending_evaluation_for_tutor_turn,
)
from knowledge_engine.src.node_deep_dive.content_assets import merge_content_assets
from knowledge_engine.src.node_deep_dive.dialog_context import (
    PINNED_CONTEXT_TAG,
    build_dynamic_suffix,
    build_layer1_explicit_cache_context,
    build_layer2_session_state_context,
    combine_node_context_layers,
    set_dialogue_anchor,
)
from knowledge_engine.src.node_deep_dive.dialog_ids import (
    patch_last_tutor_history_content,
    sync_session_history_turns,
)
from knowledge_engine.src.node_deep_dive.init_context import (
    build_fast_track_tutor_message,
    fast_track_overlap_ratio,
    overlap_mastered_labels,
    regenerate_node_init_context,
)
from knowledge_engine.src.node_deep_dive.learning_loop import (
    build_mastery_dashboard,
)
from knowledge_engine.src.node_deep_dive.lecture_coverage import (
    assess_lecture_coverage,
    coverage_flag_payload_block,
)
from knowledge_engine.src.node_deep_dive.lecture_coverage_registry import (
    merge_lecture_coverage_from_dense,
)
from knowledge_engine.src.node_deep_dive.lecture_scope import (
    dialogue_focus_text,
    resolve_lecture_scope,
)
from knowledge_engine.src.node_deep_dive.lecture_search_orchestrator import (
    collect_lecture_allowed_urls,
    fetch_verified_external_sources,
    format_verified_external_sources_block,
    is_search_tool_only_response,
    merge_verified_sources,
    parse_search_external_materials_request,
)
from knowledge_engine.src.node_deep_dive.memory_schemas import SessionMemory, UserIntent
from knowledge_engine.src.node_deep_dive.prompt_types import (
    InteractionPromptMode,
    PromptComposeContext,
)
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
from knowledge_engine.src.node_deep_dive.session_store import (
    get_all_sessions_for_curriculum,
    get_session,
    persist_session_memory,
    repair_history_with_memory,
    save_session,
)
from knowledge_engine.src.node_deep_dive.step_pipeline import (
    build_tutor_behavior_state_block,
    rotate_window_after_message,
)
from knowledge_engine.src.node_deep_dive.term_registry import merge_introduced_terms
from knowledge_engine.src.node_deep_dive.tiered_memory import (
    append_to_active_window,
    derive_node_status,
    init_session_memory,
)
from knowledge_engine.src.node_deep_dive.tutor_dialogue import (
    coerce_deep_dive_llm_output,
    compose_tutor_dialogue_from_output,
    deep_dive_llm_output_from_chat_text,
    resolve_tutor_display_message,
)
from knowledge_engine.src.node_deep_dive.tutor_prompt_builder import (
    build_intro_system,
    compose_system_prompt,
)
from knowledge_engine.src.node_deep_dive.tutor_reply_sanitize import (
    sanitize_tutor_message_for_transition,
)
from knowledge_engine.src.node_deep_dive.tutor_source_citations import (
    build_tutor_source_registry,
    coerce_references_to_registry,
    scrub_content_references,
)
from knowledge_engine.src.node_deep_dive.user_mastery_profile import (
    get_curriculum_user_mastery_map,
    merge_mastery_from_session_memory,
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
from knowledge_engine.web.llm_text_repair import repair_llm_display_text

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
    """Плотная лекция в чат — кнопка / явный запрос; не обычный ответ на intro-вопрос."""
    if lecture_button_pressed:
        return True
    explicit_lecture = _lecture_request(intent, memory.learning_mode, user_msg)
    if memory.learning_phase == "intro_assessment":
        return explicit_lecture
    if intent == "INTENT_FINALIZE":
        return False
    return explicit_lecture


def resolve_interaction_prompt_mode(
    memory: SessionMemory,
    intent: UserIntent,
    user_msg: str,
) -> InteractionPromptMode:
    """Режим system prompt (не путать с memory.learning_mode)."""
    if memory.learning_phase == "intro_assessment" and not (user_msg or "").strip():
        return InteractionPromptMode.INTRO
    if intent == "INTENT_EXPLAIN":
        return InteractionPromptMode.LECTURE_CHAT
    if _lecture_request(intent, memory.learning_mode, user_msg):
        return InteractionPromptMode.LECTURE_CHAT
    return InteractionPromptMode.DIALOGUE_FEEDBACK


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


def _tutor_message_is_lecture_stub(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 280:
        return True
    low = t.lower()
    return any(m in low for m in _LECTURE_STUB_MARKERS) and len(t) < 900


def _compose_dense_chat_message(dense: DenseMaterialOutput) -> str:
    from knowledge_engine.services.lecture_body_format import (
        append_checkpoint_to_lecture_body,
        strip_lecture_credit_scoreboard,
    )

    body = (dense.lecture_body or "").strip() or (dense.summary or "").strip()
    body = repair_llm_display_text(strip_lecture_credit_scoreboard(body))
    checkpoint = (dense.checkpoint_prompt or "").strip()
    # Keep question in the same message body (streamed + persisted together).
    body = append_checkpoint_to_lecture_body(body, checkpoint)
    return body[:12_000]


def _ensure_lecture_in_tutor_message(
    tutor: str,
    content_summary: str,
    lecture_body: str = "",
) -> str:
    merged = (tutor or "").strip()
    fallback = (lecture_body or "").strip() or (content_summary or "").strip()
    if fallback and _tutor_message_is_lecture_stub(merged):
        if (
            merged
            and len(merged) >= 80
            and not any(m in merged.lower() for m in _LECTURE_STUB_MARKERS)
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


async def _apply_lazy_grounding_for_init(
    req: NodeDeepDiveRequest,
) -> NodeDeepDiveRequest:
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

    from knowledge_engine.src.node_deep_dive.diagram_session import (
        curriculum_node_to_data_input,
        refresh_node_session_diagrams_from_articles,
    )

    status = (node.grounding_status or "").strip()
    reground_academic = False
    if status == "grounded" and node.source_ref:
        from knowledge_engine.src.curriculum.source_registry import (
            resolve_sources_for_node,
        )

        meta_pre = get_curriculum_meta(req.curriculum_id) or {}
        policy_pre = resolve_source_policy(
            meta_pre.get("source_policy"),
            str(meta_pre.get("generation_mode") or "fast"),
            default="hybrid",
        )
        mapped_rows = resolve_sources_for_node(graph, node.node_id)
        academic_tiers = frozenset(
            {"consensus", "arxiv", "semantic_scholar", "searxng_science"}
        )
        has_academic = any(
            (r.get("source_tier") or "").strip().lower() in academic_tiers
            for r in mapped_rows
        )
        if policy_pre in ("hybrid", "academic_only") and not has_academic:
            reground_academic = True
            trace(
                f"NODE_DIVE lazy re-ground ▶ | node={node.node_id} "
                "grounded but mapped sources lack academic/consensus — re-search"
            )
        else:
            refresh_node_session_diagrams_from_articles(
                req.curriculum_id,
                curriculum_node_to_data_input(node),
                rebuild=True,
            )
            trace(
                f"NODE_DIVE lazy grounding ⊘ | node={node.node_id} "
                "already grounded — skip search (diagram refresh only)"
            )
            return _merge_node_data_from_graph(req, node)
    if status != "pending_grounding" and not reground_academic:
        refresh_node_session_diagrams_from_articles(
            req.curriculum_id,
            curriculum_node_to_data_input(node),
            rebuild=True,
        )
        return _merge_node_data_from_graph(req, node)

    from knowledge_engine.services.node_grounding_lock import node_grounding_lock

    async with node_grounding_lock(req.curriculum_id, node.node_id) as lock_held:
        raw = get_curriculum_graph(req.curriculum_id)
        if raw:
            graph = CurriculumGraph.model_validate(raw)
            node = next(
                (n for n in graph.nodes if n.node_id == req.node_data.node_id),
                node,
            )
        status = (node.grounding_status or "").strip()
        if not lock_held or (
            status == "grounded" and node.source_ref and not reground_academic
        ):
            trace(
                f"NODE_DIVE lazy grounding ⊘ | node={node.node_id} "
                "grounded after lock wait — skip search"
            )
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
            reground_academic=reground_academic,
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


def _lecture_rag_inspector_from_memory(memory: SessionMemory | None) -> list[dict]:
    if memory is None:
        return []
    return list(memory.lecture_rag_inspector or [])[:16]


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
    system = build_intro_system()
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
        IntroAssessmentContract,
        "node_deep_dive / intro_assessment",
        rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
        chat_manager=None,
        chat_label="node_deep_dive/intro_assessment",
        handoff_summary="",
        session_registry=chat_mgr,
        models=gemini_tutor_model_chain(),
        http_timeout_sec=GEMINI_TUTOR_TIMEOUT_SEC,
        max_output_tokens=GEMINI_INTRO_MAX_OUTPUT_TOKENS,
    )
    tutor = (out.tutor_message or "").strip()
    if tutor:
        chat_mgr.record_turn(
            "node_deep_dive/intro_assessment",
            user_payload[:8000],
            tutor,
        )
    trace(
        "NODE_DIVE этап 2/2 intro ✓ | learning_phase=intro_assessment | в чате первый вопрос"
    )
    return IntroAssessmentOutput.model_validate(out.model_dump())


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
    node_content: NodeContentBlock | None = None,
    stream_callback: Callable[[str], None] | None = None,
    node_session_key: str = "",
    *,
    strip_chat_history: bool = False,
) -> DeepDiveLLMOutput:
    from knowledge_engine.src.node_deep_dive.subconcept_invariants import (
        format_subconcept_hard_anchor,
    )

    node_for_tutor = enrich_node_learning_materials_from_graph(node, curriculum_id)
    node_ctx = format_node_curriculum_context_for_tutor(node_for_tutor, curriculum_id)
    from knowledge_engine.src.node_deep_dive.prompt_factory import (
        is_factory_control_mode,
        parse_tutor_mode_prefix,
        select_system_prompt_and_mode,
    )

    raw_user_msg = user_msg
    cleaned_user_msg, factory_mode = parse_tutor_mode_prefix(user_msg)
    # Behavior/chip classify keep the prefix; LLM/RAG see the cleaned body.
    user_msg = cleaned_user_msg or user_msg
    dlg_focus = dialogue_focus_text(user_msg, memory)
    prompt_mode = resolve_interaction_prompt_mode(memory, intent, user_msg)
    use_lite_layers = prompt_mode == InteractionPromptMode.DIALOGUE_FEEDBACK
    behavior_block = build_tutor_behavior_state_block(
        intent,
        action,
        memory.learning_mode,
        memory.learning_phase,
        raw_user_msg,
        has_user_focus=bool(dlg_focus),
        memory=memory,
        node_layer=str(getattr(node, "layer", "") or ""),
    )
    ensure_sub_concept_map(memory, node_for_tutor)
    nid = (memory.next_question_concept_id or "").strip()
    focus_row = find_sub_concept(memory, nid) if nid else None
    if focus_row is None or focus_row.status == "verified":
        focus_sc = select_next_sub_concept(memory)
        memory.next_question_concept_id = focus_sc.id if focus_sc is not None else ""
    concept_block = format_concept_map_for_tutor(
        memory,
        focus_id=memory.next_question_concept_id,
    )
    hard_anchor = format_subconcept_hard_anchor(memory)
    layer1 = build_layer1_explicit_cache_context(
        memory,
        node_for_tutor,
        intent,
        action,
        node_ctx,
        curriculum_id=(curriculum_id or "").strip(),
        node_content=node_content,
        is_lightweight=use_lite_layers,
    )
    atoms_block = ""
    try:
        from knowledge_engine.services.dialog_atoms_rag import (
            retrieve_dialog_knowledge_atoms,
        )

        atoms_block = retrieve_dialog_knowledge_atoms(
            user_msg,
            node_for_tutor,
            (curriculum_id or "").strip(),
        )
    except Exception as exc:
        trace(f"DIALOG_ATOMS skip | {exc}")
    layer2 = build_layer2_session_state_context(
        memory,
        node_for_tutor,
        intent,
        user_focus=dlg_focus,
        curriculum_id=(curriculum_id or "").strip(),
        is_lightweight=use_lite_layers,
        concept_map_block=concept_block if use_lite_layers else "",
        knowledge_atoms_block=atoms_block,
    )
    pinned_body = combine_node_context_layers(layer1, layer2)
    pinned = (
        pinned_body
        if PINNED_CONTEXT_TAG in pinned_body
        else f"{PINNED_CONTEXT_TAG}\n\n{pinned_body}"
    ).strip()
    movable_body = behavior_block
    if concept_block and not use_lite_layers:
        movable_body = f"{behavior_block}\n\n{concept_block}"
    if hard_anchor:
        movable_body = f"{hard_anchor}\n\n{movable_body}"
    tutor_label = "node_deep_dive/tutor"
    tutor_chat = chat_mgr.get(tutor_label)
    include_sliding_window = (
        False if strip_chat_history else (tutor_chat is None or tutor_chat.turns <= 0)
    )
    compose_ctx = PromptComposeContext(memory=memory)
    default_system = compose_system_prompt(prompt_mode, context=compose_ctx)
    system, factory_mode, _ = select_system_prompt_and_mode(
        raw_user_msg,
        default_system_prompt=default_system,
    )
    if is_factory_control_mode(factory_mode):
        trace(f"PROMPT_FACTORY | mode={factory_mode} | isolated system prompt")
    dynamic = build_dynamic_suffix(
        memory,
        user_msg,
        include_window=include_sliding_window,
    )
    if strip_chat_history and hard_anchor:
        dynamic = f"{hard_anchor}\n\n{dynamic}".strip()
    prompt_trace = PromptTraceContext(
        node_session_key=(node_session_key or "").strip(),
        metrics=PromptTraceMetrics(
            pinned_len=len(pinned),
            behavior_state_len=len(behavior_block),
            recency_tail_len=compose_ctx.last_recency_len,
            user_message_len=len((user_msg or "").strip()),
        ),
    )
    raw = run_gemini_structured_with_chain(
        GEMINI_TUTOR_MODEL,
        system,
        movable_body,
        anchor,
        DeepDiveTutorContract,
        label,
        rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
        chat_manager=chat_mgr,
        chat_label=tutor_label,
        handoff_summary="" if strip_chat_history else handoff,
        delta_user_message=dynamic,
        pinned_context=pinned,
        stream_callback=stream_callback,
        models=gemini_tutor_model_chain(),
        http_timeout_sec=GEMINI_TUTOR_TIMEOUT_SEC,
        max_output_tokens=GEMINI_TUTOR_MAX_OUTPUT_TOKENS,
        prompt_trace=prompt_trace,
        layer1_context=layer1,
        layer2_context=layer2,
        node_session_key=(node_session_key or "").strip()
        or f"{curriculum_id}/{node_for_tutor.node_id}",
    )
    return DeepDiveLLMOutput.model_validate(raw.model_dump())


def _merge_content(
    prev: NodeContentBlock,
    llm: DeepDiveLLMOutput,
    is_init: bool,
    *,
    curriculum_id: str = "",
    node: NodeDataInput | None = None,
) -> NodeContentBlock:
    if is_init:
        return NodeContentBlock()
    summary = (llm.summary or "").strip() or None
    refs = list(llm.references or [])
    if node is not None and refs:
        registry = build_tutor_source_registry(curriculum_id, node, prev)
        refs = coerce_references_to_registry(refs, registry)
    elif refs:
        refs = []
    return merge_content_assets(
        prev,
        referenced_diagram_id=(llm.referenced_diagram_id or None),
        references=refs if refs else None,
        summary=summary,
    )


def _apply_dense_material(
    content: NodeContentBlock, dense: DenseMaterialOutput
) -> NodeContentBlock:
    summary = (dense.summary or "").strip() or None
    return merge_content_assets(
        content,
        referenced_diagram_id=(dense.referenced_diagram_id or None),
        code_snippets=list(dense.code_snippets or []),
        references=dense.references if dense.references else None,
        summary=summary,
    )


def _coerce_llm_status(raw: str | None) -> NodeStatus | None:
    s = (raw or "").strip()
    if s in (
        "unexplored",
        "in_progress",
        "deep_understanding",
        "mastered",
        "gap",
        "passed_by_equivalence",
    ):
        return s
    return None


def _anchor_turn_content(memory: SessionMemory) -> str:
    at = memory.anchor_turn
    if isinstance(at, dict):
        return (at.get("content") or "").strip()
    return (str(at or "")).strip() if at else ""


def _session_needs_lazy_intro(session, memory: SessionMemory, user_msg: str) -> bool:
    if session.node_status != "unexplored":
        return False
    if _anchor_turn_content(memory):
        return False
    raw = (user_msg or "").strip()
    if not raw:
        return False
    if raw.startswith("[mode:lecture]"):
        return False
    stripped = raw.replace("[mode:lecture]", "", 1).strip() or raw
    if _is_explicit_lecture_request(stripped):
        return False
    return True


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
    if session.memory is not None:
        return session.memory
    mem = init_session_memory(node, rag_text)
    session.memory = mem
    return mem


async def fetch_node_init_rag_facts(
    req: NodeDeepDiveRequest,
) -> tuple[str, int, list[str]]:
    """Directional RAG только (без intro payload) — можно overlap с lazy grounding."""
    from knowledge_engine.src.memory.light_rag import (
        sync_profile_from_markdown_if_needed,
    )

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
        "(векторный поиск + cross-encoder; длинные факты — Gemma-сжатие)"
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
    dir_tags = sorted(
        {(f.direction or "").strip() for f in rag_resp.facts if f.direction}
    )
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
) -> tuple[str, ChatSessionManager, int, list[str]]:
    """Memory + RAG после merge grounding; intro Gemini — при первом chat."""
    node = req.node_data
    anchor = _anchor(req.curriculum_id, node.node_id)
    session = get_session(req.curriculum_id, node.node_id)
    preserved_cov = dict(
        (session.memory.covered_subtopics if session.memory is not None else None) or {}
    )
    preserved_terms = list(
        (session.memory.introduced_terms if session.memory is not None else None) or []
    )
    session.history = []
    session.memory = init_session_memory(
        node,
        rag_facts_text,
        preserved_covered_subtopics=preserved_cov,
        preserved_introduced_terms=preserved_terms,
    )
    chat_mgr = ChatSessionManager.from_memory_blob(anchor, session.memory.chat_sessions)
    chat_mgr.clear_all("init")
    session.memory.chat_sessions = chat_mgr.to_memory_blob()
    init_registry = build_session_source_registry(
        req.curriculum_id,
        list(node.mapped_source_ids or []),
    )
    from knowledge_engine.src.node_deep_dive.content_assets import (
        hydrate_content_diagrams_from_articles,
    )

    content = hydrate_content_diagrams_from_articles(
        NodeContentBlock(),
        node,
        req.curriculum_id,
    )
    save_session(
        req.curriculum_id,
        node.node_id,
        "unexplored",
        content,
        [],
        rag_fact_labels=rag_fact_labels,
        memory=session.memory,
        source_registry=init_registry,
    )
    return anchor, chat_mgr, rag_facts_count, rag_fact_labels


async def prepare_node_init_rag(
    req: NodeDeepDiveRequest,
) -> tuple[str, ChatSessionManager, int, list[str]]:
    """Этап 1 init: RAG + memory (async), без intro Gemini."""
    rag_facts_text, rag_facts_count, rag_fact_labels = await fetch_node_init_rag_facts(
        req
    )
    anchor, chat_mgr, rag_facts_count, rag_fact_labels = (
        await finalize_node_init_after_grounding(
            req,
            rag_facts_text,
            rag_facts_count,
            rag_fact_labels,
        )
    )
    return anchor, chat_mgr, rag_facts_count, rag_fact_labels


async def complete_node_prepare_response(
    req: NodeDeepDiveRequest,
    rag_facts_count: int,
    rag_fact_labels: list[str],
) -> NodeDeepDiveResponse:
    """Ответ init: RAG/memory готовы, первый вопрос — только после chat (Начать)."""
    node = req.node_data
    session = get_session(req.curriculum_id, node.node_id)
    mem = session.memory
    status: NodeStatus = "unexplored"
    from knowledge_engine.src.node_deep_dive.content_assets import (
        hydrate_content_diagrams_from_articles,
    )

    content = hydrate_content_diagrams_from_articles(
        session.content,
        node,
        req.curriculum_id,
    )
    before_n = len(session.content.diagrams or [])
    after_n = len(content.diagrams or [])
    if after_n > before_n:
        save_session(
            req.curriculum_id,
            node.node_id,
            session.node_status,
            content,
            session.history,
            memory=mem,
        )
    source_registry = build_session_source_registry(
        req.curriculum_id,
        list(node.mapped_source_ids or []),
    )
    content = scrub_content_references(content, source_registry)
    dash = build_mastery_dashboard(mem, status)
    key = _anchor(req.curriculum_id, node.node_id)
    base = NodeDeepDiveResponse(
        node_id=node.node_id,
        node_status=status,
        content=content,
        tutor_message="",
        history=[],
        new_gap_to_record=None,
        session_key=key,
        rag_facts_count=rag_facts_count,
        rag_fact_labels=rag_fact_labels,
        topic_mastery_score=mem.topic_mastery_score if mem else 0,
        concepts_matrix=list(mem.concepts_matrix) if mem else [],
        mastery_dashboard=dash,
        coverage_summary=dash.coverage_summary,
        learning_phase=mem.learning_phase if mem else "intro_assessment",
        learning_mode=mem.learning_mode if mem else "lecture",
        source_registry=source_registry,
        lecture_rag_inspector=_lecture_rag_inspector_from_memory(mem),
    )
    trace(
        f"NODE_DIVE ✓ init prepare | {req.curriculum_id}/{node.node_id} | "
        "status=unexplored (lazy intro)"
    )
    return enrich_node_deep_dive_response(base, source_registry)


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


async def _mark_passed_by_equivalence(
    req: NodeDeepDiveRequest,
    session,
    anchor: str,
    user_msg: str,
    rag_facts_count: int,
    rag_fact_labels: list[str],
) -> NodeDeepDiveResponse:
    node = req.node_data
    memory = session.memory
    tutor = (
        "Зафиксировали: тема отмечена как уже известная (passed_by_equivalence). "
        "Можно перейти к следующим нодам на карте."
    )
    if memory is not None:
        memory.intro_question_pending = False
        merge_mastery_from_session_memory(
            req.curriculum_id,
            memory,
            overlap_mastered_labels(
                node,
                get_curriculum_user_mastery_map(req.curriculum_id),
            ),
            score=0.92,
        )
    llm_out = deep_dive_llm_output_from_chat_text(
        tutor,
        node_status="passed_by_equivalence",
    )
    return await _finalize_node_deep_dive(
        req,
        session,
        anchor,
        session.content,
        llm_out,
        tutor,
        rag_facts_count,
        rag_fact_labels,
        None,
        "chat",
    )


async def _deliver_lazy_intro(
    req: NodeDeepDiveRequest,
    session,
    memory: SessionMemory,
    anchor: str,
    chat_mgr: ChatSessionManager,
    user_msg: str,
    rag_facts_count: int,
    rag_fact_labels: list[str],
) -> NodeDeepDiveResponse:
    node = req.node_data
    cid = req.curriculum_id
    intro_payload = regenerate_node_init_context(
        cid, node, memory, user_message=user_msg
    )
    mastery_map = get_curriculum_user_mastery_map(cid)
    ratio = fast_track_overlap_ratio(node, mastery_map)
    labels = overlap_mastered_labels(node, mastery_map)

    if ratio >= 0.7 and labels:
        tutor = build_fast_track_tutor_message(node, labels)
        trace(f"NODE_DIVE lazy intro fast-track | overlap={ratio:.0%}")
    else:
        trace(
            "NODE_DIVE lazy intro ▶ | regenerate_node_init_context + intro_assessment"
        )
        intro_out = await run_blocking(
            pool_llm_sync(),
            _invoke_intro_assessment,
            intro_payload,
            anchor,
            chat_mgr,
        )
        tutor = (intro_out.tutor_message or "").strip()
        if not tutor:
            tutor = (
                f"Один практический кейс по «{node.title}»: "
                "опишите главный риск или механику в 3–5 предложений."
            )

    set_dialogue_anchor(memory, node, tutor)
    focus = select_next_sub_concept(memory)
    set_pending_evaluation_for_tutor_turn(memory, focus.id if focus else "")
    memory.intro_question_pending = True
    memory.chat_sessions = chat_mgr.to_memory_blob()
    llm_out = deep_dive_llm_output_from_chat_text(
        tutor,
        node_status="unexplored",
    )
    return await _finalize_node_deep_dive(
        req,
        session,
        anchor,
        session.content,
        llm_out,
        tutor,
        rag_facts_count,
        rag_fact_labels,
        None,
        "chat",
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
    llm_out = deep_dive_llm_output_from_chat_text(
        tutor,
        node_status="unexplored",
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
    tutor = repair_llm_display_text((tutor or "").strip())
    trace(f"NODE_DIVE ▶ {action} | {req.curriculum_id}/{node.node_id}")
    session.content = content

    memory = session.memory
    if memory is not None and llm_out is not None:
        llm_out = orchestrate_tutor_llm_output(
            memory,
            llm_out,
            user_message=(req.user_message or "").strip(),
            node_layer=str(getattr(node, "layer", "") or ""),
        )
        if llm_out.ready_for_transition:
            tutor = sanitize_tutor_message_for_transition(
                compose_tutor_dialogue_from_output(llm_out) or tutor or ""
            )
            repacked = deep_dive_llm_output_from_chat_text(
                tutor,
                node_status=llm_out.node_status,
            )
            llm_out = llm_out.model_copy(
                update={
                    "feedback_on_answer": repacked.feedback_on_answer,
                    "technical_explanation": repacked.technical_explanation,
                    "follow_up_question": repacked.follow_up_question,
                }
            )
    if memory is not None and (tutor or "").strip():
        if not llm_out.ready_for_transition:
            follow = (llm_out.follow_up_question or "").strip()
            qid = (llm_out.question_sub_concept_id or "").strip()
            if follow and qid:
                cid = set_pending_evaluation_for_tutor_turn(memory, qid)
                if cid:
                    trace(
                        f"NODE_DIVE pending question set | concept={cid} "
                        "(question_sub_concept_id)"
                    )
            elif follow:
                trace("WARN finalize | follow_up_question без question_sub_concept_id")
        else:
            trace(
                f"NODE_DIVE topic transition | "
                f"step={llm_out.suggested_next_step or 'next_node'} "
                "(no pending technical question)"
            )
        merge_introduced_terms(memory, list(llm_out.introduced_terms or []))
        from knowledge_engine.src.node_deep_dive.tutor_memory_content import (
            tutor_content_for_active_window,
        )

        window_tutor = tutor_content_for_active_window(
            llm_out, fallback_compose_text=tutor
        )
        append_to_active_window(memory, "tutor", window_tutor or tutor)
        rotate_window_after_message(memory, anchor)
        from knowledge_engine.src.curriculum.global_tracker import infer_question_angle
        from knowledge_engine.src.node_deep_dive.concept_map import (
            list_verified_sub_concept_ids,
        )

        memory.last_tutor_question_angle = infer_question_angle(tutor)
        verified_ids = list_verified_sub_concept_ids(memory)[:8]
        llm_out = llm_out.model_copy(
            update={"verified_sub_concept_ids": verified_ids},
        )

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
        if llm_status == "passed_by_equivalence":
            status = "passed_by_equivalence"
        elif llm_status == "mastered" and status != "gap":
            status = "mastered"
        elif llm_status == "gap":
            status = "gap"
        elif llm_status == "unexplored" or memory.intro_question_pending:
            status = "unexplored"
    else:
        status = _coerce_llm_status(llm_out.node_status) or "in_progress"

    if memory is not None and status not in ("unexplored", "gap"):
        score = max(0.55, memory.topic_mastery_score / 100.0)
        if status == "passed_by_equivalence":
            score = 0.92
        merge_mastery_from_session_memory(req.curriculum_id, memory, score=score)

    labels_for_store = list(rag_fact_labels)
    if not labels_for_store:
        prev_blob = get_all_sessions_for_curriculum(req.curriculum_id).get(
            node.node_id, {}
        )
        labels_for_store = list(prev_blob.get("rag_fact_labels") or [])

    source_registry = build_session_source_registry(
        req.curriculum_id,
        list(node.mapped_source_ids or []),
    )
    content = scrub_content_references(content, source_registry)

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
    cov = dash.coverage_summary
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
        coverage_summary=cov,
        learning_phase=mem.learning_phase if mem else "intro_assessment",
        learning_mode=mem.learning_mode if mem else "lecture",
        source_registry=source_registry,
        lecture_rag_inspector=_lecture_rag_inspector_from_memory(mem),
    )
    return enrich_node_deep_dive_response(base, source_registry)


async def finalize_graph_chat_response(state: dict[str, Any]) -> NodeDeepDiveResponse:
    """Post-graph finalize: save session and build API response (no turn window writes)."""
    req = state["request"]
    memory = state["memory"]
    node = req.node_data
    action = (req.user_action or "").strip().lower()
    content = state.get("content") or NodeContentBlock()
    llm_out = coerce_deep_dive_llm_output(state.get("llm_out"))
    tutor = resolve_tutor_display_message(
        llm_out,
        (state.get("tutor_message") or "").strip(),
    )
    pipeline_gap = state.get("pipeline_gap")
    rag_facts_count = int(state.get("rag_facts_count") or 0)
    rag_fact_labels = list(state.get("rag_fact_labels") or [])

    trace(f"NODE_DIVE ▶ {action} | {req.curriculum_id}/{node.node_id}")
    if llm_out is None:
        llm_out = deep_dive_llm_output_from_chat_text(tutor)

    history = list(state.get("session_history") or [])
    if not history:
        session = get_session(req.curriculum_id, node.node_id)
        if action in ("chat", "verify"):
            history = sync_session_history_turns(
                session.history,
                memory,
                user_message=(req.user_message or "").strip(),
                tutor_message=tutor,
            )
        else:
            history = sync_session_history_turns(
                session.history,
                memory,
                tutor_message=tutor,
            )
        history = repair_history_with_memory(history, memory)

    history = patch_last_tutor_history_content(history, tutor)

    verified_ids = list_verified_sub_concept_ids(memory)[:8]
    llm_out = llm_out.model_copy(update={"verified_sub_concept_ids": verified_ids})

    gap = (llm_out.new_gap_to_record or "").strip() or pipeline_gap or None
    if gap:
        await _record_gap_if_needed(gap, node.node_id)

    status: NodeStatus
    if action == "init":
        status = "unexplored"
    elif memory is not None:
        status = derive_node_status(memory, gap)
        llm_status = _coerce_llm_status(llm_out.node_status)
        if llm_status == "passed_by_equivalence":
            status = "passed_by_equivalence"
        elif llm_status == "mastered" and status != "gap":
            status = "mastered"
        elif llm_status == "gap":
            status = "gap"
        elif llm_status == "unexplored" or memory.intro_question_pending:
            status = "unexplored"
    else:
        status = _coerce_llm_status(llm_out.node_status) or "in_progress"

    if memory is not None and status not in ("unexplored", "gap"):
        score = max(0.55, memory.topic_mastery_score / 100.0)
        if status == "passed_by_equivalence":
            score = 0.92
        merge_mastery_from_session_memory(req.curriculum_id, memory, score=score)

    labels_for_store = list(rag_fact_labels)
    if not labels_for_store:
        prev_blob = get_all_sessions_for_curriculum(req.curriculum_id).get(
            node.node_id, {}
        )
        labels_for_store = list(prev_blob.get("rag_fact_labels") or [])

    source_registry = build_session_source_registry(
        req.curriculum_id,
        list(node.mapped_source_ids or []),
    )
    content = scrub_content_references(content, source_registry)

    if memory is not None and tutor:
        memory.last_tutor_display_message = tutor[:12_000]
        memory.last_tutor_follow_up_question = (
            llm_out.follow_up_question or ""
        ).strip()[:2000]

    key = save_session(
        req.curriculum_id,
        node.node_id,
        status,
        content,
        history,
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
    display_history = patch_last_tutor_history_content(display_history, tutor)
    mem = stored.memory
    dash = build_mastery_dashboard(mem, status)
    cov = dash.coverage_summary
    dlg_fb = (llm_out.feedback_on_answer or "").strip()
    dlg_tech = (llm_out.technical_explanation or "").strip()
    dlg_fu = (llm_out.follow_up_question or "").strip()
    from knowledge_engine.src.node_deep_dive.concept_map import (
        classify_gloss_fork_choice,
        is_full_depth_closure,
        sub_concept_coverage_complete,
    )

    ready_tr = bool(getattr(llm_out, "ready_for_transition", False))
    chip = classify_gloss_fork_choice((req.user_message or "").strip())
    if chip in ("how", "mech"):
        ready_tr = False
    elif mem is not None:
        pending = (mem.pending_evaluation_concept_id or "").strip()
        if pending and not ready_tr:
            # Awaiting deep-dive control answer — keep chips off.
            ready_tr = False
        elif not ready_tr:
            layer = str(getattr(node, "layer", "") or "foundation")
            if is_full_depth_closure(mem, layer) and (
                sub_concept_coverage_complete(mem)
                or (mem.learning_phase or "") == "pathway_decision"
            ):
                ready_tr = True
            elif (
                sub_concept_coverage_complete(mem)
                or (mem.learning_phase or "") == "pathway_decision"
            ) and chip == "gloss":
                ready_tr = True
            elif (
                sub_concept_coverage_complete(mem)
                or (mem.learning_phase or "") == "pathway_decision"
            ) and not pending:
                # Threshold met (optional layers may still be open) — allow chips.
                ready_tr = True
    base = NodeDeepDiveResponse(
        node_id=node.node_id,
        node_status=status,
        content=content,
        tutor_message=tutor,
        tutor_dialogue_feedback=dlg_fb,
        tutor_dialogue_technical=dlg_tech,
        tutor_dialogue_follow_up=dlg_fu,
        quick_replies=[
            str(x).strip()
            for x in (getattr(llm_out, "quick_replies", None) or [])
            if str(x).strip()
        ][:4],
        ready_for_transition=ready_tr,
        last_eval_directive=(
            (getattr(mem, "last_eval_directive", None) or "").strip()[:64]
            if mem
            else ""
        ),
        history=display_history,
        new_gap_to_record=gap,
        session_key=key,
        rag_facts_count=rag_facts_count,
        rag_fact_labels=rag_fact_labels,
        topic_mastery_score=mem.topic_mastery_score if mem else 0,
        concepts_matrix=list(mem.concepts_matrix) if mem else [],
        mastery_dashboard=dash,
        coverage_summary=cov,
        learning_phase=mem.learning_phase if mem else "intro_assessment",
        learning_mode=mem.learning_mode if mem else "lecture",
        source_registry=source_registry,
        lecture_rag_inspector=_lecture_rag_inspector_from_memory(mem),
    )
    return enrich_node_deep_dive_response(base, source_registry)


async def run_init_prepare_turn(state: dict[str, Any]) -> dict[str, Any]:
    """Graph init node: grounding ∥ RAG → memory + content (lazy intro on chat)."""
    req: NodeDeepDiveRequest = state["request"]
    node = req.node_data
    trace("NODE_DIVE init parallel ▶ | lazy_grounding ∥ directional RAG (prepare only)")
    from knowledge_engine.config import (
        CURRICULUM_TARGETED_NODE_GROUNDING_ENABLED,
        KE_NODE_DIVE_INIT_ASYNC_TIMEOUT_SEC,
        KE_NODE_DIVE_INIT_GROUNDING_MIN_TIMEOUT_SEC,
        KE_RAG_TIMEOUT_SEC,
    )

    init_timeout = float(KE_NODE_DIVE_INIT_ASYNC_TIMEOUT_SEC)
    if CURRICULUM_TARGETED_NODE_GROUNDING_ENABLED:
        init_timeout = max(init_timeout, KE_NODE_DIVE_INIT_GROUNDING_MIN_TIMEOUT_SEC)

    try:
        grounded_req, rag_facts = await asyncio.gather(
            asyncio.wait_for(_apply_lazy_grounding_for_init(req), timeout=init_timeout),
            asyncio.wait_for(
                fetch_node_init_rag_facts(req), timeout=KE_RAG_TIMEOUT_SEC
            ),
        )
    except asyncio.TimeoutError:
        from knowledge_engine.src.retrieval.consensus_session import (
            shutdown_shared_consensus_session,
        )

        trace(
            f"NODE_DIVE init timeout ▶ | lazy_grounding>{init_timeout:.0f}s — "
            "closing Consensus browser"
        )
        await shutdown_shared_consensus_session()
        raise
    await finalize_node_init_after_grounding(grounded_req, *rag_facts)
    rag_facts_count = int(rag_facts[1])
    rag_fact_labels = list(rag_facts[2])
    session = get_session(grounded_req.curriculum_id, node.node_id)
    anchor = _anchor(grounded_req.curriculum_id, node.node_id)
    return {
        **state,
        "request": grounded_req,
        "memory": session.memory,
        "content": session.content,
        "anchor": anchor,
        "tutor_message": "",
        "llm_out": DeepDiveLLMOutput(node_status="unexplored"),
        "rag_facts_count": rag_facts_count,
        "rag_fact_labels": rag_fact_labels,
        "session_history": [],
    }


async def run_lazy_intro_turn(state: dict[str, Any]) -> dict[str, Any]:
    """Graph lazy intro: intro_assessment or fast-track before normal chat turns."""
    req: NodeDeepDiveRequest = state["request"]
    memory: SessionMemory = state["memory"]
    anchor: str = state["anchor"]
    node = req.node_data
    cid = req.curriculum_id
    user_msg = (req.user_message or "").strip()
    chat_mgr = ChatSessionManager.from_memory_blob(anchor, memory.chat_sessions)
    intro_payload = regenerate_node_init_context(
        cid, node, memory, user_message=user_msg
    )
    mastery_map = get_curriculum_user_mastery_map(cid)
    ratio = fast_track_overlap_ratio(node, mastery_map)
    labels = overlap_mastered_labels(node, mastery_map)

    if ratio >= 0.7 and labels:
        tutor = build_fast_track_tutor_message(node, labels)
        trace(f"NODE_DIVE lazy intro fast-track | overlap={ratio:.0%}")
    else:
        trace(
            "NODE_DIVE lazy intro ▶ | regenerate_node_init_context + intro_assessment"
        )
        intro_out = await run_blocking(
            pool_llm_sync(),
            _invoke_intro_assessment,
            intro_payload,
            anchor,
            chat_mgr,
        )
        tutor = (intro_out.tutor_message or "").strip()
        if not tutor:
            tutor = (
                f"Один практический кейс по «{node.title}»: "
                "опишите главный риск или механику в 3–5 предложений."
            )

    set_dialogue_anchor(memory, node, tutor)
    focus = select_next_sub_concept(memory)
    set_pending_evaluation_for_tutor_turn(memory, focus.id if focus else "")
    memory.intro_question_pending = True
    memory.chat_sessions = chat_mgr.to_memory_blob()
    llm_out = deep_dive_llm_output_from_chat_text(
        tutor,
        node_status="unexplored",
    )
    return {
        **state,
        "memory": memory,
        "tutor_message": tutor,
        "llm_out": llm_out,
    }


async def run_equivalence_turn(state: dict[str, Any]) -> dict[str, Any]:
    """Graph: user marks node as already known."""
    req: NodeDeepDiveRequest = state["request"]
    memory = state.get("memory")
    node = req.node_data
    tutor = (
        "Зафиксировали: тема отмечена как уже известная (passed_by_equivalence). "
        "Можно перейти к следующим нодам на карте."
    )
    if memory is not None:
        memory.intro_question_pending = False
        merge_mastery_from_session_memory(
            req.curriculum_id,
            memory,
            overlap_mastered_labels(
                node,
                get_curriculum_user_mastery_map(req.curriculum_id),
            ),
            score=0.92,
        )
    llm_out = deep_dive_llm_output_from_chat_text(
        tutor,
        node_status="passed_by_equivalence",
    )
    return {
        **state,
        "memory": memory,
        "tutor_message": tutor,
        "llm_out": llm_out,
    }


async def run_node_deep_dive(
    req: NodeDeepDiveRequest,
    stream_callback: Callable[[str], None] | None = None,
) -> NodeDeepDiveResponse:
    if not is_gemini_available():
        raise GeminiUnavailableError("Gemini недоступен для Node Deep-Dive")

    action = req.user_action
    node = req.node_data
    if action in ("chat", "verify") and not (req.user_message or "").strip():
        raise ValueError("user_message обязателен для chat и verify")

    anchor = _anchor(req.curriculum_id, node.node_id)
    raw_user = (req.user_message or "").strip()
    stream_for_llm = stream_callback
    stream_was_wrapped = False
    if action in ("chat", "verify") and stream_callback is not None and raw_user:
        stream_for_llm, _ = wrap_stream_callback_for_competency_extraction(
            stream_callback,
            req.curriculum_id,
            raw_user,
            node,
        )
        stream_was_wrapped = True

    from knowledge_engine.src.node_deep_dive.graph import get_compiled_tutor_graph

    trace(f"NODE_DIVE ▶ LangGraph | action={action}")
    final_state = await get_compiled_tutor_graph().ainvoke(
        {"request": req},
        config={
            "configurable": {
                "thread_id": anchor,
                "stream_callback": stream_for_llm,
            },
        },
    )
    resp = final_state.get("response")
    if resp is None:
        raise RuntimeError("NODE_DIVE graph finished without NodeDeepDiveResponse")

    tutor = (final_state.get("tutor_message") or "").strip()
    if action in ("chat", "verify") and raw_user and not stream_was_wrapped:
        schedule_competency_extraction(
            req.curriculum_id,
            raw_user,
            node,
            tutor_preview=tutor,
        )
    return resp


async def run_dense_lecture_turn(
    state: dict[str, Any],
    stream_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Full dense lecture path for LangGraph ``dense_lecture`` node."""
    req: NodeDeepDiveRequest = state["request"]
    memory: SessionMemory = state["memory"]
    anchor: str = state["anchor"]
    node = req.node_data
    content = state.get("content") or NodeContentBlock()
    raw_user = (req.user_message or "").strip()
    from knowledge_engine.src.node_deep_dive.prompt_factory import (
        parse_tutor_mode_prefix,
    )

    user_msg, mode = parse_tutor_mode_prefix(raw_user)
    lecture_pressed = mode == "lecture"

    trace(
        f"NODE_DIVE dense_material ▶ | phase={memory.learning_phase} | "
        f"модель={GEMINI_TUTOR_MODEL}"
    )
    session = get_session(req.curriculum_id, node.node_id)
    node_for_lecture = enrich_node_learning_materials_from_graph(
        node, req.curriculum_id
    )
    lecture_scope, focus_text = resolve_lecture_scope(
        user_msg,
        memory,
        lecture_button_pressed=lecture_pressed,
    )
    content_summary = (content.summary or "").strip()
    coverage = assess_lecture_coverage(
        memory,
        session.history,
        user_msg,
        lecture_scope,
        focus_text,
        lecture_pressed,
        content_summary=content_summary,
    )
    trace(
        f"NODE_DIVE coverage | covered={coverage.is_topic_already_covered} "
        f"notice={coverage.should_return_coverage_notice} "
        f"scope={lecture_scope}"
    )
    if coverage.should_return_coverage_notice:
        trace("NODE_DIVE dense_lecture skip | coverage_notice (router should handle)")
        return state

    topic_flag = coverage.is_topic_already_covered
    coverage_payload = coverage_flag_payload_block(coverage, memory)
    rag_query = (
        focus_text if lecture_scope == "targeted_lecture" and focus_text else user_msg
    )
    rag_result = await retrieve_lecture_rag_context(
        node_for_lecture, rag_query, req.curriculum_id
    )
    rag_context = rag_result.context
    rag_citation_registry = (rag_result.citation_registry_block or "").strip()
    memory.lecture_rag_inspector = list(rag_result.inspector_chunks or [])
    verified_sources: list = []
    if not LECTURE_EXTERNAL_SEARCH_ENABLED:
        trace(
            "[LECTURE_PIPELINE] External search disabled "
            "(LECTURE_EXTERNAL_SEARCH_ENABLED=false)"
        )
    elif should_bypass_primary_external_search(rag_result.stats):
        log_external_search_bypass(rag_result.stats)
    else:
        log_external_search_run(rag_result.stats)
        verified_sources = await fetch_verified_external_sources(
            node_for_lecture,
            rag_query,
            req.curriculum_id,
        )

    chat_mgr = ChatSessionManager.from_memory_blob(anchor, memory.chat_sessions)
    curriculum_id = req.curriculum_id

    def _dense_material_job() -> DenseMaterialOutput:
        block = format_verified_external_sources_block(verified_sources)
        urls = collect_lecture_allowed_urls(
            verified_sources,
            rag_context,
            node_for_lecture,
            curriculum_id,
            skip_graph_enrich=True,
        )
        return generate_dense_material(
            node_for_lecture,
            memory,
            memory.rag_profile_compressed,
            anchor,
            chat_mgr,
            user_msg,
            rag_context,
            curriculum_id,
            lecture_scope,
            focus_text,
            stream_callback,
            topic_flag,
            coverage_payload,
            block,
            urls,
            node_content=content,
            rag_citation_registry=rag_citation_registry,
        )

    trace(
        "NODE_DIVE dense_material ▶ Gemini | "
        f"RAG_CONTEXT={len(rag_context or '')} (ke-pool-llm-lecture)"
    )
    dense = await run_blocking(pool_llm_lecture(), _dense_material_job)
    tool_q = parse_search_external_materials_request(dense.lecture_body or "")
    if tool_q or is_search_tool_only_response(dense.lecture_body or ""):
        trace(
            "[LECTURE_PIPELINE] Tool search_external_materials — "
            "running external search (bypass guard)"
        )
        extra_sources = await fetch_verified_external_sources(
            node_for_lecture,
            rag_query,
            curriculum_id,
            query_override=tool_q or rag_query,
        )
        if extra_sources:
            verified_sources = merge_verified_sources(
                verified_sources,
                extra_sources,
            )
            verified_block = format_verified_external_sources_block(verified_sources)
            allowed_urls = collect_lecture_allowed_urls(
                verified_sources,
                rag_context,
                node_for_lecture,
                curriculum_id,
                skip_graph_enrich=True,
            )

            def _delta_dense_job() -> DenseMaterialOutput:
                return generate_dense_material(
                    node_for_lecture,
                    memory,
                    memory.rag_profile_compressed,
                    anchor,
                    chat_mgr,
                    user_msg,
                    rag_context,
                    curriculum_id,
                    lecture_scope,
                    focus_text,
                    stream_callback,
                    topic_flag,
                    coverage_payload,
                    verified_block,
                    allowed_urls,
                    external_search_delta=True,
                    node_content=content,
                    rag_citation_registry=rag_citation_registry,
                )

            delta_dense = await run_blocking(pool_llm_lecture(), _delta_dense_job)
            if not is_search_tool_only_response(delta_dense.lecture_body or ""):
                if is_search_tool_only_response(dense.lecture_body or ""):
                    dense = delta_dense
                else:
                    dense = merge_dense_material_delta(dense, delta_dense)

    content = _apply_dense_material(content, dense)

    def _persist_memory() -> None:
        persist_session_memory(curriculum_id, node.node_id, memory)

    merge_lecture_coverage_from_dense(
        memory,
        dense,
        focus_text=focus_text,
        lecture_scope=lecture_scope,
        persist=_persist_memory,
    )
    from knowledge_engine.src.node_deep_dive.term_registry import merge_introduced_terms

    merge_introduced_terms(
        memory,
        list(dense.introduced_terms or []),
        persist=_persist_memory,
    )
    memory.learning_phase = "dense_material"
    memory.pathway_bridge = (dense.bridge_to_next or "").strip()
    memory.chat_sessions = chat_mgr.to_memory_blob()

    tutor = _compose_dense_chat_message(dense)
    if _tutor_message_is_lecture_stub(tutor):
        tutor = _ensure_lecture_in_tutor_message(
            tutor,
            content.summary,
            (dense.lecture_body or "").strip(),
        )
    llm_out = deep_dive_llm_output_from_chat_text(tutor)
    return {
        **state,
        "memory": memory,
        "content": content,
        "tutor_message": tutor,
        "llm_out": llm_out,
    }


async def iter_node_deep_dive_chat_stream(
    req: NodeDeepDiveRequest,
):
    """SSE: token events + complete/error (только action=chat, dialogue path со stream)."""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def on_token(text: str) -> None:
        loop.call_soon_threadsafe(
            q.put_nowait,
            {"type": "token", "text": text},
        )

    async def worker() -> None:
        try:
            resp = await run_node_deep_dive(req, stream_callback=on_token)
            enriched = enrich_node_deep_dive_response(
                resp,
                list(resp.source_registry),
            )
            await q.put({"type": "complete", "result": enriched.model_dump()})
        except Exception as exc:
            from knowledge_engine.ui.errors import trace_exception

            detail = trace_exception(exc, "NODE_DIVE")
            await q.put(
                {
                    "type": "error",
                    "detail": detail,
                    "error_type": type(exc).__name__,
                }
            )
        finally:
            await q.put(None)

    task = asyncio.create_task(worker())
    while True:
        item = await q.get()
        if item is None:
            break
        yield item
    await task
