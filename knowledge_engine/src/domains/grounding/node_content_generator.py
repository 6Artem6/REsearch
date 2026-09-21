"""Плотный материал ноды (Lite / tutor chain, не Reasoner)."""

from __future__ import annotations

from collections.abc import Callable

from knowledge_engine.src.adapters.llm_providers.gemini_stateless import (
    gemini_tutor_model_chain,
    run_gemini_structured_with_chain,
)
from knowledge_engine.src.adapters.llm_providers.local_llm_stateless import (
    run_local_structured,
)
from knowledge_engine.src.config.settings import (
    GEMINI_RPM_PAUSE_SEC,
    GEMINI_TUTOR_MODEL,
    LECTURE_GENERATION_TEMPERATURE,
    LECTURE_GENERATION_TIMEOUT_SEC,
    LECTURE_MAX_OUTPUT_TOKENS,
    MAIN_MODEL,
)
from knowledge_engine.src.domains.grounding.chat_session_manager import (
    ChatSessionManager,
)
from knowledge_engine.src.domains.grounding.code_snippet_heuristic import (
    filter_code_snippets,
)
from knowledge_engine.src.domains.grounding.lecture_body_format import (
    format_self_check_block,
    sanitize_lecture_body_markdown,
    strip_lecture_credit_scoreboard,
    strip_trailing_checkpoint_from_lecture_body,
)
from knowledge_engine.src.domains.grounding.memory_schemas import (
    SessionMemory,
    build_sub_concept_status_lookup,
)
from knowledge_engine.src.domains.grounding.prompt_types import InteractionPromptMode
from knowledge_engine.src.domains.grounding.schemas import (
    DenseMaterialOutput,
    NodeContentBlock,
    NodeDataInput,
)
from knowledge_engine.src.domains.grounding.tiered_memory import (
    build_handoff_summary,
    format_matrix_for_llm,
)
from knowledge_engine.src.domains.grounding.tutor import (
    StructuredLectureResponse,
    StructuredLectureResponseWithPlanContract,
    TopicQnaLectureResponse,
    structured_lecture_to_dense,
)
from knowledge_engine.src.domains.grounding.tutor_prompt_builder import (
    build_critical_rules_recency_tail,
    build_dense_system,
)
from knowledge_engine.src.shared.retrieval.lecture_rag_context import (
    build_lecture_generation_payload,
)


def _dense_system_instruction(
    lecture_scope: str,
    topic_already_covered: bool = False,
    *,
    memory: SessionMemory | None = None,
    interaction_axis: str = "lecture_self_check",
) -> str:
    return build_dense_system(
        targeted=lecture_scope == "targeted_lecture",
        topic_already_covered=topic_already_covered,
        memory=memory,
        interaction_axis=interaction_axis,
    )


def _sanitize_dense_output(
    dense: DenseMaterialOutput,
    allowed_urls: set[str],
) -> DenseMaterialOutput:
    """Оставить в references только URL из allow-list; починить ```python``` в lecture_body."""
    _ = allowed_urls
    body = sanitize_lecture_body_markdown(dense.lecture_body or "")
    body = strip_lecture_credit_scoreboard(body)
    checkpoint = (dense.follow_up_question or "").strip()
    body = strip_trailing_checkpoint_from_lecture_body(body, checkpoint)
    snippets = [
        sanitize_lecture_body_markdown(s) if "```" in (s or "") else (s or "")
        for s in (dense.code_snippets or [])
    ]
    return dense.model_copy(
        update={
            "lecture_body": body,
            "code_snippets": [s for s in snippets if (s or "").strip()],
        }
    )


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
    stream_callback: Callable[[str], None] | None = None,
    topic_already_covered: bool = False,
    coverage_payload: str = "",
    verified_sources_block: str = "",
    allowed_urls: set[str] | None = None,
    external_search_delta: bool = False,
    node_content: NodeContentBlock | None = None,
    rag_citation_registry: str = "",
    interaction_axis: str = "lecture_self_check",
) -> DenseMaterialOutput:
    from knowledge_engine.src.core.run_log import trace

    matrix = format_matrix_for_llm(memory.concepts_matrix)
    scope = (lecture_scope or "full_node_lecture").strip()
    focus = (focus_text or "").strip()
    rag_query = focus if scope == "targeted_lecture" and focus else user_query
    recency_rules = build_critical_rules_recency_tail(
        mode=InteractionPromptMode.LECTURE_DENSE,
        memory=memory,
        topic_already_covered=topic_already_covered,
    )
    payload = build_lecture_generation_payload(
        node,
        rag_profile,
        rag_query,
        rag_context,
        matrix,
        memory.rolling_dialogue_summary or "",
        curriculum_id,
        verified_sources_block=verified_sources_block,
        external_search_delta=external_search_delta,
        node_content=node_content,
        memory=memory,
        rag_citation_registry=rag_citation_registry,
        coverage_payload=coverage_payload,
        lecture_scope=scope,
        focus_text=focus,
        recency_rules=recency_rules,
    )
    from knowledge_engine.src.domains.grounding.subconcept_invariants import (
        format_subconcept_hard_anchor,
    )

    anchor_block = format_subconcept_hard_anchor(memory)
    if anchor_block:
        payload = f"{anchor_block}\n\n{payload}"
    mgr = chat_manager or ChatSessionManager.from_memory_blob(
        anchor, memory.chat_sessions
    )
    handoff = build_handoff_summary(memory)

    is_topic_qna = (interaction_axis or "").strip().lower() == "topic_qna"
    # RU: follow_up_question существует ТОЛЬКО в StructuredLectureResponse —
    # для topic_qna используется TopicQnaLectureResponse, у которой этого
    # поля вообще нет в JSON Schema, так что self-check вопрос физически
    # невозможно записать (см. _DenseLectureFieldsBase в tutor.py). Раньше
    # это запрещалось только текстом инструкции (system prompt + Field
    # description) — Gemini всё равно его игнорировал (live-тест на sql_cte):
    # известный на момент запроса инвариант должен быть
    # закодирован в схеме, а не в условной инструкции.
    if is_topic_qna:
        response_schema = TopicQnaLectureResponse
    else:
        from knowledge_engine.src.config.settings import ENABLE_TUTOR_EXECUTION_PLAN

        response_schema = (
            StructuredLectureResponseWithPlanContract
            if ENABLE_TUTOR_EXECUTION_PLAN
            else StructuredLectureResponse
        )

    system = _dense_system_instruction(
        scope,
        topic_already_covered=topic_already_covered,
        memory=memory,
        interaction_axis=interaction_axis,
    )
    if is_topic_qna:
        # RU (Interaction Axis — см. docs/STEERING_AND_TOPIC_QNA_ROADMAP.md):
        # оставшиеся поведенческие правила (не задавать встречных вопросов,
        # отвечать по RAG, короткий формат) не кодируются в схеме — только
        # они и остаются текстом. select_interaction_axis_system_prompt
        # возвращает None для "lecture_self_check" — ветка не выполняется.
        from knowledge_engine.src.domains.grounding.prompt_factory import (
            select_interaction_axis_system_prompt,
        )

        axis_prompt = select_interaction_axis_system_prompt(interaction_axis)
        if axis_prompt:
            system = f"{system}\n\n{axis_prompt.strip()}"
    trace(
        f"NODE_DIVE dense_material ▶ Gemini | scope={scope} "
        f"focus_len={len(focus)} RAG_CONTEXT len={len(rag_context or '')} "
        f"verified_block={len(verified_sources_block or '')} "
        f"delta={external_search_delta} | payload_len={len(payload)} "
        f"system_len={len(system)} max_output_tokens={LECTURE_MAX_OUTPUT_TOKENS}"
    )
    try:
        structured = run_gemini_structured_with_chain(
            GEMINI_TUTOR_MODEL,
            system,
            payload,
            anchor,
            response_schema,
            "node_deep_dive / dense_material",
            rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
            chat_manager=mgr,
            chat_label="node_deep_dive/dense_material",
            handoff_summary=handoff,
            stream_callback=stream_callback,
            stream_text_field="lecture_body",
            models=gemini_tutor_model_chain(),
            http_timeout_sec=LECTURE_GENERATION_TIMEOUT_SEC,
            max_output_tokens=LECTURE_MAX_OUTPUT_TOKENS,
            temperature=LECTURE_GENERATION_TEMPERATURE,
            interaction_axis=interaction_axis,
            sub_concept_id=memory.asked_question_sub_concept_id,
            sub_concept_status_lookup=build_sub_concept_status_lookup(
                memory.sub_concepts
            ),
        )
    except Exception as exc:
        trace(
            f"NODE_DIVE dense_material fallback Cloud LLM json_schema | "
            f"{response_schema.__name__} | {exc}"
        )
        structured = run_local_structured(
            MAIN_MODEL,
            response_schema,
            system,
            payload,
            anchor,
            "node_deep_dive/dense_material",
            temperature=LECTURE_GENERATION_TEMPERATURE,
        )
        try:
            from knowledge_engine.src.shared.session_prompt_trace import (
                log_local_llm_exchange,
            )

            log_local_llm_exchange(
                trace_label="node_deep_dive/dense_material",
                model_name=MAIN_MODEL,
                system_instruction=system,
                user_payload=payload,
                output_text=structured.model_dump_json(indent=2),
            )
        except Exception:
            pass
    memory.chat_sessions = mgr.to_memory_blob()
    result = structured_lecture_to_dense(structured, allowed_urls=allowed_urls or set())
    result = _sanitize_dense_output(result, allowed_urls or set())
    # Stream may have finished on raw lecture_body before host strips PART 2.
    if stream_callback is not None:
        checkpoint = (result.follow_up_question or "").strip()
        marker_block = format_self_check_block(checkpoint)
        if marker_block:
            raw_body = sanitize_lecture_body_markdown(
                getattr(structured, "lecture_body", None) or ""
            )
            raw_body = strip_lecture_credit_scoreboard(raw_body)
            if "**Самопроверка:**" not in raw_body and marker_block not in raw_body:
                stream_callback("\n\n" + marker_block)
    trace("NODE_DIVE dense_material ✓ | лекция в чат + панель (summary/diagram/refs)")
    return result


def merge_dense_material_delta(
    base: DenseMaterialOutput,
    delta: DenseMaterialOutput,
) -> DenseMaterialOutput:
    """Склеить дельту после Stage-2 search; references/code_snippets из delta при наличии."""
    body_parts = [
        (base.lecture_body or "").strip(),
        (delta.lecture_body or "").strip(),
    ]
    body = "\n\n".join(p for p in body_parts if p)
    refs = list(base.references or [])
    for r in delta.references or []:
        if r.url or r.title:
            refs.append(r)
    snippets = filter_code_snippets(base.code_snippets or [])
    for s in delta.code_snippets or []:
        if (s or "").strip():
            snippets.append(s.strip())
    snippets = filter_code_snippets(snippets)
    ref_id = (delta.referenced_diagram_id or "").strip() or (
        (base.referenced_diagram_id or "").strip() or None
    )
    summary = (delta.summary or "").strip() or (base.summary or "").strip()
    bridge = (delta.bridge_to_next or "").strip() or (base.bridge_to_next or "").strip()
    checkpoint = (delta.follow_up_question or "").strip() or (
        base.follow_up_question or ""
    ).strip()
    concepts = list(base.extracted_concepts or [])
    for c in delta.extracted_concepts or []:
        if c.key and c.summary:
            concepts.append(c)
    terms = list(base.introduced_terms or [])
    for t in delta.introduced_terms or []:
        if (t or "").strip():
            terms.append(str(t).strip())
    return base.model_copy(
        update={
            "lecture_body": body,
            "summary": summary,
            "referenced_diagram_id": ref_id,
            "references": refs[:6],
            "code_snippets": snippets[:4],
            "bridge_to_next": bridge,
            "follow_up_question": checkpoint,
            "extracted_concepts": concepts[:5],
            "introduced_terms": terms[:24],
        }
    )
