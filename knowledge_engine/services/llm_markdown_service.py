"""Преобразование markdown от LLM в HTML для web-клиентов."""

from __future__ import annotations

from typing import Any

from knowledge_engine.src.node_deep_dive.schemas import (
    NodeContentBlock,
    NodeDeepDiveResponse,
)
from knowledge_engine.src.node_deep_dive.session_store import repair_history_with_memory
from knowledge_engine.web.linkify import markdown_document_html
from knowledge_engine.web.llm_text_repair import (
    repair_diagram_markdown,
    repair_llm_display_text,
)


def llm_markdown_to_html(
    text: str,
    source_registry: list[dict[str, Any]] | None = None,
) -> str:
    """Единая точка: Reasoner / тьютор / explainer markdown → HTML."""
    return markdown_document_html(
        repair_llm_display_text(text),
        source_registry,
    )


def present_history_for_client(
    history: list[dict[str, str]] | None,
    source_registry: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """История диалога с content_html для реплик тьютора (LLM)."""
    out: list[dict[str, str]] = []
    for item in history or []:
        role = str(item.get("role") or "tutor").strip()
        if role not in ("user", "tutor"):
            role = "tutor"
        content = repair_llm_display_text(str(item.get("content") or ""))
        row: dict[str, str] = {"role": role, "content": content}
        if role == "tutor" and content.strip():
            row["content_html"] = llm_markdown_to_html(content, source_registry)
        out.append(row)
    return out


def present_node_content_for_client(
    content: NodeContentBlock,
    source_registry: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    data = content.model_dump()
    summary = repair_llm_display_text((data.get("summary") or "").strip())
    if summary:
        data["summary"] = summary
        data["summary_html"] = llm_markdown_to_html(summary, source_registry)
    diagram = repair_diagram_markdown(data.get("diagram") or "")
    if diagram:
        data["diagram"] = diagram
    return data


def enrich_node_deep_dive_response(
    resp: NodeDeepDiveResponse,
    source_registry: list[dict[str, Any]] | None = None,
) -> NodeDeepDiveResponse:
    """Добавляет HTML-поля к ответу interact (не меняет сырой markdown в store)."""
    from knowledge_engine.src.node_deep_dive.tutor_dialogue import (
        compose_tutor_dialogue_message,
    )

    registry = list(source_registry or resp.source_registry or [])
    composed = compose_tutor_dialogue_message(
        feedback_on_answer=resp.tutor_dialogue_feedback,
        technical_explanation=resp.tutor_dialogue_technical,
        follow_up_question=resp.tutor_dialogue_follow_up,
    )
    tutor_raw = repair_llm_display_text(composed.strip() or (resp.tutor_message or ""))
    content_data = present_node_content_for_client(resp.content, registry)
    content = NodeContentBlock.model_validate(content_data)
    tutor_html = ""
    if tutor_raw.strip():
        tutor_html = llm_markdown_to_html(tutor_raw, registry)
    history = present_history_for_client(resp.history, registry)
    return NodeDeepDiveResponse(
        node_id=resp.node_id,
        node_status=resp.node_status,
        content=content,
        tutor_message=tutor_raw,
        tutor_message_html=tutor_html,
        tutor_dialogue_feedback=resp.tutor_dialogue_feedback,
        tutor_dialogue_technical=resp.tutor_dialogue_technical,
        tutor_dialogue_follow_up=resp.tutor_dialogue_follow_up,
        quick_replies=list(resp.quick_replies or [])[:4],
        ready_for_transition=bool(resp.ready_for_transition),
        last_eval_directive=(resp.last_eval_directive or "").strip()[:64],
        history=history,
        new_gap_to_record=resp.new_gap_to_record,
        session_key=resp.session_key,
        rag_facts_count=resp.rag_facts_count,
        rag_fact_labels=resp.rag_fact_labels,
        topic_mastery_score=resp.topic_mastery_score,
        concepts_matrix=list(resp.concepts_matrix),
        mastery_dashboard=resp.mastery_dashboard,
        coverage_summary=resp.coverage_summary,
        learning_phase=resp.learning_phase,
        learning_mode=resp.learning_mode,
        source_registry=list(resp.source_registry),
        mapped_source_ids=list(resp.mapped_source_ids or []),
        lecture_rag_inspector=list(resp.lecture_rag_inspector or [])[:16],
    )


def enrich_session_blob_for_client(
    blob: dict[str, Any],
    *,
    node_id: str = "",
    curriculum_id: str = "",
) -> dict[str, Any]:
    """Сессия ноды из store → для workspace API (summary_html, content_html в history)."""
    from knowledge_engine.services.node_source_registry import (
        filter_source_registry,
    )
    from knowledge_engine.src.node_deep_dive.learning_loop import (
        build_mastery_dashboard,
    )
    from knowledge_engine.src.node_deep_dive.tiered_memory import (
        memory_from_blob,
        sync_topic_mastery_score,
    )
    from knowledge_engine.src.node_deep_dive.tutor_source_citations import (
        scrub_content_references,
    )

    out = dict(blob)
    memory = memory_from_blob(out.get("memory"))
    merged_hist = repair_history_with_memory(out.get("history"), memory)
    from knowledge_engine.src.node_deep_dive.dialog_ids import (
        patch_last_tutor_history_content,
    )
    from knowledge_engine.src.node_deep_dive.tutor_dialogue import (
        recover_tutor_display_from_chat_sessions,
    )
    from knowledge_engine.src.node_deep_dive.tutor_field_limits import (
        SCHEMA_FOLLOW_UP_QUESTION_MAX,
        SCHEMA_TUTOR_MESSAGE_MAX,
    )

    full_display = ""
    recovered = ""
    if memory is not None:
        full_display = (memory.last_tutor_display_message or "").strip()
        if not full_display:
            recovered, fu_rec = recover_tutor_display_from_chat_sessions(memory)
            if recovered:
                full_display = recovered
                memory.last_tutor_display_message = recovered[:SCHEMA_TUTOR_MESSAGE_MAX]
                if fu_rec:
                    memory.last_tutor_follow_up_question = fu_rec[
                        :SCHEMA_FOLLOW_UP_QUESTION_MAX
                    ]
        if not full_display and (memory.last_tutor_follow_up_question or "").strip():
            fu = memory.last_tutor_follow_up_question.strip()
            for i in range(len(merged_hist) - 1, -1, -1):
                if merged_hist[i].get("role") != "tutor":
                    continue
                body = (merged_hist[i].get("content") or "").strip()
                if fu not in body:
                    merged_hist[i]["content"] = f"{body}\n\n{fu}".strip()
                merged_hist[i].pop("content_html", None)
                break
    if full_display:
        merged_hist = patch_last_tutor_history_content(merged_hist, full_display)
    if memory is not None and full_display:
        from knowledge_engine.src.node_deep_dive.tiered_memory import memory_to_blob

        out["memory"] = memory_to_blob(memory)
        nid = (node_id or "").strip()
        cid = (curriculum_id or "").strip()
        if nid and cid and recovered:
            from knowledge_engine.src.node_deep_dive.session_store import (
                get_session,
                save_session,
            )

            sess = get_session(cid, nid)
            save_session(
                cid,
                nid,
                str(out.get("node_status") or sess.node_status or "in_progress"),
                sess.content,
                merged_hist,
                rag_fact_labels=list(out.get("rag_fact_labels") or []),
                memory=memory,
                source_registry=list(out.get("source_registry") or []),
            )
    from knowledge_engine.src.node_deep_dive.session_store import (
        _repair_tutor_history_markdown,
    )

    _repair_tutor_history_markdown(merged_hist)
    out["history"] = merged_hist
    content_raw = out.get("content") or {}
    registry: list[dict[str, Any]] = filter_source_registry(
        list(out.get("source_registry") or [])
    )
    if isinstance(content_raw, dict):
        block = NodeContentBlock.model_validate(content_raw)
        block = scrub_content_references(block, registry)
        out["content"] = present_node_content_for_client(block, registry)
    out["source_registry"] = registry
    out["history"] = present_history_for_client(out.get("history"), registry)
    status = str(out.get("node_status") or out.get("status") or "unexplored")
    if memory:
        score = sync_topic_mastery_score(memory)
        dash = build_mastery_dashboard(memory, status)
        out["mastery_dashboard"] = dash.model_dump()
        out["coverage_summary"] = (
            dash.coverage_summary.model_dump() if dash.coverage_summary else None
        )
        out["topic_mastery_score"] = score
        out["learning_phase"] = memory.learning_phase
        out["learning_mode"] = memory.learning_mode
        out["memory_prepared"] = True
        out["lecture_rag_inspector"] = list(memory.lecture_rag_inspector or [])[:16]
        from knowledge_engine.src.node_deep_dive.concept_map import (
            sub_concept_coverage_complete,
        )

        out["last_eval_directive"] = (memory.last_eval_directive or "").strip()[:64]
        out["ready_for_transition"] = bool(
            sub_concept_coverage_complete(memory)
            or (memory.learning_phase or "") == "pathway_decision"
        )
    return out
