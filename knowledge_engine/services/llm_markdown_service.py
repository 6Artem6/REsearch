"""Преобразование markdown от LLM в HTML для web-клиентов."""

from __future__ import annotations

from typing import Any

from knowledge_engine.src.node_deep_dive.schemas import NodeContentBlock, NodeDeepDiveResponse
from knowledge_engine.src.node_deep_dive.session_store import repair_history_with_memory
from knowledge_engine.web.linkify import markdown_document_html
from knowledge_engine.web.llm_text_repair import (
    repair_diagram_markdown,
    repair_llm_literal_escapes,
    repair_structured_analysis_json,
)


def llm_markdown_to_html(
    text: str,
    source_registry: list[dict[str, Any]] | None = None,
) -> str:
    """Единая точка: Reasoner / тьютор / explainer markdown → HTML."""
    repaired = repair_structured_analysis_json(
        repair_llm_literal_escapes(text),
    )
    return markdown_document_html(
        repaired,
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
        content = repair_llm_literal_escapes(str(item.get("content") or ""))
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
    summary = repair_llm_literal_escapes((data.get("summary") or "").strip())
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
    registry = list(source_registry or resp.source_registry or [])
    tutor_raw = repair_llm_literal_escapes(resp.tutor_message or "")
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
        history=history,
        new_gap_to_record=resp.new_gap_to_record,
        session_key=resp.session_key,
        rag_facts_count=resp.rag_facts_count,
        rag_fact_labels=resp.rag_fact_labels,
        topic_mastery_score=resp.topic_mastery_score,
        concepts_matrix=list(resp.concepts_matrix),
        mastery_dashboard=resp.mastery_dashboard,
        learning_phase=resp.learning_phase,
        learning_mode=resp.learning_mode,
        source_registry=list(resp.source_registry),
    )


def enrich_session_blob_for_client(blob: dict[str, Any]) -> dict[str, Any]:
    """Сессия ноды из store → для workspace API (summary_html, content_html в history)."""
    from knowledge_engine.services.node_source_registry import build_registry_from_references
    from knowledge_engine.src.node_deep_dive.learning_loop import build_mastery_dashboard
    from knowledge_engine.src.node_deep_dive.tiered_memory import memory_from_blob

    out = dict(blob)
    memory = memory_from_blob(out.get("memory"))
    merged_hist = repair_history_with_memory(out.get("history"), memory)
    out["history"] = merged_hist
    content_raw = out.get("content") or {}
    registry: list[dict[str, Any]] = list(out.get("source_registry") or [])
    if isinstance(content_raw, dict):
        block = NodeContentBlock.model_validate(content_raw)
        if not registry:
            registry = build_registry_from_references(block.references)
        out["content"] = present_node_content_for_client(block, registry)
    out["source_registry"] = registry
    out["history"] = present_history_for_client(out.get("history"), registry)
    status = str(out.get("status") or "in_progress")
    if memory:
        dash = build_mastery_dashboard(memory, status)
        out["mastery_dashboard"] = dash.model_dump()
        out["topic_mastery_score"] = memory.topic_mastery_score
        out["learning_phase"] = memory.learning_phase
        out["learning_mode"] = memory.learning_mode
    return out
