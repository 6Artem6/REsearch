"""Пояснение выделенного фрагмента в материале ноды (как v08 explain + registry)."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.services.llm_markdown_service import llm_markdown_to_html
from knowledge_engine.services.node_source_registry import registry_for_prompt
from knowledge_engine.src.node_deep_dive.interaction_prompt_layout import (
    BLOCK_DYNAMIC_HEADER,
    BLOCK_RAG_TAG,
    BLOCK_SEMI_STATIC_HEADER,
    BLOCK_STATIC_PRESET_HEADER,
    BLOCK_USER_QUERY_TAG,
    LAYOUT_AND_TYPOGRAPHY_RULES,
    PINNED_REGISTRY_TAG,
    PROMPT_CITATION_ID_RULES,
)
from knowledge_engine.src.processors.explainer import (
    DEFAULT_EXPLAIN_QUESTION,
    ExplainSourceRef,
)
from knowledge_engine.src.processors.source_anchors import strip_source_anchor_tags

_SOURCE_TAG_RE = re.compile(r"\[S(\d+)\]", re.I)
_R_TAG_RE = re.compile(r"\[(R\d+)\]", re.I)

_NODE_EXPLAIN_SYSTEM = f"""{BLOCK_STATIC_PRESET_HEADER}
You are a research analyst / tutor Explainer. The student highlighted a stuck fragment
in node learning material and wants a precise explanation.

{RUSSIAN_OUTPUT_RULE}

TASK: answer the student's question using the context triad when present:
1) target_anchor / EXACT LECTURE SOURCE CHUNKS / highlighted_text (where the stuck point is)
2) fundamental_invariants — PRINCIPLE / MECHANIC knowledge atoms (why it works this way)
3) causal_facts — cause→effect links and boundaries from the fact graph / profile
Use SOURCE REGISTRY [S*] only to fill gaps missing from that triad. Do not paraphrase the highlight.

{PROMPT_CITATION_ID_RULES}

{LAYOUT_AND_TYPOGRAPHY_RULES}

RULES:
1. Do NOT paraphrase or restate the highlighted text.
2. Prefer target_anchor / [R*] chunks first; then invariants + causal_facts; then [S*] registry snippets.
3. Cover the student's question fully, but stay dense — no filler, no generic CS lecture, no padding.
4. If payload has [SHARED_SESSION_CONTEXT] / fact_manifest — honor agreed_concepts and current_subtopic.
5. List real [R*] and/or [S*] used in cited_source_ids (do not invent new id schemes for atoms/facts).
6. Structure (Russian section headers in output):
   - **Почему так**: causal chain + invariant(s) that answer the stuck point.
   - **Как стыкуется с фрагментом**: how that mechanism maps onto the highlighted lecture fragment ([R*]/[S*] when available).
   - **В двух словах**: engineer takeaway in ≤3 sentences.
"""
"""
RU (пояснение): Explainer — triad Anchor/Invariants/Causal; RU headers; без воды.
"""


class NodeExplainResult(BaseModel):
    explanation: str = ""
    source_ref: ExplainSourceRef = Field(default_factory=ExplainSourceRef)


def _extract_source_ids(text: str) -> list[str]:
    ids: list[str] = []
    for m in _SOURCE_TAG_RE.finditer(text or ""):
        sid = f"S{m.group(1)}"
        if sid not in ids:
            ids.append(sid)
    return ids


def _extract_rag_chunk_ids(text: str) -> list[str]:
    ids: list[str] = []
    for m in _R_TAG_RE.finditer(text or ""):
        rid = m.group(1).strip().upper()
        if not rid.startswith("R"):
            rid = f"R{rid}"
        if rid not in ids:
            ids.append(rid)
    return ids


def _source_ref_from_rag_inspector_row(row: dict[str, Any]) -> ExplainSourceRef:
    rid = str(row.get("rag_id") or "").strip()
    return ExplainSourceRef(
        title=str(row.get("title") or rid or "RAG chunk"),
        url=str(row.get("url") or ""),
        source_id=rid,
    )


def _resolve_explain_source_ref(
    cited_source_ids: list[str],
    resolved_r_chunks: list[dict[str, Any]],
    registry_fallback: ExplainSourceRef,
) -> ExplainSourceRef:
    by_rid: dict[str, dict[str, Any]] = {}
    for row in resolved_r_chunks:
        rid = str(row.get("rag_id") or "").strip().upper()
        if rid:
            by_rid[rid] = row
    for raw in cited_source_ids:
        cid = str(raw or "").strip().upper()
        if not cid:
            continue
        if cid.startswith("R") and cid in by_rid:
            return _source_ref_from_rag_inspector_row(by_rid[cid])
    for raw in cited_source_ids:
        cid = str(raw or "").strip().upper()
        if cid.startswith("R") and resolved_r_chunks:
            for row in resolved_r_chunks:
                if str(row.get("rag_id") or "").strip().upper() == cid:
                    return _source_ref_from_rag_inspector_row(row)
    if resolved_r_chunks:
        return _source_ref_from_rag_inspector_row(resolved_r_chunks[0])
    return registry_fallback


def _registry_entry(
    registry: list[dict[str, Any]], source_id: str
) -> dict[str, Any] | None:
    want = (source_id or "").strip().upper()
    for entry in registry:
        if not isinstance(entry, dict):
            continue
        sid = str(entry.get("id") or entry.get("source_id") or "").strip().upper()
        if sid == want:
            return entry
    return None


def _material_from_registry(
    registry: list[dict[str, Any]],
    hint_ids: list[str],
) -> tuple[str, ExplainSourceRef]:
    entry: dict[str, Any] | None = None
    for sid in hint_ids:
        entry = _registry_entry(registry, sid)
        if entry:
            break
    if not entry and registry:
        entry = registry[0]
    if not entry:
        return "(нет источников в registry)", ExplainSourceRef()

    sid = str(entry.get("id") or "")
    title = str(entry.get("title") or "Source")
    url = str(entry.get("url") or "")
    snippet = (entry.get("snippet") or "")[:4000]
    material = f"--- SOURCE [{sid}] ---\n" f"Title: {title}\nURL: {url}\n\n{snippet}"
    return material, ExplainSourceRef(title=title, url=url, source_id=sid)


def _build_node_explain_payload(
    node_title: str,
    selected_text: str,
    user_question: str,
    surrounding_paragraph: str,
    summary_excerpt: str,
    rag_profile: str,
    source_registry: list[dict[str, Any]],
    *,
    memory: Any | None = None,
    curriculum_id: str = "",
    node: Any | None = None,
) -> tuple[str, ExplainSourceRef, list[dict[str, Any]]]:
    selected = strip_source_anchor_tags((selected_text or "").strip())
    if len(selected) < 2:
        raise ValueError("selected_text is too short")

    question = (user_question or "").strip() or DEFAULT_EXPLAIN_QUESTION
    raw_surrounding = (surrounding_paragraph or "").strip()
    rag_hint_ids = _extract_rag_chunk_ids(selected_text) + _extract_rag_chunk_ids(
        raw_surrounding
    )
    s_hint_ids = _extract_source_ids(selected_text) + _extract_source_ids(
        raw_surrounding
    )

    resolved_r_chunks: list[dict[str, Any]] = []
    if memory is not None and rag_hint_ids:
        from knowledge_engine.services.lecture_rag_context import (
            lookup_lecture_rag_inspector_chunks,
        )

        inspector = list(getattr(memory, "lecture_rag_inspector", None) or [])
        resolved_r_chunks = lookup_lecture_rag_inspector_chunks(
            inspector,
            rag_hint_ids,
        )

    from knowledge_engine.services.explain_context_bundle import (
        build_explain_context_bundle,
    )

    bundle = build_explain_context_bundle(
        selected_text=selected,
        user_question=question,
        surrounding_paragraph=raw_surrounding,
        resolved_r_chunks=resolved_r_chunks,
        rag_profile=rag_profile,
        curriculum_id=curriculum_id,
        node=node,
        node_title=node_title,
    )

    material, source_ref = _material_from_registry(source_registry, s_hint_ids)
    if resolved_r_chunks:
        source_ref = _source_ref_from_rag_inspector_row(resolved_r_chunks[0])

    registry_block = registry_for_prompt(source_registry)
    summary = strip_source_anchor_tags((summary_excerpt or "").strip()[:6000])
    rag = strip_source_anchor_tags((rag_profile or "").strip()[:2000])

    block2_parts: list[str] = [
        BLOCK_SEMI_STATIC_HEADER,
        f"### node_title\n{node_title}",
        f"{PINNED_REGISTRY_TAG}\n### SOURCE REGISTRY\n{registry_block}",
        f"### node_summary_excerpt\n{summary or '(пусто)'}",
    ]
    block2 = "\n\n".join(block2_parts)

    block3_parts: list[str] = [BLOCK_DYNAMIC_HEADER]
    if memory is not None:
        from knowledge_engine.src.node_deep_dive.dialog_context import (
            build_shared_session_context_block,
        )

        session_block = build_shared_session_context_block(
            memory,
            include_sliding_window=False,
        )
        if session_block:
            block3_parts.append(session_block)
    if rag:
        block3_parts.append(f"### rag_profile\n{rag}")

    rag_block_parts: list[str] = []
    if bundle.anchor_block:
        rag_block_parts.append(bundle.anchor_block)
    elif resolved_r_chunks:
        from knowledge_engine.services.lecture_rag_context import (
            format_highlight_rag_chunks_block,
        )

        rag_block_parts.append(format_highlight_rag_chunks_block(resolved_r_chunks))
    if bundle.invariants_block:
        rag_block_parts.append(bundle.invariants_block)
    if bundle.causal_block:
        rag_block_parts.append(bundle.causal_block)
    if material and material != "(нет источников в registry)":
        rag_block_parts.append(f"### whitelist_source_snippet\n{material}")
    if rag_block_parts:
        block3_parts.append(f"{BLOCK_RAG_TAG}\n" + "\n\n".join(rag_block_parts))
    block3_parts.append(
        f"### highlighted_text\n{selected}\n\n"
        f"### context_paragraph\n{raw_surrounding[:2000]}"
    )
    block3_parts.append(f"{BLOCK_USER_QUERY_TAG}\n### user_question\n{question}")
    block3 = "\n\n".join(block3_parts)

    user_payload = f"{block2}\n\n{block3}"
    return user_payload, source_ref, resolved_r_chunks


def _invoke_node_explain_gemini(
    user_payload: str,
    anchor: str,
    stream_callback: Callable[[str], None] | None = None,
):
    from knowledge_engine.schemas.llm_contracts.tutor import NodeExplainContract
    from knowledge_engine.src.analytics.gemini_v07 import run_gemini_lite_structured

    return run_gemini_lite_structured(
        _NODE_EXPLAIN_SYSTEM,
        user_payload,
        anchor,
        NodeExplainContract,
        "node_selection_explain",
        stream_callback=stream_callback,
    )


def run_node_selection_explain(
    node_title: str,
    selected_text: str,
    user_question: str,
    surrounding_paragraph: str,
    summary_excerpt: str,
    rag_profile: str,
    source_registry: list[dict[str, Any]],
    anchor: str,
    *,
    memory: Any | None = None,
    curriculum_id: str = "",
    node: Any | None = None,
) -> NodeExplainResult:
    user_payload, registry_ref, resolved_r = _build_node_explain_payload(
        node_title,
        selected_text,
        user_question,
        surrounding_paragraph,
        summary_excerpt,
        rag_profile,
        source_registry,
        memory=memory,
        curriculum_id=curriculum_id,
        node=node,
    )
    contract = _invoke_node_explain_gemini(user_payload, anchor, None)
    explanation = (contract.explanation or "").strip()
    source_ref = _resolve_explain_source_ref(
        contract.cited_source_ids,
        resolved_r,
        registry_ref,
    )
    return NodeExplainResult(explanation=explanation, source_ref=source_ref)


def explain_result_to_api_dict(
    result: NodeExplainResult,
    source_registry: list[dict[str, Any]],
) -> dict[str, Any]:
    explanation_html = llm_markdown_to_html(result.explanation, source_registry)
    return {
        "explanation": result.explanation,
        "explanation_html": explanation_html,
        "source_ref": {
            "title": result.source_ref.title,
            "url": result.source_ref.url,
            "source_id": result.source_ref.source_id,
        },
        "default_question": DEFAULT_EXPLAIN_QUESTION,
    }


async def iter_node_selection_explain_stream(
    node_title: str,
    selected_text: str,
    user_question: str,
    surrounding_paragraph: str,
    summary_excerpt: str,
    rag_profile: str,
    source_registry: list[dict[str, Any]],
    anchor: str,
    *,
    memory: Any | None = None,
    curriculum_id: str = "",
    node: Any | None = None,
):
    """SSE: token + complete/error для пояснения выделения."""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def on_token(text: str) -> None:
        loop.call_soon_threadsafe(
            q.put_nowait,
            {"type": "token", "text": text},
        )

    async def worker() -> None:
        try:
            user_payload, registry_ref, resolved_r = _build_node_explain_payload(
                node_title,
                selected_text,
                user_question,
                surrounding_paragraph,
                summary_excerpt,
                rag_profile,
                source_registry,
                memory=memory,
                curriculum_id=curriculum_id,
                node=node,
            )
            contract = _invoke_node_explain_gemini(
                user_payload,
                anchor,
                on_token,
            )
            explanation = (contract.explanation or "").strip()
            source_ref = _resolve_explain_source_ref(
                contract.cited_source_ids,
                resolved_r,
                registry_ref,
            )
            result = NodeExplainResult(explanation=explanation, source_ref=source_ref)
            payload = explain_result_to_api_dict(result, source_registry)
            await q.put({"type": "complete", "result": payload})
        except Exception as exc:
            await q.put({"type": "error", "detail": str(exc)})
        finally:
            await q.put(None)

    task = asyncio.create_task(worker())
    while True:
        item = await q.get()
        if item is None:
            break
        yield item
    await task
