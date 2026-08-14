"""Contextual Explainer — Gemini Lite по выделенному фрагменту и глубокому контексту статьи."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.services.v07_run_store import v07_run_store
from knowledge_engine.src.node_deep_dive.interaction_prompt_layout import (
    BLOCK_STATIC_PRESET_HEADER,
    LAYOUT_AND_TYPOGRAPHY_RULES,
)
from knowledge_engine.src.processors.source_anchors import strip_source_anchor_tags

_SOURCE_TAG_RE = re.compile(r"\[S(\d+)\]", re.I)
_WORD_RE = re.compile(r"\w+", re.UNICODE)

EXPLAINER_SYSTEM = f"""{BLOCK_STATIC_PRESET_HEADER}
You are a research analyst. The user highlighted a phrase in a short summary and wants deeper explanation.

{RUSSIAN_OUTPUT_RULE}
User questions may be in English (to preserve terms); the answer body must still be fully in Russian.
Keep article terms (microservices, HNSW, RPC) in the original form when appropriate.

TASK:
Read RAW SOURCE TEXT from the paper. Extract mechanisms, causes, and definitions the authors rely on.

{LAYOUT_AND_TYPOGRAPHY_RULES}

RULES:
1. Do NOT paraphrase the highlighted summary text.
2. Pull details the summary omitted (protocols, constraints, experiments, formulas).
3. If the source lacks depth for \"why/how\" — state clearly in Russian that the paper names the fact without mechanism, then give a short standard CS gloss.

Structure (Russian section headers):
- **Detail from source ([S_x])**: what do the authors describe?
- **In short**: engineer-friendly takeaway.
"""
"""
RU (пояснение): Explain v08 статьи — RAW SOURCE, [S*], структура «из источника / коротко».
"""

DEFAULT_EXPLAIN_QUESTION = "Объясни, что это значит?"

_EXPAND_WINDOW = 1
_MAX_EXPANDED_CHARS = 14_000


class ExplainSourceRef(BaseModel):
    title: str = ""
    url: str = ""
    source_id: str = ""


class ExplainResult(BaseModel):
    explanation: str = ""
    source_ref: ExplainSourceRef = Field(default_factory=ExplainSourceRef)
    matched_chunk_id: str = ""
    source_chunk_preview: str = ""
    expanded_chunk_ids: list[str] = Field(default_factory=list)


def _token_set(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text or "") if len(w) > 2}


def _extract_source_ids(text: str) -> list[str]:
    ids: list[str] = []
    for m in _SOURCE_TAG_RE.finditer(text or ""):
        sid = f"S{m.group(1)}"
        if sid not in ids:
            ids.append(sid)
    return ids


def _registry_entry(
    registry: list[dict[str, Any]], source_id: str
) -> dict[str, Any] | None:
    want = (source_id or "").strip().upper()
    if not want:
        return None
    for entry in registry:
        if not isinstance(entry, dict):
            continue
        sid = (entry.get("source_id") or entry.get("id") or "").strip().upper()
        if sid == want:
            return entry
    return None


def _source_key(chunk: dict[str, Any]) -> str:
    anchor = (chunk.get("source_anchor") or "").strip().upper()
    if anchor:
        return anchor
    doc_id = str(chunk.get("doc_id") or "").strip()
    return f"DOC:{doc_id}" if doc_id else "UNKNOWN"


def _build_chunk_catalog(
    result: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Чанки из structured_chunks; fallback — documents.raw_markdown. doc_by_id для расширения."""
    chunks: list[dict[str, Any]] = []
    documents = result.get("documents") or []
    doc_by_id: dict[str, dict[str, Any]] = {}
    for d in documents:
        if isinstance(d, dict) and d.get("doc_id"):
            doc_by_id[str(d["doc_id"])] = d

    seq = 0
    for raw in result.get("structured_chunks") or []:
        if not isinstance(raw, dict):
            continue
        text = strip_source_anchor_tags(str(raw.get("text") or ""))
        if len(text) < 20:
            continue
        doc_id = str(raw.get("doc_id") or "")
        doc = doc_by_id.get(doc_id) or {}
        chunks.append(
            {
                "chunk_id": str(raw.get("chunk_id") or ""),
                "doc_id": doc_id,
                "text": text[:8000],
                "source_anchor": str(raw.get("source_anchor") or "").strip(),
                "source_url": str(doc.get("source_url") or ""),
                "title": str(doc.get("title") or "paper"),
                "p99_relevance_score": float(raw.get("p99_relevance_score") or 0),
                "seq": seq,
            }
        )
        seq += 1

    if chunks:
        return chunks, doc_by_id

    for doc in documents:
        if not isinstance(doc, dict):
            continue
        body = strip_source_anchor_tags(str(doc.get("raw_markdown") or ""))
        if len(body) < 80:
            continue
        chunks.append(
            {
                "chunk_id": f"doc:{doc.get('doc_id', '')}",
                "doc_id": str(doc.get("doc_id") or ""),
                "text": body[:6000],
                "source_anchor": "",
                "source_url": str(doc.get("source_url") or ""),
                "title": str(doc.get("title") or "paper"),
                "p99_relevance_score": 0.0,
                "seq": seq,
            }
        )
        seq += 1
    return chunks, doc_by_id


def _score_chunk(
    chunk: dict[str, Any],
    selected_text: str,
    surrounding: str,
    hint_ids: list[str],
) -> float:
    text = chunk.get("text") or ""
    score = 0.0
    sel = (selected_text or "").strip()
    sur = (surrounding or "").strip()
    anchor = (chunk.get("source_anchor") or "").strip().upper()

    if chunk.get("doc_id") == "reasoner_answer":
        score += 15.0

    if hint_ids and anchor:
        if anchor in [h.upper() for h in hint_ids]:
            score += 80.0

    if sel and sel in text:
        score += 40.0
    if sur and len(sur) > 30 and sur[:200] in text:
        score += 25.0

    sel_tokens = _token_set(sel)
    if sel_tokens:
        chunk_tokens = _token_set(text)
        overlap = len(sel_tokens & chunk_tokens)
        score += min(overlap * 2.0, 30.0)

    score += float(chunk.get("p99_relevance_score") or 0) * 2.0
    return score


def _chunks_for_source(
    chunks: list[dict[str, Any]], source_id: str
) -> list[dict[str, Any]]:
    want = (source_id or "").strip().upper()
    if not want:
        return []
    matched = [
        c for c in chunks if (c.get("source_anchor") or "").strip().upper() == want
    ]
    matched.sort(key=lambda c: c.get("seq", 0))
    return matched


def _expand_from_document(
    doc: dict[str, Any],
    selected_text: str,
    *,
    max_chars: int = 6000,
) -> str:
    body = strip_source_anchor_tags(str(doc.get("raw_markdown") or ""))
    if len(body) < 80:
        return ""
    sel = (selected_text or "").strip()
    needle = sel[:120] if len(sel) > 120 else sel
    idx = body.lower().find(needle.lower()) if needle else -1
    if idx >= 0:
        start = max(0, idx - 2800)
        end = min(len(body), idx + len(needle) + 2800)
        return body[start:end][:max_chars]
    return body[:max_chars]


def _retrieve_expanded_chunks(
    chunks: list[dict[str, Any]],
    primary: dict[str, Any],
    *,
    window: int = _EXPAND_WINDOW,
) -> list[dict[str, Any]]:
    key = _source_key(primary)
    same = [c for c in chunks if _source_key(c) == key]
    if not same:
        return [primary]
    same.sort(key=lambda c: c.get("seq", 0))
    primary_id = str(primary.get("chunk_id") or "")
    idx = 0
    for i, c in enumerate(same):
        if str(c.get("chunk_id") or "") == primary_id:
            idx = i
            break
    start = max(0, idx - window)
    end = min(len(same), idx + window + 1)
    return same[start:end]


def _format_expanded_source_material(
    expanded: list[dict[str, Any]],
    primary: dict[str, Any],
    source_label: str,
) -> str:
    primary_id = str(primary.get("chunk_id") or "")
    parts: list[str] = []
    for ch in expanded:
        cid = str(ch.get("chunk_id") or "")
        role = "PRIMARY MATCH" if cid == primary_id else "NEIGHBOR CHUNK (same paper)"
        parts.append(f"--- {role} | chunk_id={cid} ---\n{ch.get('text') or ''}")
    header = f"Expanded Source Material [{source_label}] — raw excerpts from indexed paper chunks:\n"
    return header + "\n\n".join(parts)


def _build_expanded_source_material(
    result: dict[str, Any],
    chunks: list[dict[str, Any]],
    doc_by_id: dict[str, dict[str, Any]],
    primary: dict[str, Any],
    selected_text: str,
    hint_ids: list[str],
) -> tuple[str, list[str]]:
    source_id = (primary.get("source_anchor") or "").strip()
    if not source_id and hint_ids:
        source_id = hint_ids[0]

    label = source_id or _source_key(primary)
    if source_id:
        pool = _chunks_for_source(chunks, source_id)
        if pool:
            best_in_pool = max(
                pool,
                key=lambda c: _score_chunk(c, selected_text, "", hint_ids),
            )
            primary = best_in_pool

    expanded = _retrieve_expanded_chunks(chunks, primary, window=_EXPAND_WINDOW)
    material = _format_expanded_source_material(expanded, primary, label)
    chunk_ids = [str(c.get("chunk_id") or "") for c in expanded if c.get("chunk_id")]

    total_len = sum(len(c.get("text") or "") for c in expanded)
    if total_len < 500 or len(expanded) <= 1:
        doc_id = str(primary.get("doc_id") or "")
        doc = doc_by_id.get(doc_id)
        if doc:
            extra = _expand_from_document(doc, selected_text)
            if extra:
                material += (
                    "\n\n--- Additional excerpt from full paper markdown (parent document) ---\n"
                    + extra
                )

    if len(material) > _MAX_EXPANDED_CHARS:
        material = material[:_MAX_EXPANDED_CHARS] + "\n… [truncated]"

    return material, chunk_ids


def find_matched_chunk(
    result: dict[str, Any],
    selected_text: str,
    surrounding_paragraph: str,
) -> tuple[dict[str, Any], ExplainSourceRef]:
    registry = result.get("source_registry") or []
    chunks, _doc_by_id = _build_chunk_catalog(result)
    hint_ids = _extract_source_ids(selected_text) + _extract_source_ids(
        surrounding_paragraph
    )

    pool = chunks
    if hint_ids:
        hinted = _chunks_for_source(chunks, hint_ids[0])
        if hinted:
            pool = hinted

    best: dict[str, Any] | None = None
    best_score = -1.0
    for ch in pool:
        sc = _score_chunk(ch, selected_text, surrounding_paragraph, hint_ids)
        if sc > best_score:
            best_score = sc
            best = ch

    if not best and chunks:
        best = max(
            chunks,
            key=lambda c: _score_chunk(
                c, selected_text, surrounding_paragraph, hint_ids
            ),
        )

    source_id = (best.get("source_anchor") or "").strip() if best else ""
    if not source_id and hint_ids:
        source_id = hint_ids[0]

    ref_entry = _registry_entry(registry, source_id) if source_id else None
    if not ref_entry and registry and source_id:
        for entry in registry:
            if isinstance(entry, dict):
                ref_entry = entry
                source_id = str(entry.get("source_id") or entry.get("id") or "")
                break

    title = str((ref_entry or {}).get("title") or (best or {}).get("title") or "Source")
    url = str((ref_entry or {}).get("url") or (best or {}).get("source_url") or "")
    if not source_id and ref_entry:
        source_id = str(ref_entry.get("source_id") or ref_entry.get("id") or "")

    source_ref = ExplainSourceRef(title=title, url=url, source_id=source_id)
    if not best:
        return (
            {
                "chunk_id": "",
                "text": "(no indexed chunks for this run — use surrounding context only)",
                "source_anchor": source_id,
            },
            source_ref,
        )
    return best, source_ref


def run_contextual_explain(
    run_id: str,
    selected_text: str,
    user_question: str,
    surrounding_paragraph: str,
) -> ExplainResult:
    run = v07_run_store.get(run_id)
    if not run or not run.result:
        raise ValueError("Run not found or has no result yet")

    question = (user_question or "").strip() or DEFAULT_EXPLAIN_QUESTION
    selected = (selected_text or "").strip()
    if len(selected) < 2:
        raise ValueError("selected_text is too short")

    result = run.result
    chunks, doc_by_id = _build_chunk_catalog(result)
    hint_ids = _extract_source_ids(selected) + _extract_source_ids(
        surrounding_paragraph
    )

    chunk, source_ref = find_matched_chunk(result, selected, surrounding_paragraph)
    expanded_material, expanded_ids = _build_expanded_source_material(
        result,
        chunks,
        doc_by_id,
        chunk,
        selected,
        hint_ids,
    )
    chunk_label = source_ref.source_id or chunk.get("source_anchor") or "Sx"

    from knowledge_engine.schemas.llm_contracts.tutor import NodeExplainContract
    from knowledge_engine.src.analytics.gemini_v07 import run_gemini_lite_structured

    user_payload = (
        f"Highlighted Text:\n{selected}\n\n"
        f"User Question:\n{question}\n\n"
        f"Context Paragraph (from summary UI, may be shallow):\n"
        f"{(surrounding_paragraph or '').strip()[:2000]}\n\n"
        f"{expanded_material}"
    )
    anchor = f"Explain fragment for run {run_id} | source {chunk_label}"
    out = run_gemini_lite_structured(
        EXPLAINER_SYSTEM,
        user_payload,
        anchor,
        NodeExplainContract,
        "contextual_explainer",
    )
    explanation = (out.explanation or "").strip()

    return ExplainResult(
        explanation=explanation,
        source_ref=source_ref,
        matched_chunk_id=str(chunk.get("chunk_id") or ""),
        source_chunk_preview=expanded_material[:500],
        expanded_chunk_ids=expanded_ids,
    )
