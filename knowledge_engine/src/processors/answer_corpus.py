"""Сбор ссылок и чанков из финального ответа Reasoner для UI и Explainer."""

from __future__ import annotations

import re
import uuid
from typing import Any

from knowledge_engine.src.processors.source_anchors import (
    expand_source_tags_to_markdown_links,
    strip_source_anchor_tags,
)
from knowledge_engine.ui.run_log import trace

_SOURCE_TAG_RE = re.compile(r"\[S(\d+)\]", re.I)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_BARE_URL_RE = re.compile(r"(?<![(\w])https?://[^\s<>\"']+")
_ARXIV_ABS_RE = re.compile(r"https?://(?:www\.)?arxiv\.org/abs/([^\s\])]+)", re.I)
_DOI_URL_RE = re.compile(r"https?://(?:dx\.)?doi\.org/(10\.\d+/[^\s\])]+)", re.I)


def _next_source_id(registry: list[dict[str, Any]]) -> str:
    max_n = 0
    for entry in registry:
        sid = str(entry.get("id") or entry.get("source_id") or "")
        m = re.match(r"S(\d+)$", sid.upper())
        if m:
            max_n = max(max_n, int(m.group(1)))
    return f"S{max_n + 1}"


def _registry_by_url(registry: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for entry in registry:
        url = (entry.get("url") or "").strip().lower()
        if url:
            out[url.rstrip("/")] = entry
    return out


def _append_registry_entry(
    registry: list[dict[str, Any]],
    *,
    title: str,
    url: str,
    by_url: dict[str, dict[str, Any]],
) -> str:
    url = (url or "").strip()
    norm = url.lower().rstrip("/")
    if norm and norm in by_url:
        return str(by_url[norm].get("id") or by_url[norm].get("source_id") or "")
    sid = _next_source_id(registry)
    entry = {
        "id": sid,
        "source_id": sid,
        "tag": f"[{sid}]",
        "title": (title or url or "source").strip()[:300],
        "url": url,
        "doi": "",
        "authors": "",
        "snippet": "",
        "venue": "",
    }
    if _DOI_URL_RE.search(url):
        m = _DOI_URL_RE.search(url)
        if m:
            entry["doi"] = m.group(1)
    registry.append(entry)
    if norm:
        by_url[norm] = entry
    return sid


def merge_registry_from_answer_text(
    answer: str,
    registry: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Дополнить SOURCE REGISTRY ссылками из markdown и URL в ответе Reasoner."""
    text = (answer or "").strip()
    if not text:
        return list(registry or [])
    out = list(registry or [])
    by_url = _registry_by_url(out)

    for m in _MD_LINK_RE.finditer(text):
        title = m.group(1).strip()
        url = m.group(2).strip()
        if url.startswith("http"):
            _append_registry_entry(out, title=title, url=url, by_url=by_url)

    for m in _ARXIV_ABS_RE.finditer(text):
        url = m.group(0)
        _append_registry_entry(out, title=f"arXiv:{m.group(1)}", url=url, by_url=by_url)

    for m in _DOI_URL_RE.finditer(text):
        url = m.group(0)
        _append_registry_entry(out, title=f"DOI {m.group(1)}", url=url, by_url=by_url)

    for m in _BARE_URL_RE.finditer(text):
        url = m.group(0).rstrip(".,;)")
        if len(url) < 12:
            continue
        if "arxiv.org" in url or "doi.org" in url:
            continue
        _append_registry_entry(out, title=url[:80], url=url, by_url=by_url)

    return out


def _section_source_ids(section_text: str) -> list[str]:
    ids: list[str] = []
    for m in _SOURCE_TAG_RE.finditer(section_text or ""):
        sid = f"S{m.group(1)}"
        if sid not in ids:
            ids.append(sid)
    return ids


def build_answer_context_chunks(
    answer: str,
    registry: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Секции ответа Reasoner как чанки для contextual explainer."""
    text = (answer or "").strip()
    if not text:
        return []
    by_id = {
        str(e.get("id") or e.get("source_id") or ""): e
        for e in registry
        if e.get("id") or e.get("source_id")
    }
    parts = re.split(r"\n(?=#{1,3}\s)", text)
    chunks: list[dict[str, Any]] = []
    for _i, part in enumerate(parts):
        body = part.strip()
        if len(body) < 40:
            continue
        sids = _section_source_ids(body)
        anchor = sids[0] if sids else ""
        ref = by_id.get(anchor) or {}
        chunk_id = f"answer:{uuid.uuid4().hex[:10]}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "doc_id": "reasoner_answer",
                "text": strip_source_anchor_tags(body)[:12_000],
                "concepts": [],
                "code_snippets": [],
                "p99_relevance_score": 0.85,
                "source_anchor": anchor,
            }
        )
        if ref.get("url"):
            chunks[-1]["_source_url"] = ref.get("url")
            chunks[-1]["_title"] = ref.get("title")
    if not chunks and len(text) >= 40:
        chunks.append(
            {
                "chunk_id": f"answer:{uuid.uuid4().hex[:10]}",
                "doc_id": "reasoner_answer",
                "text": strip_source_anchor_tags(text)[:12_000],
                "concepts": [],
                "code_snippets": [],
                "p99_relevance_score": 0.7,
                "source_anchor": (
                    _section_source_ids(text)[0] if _section_source_ids(text) else ""
                ),
            }
        )
    return chunks


def build_answer_block_sources(answer: str) -> list[dict[str, Any]]:
    """Метаданные: блок ответа → список source id."""
    text = (answer or "").strip()
    if not text:
        return []
    blocks: list[dict[str, Any]] = []
    parts = re.split(r"\n(?=#{1,3}\s)", text)
    for i, part in enumerate(parts):
        body = part.strip()
        if len(body) < 20:
            continue
        head = body.split("\n", 1)[0].strip().lstrip("#").strip()
        blocks.append(
            {
                "block_index": i,
                "heading": head[:200],
                "source_ids": _section_source_ids(body),
            }
        )
    return blocks


def finalize_run_answer_corpus(state: dict[str, Any]) -> dict[str, Any]:
    """
    После Reasoner: реестр ссылок, кликабельные [Sx], чанки ответа для Explainer.
    """
    answer = str(state.get("user_final_answer") or "").strip()
    if not answer:
        return state

    registry = merge_registry_from_answer_text(
        answer, state.get("source_registry") or []
    )
    state["source_registry"] = registry
    state["user_final_answer"] = expand_source_tags_to_markdown_links(answer, registry)
    state["answer_block_sources"] = build_answer_block_sources(
        state["user_final_answer"]
    )

    answer_chunks = build_answer_context_chunks(state["user_final_answer"], registry)
    if answer_chunks:
        existing = list(state.get("structured_chunks") or [])
        existing_ids = {
            str(c.get("chunk_id") or "") for c in existing if isinstance(c, dict)
        }
        for ch in answer_chunks:
            if ch["chunk_id"] not in existing_ids:
                existing.append(ch)
        state["structured_chunks"] = existing

    docs = list(state.get("documents") or [])
    has_answer_doc = any(
        isinstance(d, dict) and str(d.get("doc_id") or "") == "reasoner_answer"
        for d in docs
    )
    if not has_answer_doc:
        docs.append(
            {
                "doc_id": "reasoner_answer",
                "source_url": "",
                "source_type": "trafilatura",
                "raw_markdown": state["user_final_answer"],
                "title": "Ответ Reasoner",
                "cosine_dedup_passed": True,
                "is_pdf": False,
            }
        )
        state["documents"] = docs

    trace(
        f"Answer corpus ✓ registry={len(registry)} "
        f"answer_chunks={len(answer_chunks)} blocks={len(state.get('answer_block_sources') or [])}"
    )
    return state
