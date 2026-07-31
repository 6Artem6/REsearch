"""Source registry для ноды Skill Tree (как v07 overview + [Sx] в тексте)."""

from __future__ import annotations

from typing import Any

from knowledge_engine.src.node_deep_dive.schemas import RichReferenceItem
from knowledge_engine.src.processors.source_anchors import (
    build_source_registry,
    format_registry_for_prompt,
)


def _ref_to_paper_dict(ref: RichReferenceItem | dict[str, Any]) -> dict[str, Any]:
    if isinstance(ref, RichReferenceItem):
        data = ref.model_dump()
    else:
        data = dict(ref)
    title = (data.get("title") or data.get("source_name") or "source").strip()
    url = (data.get("url") or "").strip()
    why = (data.get("why_read") or "").strip()
    focus = (data.get("key_focus") or "").strip()
    snippet = "\n".join(x for x in [why, focus] if x)
    return {
        "title": title,
        "url": url,
        "snippet": snippet[:1200],
        "source_name": (data.get("source_name") or "").strip(),
    }


def build_registry_from_references(
    refs: list[RichReferenceItem | dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    papers = [_ref_to_paper_dict(r) for r in (refs or []) if r]
    papers = [p for p in papers if (p.get("url") or "").strip()]
    if not papers:
        return []
    return build_source_registry(papers)


def build_registry_from_curriculum_library(
    curriculum_id: str,
    mapped_source_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Подмножество curriculum_sources_registry только по mapped_source_ids.
    Пустой mapped → [] (никакого фоллбэка на весь пул курса).
    """
    cid = (curriculum_id or "").strip()
    if not cid:
        return []
    from knowledge_engine.services.skill_tree_store import get_curriculum_graph

    raw = get_curriculum_graph(cid)
    if not raw:
        return []
    entries = list(raw.get("curriculum_sources_registry") or [])
    want = {str(x).strip() for x in (mapped_source_ids or []) if str(x).strip()}
    if not want:
        # Без mapped_source_ids — не показывать весь пул курса как «источники ноды».
        return []
    entries = [
        e
        for e in entries
        if str(e.get("source_id") or "").strip() in want
    ]
    papers: list[dict[str, Any]] = []
    for e in entries:
        url = (e.get("url") or "").strip()
        if not url:
            continue
        title = (e.get("title") or e.get("source_name") or url)[:400]
        snippet = (e.get("snippet") or e.get("why_read") or "").strip()[:1200]
        papers.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
                "source_name": title,
            }
        )
    if not papers:
        return []
    return build_source_registry(papers)


def build_session_source_registry(
    curriculum_id: str,
    mapped_source_ids: list[str] | None,
    content_references: list[RichReferenceItem | dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Реестр сессии ноды: только mapped_source_ids → библиотека курса;
    если mapped пуст — только явные references из content (лекция/verify), не глобальный пул.
    """
    mapped_registry = build_registry_from_curriculum_library(
        curriculum_id,
        mapped_source_ids,
    )
    if mapped_registry:
        return mapped_registry
    return build_registry_from_references(content_references)


def registry_for_prompt(registry: list[dict[str, Any]]) -> str:
    return format_registry_for_prompt(registry)
