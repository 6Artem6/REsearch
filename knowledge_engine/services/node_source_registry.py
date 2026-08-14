"""Source registry для ноды Skill Tree (как v07 overview + [Sx] в тексте)."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from knowledge_engine.src.node_deep_dive.schemas import RichReferenceItem
from knowledge_engine.src.processors.source_anchors import (
    build_source_registry,
    format_registry_for_prompt,
)

# RFC 2606 / типичные заглушки LLM — никогда не в реестр сессии и не в [Sx].
_DISALLOWED_REGISTRY_HOSTS = frozenset(
    {
        "example.com",
        "example.org",
        "example.net",
        "example.edu",
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
    }
)


def is_disallowed_source_url(url: str) -> bool:
    raw = (url or "").strip()
    if len(raw) < 8 or not raw.lower().startswith(("http://", "https://")):
        return True
    try:
        host = (urlparse(raw).hostname or "").lower()
    except ValueError:
        return True
    if not host:
        return True
    if host in _DISALLOWED_REGISTRY_HOSTS:
        return True
    if host.endswith(".example") or host.endswith(".invalid") or host.endswith(".test"):
        return True
    return False


def filter_source_registry(
    registry: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in registry or []:
        if not isinstance(entry, dict):
            continue
        url = (entry.get("url") or "").strip()
        if is_disallowed_source_url(url):
            continue
        out.append(entry)
    return out


def _ref_to_paper_dict(
    ref: RichReferenceItem | dict[str, Any],
) -> dict[str, Any] | None:
    if isinstance(ref, RichReferenceItem):
        data = ref.model_dump()
    else:
        data = dict(ref)
    title = (data.get("title") or data.get("source_name") or "source").strip()
    url = (data.get("url") or "").strip()
    if is_disallowed_source_url(url):
        return None
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
    papers = []
    for r in refs or []:
        if not r:
            continue
        p = _ref_to_paper_dict(r)
        if p and (p.get("url") or "").strip():
            papers.append(p)
    if not papers:
        return []
    return filter_source_registry(build_source_registry(papers))


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
    want_ids = [str(x).strip() for x in (mapped_source_ids or []) if str(x).strip()]
    if not want_ids:
        # Без mapped_source_ids — не показывать весь пул курса как «источники ноды».
        return []
    # Одна каноническая строка на src_* (как resolve_sources_for_node), не все дубли в registry.
    from knowledge_engine.src.curriculum.source_registry import resolve_sources_for_node

    entries = resolve_sources_for_node(raw, "", want_ids)
    papers: list[dict[str, Any]] = []
    for e in entries:
        url = (e.get("url") or "").strip()
        if not url or is_disallowed_source_url(url):
            continue
        title = (e.get("title") or e.get("source_name") or url)[:400]
        snippet = (e.get("snippet") or e.get("why_read") or "").strip()[:1200]
        papers.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
                "source_name": title,
                "course_source_id": str(e.get("source_id") or "").strip(),
            }
        )
    if not papers:
        return []
    return filter_source_registry(build_source_registry(papers))


def build_session_source_registry(
    curriculum_id: str,
    mapped_source_ids: list[str] | None,
    content_references: list[RichReferenceItem | dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Реестр [S1]… для сессии ноды: ТОЛЬКО curriculum_sources_registry по mapped_source_ids.
    Ответы LLM в content.references НЕ создают реестр (защита от example.com и выдумок).
    """
    _ = content_references
    mapped_registry = build_registry_from_curriculum_library(
        curriculum_id,
        mapped_source_ids,
    )
    return filter_source_registry(mapped_registry)


def registry_for_prompt(registry: list[dict[str, Any]]) -> str:
    return format_registry_for_prompt(registry)
