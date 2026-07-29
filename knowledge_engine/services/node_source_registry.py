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


def registry_for_prompt(registry: list[dict[str, Any]]) -> str:
    return format_registry_for_prompt(registry)
