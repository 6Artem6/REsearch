"""Источники тьютора: только SOURCE REGISTRY / verified catalog, не URL в prose."""

from __future__ import annotations

from typing import Any

from knowledge_engine.services.node_source_registry import build_session_source_registry
from knowledge_engine.src.node_deep_dive.schemas import (
    NodeContentBlock,
    NodeDataInput,
    RichReferenceItem,
)
from knowledge_engine.src.processors.source_anchors import (
    format_registry_for_prompt,
    retarget_source_anchor_tags,
)
from knowledge_engine.utils.link_sanitizer import normalize_lecture_url


def build_tutor_source_registry(
    curriculum_id: str,
    node: NodeDataInput,
    content: NodeContentBlock | None = None,
) -> list[dict[str, Any]]:
    """Реестр [S1]… только из библиотеки курса (mapped_source_ids), не из LLM references."""
    _ = content
    return build_session_source_registry(
        curriculum_id,
        list(node.mapped_source_ids or []),
    )


def format_tutor_source_registry_light(registry: list[dict[str, Any]]) -> str:
    """Краткий реестр для dialogue Lite (без 400-символьных snippets)."""
    if not registry:
        return "### SOURCE REGISTRY (lite)\n" "(пусто) references: []\n"
    lines = ["### SOURCE REGISTRY (lite)"]
    for entry in registry:
        sid = entry.get("id") or "?"
        title = (entry.get("title") or "source").strip()[:200]
        url = (entry.get("url") or "—").strip()[:200]
        lines.append(f"- [{sid}] {title} ({url})")
    lines.append("Цитаты в tutor_message: только [S1]…; URL только в JSON references.")
    return "\n".join(lines)


def format_tutor_source_registry_pinned(registry: list[dict[str, Any]]) -> str:
    if not registry:
        return (
            "### SOURCE REGISTRY\n"
            "(пусто — у ноды нет mapped_source_ids в библиотеке курса)\n"
            "Поле JSON `references` оставь пустым []. Не создавай [S1] и не вставляй http в текст.\n"
        )
    body = format_registry_for_prompt(registry)
    return (
        f"{body}\n\n"
        "=== CITATION POLICY (обязательно) ===\n"
        "- В tutor_message / lecture_body: только inline-теги [S1], [S2], … из реестра выше; "
        "также [diagram-N] / [code-N] из материалов ноды.\n"
        "- ЗАПРЕЩЕНО: любые http/https, Markdown `[текст](url)`, голые URL в тексте ответа.\n"
        "- Внешние источники — ТОЛЬКО в JSON `references` (dialogue) или `used_sources` (лекция): "
        "скопируй title и url ТОЧНО из строки реестра; asset_id = id реестра (S1, S2, …).\n"
        "- Если источник не цитировался — не добавляй его в references.\n"
        "- Если реестр пуст — references: [].\n"
    )


def _registry_by_id(registry: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(e.get("id") or "").strip(): e for e in registry if e.get("id")}


def _registry_by_url(registry: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for e in registry:
        url = normalize_lecture_url(str(e.get("url") or ""))
        if url:
            out[url] = e
    return out


def scrub_content_references(
    content: NodeContentBlock,
    registry: list[dict[str, Any]],
) -> NodeContentBlock:
    """Убрать из панели references, не прошедшие привязку к реестру курса."""
    clean = coerce_references_to_registry(list(content.references or []), registry)
    if len(clean) == len(content.references or []):
        return content
    return content.model_copy(update={"references": clean})


def retarget_content_source_anchors(
    content: NodeContentBlock,
    old_registry: list[dict[str, Any]] | None,
    new_registry: list[dict[str, Any]] | None,
) -> NodeContentBlock:
    """Синхронизировать [Sx] в summary/code с новым реестром (по URL)."""
    if not new_registry:
        return content
    summary = retarget_source_anchor_tags(
        content.summary or "",
        old_registry,
        new_registry,
    )
    snippets = [
        retarget_source_anchor_tags(s, old_registry, new_registry)
        for s in (content.code_snippets or [])
    ]
    diagram = retarget_source_anchor_tags(
        content.diagram or "",
        old_registry,
        new_registry,
    )
    updates: dict[str, Any] = {}
    if summary != content.summary:
        updates["summary"] = summary
    if snippets != list(content.code_snippets or []):
        updates["code_snippets"] = snippets
    if diagram != content.diagram:
        updates["diagram"] = diagram
    if not updates:
        return content
    return content.model_copy(update=updates)


def retarget_dialog_history_source_anchors(
    history: list[dict[str, str]] | None,
    old_registry: list[dict[str, Any]] | None,
    new_registry: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    if not history or not new_registry:
        return list(history or [])
    out: list[dict[str, str]] = []
    for row in history:
        if not isinstance(row, dict):
            continue
        row_copy = dict(row)
        for key in ("content", "tutor_message", "lecture_body", "message"):
            raw = row_copy.get(key)
            if not isinstance(raw, str) or "[S" not in raw:
                continue
            fixed = retarget_source_anchor_tags(raw, old_registry, new_registry)
            if fixed != raw:
                row_copy[key] = fixed
        out.append(row_copy)
    return out


def registry_entry_to_rich_reference(entry: dict[str, Any]) -> RichReferenceItem | None:
    url = (entry.get("url") or "").strip()
    if len(url) < 8:
        return None
    from knowledge_engine.services.node_source_registry import is_disallowed_source_url

    if is_disallowed_source_url(url):
        return None
    sid = str(entry.get("id") or "").strip()
    title = (entry.get("title") or entry.get("source_name") or url).strip()[:400]
    snippet = (entry.get("snippet") or "").strip()
    why = snippet[:1200] if snippet else ""
    focus = ""
    if " | " in snippet:
        parts = snippet.split(" | ", 1)
        why = parts[0][:1200]
        focus = parts[1][:800] if len(parts) > 1 else ""
    return RichReferenceItem(
        asset_id=sid or "",
        source_name=title[:300],
        url=url,
        title=title,
        why_read=why,
        key_focus=focus,
        read_time_minutes=0,
    )


def coerce_references_to_registry(
    refs: list[RichReferenceItem] | None,
    registry: list[dict[str, Any]],
) -> list[RichReferenceItem]:
    """
    Оставить только ссылки, совпадающие с SOURCE REGISTRY (по S-id или URL).
    Выдуманные URL отбрасываются.
    """
    if not registry:
        return []
    by_id = _registry_by_id(registry)
    by_url = _registry_by_url(registry)
    out: list[RichReferenceItem] = []
    seen: set[str] = set()
    for raw in refs or []:
        aid = (raw.asset_id or "").strip().upper()
        if aid and not aid.startswith("S"):
            if aid.lower().startswith("ref-"):
                aid = ""
        url_key = normalize_lecture_url(raw.url or "")
        entry: dict[str, Any] | None = None
        if aid and aid in by_id:
            entry = by_id[aid]
        elif url_key and url_key in by_url:
            entry = by_url[url_key]
        if entry is None:
            continue
        canon_url = (entry.get("url") or "").strip()
        key = normalize_lecture_url(canon_url)
        if not key or key in seen:
            continue
        seen.add(key)
        built = registry_entry_to_rich_reference(entry)
        if built is None:
            continue
        if (raw.why_read or "").strip():
            built = built.model_copy(update={"why_read": raw.why_read.strip()[:1200]})
        if (raw.key_focus or "").strip():
            built = built.model_copy(update={"key_focus": raw.key_focus.strip()[:800]})
        if raw.read_time_minutes:
            built = built.model_copy(
                update={"read_time_minutes": raw.read_time_minutes}
            )
        out.append(built)
        if len(out) >= 6:
            break
    return out
