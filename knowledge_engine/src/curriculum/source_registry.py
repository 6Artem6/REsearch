"""Реестр источников курса и адресация mapped_source_ids → ноды."""

from __future__ import annotations

from typing import Any

from knowledge_engine.src.curriculum.schemas import (
    CurriculumGraph,
    CurriculumNode,
    CurriculumSourceRegistryEntry,
    RouteSourceEntry,
)


def validate_curriculum_source_links(graph: CurriculumGraph) -> list[str]:
    """Каждый mapped_source_id должен быть в curriculum_sources_registry."""
    reg_ids = {e.source_id for e in graph.curriculum_sources_registry}
    if not reg_ids and graph.route_sources:
        reg_ids = {e.source_id for e in graph.route_sources}
    errors: list[str] = []
    if len(graph.curriculum_sources_registry) < 8 and not graph.route_sources:
        errors.append(
            "curriculum_sources_registry: ожидается ≥8 источников в библиотеке курса."
        )
    for n in graph.nodes:
        mapped = [m.strip() for m in n.mapped_source_ids if (m or "").strip()]
        if not mapped:
            errors.append(f"Узел '{n.node_id}': mapped_source_ids пуст — нужны 1–3 src_id.")
            continue
        if len(mapped) > 3:
            errors.append(f"Узел '{n.node_id}': mapped_source_ids > 3.")
        for sid in mapped:
            if sid not in reg_ids:
                errors.append(
                    f"Узел '{n.node_id}': mapped_source_ids содержит несуществующий '{sid}'."
                )
    return errors


def registry_index(graph: CurriculumGraph | dict[str, Any]) -> dict[str, dict[str, Any]]:
    """source_id → entry dict (registry + legacy route_sources)."""
    out: dict[str, dict[str, Any]] = {}
    if isinstance(graph, CurriculumGraph):
        for e in graph.curriculum_sources_registry:
            out[e.source_id] = e.model_dump()
        for e in graph.route_sources:
            out.setdefault(e.source_id, e.model_dump())
        return out
    for e in graph.get("curriculum_sources_registry") or []:
        if isinstance(e, dict) and e.get("source_id"):
            out[str(e["source_id"])] = e
    for e in graph.get("route_sources") or []:
        if isinstance(e, dict) and e.get("source_id"):
            out.setdefault(str(e["source_id"]), e)
    return out


def resolve_sources_for_node(
    graph: CurriculumGraph | dict[str, Any],
    node_id: str,
    mapped_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    idx = registry_index(graph)
    ids = mapped_ids or []
    if not ids and isinstance(graph, CurriculumGraph):
        for n in graph.nodes:
            if n.node_id == node_id:
                ids = list(n.mapped_source_ids)
                break
    elif not ids and isinstance(graph, dict):
        for n in graph.get("nodes") or []:
            if str(n.get("node_id") or "") == node_id:
                ids = list(n.get("mapped_source_ids") or [])
                break
    resolved: list[dict[str, Any]] = []
    for sid in ids:
        key = (sid or "").strip()
        if key and key in idx:
            resolved.append(idx[key])
    return resolved


def format_resolved_sources_for_lecture(
    graph: CurriculumGraph | dict[str, Any],
    node_id: str,
    mapped_ids: list[str] | None = None,
) -> str:
    rows = resolve_sources_for_node(graph, node_id, mapped_ids)
    if not rows:
        return ""
    lines = ["ИСТОЧНИКИ НОДЫ ИЗ ГЛОБАЛЬНОГО РЕЕСТРА КУРСА:"]
    for r in rows:
        sid = r.get("source_id") or ""
        title = r.get("title") or r.get("source_name") or sid
        domain = r.get("whitelist_domain") or r.get("whitelist_category") or ""
        typ = r.get("source_type") or r.get("type") or ""
        url = r.get("url") or ""
        why = r.get("why_read") or ""
        lines.append(f"- [{sid}] {title}")
        if domain:
            lines.append(f"  whitelist: {domain}")
        if typ:
            lines.append(f"  type: {typ}")
        if url:
            lines.append(f"  url: {url}")
        if why:
            lines.append(f"  зачем: {why[:600]}")
        snippet = r.get("snippet") or ""
        if snippet and snippet != why:
            lines.append(f"  snippet: {snippet[:800]}")
    return "\n".join(lines)


def sync_route_sources_from_registry(graph: CurriculumGraph) -> CurriculumGraph:
    """Legacy UI: route_sources зеркалит registry с URL."""
    route: list[RouteSourceEntry] = []
    for i, e in enumerate(graph.curriculum_sources_registry, start=1):
        url = (e.url or "").strip()
        if len(url) < 8:
            continue
        sid = (e.source_id or "").strip() or f"S{i}"
        route.append(
            RouteSourceEntry(
                source_id=sid[:16],
                source_name=(e.title or e.whitelist_domain or sid)[:400],
                url=url[:2000],
                whitelist_category=(e.whitelist_domain or e.source_type or "")[:120],
                why_read=(e.why_read or "")[:800],
            )
        )
    if route:
        return graph.model_copy(update={"route_sources": route[:24]})
    return graph
