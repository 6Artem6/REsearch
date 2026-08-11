"""Локальное хранение учебных маршрутов Skill Tree (.runs)."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any

from knowledge_engine.config import PACKAGE_ROOT
from knowledge_engine.services.llm_markdown_service import (
    enrich_session_blob_for_client,
)
from knowledge_engine.src.curriculum.schemas import CurriculumGraph
from knowledge_engine.src.curriculum.source_registry import (
    cap_curriculum_sources_registry,
    normalize_stored_curriculum_graph,
    sync_route_sources_from_registry,
)
from knowledge_engine.src.node_deep_dive.session_store import (
    discover_curriculum_ids_from_sessions,
    get_all_sessions_for_curriculum,
    get_node_statuses_for_curriculum,
)

_STORE_PATH = PACKAGE_ROOT / ".runs" / "skill_tree_curricula.json"
_lock = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_doc() -> dict[str, Any]:
    if not _STORE_PATH.is_file():
        return {"curricula": [], "active_curriculum_id": ""}
    try:
        raw = json.loads(_STORE_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {"curricula": [], "active_curriculum_id": ""}
        raw.setdefault("curricula", [])
        return raw
    except Exception:
        return {"curricula": [], "active_curriculum_id": ""}


def _save_doc(doc: dict[str, Any]) -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STORE_PATH.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_curriculum_record(
    graph: CurriculumGraph | dict[str, Any],
    *,
    target_goal: str,
    generation_mode: str = "fast",
    depth_level: str = "Standard",
    user_level: str = "Intermediate/Advanced",
    source_policy: str | None = None,
) -> dict[str, Any]:
    """Сохранить или обновить граф маршрута."""
    if isinstance(graph, CurriculumGraph):
        g_obj = graph
    else:
        g_obj = CurriculumGraph.model_validate(
            normalize_stored_curriculum_graph(dict(graph))
        )
    registry = cap_curriculum_sources_registry(
        list(g_obj.curriculum_sources_registry),
        graph=g_obj,
    )
    g_obj = sync_route_sources_from_registry(
        g_obj.model_copy(update={"curriculum_sources_registry": registry})
    )
    payload = g_obj.model_dump()
    cid = str(payload.get("curriculum_id") or "").strip()
    if not cid:
        raise ValueError("curriculum_id обязателен")

    record = {
        "curriculum_id": cid,
        "target_goal": (target_goal or "").strip(),
        "generation_mode": (generation_mode or "fast").strip(),
        "depth_level": (depth_level or "Standard").strip(),
        "user_level": (user_level or "Intermediate/Advanced").strip(),
        "source_policy": (source_policy or "").strip(),
        "title": str(payload.get("title") or "").strip(),
        "description": str(payload.get("description") or "").strip(),
        "graph": payload,
        "updated_at": _now_iso(),
    }
    g = record["graph"]
    if isinstance(g, dict):
        meta = dict(g.get("meta") or {})
        meta["generation_mode"] = record["generation_mode"]
        g["meta"] = meta

    with _lock:
        doc = _load_doc()
        items = doc.get("curricula") or []
        found = False
        for i, item in enumerate(items):
            if item.get("curriculum_id") == cid:
                record["created_at"] = item.get("created_at") or _now_iso()
                items[i] = record
                found = True
                break
        if not found:
            record["created_at"] = _now_iso()
            items.append(record)
        doc["curricula"] = items
        doc["active_curriculum_id"] = cid
        _save_doc(doc)
    return record


def list_curriculum_summaries() -> list[dict[str, Any]]:
    with _lock:
        doc = _load_doc()
    out: list[dict[str, Any]] = []
    for item in doc.get("curricula") or []:
        g = item.get("graph") or {}
        out.append(
            {
                "curriculum_id": item.get("curriculum_id"),
                "title": item.get("title") or g.get("title"),
                "target_goal": item.get("target_goal"),
                "generation_mode": item.get("generation_mode"),
                "depth_level": item.get("depth_level"),
                "total_nodes": g.get("total_nodes") or len(g.get("nodes") or []),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
                "is_active": item.get("curriculum_id")
                == doc.get("active_curriculum_id"),
                "has_graph": True,
            }
        )
    known = {str(x.get("curriculum_id")) for x in out}
    for cid in discover_curriculum_ids_from_sessions():
        if cid in known:
            continue
        out.append(
            {
                "curriculum_id": cid,
                "title": f"Маршрут {cid}",
                "target_goal": "",
                "generation_mode": "",
                "depth_level": "",
                "total_nodes": 0,
                "created_at": "",
                "updated_at": "",
                "is_active": cid == doc.get("active_curriculum_id"),
                "has_graph": False,
            }
        )
    out.sort(key=lambda x: str(x.get("updated_at") or ""), reverse=True)
    return out


def get_curriculum_graph(curriculum_id: str) -> dict[str, Any] | None:
    cid = curriculum_id.strip()
    with _lock:
        doc = _load_doc()
    for item in doc.get("curricula") or []:
        if item.get("curriculum_id") == cid:
            g = item.get("graph") or None
            if g and isinstance(g, dict):
                return normalize_stored_curriculum_graph(g)
            return g
    return None


def get_curriculum_meta(curriculum_id: str) -> dict[str, Any] | None:
    cid = curriculum_id.strip()
    with _lock:
        doc = _load_doc()
    for item in doc.get("curricula") or []:
        if item.get("curriculum_id") == cid:
            return dict(item)
    return None


def patch_curriculum_graph_node(
    curriculum_id: str,
    node_id: str,
    updates: dict[str, Any],
) -> bool:
    """Merge ``updates`` into one node dict inside stored graph."""
    cid = (curriculum_id or "").strip()
    nid = (node_id or "").strip()
    if not cid or not nid or not updates:
        return False
    with _lock:
        doc = _load_doc()
        for item in doc.get("curricula") or []:
            if item.get("curriculum_id") != cid:
                continue
            graph = item.get("graph") or {}
            nodes = graph.get("nodes") or []
            found = False
            for raw in nodes:
                if not isinstance(raw, dict):
                    continue
                if str(raw.get("node_id") or "").strip() != nid:
                    continue
                raw.update(updates)
                found = True
                break
            if not found:
                return False
            graph["nodes"] = nodes
            item["graph"] = graph
            item["updated_at"] = _now_iso()
            _save_doc(doc)
            return True
    return False


def _short_neighbor_concepts(concepts: list[str] | None, max_items: int = 3) -> str:
    parts: list[str] = []
    for c in concepts or []:
        s = (c or "").strip()
        if not s:
            continue
        words = s.split()[:3]
        parts.append(" ".join(words))
        if len(parts) >= max_items:
            break
    return ", ".join(parts)


def get_node_neighbors_context(curriculum_id: str, node_id: str) -> dict[str, Any]:
    """
    Микро-контекст соседей по DAG: предшественники (prerequisites + edges) и преемники.
    """
    graph = get_curriculum_graph(curriculum_id)
    if not graph:
        return {}
    nid = (node_id or "").strip()
    if not nid:
        return {}

    nodes = graph.get("nodes") or []
    by_id: dict[str, dict[str, Any]] = {}
    for raw in nodes:
        if isinstance(raw, dict) and raw.get("node_id"):
            by_id[str(raw["node_id"]).strip()] = raw

    current = by_id.get(nid)
    if not current:
        return {}

    prereq_ids: set[str] = set()
    for p in current.get("prerequisites") or []:
        pid = str(p).strip()
        if pid and pid != nid:
            prereq_ids.add(pid)

    for e in graph.get("edges") or graph.get("dag_edges") or []:
        if not isinstance(e, dict):
            continue
        to_id = str(e.get("to_node_id") or e.get("to") or "").strip()
        fr_id = str(e.get("from_node_id") or e.get("from") or "").strip()
        if to_id == nid and fr_id and fr_id != nid:
            prereq_ids.add(fr_id)

    predecessors: list[dict[str, str]] = []
    for pid in sorted(prereq_ids):
        raw = by_id.get(pid)
        if not raw:
            continue
        predecessors.append(
            {
                "node_id": pid,
                "title": str(raw.get("title") or pid)[:300],
                "short_concepts": _short_neighbor_concepts(
                    raw.get("core_concepts") or []
                ),
            }
        )

    successors: list[dict[str, str]] = []
    for oid, raw in by_id.items():
        if oid == nid:
            continue
        prereqs = [str(p).strip() for p in (raw.get("prerequisites") or [])]
        if nid in prereqs:
            successors.append(
                {
                    "node_id": oid,
                    "title": str(raw.get("title") or oid)[:300],
                }
            )

    return {
        "current_node_id": nid,
        "current_title": str(current.get("title") or nid)[:300],
        "predecessors": predecessors,
        "successors": successors,
    }


def set_active_curriculum(curriculum_id: str) -> bool:
    cid = curriculum_id.strip()
    with _lock:
        doc = _load_doc()
        ok = any(c.get("curriculum_id") == cid for c in doc.get("curricula") or [])
        if not ok and cid not in discover_curriculum_ids_from_sessions():
            return False
        doc["active_curriculum_id"] = cid
        _save_doc(doc)
    return True


def get_active_curriculum_id() -> str:
    with _lock:
        return str(_load_doc().get("active_curriculum_id") or "").strip()


def _mapped_source_ids_for_node(graph: dict[str, Any], node_id: str) -> list[str]:
    nid = (node_id or "").strip()
    for n in graph.get("nodes") or []:
        if not isinstance(n, dict):
            continue
        if str(n.get("node_id") or n.get("id") or "").strip() == nid:
            return [
                str(x).strip()
                for x in (n.get("mapped_source_ids") or [])
                if str(x).strip()
            ]
    return []


def get_workspace_state(curriculum_id: str) -> dict[str, Any] | None:
    """
    Полный снимок для UI: граф, статусы, сессии нод (контент, диалоги, ссылки).
    """
    graph = get_curriculum_graph(curriculum_id)
    if not graph:
        return None
    meta = get_curriculum_meta(curriculum_id) or {}
    statuses = get_node_statuses_for_curriculum(curriculum_id)
    from knowledge_engine.services.node_source_registry import (
        build_session_source_registry,
    )
    from knowledge_engine.src.node_deep_dive.schemas import NodeContentBlock
    from knowledge_engine.src.node_deep_dive.tutor_source_citations import (
        scrub_content_references,
    )

    sessions: dict[str, Any] = {}
    for node_id, blob in get_all_sessions_for_curriculum(curriculum_id).items():
        b = dict(blob)
        mapped = _mapped_source_ids_for_node(graph, node_id)
        reg = build_session_source_registry(curriculum_id, mapped)
        b["source_registry"] = reg
        raw_content = b.get("content")
        if isinstance(raw_content, dict):
            block = scrub_content_references(
                NodeContentBlock.model_validate(raw_content),
                reg,
            )
            b["content"] = block.model_dump(exclude={"summary_html"})
        sessions[node_id] = enrich_session_blob_for_client(
            b,
            node_id=node_id,
            curriculum_id=curriculum_id,
        )
    return {
        "curriculum_id": curriculum_id,
        "meta": {
            "target_goal": meta.get("target_goal"),
            "generation_mode": meta.get("generation_mode"),
            "depth_level": meta.get("depth_level"),
            "title": meta.get("title") or graph.get("title"),
            "updated_at": meta.get("updated_at"),
        },
        "curriculum": graph,
        "statuses": statuses,
        "sessions": sessions,
    }
