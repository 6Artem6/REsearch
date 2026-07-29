"""Локальное хранение учебных маршрутов Skill Tree (.runs)."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any

from knowledge_engine.config import PACKAGE_ROOT
from knowledge_engine.services.llm_markdown_service import enrich_session_blob_for_client
from knowledge_engine.src.curriculum.schemas import CurriculumGraph
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
) -> dict[str, Any]:
    """Сохранить или обновить граф маршрута."""
    if isinstance(graph, CurriculumGraph):
        payload = graph.model_dump()
    else:
        payload = dict(graph)
    cid = str(payload.get("curriculum_id") or "").strip()
    if not cid:
        raise ValueError("curriculum_id обязателен")

    record = {
        "curriculum_id": cid,
        "target_goal": (target_goal or "").strip(),
        "generation_mode": (generation_mode or "fast").strip(),
        "depth_level": (depth_level or "Standard").strip(),
        "user_level": (user_level or "Intermediate/Advanced").strip(),
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
                "is_active": item.get("curriculum_id") == doc.get(
                    "active_curriculum_id"
                ),
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
            return item.get("graph") or None
    return None


def get_curriculum_meta(curriculum_id: str) -> dict[str, Any] | None:
    cid = curriculum_id.strip()
    with _lock:
        doc = _load_doc()
    for item in doc.get("curricula") or []:
        if item.get("curriculum_id") == cid:
            return dict(item)
    return None


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


def get_workspace_state(curriculum_id: str) -> dict[str, Any] | None:
    """
    Полный снимок для UI: граф, статусы, сессии нод (контент, диалоги, ссылки).
    """
    graph = get_curriculum_graph(curriculum_id)
    if not graph:
        return None
    meta = get_curriculum_meta(curriculum_id) or {}
    statuses = get_node_statuses_for_curriculum(curriculum_id)
    sessions = {
        node_id: enrich_session_blob_for_client(blob)
        for node_id, blob in get_all_sessions_for_curriculum(curriculum_id).items()
    }
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
