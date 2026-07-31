"""Персистентные сессии нод (curriculum_id + node_id)."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any

from knowledge_engine.config import PACKAGE_ROOT
from knowledge_engine.src.node_deep_dive.schemas import NodeContentBlock, NodeStatus
from knowledge_engine.src.node_deep_dive.memory_schemas import SessionMemory
from knowledge_engine.web.llm_text_repair import repair_diagram_markdown
from knowledge_engine.src.node_deep_dive.dialog_ids import (
    MSG_ID_KEY,
    clean_dialog_rows,
    ensure_msg_ids,
    max_msg_id,
    reconcile_dialog_history,
)
from knowledge_engine.src.node_deep_dive.tiered_memory import memory_from_blob, memory_to_blob

_STORE_PATH = PACKAGE_ROOT / ".runs" / "node_deep_dive_sessions.json"
_lock = threading.Lock()

_VALID_STATUSES = frozenset(
    {"in_progress", "deep_understanding", "mastered", "gap"}
)


def _session_key(curriculum_id: str, node_id: str) -> str:
    return f"{curriculum_id.strip()}::{node_id.strip()}"


def normalize_dialog_history(
    history: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    """Очистка полей; порядок реплик не меняется (хронология — reconcile / active_window)."""
    return clean_dialog_rows(history)


def _repair_memory_dialog_ids(memory: SessionMemory | None) -> None:
    if memory is None:
        return
    seq = max(int(memory.dialog_seq or 0), max_msg_id(memory.active_window))
    window, seq = ensure_msg_ids(memory.active_window, start_seq=seq)
    memory.active_window = window
    memory.dialog_seq = max(int(memory.dialog_seq or 0), seq)


def repair_history_with_memory(
    history: list[dict[str, str]] | None,
    memory: SessionMemory | None,
) -> list[dict[str, str]]:
    """Синхронизирует history с active_window: порядок + msg_id."""
    hist = clean_dialog_rows(history)
    if not memory or not memory.active_window:
        repaired, seq = reconcile_dialog_history(hist, None, start_seq=0)
        if memory is not None:
            memory.dialog_seq = max(int(memory.dialog_seq or 0), seq)
        return repaired
    _repair_memory_dialog_ids(memory)
    window = clean_dialog_rows(memory.active_window)
    start = max(int(memory.dialog_seq or 0), max_msg_id(hist))
    merged, seq = reconcile_dialog_history(hist, window, start_seq=start)
    memory.dialog_seq = max(int(memory.dialog_seq or 0), seq, max_msg_id(merged))
    return merged


class _SessionRecord:
    def __init__(
        self,
        node_status: NodeStatus = "in_progress",
        content: NodeContentBlock | None = None,
        history: list[dict[str, str]] | None = None,
        memory: SessionMemory | None = None,
    ) -> None:
        self.node_status = node_status
        self.content = content or NodeContentBlock()
        self.history = history or []
        self.memory = memory


def _load_all() -> dict[str, dict]:
    if not _STORE_PATH.is_file():
        return {}
    try:
        raw = json.loads(_STORE_PATH.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _save_all(data: dict[str, dict]) -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STORE_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _normalize_status(raw: str) -> NodeStatus:
    s = (raw or "").strip()
    if s in _VALID_STATUSES:
        return s
    return "in_progress"


def get_session(curriculum_id: str, node_id: str) -> _SessionRecord:
    key = _session_key(curriculum_id, node_id)
    with _lock:
        blob = _load_all().get(key) or {}
    content_raw = blob.get("content") or {}
    content = NodeContentBlock.model_validate(content_raw)
    if (content.diagram or "").strip():
        content = NodeContentBlock(
            **{
                **content.model_dump(),
                "diagram": repair_diagram_markdown(content.diagram),
            }
        )
    history = clean_dialog_rows(blob.get("history") or [])
    memory = memory_from_blob(blob.get("memory"))
    if memory is not None:
        _repair_memory_dialog_ids(memory)
    history = repair_history_with_memory(history, memory)
    status = _normalize_status(str(blob.get("node_status") or "in_progress"))
    return _SessionRecord(
        node_status=status,
        content=content,
        history=history,
        memory=memory,
    )


def save_session(
    curriculum_id: str,
    node_id: str,
    node_status: NodeStatus,
    content: NodeContentBlock,
    history: list[dict[str, str]],
    rag_fact_labels: list[str] | None = None,
    memory: SessionMemory | None = None,
    source_registry: list[dict[str, Any]] | None = None,
) -> str:
    key = _session_key(curriculum_id, node_id)
    labels = [str(x).strip() for x in (rag_fact_labels or []) if str(x).strip()][:8]
    norm_history = repair_history_with_memory(history, memory)
    content_dump = {
        k: v
        for k, v in content.model_dump().items()
        if k != "summary_html"
    }
    if content_dump.get("diagram"):
        content_dump["diagram"] = repair_diagram_markdown(
            str(content_dump.get("diagram") or "")
        )
    mem_blob: dict[str, Any] = {}
    if memory is not None:
        mem_blob = memory_to_blob(memory)
    with _lock:
        all_data = _load_all()
        prev = all_data.get(key) or {}
        merged_labels = labels or prev.get("rag_fact_labels") or []
        entry: dict[str, Any] = {
            "node_status": node_status,
            "content": content_dump,
            "history": norm_history[-40:],
            "rag_fact_labels": merged_labels,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if mem_blob:
            entry["memory"] = mem_blob
        if source_registry is not None:
            entry["source_registry"] = list(source_registry)[:12]
        all_data[key] = entry
        _save_all(all_data)
    return key


def get_all_sessions_for_curriculum(curriculum_id: str) -> dict[str, dict]:
    """Все сохранённые сессии нод (контент, история чата, ссылки)."""
    prefix = f"{curriculum_id.strip()}::"
    out: dict[str, dict] = {}
    with _lock:
        for key, blob in _load_all().items():
            if not key.startswith(prefix):
                continue
            node_id = key.split("::", 1)[-1]
            out[node_id] = dict(blob)
    return out


def get_node_statuses_for_curriculum(curriculum_id: str) -> dict[str, str]:
    """Статусы нод для графа."""
    prefix = f"{curriculum_id.strip()}::"
    out: dict[str, str] = {}
    with _lock:
        for key, blob in _load_all().items():
            if not key.startswith(prefix):
                continue
            node_id = key.split("::", 1)[-1]
            out[node_id] = _normalize_status(str(blob.get("node_status") or "in_progress"))
    return out


def discover_curriculum_ids_from_sessions() -> set[str]:
    """ID маршрутов из ключей сессий (curriculum_id::node_id)."""
    ids: set[str] = set()
    with _lock:
        for key in _load_all().keys():
            if "::" in key:
                ids.add(key.split("::", 1)[0].strip())
    return ids
