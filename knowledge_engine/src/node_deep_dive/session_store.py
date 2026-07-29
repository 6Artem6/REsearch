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
    """Исправляет legacy-порядок tutor→user на user→tutor в каждом обмене."""
    cleaned: list[dict[str, str]] = []
    for item in history or []:
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        if role not in ("user", "tutor"):
            role = "tutor"
        cleaned.append({"role": role, "content": content})
    fixed: list[dict[str, str]] = []
    i = 0
    while i < len(cleaned):
        if (
            i + 1 < len(cleaned)
            and i > 0
            and cleaned[i]["role"] == "tutor"
            and cleaned[i + 1]["role"] == "user"
        ):
            fixed.append(cleaned[i + 1])
            fixed.append(cleaned[i])
            i += 2
        else:
            fixed.append(cleaned[i])
            i += 1
    return fixed


def _history_key(m: dict[str, str]) -> tuple[str, str]:
    return (m["role"], m["content"])


def _history_severely_corrupt(hist: list[dict[str, str]]) -> bool:
    if len(hist) < 3:
        return False
    consec = 0
    last_role: str | None = None
    for h in hist:
        r = h["role"]
        if r == last_role:
            consec += 1
            if consec >= 2:
                return True
        else:
            consec = 1
            last_role = r
    users_run = 0
    for h in hist:
        if h["role"] == "user":
            users_run += 1
            if users_run >= 3:
                return True
        else:
            users_run = 0
    tutors_run = 0
    for h in hist:
        if h["role"] == "tutor":
            tutors_run += 1
            if tutors_run >= 3:
                return True
        else:
            tutors_run = 0
    return False


def repair_history_with_memory(
    history: list[dict[str, str]] | None,
    memory: SessionMemory | None,
) -> list[dict[str, str]]:
    """Синхронизирует history с active_window без «prefix+window» (ломал порядок)."""
    hist = normalize_dialog_history(history)
    if not memory or not memory.active_window:
        return hist
    window = normalize_dialog_history(memory.active_window)
    if not window:
        return hist

    hk = [_history_key(h) for h in hist]
    wk = [_history_key(w) for w in window]

    if len(wk) <= len(hk) and hk[-len(wk):] == wk:
        return hist

    best_overlap = 0
    for i in range(1, min(len(hk), len(wk)) + 1):
        if hk[-i:] == wk[:i]:
            best_overlap = i

    if best_overlap:
        merged = hist[:-best_overlap] + window
        return normalize_dialog_history(merged)

    if _history_severely_corrupt(hist):
        intro: list[dict[str, str]] = []
        if (
            hist
            and hist[0]["role"] == "tutor"
            and window
            and window[0]["role"] != "tutor"
        ):
            intro = [hist[0]]
        return normalize_dialog_history(intro + window)

    if hk and wk and hk[-1] == wk[0] and len(wk) > 1:
        return normalize_dialog_history(hist + window[1:])

    return normalize_dialog_history(hist + window)


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
    history = normalize_dialog_history(blob.get("history") or [])
    memory = memory_from_blob(blob.get("memory"))
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
        if source_registry:
            entry["source_registry"] = source_registry[:12]
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
