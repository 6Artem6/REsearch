"""Персистентные сессии нод (curriculum_id + node_id).

Хранилище: ``.runs/node_deep_dive_sessions.json`` (атомарная запись под lock).
Поле ``memory`` — JSON blob ``SessionMemory``; ``covered_subtopics`` и ``introduced_terms``
дублируются на верхнем уровне записи, чтобы реестры переживали re-init диалога.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any

from knowledge_engine.config import PACKAGE_ROOT
from knowledge_engine.src.node_deep_dive.dialog_ids import (
    _normalize_user_dialog_content,
    clean_dialog_rows,
    ensure_msg_ids,
    max_msg_id,
    parse_msg_id,
    reconcile_dialog_history,
)
from knowledge_engine.src.node_deep_dive.fact_manifest import merge_manifest
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    DialogueFactManifest,
    SessionMemory,
)
from knowledge_engine.src.node_deep_dive.schemas import NodeContentBlock, NodeStatus
from knowledge_engine.src.node_deep_dive.term_registry import (
    carry_introduced_terms,
    load_introduced_terms_from_blob,
    normalize_introduced_terms,
)
from knowledge_engine.src.node_deep_dive.tiered_memory import (
    memory_from_blob,
    memory_to_blob,
)
from knowledge_engine.web.llm_text_repair import repair_llm_display_text

_STORE_PATH = PACKAGE_ROOT / ".runs" / "node_deep_dive_sessions.json"
_lock = threading.RLock()

_VALID_STATUSES = frozenset(
    {
        "unexplored",
        "in_progress",
        "deep_understanding",
        "mastered",
        "gap",
        "passed_by_equivalence",
    }
)


def _session_key(curriculum_id: str, node_id: str) -> str:
    return f"{curriculum_id.strip()}::{node_id.strip()}"


def _normalize_covered_subtopics(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        key = str(k or "").strip()[:64]
        val = str(v or "").strip()[:600]
        if key and val:
            out[key] = val
    return out


def load_covered_subtopics_from_blob(blob: dict[str, Any]) -> dict[str, str]:
    """Реестр покрытия: top-level `covered_subtopics` + поле в `memory` (memory blob)."""
    top = _normalize_covered_subtopics(blob.get("covered_subtopics"))
    mem = memory_from_blob(blob.get("memory"))
    from_mem = _normalize_covered_subtopics(
        (mem.covered_subtopics if mem is not None else None) or {}
    )
    if not from_mem:
        return top
    if not top:
        return from_mem
    merged = dict(from_mem)
    for key, summary in top.items():
        merged.setdefault(key, summary)
    return merged


def carry_covered_subtopics(
    preserved: dict[str, str] | None,
    memory: SessionMemory,
) -> None:
    """Сохранить реестр при сбросе диалога / re-init memory (не обнулять прочитанное)."""
    prev = _normalize_covered_subtopics(preserved or {})
    if not prev:
        return
    current = _normalize_covered_subtopics(memory.covered_subtopics or {})
    merged = dict(prev)
    merged.update(current)
    memory.covered_subtopics = merged


def persist_session_memory(
    curriculum_id: str,
    node_id: str,
    memory: SessionMemory,
) -> None:
    """Атомарно записать memory blob (+ зеркало covered_subtopics) в store сессии ноды."""
    key = _session_key(curriculum_id, node_id)
    mem_blob = memory_to_blob(memory)
    reg = _normalize_covered_subtopics(memory.covered_subtopics)
    terms = normalize_introduced_terms(memory.introduced_terms)
    with _lock:
        all_data = _load_all()
        prev = dict(all_data.get(key) or {})
        prev["memory"] = mem_blob
        prev["covered_subtopics"] = reg
        prev["introduced_terms"] = terms
        prev["updated_at"] = datetime.now(timezone.utc).isoformat()
        all_data[key] = prev
        _save_all(all_data)


def apply_fact_manifest_patch(
    curriculum_id: str,
    node_id: str,
    expected_version: int,
    new_manifest: DialogueFactManifest,
) -> bool:
    """CAS-merge результата фоновой fact_manifest-экстракции
    (services/context_compressor_worker.py).

    Читает свежую запись сессии ПРЯМО под ``_lock`` и пишет только
    ``memory.fact_manifest``/``manifest_version`` — остальные поля записи
    (history/content/...) могли измениться, пока фоновая задача считала
    Gemini, и не должны затираться.

    ВАЖНО (найдено на живом прогоне): один ход пользователя может вызвать
    ``rotate_window_after_message`` дважды подряд (эвикция user- и
    tutor-сообщения одного и того же хода — см. ``commit_turn_node``), и
    оба enqueue происходят с ОДНОЙ и той же ``expected_manifest_version``
    (версия растёт только здесь, в фоне, а не синхронно на hot path). Это
    не гонка с пользователем — просто два job'а одного хода. Строгий
    abort-on-mismatch (как было раньше) в этом случае выбрасывал честно
    извлечённые факты второго job'а, хотя реального конфликта не было.
    ``merge_manifest`` — аддитивная операция (union списков с dedup, см.
    ``_merge_lists``), поэтому вместо abort здесь всегда мёржим
    ``new_manifest`` поверх СВЕЖЕГО (не устаревшего enqueue-time) текущего
    ``fact_manifest`` и монотонно растим версию. Несовпадение
    ``expected_version`` только логируется — данные всё равно не теряются.
    """
    from knowledge_engine.ui.run_log import trace

    key = _session_key(curriculum_id, node_id)
    with _lock:
        all_data = _load_all()
        entry = all_data.get(key)
        if not entry:
            trace(f"WORKER dialog_summarize CAS ⊘ | {key} | session not found")
            return False
        memory = memory_from_blob(entry.get("memory"))
        if memory is None:
            trace(f"WORKER dialog_summarize CAS ⊘ | {key} | no memory blob")
            return False
        if memory.manifest_version != expected_version:
            trace(
                f"WORKER dialog_summarize CAS merge (stale base) | {key} | "
                f"expected_version={expected_version} actual={memory.manifest_version}"
            )
        memory.fact_manifest = merge_manifest(memory.fact_manifest, new_manifest)
        memory.manifest_version += 1
        entry = dict(entry)
        entry["memory"] = memory_to_blob(memory)
        all_data[key] = entry
        _save_all(all_data)
        return True


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


def _collapse_adjacent_duplicate_users(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """
    Collapse only *adjacent* identical user rows (same normalized text).

    Distinct turns with the same button text (e.g. repeated [mode:lecture]) MUST
    keep separate msg_ids — never dedupe the whole history by content/hash.
    """
    out: list[dict[str, str]] = []
    for m in rows:
        if (
            out
            and (m.get("role") or "").strip() == "user"
            and (out[-1].get("role") or "").strip() == "user"
            and _normalize_user_dialog_content(str(m.get("content") or ""))
            == _normalize_user_dialog_content(str(out[-1].get("content") or ""))
            and _normalize_user_dialog_content(str(m.get("content") or ""))
        ):
            # Keep the higher msg_id at this slot.
            if (parse_msg_id(m) or 0) >= (parse_msg_id(out[-1]) or 0):
                out[-1] = m
            continue
        out.append(m)
    return out


def repair_history_with_memory(
    history: list[dict[str, str]] | None,
    memory: SessionMemory | None,
) -> list[dict[str, str]]:
    """Синхронизирует history с active_window: порядок + msg_id."""
    from knowledge_engine.src.node_deep_dive.dialog_ids import sort_by_msg_id

    hist = clean_dialog_rows(history)
    if not memory or not memory.active_window:
        repaired, seq = reconcile_dialog_history(hist, None, start_seq=0)
        repaired = sort_by_msg_id(_collapse_adjacent_duplicate_users(repaired))
        if memory is not None:
            memory.dialog_seq = max(
                int(memory.dialog_seq or 0), seq, max_msg_id(repaired)
            )
        return repaired
    _repair_memory_dialog_ids(memory)
    window = clean_dialog_rows(memory.active_window)
    start = max(int(memory.dialog_seq or 0), max_msg_id(hist))
    merged, seq = reconcile_dialog_history(hist, window, start_seq=start)
    merged = sort_by_msg_id(_collapse_adjacent_duplicate_users(merged))
    memory.dialog_seq = max(int(memory.dialog_seq or 0), seq, max_msg_id(merged))
    return merged


class _SessionRecord:
    def __init__(
        self,
        node_status: NodeStatus = "unexplored",
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
    if s in ("passed", "skipped"):
        return "passed_by_equivalence"
    return "unexplored"


def _repair_tutor_history_markdown(history: list[dict[str, str]]) -> None:
    for item in history:
        if (item.get("role") or "").strip() != "tutor":
            continue
        raw = str(item.get("content") or "")
        if not raw.strip():
            continue
        item["content"] = repair_llm_display_text(raw)
        item.pop("content_html", None)


def get_session(curriculum_id: str, node_id: str) -> _SessionRecord:
    key = _session_key(curriculum_id, node_id)
    with _lock:
        blob = _load_all().get(key) or {}
    content_raw = blob.get("content") or {}
    content = NodeContentBlock.model_validate(content_raw)
    from knowledge_engine.src.node_deep_dive.content_assets import (
        normalize_node_content_diagrams,
    )

    content = normalize_node_content_diagrams(content)
    history = clean_dialog_rows(blob.get("history") or [])
    memory = memory_from_blob(blob.get("memory"))
    registry = load_covered_subtopics_from_blob(blob if isinstance(blob, dict) else {})
    terms = load_introduced_terms_from_blob(blob if isinstance(blob, dict) else {})
    if memory is None and (registry or terms):
        memory = SessionMemory(
            covered_subtopics=registry,
            introduced_terms=terms,
        )
    elif memory is not None:
        if registry:
            carry_covered_subtopics(registry, memory)
        if terms:
            carry_introduced_terms(terms, memory)
    if memory is not None:
        _repair_memory_dialog_ids(memory)
    history = repair_history_with_memory(history, memory)
    _repair_tutor_history_markdown(history)
    status = _normalize_status(str(blob.get("node_status") or "unexplored"))
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
    if memory is not None:
        from knowledge_engine.src.node_deep_dive.dialog_ids import (
            patch_last_tutor_history_content,
        )
        from knowledge_engine.src.node_deep_dive.tutor_dialogue import (
            recover_tutor_display_from_chat_sessions,
        )
        from knowledge_engine.src.node_deep_dive.tutor_field_limits import (
            SCHEMA_FOLLOW_UP_QUESTION_MAX,
            SCHEMA_TUTOR_MESSAGE_MAX,
        )

        full = (memory.last_tutor_display_message or "").strip()
        if not full:
            full, fu = recover_tutor_display_from_chat_sessions(memory)
            if full:
                memory.last_tutor_display_message = full[:SCHEMA_TUTOR_MESSAGE_MAX]
                if fu:
                    memory.last_tutor_follow_up_question = fu[
                        :SCHEMA_FOLLOW_UP_QUESTION_MAX
                    ]
        if full:
            norm_history = patch_last_tutor_history_content(norm_history, full)
    _repair_tutor_history_markdown(norm_history)
    from knowledge_engine.src.node_deep_dive.content_assets import (
        normalize_node_content_diagrams,
    )

    content = normalize_node_content_diagrams(content)
    content_dump = {
        k: v for k, v in content.model_dump().items() if k != "summary_html"
    }
    mem_blob: dict[str, Any] = {}
    if memory is not None:
        from knowledge_engine.schemas.global_knowledge import utc_now_iso

        memory.memory_updated_at = utc_now_iso()
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
        if memory is not None:
            entry["covered_subtopics"] = _normalize_covered_subtopics(
                memory.covered_subtopics
            )
            entry["introduced_terms"] = normalize_introduced_terms(
                memory.introduced_terms
            )
        if source_registry is not None:
            entry["source_registry"] = list(source_registry)[:12]
        all_data[key] = entry
        _save_all(all_data)
        if memory is not None:
            from knowledge_engine.src.curriculum.global_tracker import (
                commit_session_and_global_registry,
            )

            commit_session_and_global_registry(
                "default",
                curriculum_id,
                node_id,
                memory,
                session_updated_at=entry["updated_at"],
                node_title="",
            )
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
            out[node_id] = _normalize_status(
                str(blob.get("node_status") or "unexplored")
            )
    return out


def discover_curriculum_ids_from_sessions() -> set[str]:
    """ID маршрутов из ключей сессий (curriculum_id::node_id)."""
    ids: set[str] = set()
    with _lock:
        for key in _load_all().keys():
            if "::" in key:
                ids.add(key.split("::", 1)[0].strip())
    return ids


def clear_node_session(curriculum_id: str, node_id: str) -> bool:
    """Удалить запись сессии ноды (контент, память, история, прогресс)."""
    key = _session_key(curriculum_id, node_id)
    with _lock:
        all_data = _load_all()
        if key not in all_data:
            return False
        del all_data[key]
        _save_all(all_data)
    return True
