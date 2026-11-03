"""Хранилище незавершённых Node Grounding Gate (``.runs/node_gate_state.json``).

Кандидаты Gate 1 и дайджесты Gate 2 стоят десятков секунд поиска и
LLM-вызовов; без хранилища перезагрузка страницы запускала discover заново.
Ключ — ``curriculum_id::node_id``. Пишет worker (``node_grounding_jobs``),
читает роут discover (процесс API) — поэтому файл, а не память процесса.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone

from knowledge_engine.src.config.settings import PACKAGE_ROOT
from knowledge_engine.src.shared.node_grounding.node_gate_contracts import (
    NodeGateState,
)

_STORE_PATH = PACKAGE_ROOT / ".runs" / "node_gate_state.json"
_lock = threading.RLock()


def _key(curriculum_id: str, node_id: str) -> str:
    return f"{curriculum_id.strip()}::{node_id.strip()}"


def _load_all() -> dict[str, dict]:
    if not _STORE_PATH.is_file():
        return {}
    try:
        raw = json.loads(_STORE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _save_all(data: dict[str, dict]) -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _STORE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(_STORE_PATH)  # API читает файл параллельно с worker


def get_gate_state(curriculum_id: str, node_id: str) -> NodeGateState | None:
    with _lock:
        raw = _load_all().get(_key(curriculum_id, node_id))
    if not raw:
        return None
    try:
        return NodeGateState.model_validate(raw)
    except ValueError:
        return None  # запись старого формата — как будто её нет


def save_gate_state(state: NodeGateState) -> None:
    stamped = state.model_copy(
        update={"updated_at": datetime.now(timezone.utc).isoformat()}
    )
    with _lock:
        data = _load_all()
        data[_key(state.curriculum_id, state.node_id)] = stamped.model_dump()
        _save_all(data)


def clear_gate_state(curriculum_id: str, node_id: str) -> bool:
    with _lock:
        data = _load_all()
        existed = data.pop(_key(curriculum_id, node_id), None) is not None
        if existed:
            _save_all(data)
    return existed


__all__ = ["clear_gate_state", "get_gate_state", "save_gate_state"]
