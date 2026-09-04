"""Эмиссия FSM stage-progress событий из узлов графа тьютора (см. prompt.txt:
"Интеграция FSM-статусов LangGraph c SSE-стримингом").

Паттерн — тот же, что уже используется для token-стриминга
(tutor_generate.py::_stream_from_config): callback лежит в
config["configurable"]["stage_callback"], не отдельный параметр вызова
графа — иначе он не переживёт resume-путь
(tutor_graph_service.py::run_or_resume передаёт ВЕСЬ config в
ainvoke(None, config=config) при восстановлении после падения).

stage_scope() — контекст-менеджер, оборачивающий ВСЁ тело узла: эмитит
RUNNING на входе, COMPLETED на успешном выходе (в т.ч. при early return
внутри with-блока — обычный Python-механизм), FAILED с пробросом
исключения дальше при ошибке. НЕ enforced timeout/abort — elapsed_sec
чисто информационный (см. schemas/fsm.py: реальные LLM-узлы стабильно
занимают 40-90s, per-node wait_for ложно бы срабатывал).

Инструментированы только узлы с реальным пользователь-заметным ожиданием
(I/O, LLM) — ingest_node/coverage_router_node/commit_turn_node — быстрые
in-memory операции, эмиссия под них добавила бы только шум в SSE-поток без
пользы для UI. SUMMARIZE-стадия из TutorStage сейчас никогда не
эмитится графом: суммаризация вынесена в фоновый job
(context_compressor_worker.py, отдельная сессия) и больше не блокирует ход —
это осознанный пробел, не забытая стадия.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from typing import Any

from knowledge_engine.schemas.fsm import FSMStatus, StageProgressEvent, TutorStage
from knowledge_engine.ui.run_log import trace


def _stage_callback_from_config(config: dict[str, Any] | None) -> Any:
    if not config:
        return None
    return (config.get("configurable") or {}).get("stage_callback")


def _turn_started_at(config: dict[str, Any] | None) -> float:
    if config:
        started = (config.get("configurable") or {}).get("turn_started_at")
        if isinstance(started, (int, float)):
            return float(started)
    return time.monotonic()


def _session_id_from_state(state: dict[str, Any]) -> str:
    req = state.get("request") if isinstance(state, dict) else None
    if req is None:
        return ""
    cid = str(getattr(req, "curriculum_id", "") or "").strip()
    node = getattr(req, "node_data", None)
    nid = str(getattr(node, "node_id", "") or "").strip()
    if not cid and not nid:
        return ""
    return f"{cid}/{nid}"


def emit_stage(
    state: dict[str, Any],
    config: dict[str, Any] | None,
    stage: TutorStage,
    status: FSMStatus,
    message: str,
    **payload: Any,
) -> None:
    """Best-effort — любая ошибка эмиссии не должна ронять узел графа."""
    callback = _stage_callback_from_config(config)
    if callback is None:
        return
    try:
        event = StageProgressEvent(
            session_id=_session_id_from_state(state),
            stage=stage,
            status=status,
            message=message,
            elapsed_sec=round(time.monotonic() - _turn_started_at(config), 3),
            payload=payload or None,
        )
        callback(event)
    except Exception as exc:
        trace(f"FSM stage_callback ✗ | {stage.value}/{status.value} | {exc}")


@contextlib.contextmanager
def stage_scope(
    state: dict[str, Any],
    config: dict[str, Any] | None,
    stage: TutorStage,
    *,
    running_message: str,
    done_message: str | None = None,
) -> Iterator[None]:
    emit_stage(state, config, stage, FSMStatus.RUNNING, running_message)
    try:
        yield
    except Exception as exc:
        emit_stage(
            state,
            config,
            stage,
            FSMStatus.FAILED,
            f"Ошибка на этапе «{stage.value}»",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    emit_stage(
        state, config, stage, FSMStatus.COMPLETED, done_message or running_message
    )
