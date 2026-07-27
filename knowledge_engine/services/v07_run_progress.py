"""Публикация частичного state v0.8 в web run store для пошагового UI."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from knowledge_engine.services.v07_run_store import V07RunStatus, v07_run_store


def _json_ready(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_json_ready(x) for x in value]
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    return value


def publish_web_run_progress(
    web_run_id: str | None,
    current_step: str,
    state: Mapping[str, Any],
    *,
    keys: Sequence[str] | None = None,
) -> None:
    """Слить поля state в run.result и обновить current_step (для poll /view)."""
    if not web_run_id:
        return
    patch: dict[str, Any] = {"current_step": current_step}
    if keys is not None:
        for key in keys:
            if key in state:
                patch[key] = _json_ready(state[key])
    else:
        for key, val in state.items():
            if val is not None and key != "current_step":
                patch[key] = _json_ready(val)
    v07_run_store.merge_result(web_run_id, patch, current_step=current_step)
    if current_step == "completed":
        v07_run_store.update(web_run_id, status=V07RunStatus.COMPLETED)
