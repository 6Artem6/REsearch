"""Полный сброс персистентной сессии ноды перед повторным init."""

from __future__ import annotations

from knowledge_engine.services.gemini_cache_manager import registry_purge_for_anchor
from knowledge_engine.src.node_deep_dive.session_store import clear_node_session
from knowledge_engine.ui.run_log import trace


def node_deep_dive_anchor(curriculum_id: str, node_id: str) -> str:
    return f"node_deep_dive:{curriculum_id.strip()}:{node_id.strip()}"


def reset_node_deep_dive_persistence(curriculum_id: str, node_id: str) -> bool:
    """
    Удалить JSON-сессию ноды и кэши Gemini для anchor.
    Возвращает True, если запись сессии существовала.
    """
    anchor = node_deep_dive_anchor(curriculum_id, node_id)
    had = clear_node_session(curriculum_id, node_id)
    registry_purge_for_anchor(anchor)
    trace(
        f"NODE_SESSION reset ✓ | {curriculum_id}/{node_id} "
        f"had_session={had} anchor={anchor}"
    )
    return had
