"""Singleton compiled graph для CLI/API (общий MemorySaver)."""

from __future__ import annotations

from typing import Any

_graph: Any = None


def get_compiled_graph() -> Any:
    global _graph
    if _graph is None:
        from knowledge_engine.graph import build_graph

        _graph = build_graph()
    return _graph
