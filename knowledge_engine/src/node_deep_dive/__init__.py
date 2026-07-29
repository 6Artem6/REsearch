"""Модуль 2 — движок глубокой проработки ноды."""

from __future__ import annotations

from knowledge_engine.src.node_deep_dive.schemas import (
    NodeDeepDiveRequest,
    NodeDeepDiveResponse,
)

__all__ = [
    "NodeDeepDiveRequest",
    "NodeDeepDiveResponse",
    "run_node_deep_dive",
]


def __getattr__(name: str):
    if name == "run_node_deep_dive":
        from knowledge_engine.src.node_deep_dive.engine import run_node_deep_dive

        return run_node_deep_dive
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
