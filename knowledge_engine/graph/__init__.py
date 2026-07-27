"""Выбор графа Knowledge Engine (v0.2 / v0.3 / v0.4)."""

from __future__ import annotations

import knowledge_engine.config as ke_config


def build_graph():
    version = (ke_config.GRAPH_VERSION or "0.4").strip()
    if version == "0.2":
        from knowledge_engine.graph.v02 import build_graph_v02

        return build_graph_v02()
    if version == "0.3":
        from knowledge_engine.graph.v03 import build_graph_v03

        return build_graph_v03()
    if version == "0.4":
        from knowledge_engine.graph.v04 import build_graph_v04

        return build_graph_v04()
    if version == "0.8":
        from knowledge_engine.src.graph import compile_v07_graph

        return compile_v07_graph()
    if version == "0.7":
        from knowledge_engine.src.graph import compile_v07_graph

        return compile_v07_graph()
    raise ValueError(
        f"Неизвестный GRAPH_VERSION={version!r} (0.2 | 0.3 | 0.4 | 0.7 | 0.8)"
    )
