"""Compatibility shim — unified Unraveling lives in graph.nodes.unraveling."""

from knowledge_engine.src.legacy.graph.nodes.unraveling import (
    unraveling_node as unraveling_node_v04,
)

__all__ = ["unraveling_node_v04"]
