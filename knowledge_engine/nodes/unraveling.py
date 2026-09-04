"""Compatibility shim — unified Unraveling lives in graph.nodes.unraveling."""

from knowledge_engine.graph.nodes.unraveling import (
    render_unraveling_markdown,
    unraveling_node,
)

__all__ = ["unraveling_node", "render_unraveling_markdown"]
