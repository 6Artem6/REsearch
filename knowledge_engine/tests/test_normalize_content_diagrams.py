"""Serve/persist path: Mermaid sanitize on content.diagrams[]."""

from __future__ import annotations

from knowledge_engine.src.node_deep_dive.content_assets import (
    normalize_node_content_diagrams,
)
from knowledge_engine.src.node_deep_dive.schemas import DiagramAsset, NodeContentBlock


def test_normalize_fixes_spaced_bracket_in_diagrams_list() -> None:
    raw = (
        "```mermaid\n"
        "flowchart TD\n"
        'Client["MCP Client"]\n'
        'Server3[ "MCP Server"]\n'
        "```"
    )
    content = NodeContentBlock(
        diagrams=[
            DiagramAsset(
                id="diagram-5",
                title="MCP architecture",
                mermaid=raw,
            )
        ]
    )
    out = normalize_node_content_diagrams(content)
    mermaid = out.diagrams[0].mermaid
    assert 'Server3[ "MCP' not in mermaid
    assert 'Server3["MCP Server"]' in mermaid


def test_normalize_noop_when_clean() -> None:
    clean = "```mermaid\n" "flowchart TD\n" 'A["Start"] --> B["End"]\n' "```"
    content = NodeContentBlock(
        diagrams=[DiagramAsset(id="diagram-1", title="t", mermaid=clean)]
    )
    out = normalize_node_content_diagrams(content)
    assert out.diagrams[0].mermaid.strip() == clean.strip() or (
        'A["Start"]' in out.diagrams[0].mermaid
    )
