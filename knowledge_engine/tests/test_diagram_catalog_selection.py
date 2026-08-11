"""Catalog-only diagram selection: no LLM Mermaid append."""

from __future__ import annotations

from knowledge_engine.src.node_deep_dive.content_assets import (
    merge_content_assets,
    resolve_referenced_diagram,
)
from knowledge_engine.src.node_deep_dive.schemas import DiagramAsset, NodeContentBlock


def _catalog_content() -> NodeContentBlock:
    return NodeContentBlock(
        diagram="flowchart TD\n  A-->B",
        diagrams=[
            DiagramAsset(
                id="diagram-1",
                title="Pipeline",
                mermaid="flowchart TD\n  A-->B",
            ),
            DiagramAsset(
                id="diagram-2",
                title="Sequence",
                mermaid="sequenceDiagram\n  Alice->>Bob: hi",
            ),
        ],
    )


def test_resolve_referenced_diagram_by_id_and_index():
    content = _catalog_content()
    assert resolve_referenced_diagram(content, "diagram-2").id == "diagram-2"
    assert resolve_referenced_diagram(content, "2").id == "diagram-2"
    assert resolve_referenced_diagram(content, "diagram:diagram-1").id == "diagram-1"
    assert resolve_referenced_diagram(content, "missing") is None
    assert resolve_referenced_diagram(content, None) is None
    assert resolve_referenced_diagram(NodeContentBlock(), "diagram-1") is None


def test_merge_selects_catalog_diagram_without_appending():
    prev = _catalog_content()
    before_n = len(prev.diagrams)
    out = merge_content_assets(prev, referenced_diagram_id="diagram-2")
    assert len(out.diagrams) == before_n
    assert out.diagrams[0].id == "diagram-2"
    assert "Alice->>Bob" in out.diagram
    assert "Hallucination" not in out.diagram
    # Panel current matches selected asset body (may be fence-normalized).
    assert "sequenceDiagram" in out.diagram


def test_merge_ignores_raw_mermaid_and_unknown_id():
    prev = _catalog_content()
    before_ids = [d.id for d in prev.diagrams]
    hallucinated = "flowchart LR\n  Fake-->Hallucination"
    out = merge_content_assets(
        prev,
        referenced_diagram_id="diagram-99",
        diagram=hallucinated,  # deprecated kw; must be ignored
    )
    assert [d.id for d in out.diagrams] == before_ids
    assert "Hallucination" not in (out.diagram or "")
    assert out.diagram == prev.diagram


def test_dialogue_and_lecture_prompts_forbid_mermaid_generation():
    from knowledge_engine.src.node_deep_dive.lecture_prompt_en import (
        DIAGRAM_SELECTION_RULES,
        LECTURE_SYSTEM_PROMPT,
        PINNED_DIAGRAMS_GUIDING_RULES,
    )
    from knowledge_engine.src.node_deep_dive.tutor_prompt_builder import (
        build_dialogue_system,
    )

    assert "NEVER generate or write raw Mermaid" in DIAGRAM_SELECTION_RULES
    assert "referenced_diagram_id" in DIAGRAM_SELECTION_RULES
    assert DIAGRAM_SELECTION_RULES in LECTURE_SYSTEM_PROMPT
    assert "referenced_diagram_id" in PINNED_DIAGRAMS_GUIDING_RULES
    assert "never invent Mermaid" in PINNED_DIAGRAMS_GUIDING_RULES
    dialogue = build_dialogue_system()
    assert "NEVER generate or write raw Mermaid" in dialogue
    assert "referenced_diagram_id" in dialogue
