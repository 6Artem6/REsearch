"""Схемы из article_diagrams → content.diagrams в сессии ноды."""

from __future__ import annotations

from knowledge_engine.src.curriculum.schemas import CurriculumNode
from knowledge_engine.src.node_deep_dive.content_assets import (
    _diagrams_from_block,
    hydrate_content_diagrams_from_articles,
)
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput
from knowledge_engine.src.node_deep_dive.session_store import get_session, save_session


def curriculum_node_to_data_input(
    node: CurriculumNode | NodeDataInput | dict,
) -> NodeDataInput:
    if isinstance(node, NodeDataInput):
        return node
    if isinstance(node, dict):
        node = CurriculumNode.model_validate(node)
    return NodeDataInput.model_validate(
        {
            "node_id": node.node_id,
            "title": node.title,
            "layer": node.layer,
            "category": node.category,
            "brief_summary": node.brief_summary,
            "core_concepts": list(node.core_concepts or []),
            "prerequisites": list(node.prerequisites or []),
            "mapped_source_ids": list(node.mapped_source_ids or []),
            "primary_source_id": node.primary_source_id or "",
            "source_ref": node.source_ref,
            "node_curriculum_breakdown": node.node_curriculum_breakdown,
            "learning_goal": node.learning_goal or "",
            "learning_materials": node.learning_materials,
        }
    )


def refresh_node_session_diagrams_from_articles(
    curriculum_id: str,
    node: CurriculumNode | NodeDataInput,
    *,
    extra_urls: list[str] | None = None,
    rebuild: bool = False,
) -> int:
    """
    Подтянуть VLM-схемы в session.content.diagrams (mapped + extra_urls ingest).
    rebuild=True — заменить article-диаграммы (после смены mapped/grounding), не только append.
    Возвращает число диаграмм в content после обновления.
    """
    cid = (curriculum_id or "").strip()
    if isinstance(node, CurriculumNode):
        nd = curriculum_node_to_data_input(node)
    else:
        nd = node
    nid = (nd.node_id or "").strip()
    if not cid or not nid:
        return 0

    session = get_session(cid, nid)
    before = len(_diagrams_from_block(session.content))
    base = session.content
    if rebuild:
        base = base.model_copy(update={"diagrams": [], "diagram": ""})
    content = hydrate_content_diagrams_from_articles(
        base,
        nd,
        cid,
        extra_urls=extra_urls,
    )
    after = len(_diagrams_from_block(content))
    should_save = (rebuild and after != before) or (not rebuild and after > before)
    if should_save:
        save_session(
            cid,
            nid,
            session.node_status,
            content,
            session.history,
            memory=session.memory,
        )
    return after
