"""TOC → section heading для MAP-окон."""

from __future__ import annotations

from knowledge_engine.services.article_ingestion.annotated_article_ops import (
    _norm_p,
    sorted_p_ids,
)
from knowledge_engine.services.article_ingestion.triage_schemas import TOCNode
from knowledge_engine.services.parsers.html_annotator import AnnotatedArticle


def _p_sort_key(pid: str) -> int:
    t = _norm_p(pid)
    try:
        return int(t.split("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def build_p_index(paragraph_map: dict[str, str]) -> dict[str, int]:
    ordered = sorted_p_ids(paragraph_map)
    return {pid: i for i, pid in enumerate(ordered)}


def resolve_section_heading_for_paragraph_ids(
    paragraph_ids: list[str],
    toc_nodes: list[TOCNode],
    *,
    paragraph_map: dict[str, str] | None = None,
) -> str:
    if not paragraph_ids:
        return ""
    idx = build_p_index(paragraph_map or {})
    if not idx:
        idx = {_norm_p(p): _p_sort_key(p) for p in paragraph_ids}

    anchor_p = min(paragraph_ids, key=lambda p: idx.get(_norm_p(p), _p_sort_key(p)))
    anchor_i = idx.get(_norm_p(anchor_p), _p_sort_key(anchor_p))

    best: TOCNode | None = None
    best_i = -1
    for node in toc_nodes:
        sp = _norm_p(node.start_p_id)
        si = idx.get(sp, -1)
        if si < 0 or si > anchor_i:
            continue
        if si >= best_i:
            best_i = si
            best = node
    if best is not None and (best.title or "").strip():
        return (best.title or "").strip()[:300]
    return ""


def infer_article_title(
    *,
    annotated: AnnotatedArticle,
    toc_nodes: list[TOCNode],
    source_id: str = "",
    page_url: str = "",
    explicit_title: str = "",
) -> str:
    explicit = (explicit_title or "").strip()
    if explicit and not explicit.lower().startswith("http"):
        return explicit[:300]

    by_start = sorted(
        toc_nodes,
        key=lambda n: (_p_sort_key(n.start_p_id), n.level),
    )
    for node in by_start:
        if node.level == 1 and (node.title or "").strip():
            return (node.title or "").strip()[:300]
    for node in by_start:
        if (node.title or "").strip():
            return (node.title or "").strip()[:300]

    order = sorted_p_ids(annotated.paragraph_map)
    if order:
        first_text = (annotated.paragraph_map.get(order[0]) or "").strip()
        if 8 <= len(first_text) <= 200 and not first_text.endswith("."):
            return first_text[:300]

    sid = (source_id or "").strip()
    if sid and not sid.lower().startswith("http") and "://" not in sid:
        return sid[:300]
    return (page_url or annotated.page_url or "")[:300]
