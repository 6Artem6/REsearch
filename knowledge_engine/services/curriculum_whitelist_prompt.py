"""Текст белого списка для промптов Curriculum / Node Deep-Dive."""

from __future__ import annotations

from typing import Any

from knowledge_engine.config import CURRICULUM_DEEP_NODE_MAX_HITS
from knowledge_engine.src.curriculum.schemas import (
    LearningMaterials,
    NodeCurriculumBreakdown,
    NodeSourceRef,
    PrimaryWhitelistSource,
)
from knowledge_engine.src.curriculum.source_registry import format_resolved_sources_for_lecture
from knowledge_engine.src.source_evaluator.evaluator import format_whitelist_for_reasoner_prompt


def curriculum_whitelist_prompt_block() -> str:
    return (
        "ОРИЕНТИРЫ АВТОРИТЕТНЫХ ИСТОЧНИКОВ (стартовый whitelist + пополняемый архив):\n"
        f"{format_whitelist_for_reasoner_prompt()}\n\n"
        "ТРЕБОВАНИЯ К ИСТОЧНИКАМ:\n"
        "1. Предлагай глубокие инженерные материалы; домен может быть вне списка — "
        "он пройдёт Lite-валидацию и пополнит архив.\n"
        "2. Для нод с Search-First используй source_ref и выдержки из уже проверенных материалов.\n"
        "3. core_concepts — из содержания статьи/главы, не из SEO-обзоров.\n"
    )


def format_node_lesson_plan_for_lecture(node: Any, curriculum_id: str = "") -> str:
    """План урока из Flash: breakdown + выдержки source_ref (Search-First)."""
    node = _merge_node_plan_from_graph(node, curriculum_id)
    bd = getattr(node, "node_curriculum_breakdown", None)
    sr = getattr(node, "source_ref", None)
    if bd is None and sr is None:
        return ""

    key_concepts: list[str] = []
    arch = ""
    if bd is not None:
        key_concepts = list(getattr(bd, "key_concepts", None) or [])[:24]
        arch = (getattr(bd, "architectural_focus", None) or "").strip()
    if not key_concepts:
        key_concepts = list(getattr(node, "core_concepts", None) or [])[:8]

    url = (getattr(sr, "url", None) or "").strip()
    extracts = list(getattr(sr, "relevant_extracts", None) or [])[:12]
    extract_text = "\n".join(f"- {e}" for e in extracts if e)

    lines = [
        "Ты — IT-Тьютор.\n",
        f"ТЕМА УРОКА: {getattr(node, 'title', '')}",
        f"КЛЮЧЕВЫЕ ПОНЯТИЯ ДЛЯ РАЗБОРА: {', '.join(key_concepts)}",
    ]
    if arch:
        lines.append(f"АРХИТЕКТУРНЫЙ ФОКУС: {arch}")
    if url or extract_text:
        lines.append("\nИСТОЧНИК И ВЫДЕРЖКИ ИЗ СТАТЬИ/ИССЛЕДОВАНИЯ:")
        if url:
            lines.append(f"- Ссылка: {url}")
        if extract_text:
            lines.append(f"- Текст выдержки:\n{extract_text}")
    lines.append(
        "\nЗАДАЧА:\n"
        "Напиши подробную, плотную лекцию (от 400 слов). Объясни все ключевые понятия урока, "
        "опираясь напрямую на текст выдержки из источника выше."
    )
    return "\n".join(lines)


def _merge_node_plan_from_graph(node: Any, curriculum_id: str) -> Any:
    cid = (curriculum_id or "").strip()
    if not cid:
        return node
    from knowledge_engine.services.skill_tree_store import get_curriculum_graph

    graph = get_curriculum_graph(cid)
    if not graph:
        return node
    for raw in graph.get("nodes") or []:
        if str(raw.get("node_id") or "") != getattr(node, "node_id", ""):
            continue
        updates: dict[str, Any] = {}
        if raw.get("source_ref"):
            try:
                updates["source_ref"] = NodeSourceRef.model_validate(raw["source_ref"])
            except Exception:
                pass
        if raw.get("node_curriculum_breakdown"):
            try:
                updates["node_curriculum_breakdown"] = NodeCurriculumBreakdown.model_validate(
                    raw["node_curriculum_breakdown"]
                )
            except Exception:
                pass
        if updates:
            return node.model_copy(update=updates)
        return node
    return node


_TUTOR_NEIGHBORHOOD_RULES = (
    "Маршрут и окружение (блок NEIGHBORHOOD_CONTEXT в user payload):\n"
    "1. Связь с прошлым: при объяснении опирайся на концепты из «ПРЕДШЕСТВУЮЩИХ ТЕМ», "
    "если это облегчает понимание.\n"
    "2. Граница будущего: НЕ объясняй подробно концепты из «СЛЕДУЮЩИХ ТЕМ». "
    "Если пользователь спрашивает про них — кратко ответь и укажи, что детально "
    "это будет на следующем шаге маршрута.\n"
)


def format_neighborhood_context_block(curriculum_id: str, node_id: str) -> str:
    cid = (curriculum_id or "").strip()
    nid = (node_id or "").strip()
    if not cid or not nid:
        return ""
    from knowledge_engine.services.skill_tree_store import get_node_neighbors_context

    ctx = get_node_neighbors_context(cid, nid)
    if not ctx:
        return ""

    current_title = (ctx.get("current_title") or nid).strip()
    preds = ctx.get("predecessors") or []
    succs = ctx.get("successors") or []
    if not preds and not succs:
        return ""

    lines = [
        "=== МАРШРУТ И ОКРУЖЕНИЕ ТЕМЫ ===",
        "ПРЕДШЕСТВУЮЩИЕ ТЕМЫ (пользователь уже прошёл фундаментальные вещи):",
    ]
    if preds:
        for p in preds:
            title = (p.get("title") or p.get("node_id") or "").strip()
            concepts = (p.get("short_concepts") or "").strip()
            suffix = f" ({concepts})" if concepts else ""
            lines.append(f"- [Prereq]: {title}{suffix}")
    else:
        lines.append("- (нет явных предшественников в графе)")

    lines.append("")
    lines.append("ТЕКУЩАЯ ТЕМА (фокус текущего диалога):")
    lines.append(f"- {current_title}")

    lines.append("")
    lines.append(
        "СЛЕДУЮЩИЕ ТЕМЫ (НЕ углубляйся сюда — это темы будущих уроков):"
    )
    if succs:
        for s in succs:
            title = (s.get("title") or s.get("node_id") or "").strip()
            lines.append(f"- [Next]: {title}")
    else:
        lines.append("- (нет явных следующих тем в графе)")

    return "\n".join(lines)


def format_node_curriculum_context_for_tutor(node: Any, curriculum_id: str = "") -> str:
    """Контекст ноды для chat-тьютора (без дублирования роли из system prompt)."""
    parts: list[str] = []
    nid = str(getattr(node, "node_id", "") or "").strip()
    neighborhood = format_neighborhood_context_block(curriculum_id, nid)
    if neighborhood.strip():
        parts.append(neighborhood.strip())

    plan = format_node_lesson_plan_for_lecture(node, curriculum_id)
    if plan.strip():
        body = plan.replace("Ты — IT-Тьютор.\n", "").replace("Ты — IT-Тьютор.", "").strip()
        if body:
            parts.append(body)

    if not parts:
        return ""
    return "\n\n".join(parts)


def format_primary_whitelist_foundation(node: Any, curriculum_id: str = "") -> str:
    lesson = format_node_lesson_plan_for_lecture(node, curriculum_id)
    if lesson.strip():
        return lesson + "\n"
    foundation = format_resolved_sources_for_node_from_node(node, curriculum_id)
    if foundation.strip():
        return foundation + "\n"
    lm = getattr(node, "learning_materials", None)
    if not lm or not getattr(lm, "primary_whitelist_source", None):
        return ""
    p = lm.primary_whitelist_source
    concepts = "\n".join(f"  - {c}" for c in p.core_concepts[:12])
    return (
        "ФУНДАМЕНТАЛЬНЫЙ ИСТОЧНИК УРОКА (из белого списка маршрута):\n"
        f"- Ресурс: {p.source_name.strip()}\n"
        f"- Раздел/глава: {p.chapter_or_article.strip()}\n"
        f"- Ключевые концепции источника:\n{concepts}\n"
    )


def format_resolved_sources_for_node_from_node(node: Any, curriculum_id: str) -> str:
    cid = (curriculum_id or "").strip()
    if not cid:
        return ""
    from knowledge_engine.services.skill_tree_store import get_curriculum_graph

    graph = get_curriculum_graph(cid)
    if not graph:
        return ""
    nid = getattr(node, "node_id", "") or ""
    mapped = list(getattr(node, "mapped_source_ids", None) or [])
    return format_resolved_sources_for_lecture(graph, nid, mapped)


def enrich_node_learning_materials_from_graph(
    node: Any,
    curriculum_id: str,
) -> Any:
    """Подтянуть learning_materials из сохранённого графа, если UI не передал."""
    has_plan = bool(
        getattr(node, "source_ref", None) or getattr(node, "node_curriculum_breakdown", None)
    )
    if not has_plan:
        lm = getattr(node, "learning_materials", None)
        has_lm = lm and getattr(lm, "primary_whitelist_source", None)
        has_mapped = bool(getattr(node, "mapped_source_ids", None))
        if has_lm and has_mapped:
            return node
    cid = (curriculum_id or "").strip()
    if not cid:
        return node
    from knowledge_engine.services.skill_tree_store import get_curriculum_graph

    graph = get_curriculum_graph(cid)
    if not graph:
        return node
    for raw in graph.get("nodes") or []:
        if str(raw.get("node_id") or "") != getattr(node, "node_id", ""):
            continue
        updates: dict[str, Any] = {}
        if raw.get("mapped_source_ids"):
            updates["mapped_source_ids"] = list(raw.get("mapped_source_ids") or [])[
                :CURRICULUM_DEEP_NODE_MAX_HITS
            ]
        if raw.get("learning_goal"):
            updates["learning_goal"] = str(raw.get("learning_goal") or "")[:600]
        if raw.get("primary_source_id"):
            updates["primary_source_id"] = str(raw.get("primary_source_id") or "")[:16]
        if raw.get("source_ref"):
            try:
                updates["source_ref"] = NodeSourceRef.model_validate(raw["source_ref"])
            except Exception:
                pass
        if raw.get("node_curriculum_breakdown"):
            try:
                updates["node_curriculum_breakdown"] = NodeCurriculumBreakdown.model_validate(
                    raw["node_curriculum_breakdown"]
                )
            except Exception:
                pass
        pws = primary_whitelist_from_graph_node(raw)
        if pws:
            updates["learning_materials"] = LearningMaterials(primary_whitelist_source=pws)
        if updates:
            return node.model_copy(update=updates)
        return node
    return node


def primary_whitelist_from_graph_node(raw: dict) -> PrimaryWhitelistSource | None:
    lm = raw.get("learning_materials") or {}
    if not isinstance(lm, dict):
        return None
    p = lm.get("primary_whitelist_source")
    if not isinstance(p, dict):
        return None
    try:
        return PrimaryWhitelistSource.model_validate(p)
    except Exception:
        return None
