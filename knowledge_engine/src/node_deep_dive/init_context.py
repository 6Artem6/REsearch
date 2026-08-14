"""Lazy intro: контекст первого вопроса, DAG, fast-track."""

from __future__ import annotations

from knowledge_engine.services.curriculum_whitelist_prompt import (
    format_neighborhood_context_block,
)
from knowledge_engine.services.skill_tree_store import get_node_neighbors_context
from knowledge_engine.src.node_deep_dive.memory_schemas import SessionMemory
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput
from knowledge_engine.src.node_deep_dive.session_store import get_session
from knowledge_engine.src.node_deep_dive.tiered_memory import (
    build_tiered_context_payload,
)
from knowledge_engine.src.node_deep_dive.tutor_behavior_state import (
    build_tutor_behavior_state,
    format_tutor_behavior_state_block,
)
from knowledge_engine.src.node_deep_dive.user_mastery_profile import (
    format_competency_pinned_block,
    get_curriculum_competency_profile,
    mastered_entities_set,
    rebuild_competency_profile_from_sessions,
)

_BEGIN_MARKERS = (
    "начать",
    "start",
    "[begin]",
    "приступим",
    "давай",
    "go",
)

_SKIP_MARKERS = (
    "уже знаю",
    "знаю тему",
    "пропустить",
    "пропусти ноду",
    "не нужно",
    "equivalence",
    "passed",
)

_FAST_TRACK_RATIO = 0.7


def is_begin_user_message(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    if t in _BEGIN_MARKERS or t.startswith("[begin]"):
        return True
    return any(m in t for m in ("начать", "приступ", "start lesson"))


def user_declines_node_equivalence(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    return any(m in t for m in _SKIP_MARKERS)


def _normalize_concept_key(text: str) -> str:
    return " ".join((text or "").lower().split())[:200]


def _node_knowledge_entities(node: NodeDataInput) -> list[str]:
    out: list[str] = []
    for c in node.core_concepts or []:
        key = _normalize_concept_key(str(c))
        if key:
            out.append(key)
    title = _normalize_concept_key(node.title or "")
    if title:
        out.append(title)
    return out


def fast_track_overlap_ratio(
    node: NodeDataInput,
    mastery_map: dict[str, float],
    *,
    threshold: float = 0.55,
) -> float:
    entities = _node_knowledge_entities(node)
    if not entities:
        return 0.0
    mastered = mastered_entities_set(mastery_map, threshold=threshold)
    if not mastered:
        return 0.0
    hit = 0
    for ent in entities:
        if ent in mastered:
            hit += 1
            continue
        for m in mastered:
            if ent in m or m in ent:
                hit += 1
                break
    return hit / len(entities)


def build_fast_track_tutor_message(
    node: NodeDataInput, overlap_entities: list[str]
) -> str:
    sample = ", ".join(overlap_entities[:6]) or node.title
    return (
        f"Судя по нашим прошлым обсуждениям, ты уже знаком с: {sample}. "
        "Хочешь сразу переключиться на глубокий кейс/практику "
        "или всё же сделаем экспресс-проверку? "
        "(Ответь «практика» / «проверка» или «уже знаю тему — пропустить».)"
    )


def format_parent_nodes_summary(curriculum_id: str, node_id: str) -> str:
    """Статус и концепты предшественников из сессий."""
    ctx = get_node_neighbors_context(curriculum_id, node_id)
    preds = ctx.get("predecessors") or []
    if not preds:
        return "(нет предшественников в графе)"
    lines: list[str] = []
    for p in preds:
        pid = str(p.get("node_id") or "").strip()
        title = str(p.get("title") or pid).strip()
        sess = get_session(curriculum_id, pid)
        st = sess.node_status
        concepts = str(p.get("short_concepts") or "").strip()
        mem = sess.memory
        verified: list[str] = []
        if mem is not None:
            for row in mem.concepts_matrix:
                if row.status == "verified" or row.mastery_score >= 50:
                    verified.append(row.concept)
        extra = ", ".join(verified[:5])
        bits = [f"{title} [status={st}]"]
        if concepts:
            bits.append(f"концепты: {concepts}")
        if extra:
            bits.append(f"усвоено: {extra}")
        lines.append("- " + "; ".join(bits))
    return "\n".join(lines)


def format_child_nodes_summary(curriculum_id: str, node_id: str) -> str:
    ctx = get_node_neighbors_context(curriculum_id, node_id)
    succs = ctx.get("successors") or []
    if not succs:
        return "(следующие темы не заданы)"
    return ", ".join(
        str(s.get("title") or s.get("node_id") or "").strip() for s in succs[:8]
    )


def regenerate_node_init_context(
    curriculum_id: str,
    node: NodeDataInput,
    memory: SessionMemory,
    *,
    user_message: str = "",
    refresh_mastery: bool = True,
) -> str:
    """
    Собрать payload для intro / re-eval с актуальным сквозным профилем и DAG.
    """
    if refresh_mastery:
        rebuild_competency_profile_from_sessions(curriculum_id)
    profile = get_curriculum_competency_profile(curriculum_id)

    neighborhood = format_neighborhood_context_block(curriculum_id, node.node_id)
    parent_summary = format_parent_nodes_summary(curriculum_id, node.node_id)
    child_summary = format_child_nodes_summary(curriculum_id, node.node_id)
    mastery_block = format_competency_pinned_block(node, curriculum_id, profile=profile)
    diagram_block = ""
    from knowledge_engine.services.article_diagram_context import (
        build_pinned_diagrams_for_node,
    )

    diagram_block = build_pinned_diagrams_for_node(node, curriculum_id)

    behavior = build_tutor_behavior_state(
        "ANSWER",
        "begin",
        memory.learning_mode,
        memory.learning_phase,
        user_message,
        has_user_focus=False,
        memory=memory,
        node_layer=str(getattr(node, "layer", "") or ""),
    )
    behavior_block = format_tutor_behavior_state_block(behavior)

    base = build_tiered_context_payload(
        memory,
        node,
        "ANSWER",
        "begin",
        behavior_block,
        user_message,
    )

    dag_instruction = (
        "### dag_tutor_instruction\n"
        f"Ты обучаешь ноде «{node.title}». "
        f"Пользователь УЖЕ прошёл: {parent_summary}. "
        "НЕ задавай базовые вопросы по этим прошлым темам. "
        f"Сделай мостик от пройденного к текущей теме и подготовь к: {child_summary}.\n"
    )

    extra_parts: list[str] = []
    if mastery_block:
        extra_parts.append(mastery_block)
    if diagram_block:
        extra_parts.append(diagram_block)
    extra = "\n\n".join(
        [
            dag_instruction,
            f"### parent_nodes_summary\n{parent_summary}",
            f"### child_nodes_summary\n{child_summary}",
            *extra_parts,
        ]
    )
    if neighborhood.strip():
        extra = f"### neighborhood_context\n{neighborhood.strip()}\n\n{extra}"

    return f"{base}\n\n{extra}".strip()


def overlap_mastered_labels(
    node: NodeDataInput,
    mastery_map: dict[str, float],
    *,
    threshold: float = 0.55,
) -> list[str]:
    from knowledge_engine.src.node_deep_dive.user_mastery_profile import (
        mastered_entities_set,
    )

    mastered = mastered_entities_set(mastery_map, threshold=threshold)
    labels: list[str] = []
    for ent in _node_knowledge_entities(node):
        if ent in mastered:
            labels.append(ent)
            continue
        for m in mastered:
            if ent in m or m in ent:
                labels.append(ent)
                break
    return labels[:8]
