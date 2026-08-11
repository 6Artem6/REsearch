"""Слои контекста тьютора: Static Prefix → Anchor & Manifest → Dynamic Suffix."""

from __future__ import annotations

from knowledge_engine.src.node_deep_dive.concept_map import (
    format_concept_map_for_tutor,
    format_diagram_repeat_guard,
)
from knowledge_engine.src.node_deep_dive.fact_manifest import format_fact_manifest_block
from knowledge_engine.src.node_deep_dive.memory_schemas import SessionMemory
from knowledge_engine.src.node_deep_dive.schemas import NodeContentBlock, NodeDataInput
from knowledge_engine.src.node_deep_dive.tiered_memory import (
    format_matrix_for_llm,
    format_window_for_llm,
)
from knowledge_engine.src.node_deep_dive.tutor_behavior_state import (
    build_tutor_behavior_state,
    format_tutor_behavior_state_block,
)

SLIDING_WINDOW_MAX = 6
PINNED_CONTEXT_TAG = "[PINNED_NODE_CONTEXT]"
LAYER1_CONTEXT_TAG = "[LAYER1_NODE_STATIC]"
LAYER2_CONTEXT_TAG = "[LAYER2_SESSION_STATE]"


def set_dialogue_anchor(
    memory: SessionMemory,
    node: NodeDataInput,
    tutor_content: str,
) -> None:
    text = (tutor_content or "").strip()
    if not text:
        return
    memory.anchor_turn = {
        "role": "tutor",
        "content": text[:2000],
        "node_title": (node.title or "")[:400],
        "node_summary": (node.brief_summary or "")[:800],
        "node_id": (node.node_id or "")[:64],
    }


def format_anchor_block(memory: SessionMemory) -> str:
    anchor = memory.anchor_turn or {}
    if not anchor:
        return "### anchor_turn\n(нет — intro ещё не завершён)"
    parts = [
        "### anchor_turn",
        f"node: {anchor.get('node_title', '')}",
        (anchor.get("node_summary") or "")[:800],
        f"{anchor.get('role', 'tutor')}: {(anchor.get('content') or '')[:2000]}",
    ]
    return "\n".join(p for p in parts if p).strip()


def build_static_prefix(
    memory: SessionMemory,
    node: NodeDataInput,
    intent: str,
    action: str,
    node_curriculum_block: str,
    *,
    lightweight: bool = False,
    include_session_volatile: bool = True,
) -> str:
    """[Static Prefix] — для implicit / explicit cache (без session-volatile при layer1)."""
    concepts_list = "\n".join(f"- {c}" for c in node.core_concepts)
    rag = (memory.rag_profile_compressed or "").strip()
    if lightweight and len(rag) > 480:
        rag = rag[:479].rstrip() + "…"
    blocks = [
        "[STATIC_PREFIX]",
        f"### user_action\n{action}",
        f"### node_id\n{node.node_id}",
        f"### title\n{node.title}",
        f"### layer\n{node.layer}",
        f"### category\n{node.category}",
        f"### core_concepts_list\n{concepts_list}",
        f"### brief_summary\n{(node.brief_summary or '')[:400]}",
    ]
    if not lightweight:
        blocks.extend(
            [
                f"### layer_1_compressed_rag_profile\n{rag}",
                f"### layer_2_core_concepts_matrix\n{format_matrix_for_llm(memory.concepts_matrix)}",
            ]
        )
    else:
        blocks.append(f"### rag_profile_hint\n{rag[:480] if rag else '(нет)'}")
    if include_session_volatile:
        blocks.extend(
            [
                f"### topic_mastery_score\n{memory.topic_mastery_score}%",
                f"### learning_phase\n{memory.learning_phase}",
                f"### learning_mode\n{memory.learning_mode}",
                f"### detected_user_intent\n{intent}",
            ]
        )
    if not lightweight and (node_curriculum_block or "").strip():
        blocks.append(
            f"### node_curriculum_from_graph\n{node_curriculum_block.strip()}"
        )
    return "\n\n".join(blocks)


def format_session_volatile_block(
    memory: SessionMemory,
    intent: str,
    *,
    user_focus: str = "",
) -> str:
    lines = [
        f"### topic_mastery_score\n{memory.topic_mastery_score}%",
        f"### learning_phase\n{memory.learning_phase}",
        f"### learning_mode\n{memory.learning_mode}",
        f"### detected_user_intent\n{intent}",
    ]
    if (user_focus or "").strip():
        lines.append(f"### user_focus_topic\n{user_focus.strip()}")
    return "\n".join(lines)


def combine_node_context_layers(layer1: str, layer2: str) -> str:
    parts = [p.strip() for p in (layer1, layer2) if (p or "").strip()]
    return "\n\n".join(parts)


def build_layer1_explicit_cache_context(
    memory: SessionMemory,
    node: NodeDataInput,
    intent: str,
    action: str,
    node_curriculum_block: str,
    *,
    curriculum_id: str = "",
    node_content: NodeContentBlock | None = None,
    is_lightweight: bool = False,
) -> str:
    """
    Слой 1: статика темы ноды (кандидат в explicit cache).
    Без manifest/mastery/concept_map и без session-volatile полей.
    """
    static = build_static_prefix(
        memory,
        node,
        intent,
        action,
        node_curriculum_block,
        lightweight=is_lightweight,
        include_session_volatile=False,
    )
    parts: list[str] = [LAYER1_CONTEXT_TAG, static]

    cid = (curriculum_id or "").strip()
    if is_lightweight:
        if cid:
            from knowledge_engine.src.node_deep_dive.tutor_source_citations import (
                build_tutor_source_registry,
                format_tutor_source_registry_light,
            )

            src_registry = build_tutor_source_registry(cid, node, node_content)
            parts.append(format_tutor_source_registry_light(src_registry))
        return "\n\n".join(parts).strip()

    if cid:
        from knowledge_engine.services.article_diagram_context import (
            build_figure_registry_for_node,
            build_pinned_diagrams_for_node,
        )

        diagrams_block = build_pinned_diagrams_for_node(node, curriculum_id)
        if diagrams_block:
            parts.append(diagrams_block)
        registry_block = build_figure_registry_for_node(node, curriculum_id)
        if registry_block:
            parts.append(registry_block)
    from knowledge_engine.src.node_deep_dive.tutor_diagram_citations import (
        build_diagram_catalog,
        format_diagram_catalog_block,
    )

    parts.append(format_diagram_catalog_block(build_diagram_catalog(node_content)))
    from knowledge_engine.src.node_deep_dive.node_materials_context import (
        format_available_node_materials_block,
    )

    materials_block = format_available_node_materials_block(node, node_content)
    if materials_block:
        parts.append(materials_block)
    if cid:
        from knowledge_engine.src.node_deep_dive.tutor_source_citations import (
            build_tutor_source_registry,
            format_tutor_source_registry_pinned,
        )

        src_registry = build_tutor_source_registry(cid, node, node_content)
        parts.append(format_tutor_source_registry_pinned(src_registry))
    return "\n\n".join(parts).strip()


def build_layer2_session_state_context(
    memory: SessionMemory,
    node: NodeDataInput,
    intent: str,
    *,
    user_focus: str = "",
    curriculum_id: str = "",
    is_lightweight: bool = False,
    concept_map_block: str = "",
    knowledge_atoms_block: str = "",
) -> str:
    """Слой 2: manifest, mastery, concept_map, global — меняется по ходу сессии."""
    parts: list[str] = [
        LAYER2_CONTEXT_TAG,
        format_session_volatile_block(memory, intent, user_focus=user_focus),
        build_anchor_and_manifest(memory),
    ]
    if (concept_map_block or "").strip():
        parts.append(concept_map_block.strip())
    if (knowledge_atoms_block or "").strip():
        parts.append(knowledge_atoms_block.strip())

    cid = (curriculum_id or "").strip()
    if is_lightweight:
        if cid:
            from knowledge_engine.src.curriculum.global_tracker import (
                format_global_learned_block,
                get_global_verified_subconcepts_delta,
            )

            _state, delta = get_global_verified_subconcepts_delta(
                "default",
                cid,
                node.node_id,
                current_node=node,
            )
            global_block = format_global_learned_block(
                delta,
                exclude_node_id=node.node_id,
            )
            if global_block:
                parts.append(global_block)
        return "\n\n".join(parts).strip()

    mastery_block = ""
    if cid:
        from knowledge_engine.src.node_deep_dive.user_mastery_profile import (
            format_competency_pinned_block,
        )

        mastery_block = format_competency_pinned_block(node, curriculum_id)
    global_block = ""
    if cid:
        from knowledge_engine.src.curriculum.global_tracker import (
            format_global_learned_block,
            get_global_verified_subconcepts_delta,
        )

        _state, delta = get_global_verified_subconcepts_delta(
            "default",
            cid,
            node.node_id,
            current_node=node,
        )
        global_block = format_global_learned_block(
            delta,
            exclude_node_id=node.node_id,
        )
    if mastery_block:
        parts.append(mastery_block)
    if global_block:
        parts.append(global_block)
    diagram_guard = format_diagram_repeat_guard(memory)
    if diagram_guard:
        parts.append(diagram_guard)
    return "\n\n".join(parts).strip()


def build_active_node_context(
    memory: SessionMemory,
    node: NodeDataInput,
    intent: str,
    action: str,
    node_curriculum_block: str,
    *,
    user_focus: str = "",
    curriculum_id: str = "",
    node_content: NodeContentBlock | None = None,
    is_lightweight: bool = False,
    concept_map_block: str = "",
) -> str:
    """Нода + RAG + session state — склейка layer1 + layer2 (совместимость API)."""
    layer1 = build_layer1_explicit_cache_context(
        memory,
        node,
        intent,
        action,
        node_curriculum_block,
        curriculum_id=curriculum_id,
        node_content=node_content,
        is_lightweight=is_lightweight,
    )
    layer2 = build_layer2_session_state_context(
        memory,
        node,
        intent,
        user_focus=user_focus,
        curriculum_id=curriculum_id,
        is_lightweight=is_lightweight,
        concept_map_block=concept_map_block,
    )
    legacy = combine_node_context_layers(layer1, layer2)
    if PINNED_CONTEXT_TAG not in legacy:
        return f"{PINNED_CONTEXT_TAG}\n\n{legacy}".strip()
    return legacy


def build_anchor_and_manifest(memory: SessionMemory) -> str:
    return "\n\n".join(
        [
            "[ANCHOR_AND_MANIFEST]",
            format_anchor_block(memory),
            format_fact_manifest_block(memory),
        ]
    )


SHARED_SESSION_CONTEXT_TAG = "[SHARED_SESSION_CONTEXT]"


def build_shared_session_context_block(
    memory: SessionMemory | None,
    *,
    user_message: str = "",
    include_sliding_window: bool = True,
) -> str:
    """
    Единый блок для Tutor-adjacent routes (dense lecture, selection explain):
    anchor + fact_manifest + (опционально) sliding_window и current_user_message.
    """
    if memory is None:
        return ""
    parts: list[str] = [SHARED_SESSION_CONTEXT_TAG, build_anchor_and_manifest(memory)]
    if include_sliding_window and (memory.active_window or []):
        parts.append(build_sliding_window_block(memory))
    msg = (user_message or "").strip()
    if msg:
        parts.append(f"### current_user_message\n{msg}")
    return "\n\n".join(p for p in parts if p.strip()).strip()


def build_tutor_source_registry_pinned_block(
    curriculum_id: str,
    node: NodeDataInput,
    node_content: NodeContentBlock | None,
) -> str:
    cid = (curriculum_id or "").strip()
    if not cid:
        return ""
    from knowledge_engine.src.node_deep_dive.tutor_source_citations import (
        build_tutor_source_registry,
        format_tutor_source_registry_pinned,
    )

    registry = build_tutor_source_registry(cid, node, node_content)
    return (format_tutor_source_registry_pinned(registry) or "").strip()


def build_sliding_window_block(memory: SessionMemory) -> str:
    window = list(memory.active_window or [])[-SLIDING_WINDOW_MAX:]
    body = format_window_for_llm(window)
    return f"### sliding_window_last_{len(window)}\n{body}"


def build_dynamic_suffix(
    memory: SessionMemory,
    user_message: str,
    *,
    include_window: bool = True,
    recency_rules: str = "",
) -> str:
    parts = ["[DYNAMIC_SUFFIX]"]
    if include_window:
        parts.append(build_sliding_window_block(memory))
    from knowledge_engine.src.curriculum.global_tracker import (
        format_last_question_angle_hint,
    )

    angle_hint = format_last_question_angle_hint(memory.last_tutor_question_angle)
    if angle_hint:
        parts.append(angle_hint.strip())
    rules = (recency_rules or "").strip()
    if rules:
        parts.append(
            "### CRITICAL_RULES_RECENCY (read immediately before user message)\n"
            f"{rules}"
        )
    msg = (user_message or "").strip()
    if msg:
        parts.append(f"### current_user_message\n{msg}")
    return "\n\n".join(parts)


def build_tutor_invocation_payload(
    memory: SessionMemory,
    node: NodeDataInput,
    intent: str,
    action: str,
    user_message: str,
    node_curriculum_block: str,
    *,
    has_user_focus: bool = False,
) -> str:
    state = build_tutor_behavior_state(
        intent,
        action,
        memory.learning_mode,
        memory.learning_phase,
        user_message,
        has_user_focus=has_user_focus,
        memory=memory,
        node_layer=str(getattr(node, "layer", "") or ""),
    )
    static = build_static_prefix(memory, node, intent, action, node_curriculum_block)
    anchor_manifest = build_anchor_and_manifest(memory)
    behavior = format_tutor_behavior_state_block(state)
    concept_map = format_concept_map_for_tutor(memory)
    dynamic = build_dynamic_suffix(memory, user_message)
    parts = [static, anchor_manifest, behavior]
    if concept_map:
        parts.append(concept_map)
    parts.append(dynamic)
    return "\n\n".join(parts)
