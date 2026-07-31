"""Слои контекста тьютора: Static Prefix → Anchor & Manifest → Dynamic Suffix."""

from __future__ import annotations

from knowledge_engine.src.node_deep_dive.fact_manifest import format_fact_manifest_block
from knowledge_engine.src.node_deep_dive.memory_schemas import SessionMemory
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput
from knowledge_engine.src.node_deep_dive.tiered_memory import format_matrix_for_llm, format_window_for_llm
from knowledge_engine.src.node_deep_dive.tutor_behavior_state import (
    build_tutor_behavior_state,
    format_tutor_behavior_state_block,
)

SLIDING_WINDOW_MAX = 6


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
) -> str:
    """[Static Prefix] — для implicit cache Gemini."""
    concepts_list = "\n".join(f"- {c}" for c in node.core_concepts)
    blocks = [
        "[STATIC_PREFIX]",
        f"### user_action\n{action}",
        f"### node_id\n{node.node_id}",
        f"### title\n{node.title}",
        f"### layer\n{node.layer}",
        f"### category\n{node.category}",
        f"### core_concepts_list\n{concepts_list}",
        f"### brief_summary\n{node.brief_summary}",
        f"### layer_1_compressed_rag_profile\n{memory.rag_profile_compressed}",
        f"### layer_2_core_concepts_matrix\n{format_matrix_for_llm(memory.concepts_matrix)}",
        f"### topic_mastery_score\n{memory.topic_mastery_score}%",
        f"### learning_phase\n{memory.learning_phase}",
        f"### learning_mode\n{memory.learning_mode}",
        f"### detected_user_intent\n{intent}",
    ]
    if (node_curriculum_block or "").strip():
        blocks.append(f"### node_curriculum_from_graph\n{node_curriculum_block.strip()}")
    return "\n\n".join(blocks)


def build_anchor_and_manifest(memory: SessionMemory) -> str:
    return "\n\n".join(
        [
            "[ANCHOR_AND_MANIFEST]",
            format_anchor_block(memory),
            format_fact_manifest_block(memory),
        ]
    )


def build_sliding_window_block(memory: SessionMemory) -> str:
    window = list(memory.active_window or [])[-SLIDING_WINDOW_MAX:]
    body = format_window_for_llm(window)
    return f"### sliding_window_last_{len(window)}\n{body}"


def build_dynamic_suffix(
    memory: SessionMemory,
    user_message: str,
    *,
    include_window: bool = True,
) -> str:
    parts = ["[DYNAMIC_SUFFIX]"]
    if include_window:
        parts.append(build_sliding_window_block(memory))
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
    )
    static = build_static_prefix(memory, node, intent, action, node_curriculum_block)
    anchor_manifest = build_anchor_and_manifest(memory)
    behavior = format_tutor_behavior_state_block(state)
    dynamic = build_dynamic_suffix(memory, user_message)
    return "\n\n".join([static, anchor_manifest, behavior, dynamic])
