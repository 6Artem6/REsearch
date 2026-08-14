"""Strict sub-concept state invariants (lecture / question / evaluation binding)."""

from __future__ import annotations

from knowledge_engine.src.node_deep_dive.concept_map_state import (
    find_sub_concept,
    select_next_sub_concept,
)
from knowledge_engine.src.node_deep_dive.memory_schemas import SessionMemory
from knowledge_engine.src.node_deep_dive.schemas import DeepDiveLLMOutput
from knowledge_engine.ui.run_log import trace

# English system/payload instruction (see .cursor/rules/llm-system-prompts-english.mdc)
SUBCONCEPT_HARD_ANCHOR_RULE = (
    "=== SUBCONCEPT HARD ANCHOR (mandatory) ===\n"
    "Generate lecture and follow-up question EXCLUSIVELY for the active sub-topic "
    "id=`{active_id}` («{active_label}»).\n"
    "FORBIDDEN: choose the lecture topic from chat_history / prior turns.\n"
    "Ignore earlier dialogue when selecting which sub-concept to teach.\n"
    "`question_sub_concept_id` MUST equal `{active_id}` when `follow_up_question` is set.\n"
)


def resolve_active_subconcept_id(memory: SessionMemory) -> str:
    """Active generation focus (lecture + next question), not the asked-eval pointer alone."""
    nxt = select_next_sub_concept(memory)
    if nxt is not None:
        return nxt.id
    return (memory.next_question_concept_id or "").strip()


def resolve_active_subconcept_label(memory: SessionMemory, active_id: str = "") -> str:
    cid = (active_id or resolve_active_subconcept_id(memory)).strip()
    if not cid:
        return ""
    row = find_sub_concept(memory, cid)
    return ((row.label if row else "") or cid).strip()


def format_subconcept_hard_anchor(memory: SessionMemory) -> str:
    active_id = resolve_active_subconcept_id(memory)
    if not active_id:
        return ""
    label = resolve_active_subconcept_label(memory, active_id)
    return SUBCONCEPT_HARD_ANCHOR_RULE.format(
        active_id=active_id,
        active_label=label or active_id,
    )


def enforce_question_sub_concept_invariant(
    memory: SessionMemory,
    llm_out: DeepDiveLLMOutput,
) -> tuple[DeepDiveLLMOutput, bool]:
    """
    Guard: question_sub_concept_id must match active generation focus.

    Returns (possibly repaired output, drifted).
    On drift, forces question_sub_concept_id → active_id (caller may retry LLM).
    """
    if llm_out.ready_for_transition:
        return llm_out, False
    follow = (llm_out.follow_up_question or "").strip()
    if not follow:
        return llm_out, False
    active_id = resolve_active_subconcept_id(memory)
    if not active_id:
        return llm_out, False
    qid = (llm_out.question_sub_concept_id or "").strip()
    if qid == active_id:
        return llm_out, False
    trace(
        "STATE_DRIFT | question_sub_concept_id="
        f"{qid or '∅'} != active={active_id} — force-correct + retry signal"
    )
    repaired = llm_out.model_copy(update={"question_sub_concept_id": active_id})
    return repaired, True
