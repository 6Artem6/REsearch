"""Host Layer Drill orchestrator: schema selection, markdown assembly, LLM mapping."""

from __future__ import annotations

from dataclasses import dataclass

from knowledge_engine.schemas.drill_schemas import (
    ActiveDrillStepResponse,
    DrillStepResponse,
    LayerCompletionTutorOutput,
    StandardDrillTutorOutput,
)
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    LayerDrillSession,
    SessionMemory,
)
from knowledge_engine.src.node_deep_dive.schemas import DeepDiveLLMOutput

_ACTIVE_DRILL_JSON_TAIL = (
    "=== JSON OUTPUT (ActiveDrillStepResponse) — HARD ===\n"
    "Alias: StandardDrillTutorOutput. "
    "Ignore DeepDiveTutorContract / DeepDiveDeepAnalysisContract.\n"
    "Return strictly valid JSON matching ActiveDrillStepResponse and nothing else.\n"
    "Required keys: audit, status_header, theory_body, next_question.\n"
    "Fill audit FIRST (single flat TechnicalConceptAudit, not oneOf):\n"
    "  EXACT → confirmation; unused correction_breakdown / praise_points empty.\n"
    "  PARTIAL → praise_points (correct theses) + correction_breakdown "
    "(missing fragment); unused confirmation = empty string.\n"
    "  NEEDS_CORRECTION / MISUNDERSTANDING → correction_breakdown; "
    "confirmation empty; praise_points optional.\n"
    "Match last_eval_directive (PASSED_* → EXACT; PROBE_* → NEEDS_CORRECTION).\n"
    "Do not emit 📋/🎯 or feedback_on_answer — Host assembles those.\n"
    "There is NO top-level feedback_on_previous praise field.\n"
    "There is NO summary_feedback, suggested_action_chips, node_completed_message, "
    "or ready_for_transition field — do not invent them.\n"
    "theory_body: aim for ~300 Russian words; MUST be at least 150 "
    "(host warns if under 300, does not reject).\n"
    "next_question MUST be exactly one checkpoint question (must contain «?»).\n"
    "User-facing string fields MUST be in natural Russian.\n"
)

_LAYER_COMPLETION_JSON_TAIL = (
    "=== JSON OUTPUT (LayerCompletionTutorOutput) — HARD ===\n"
    "Ignore DeepDiveTutorContract / DeepDiveDeepAnalysisContract / "
    "ActiveDrillStepResponse.\n"
    "Return strictly valid JSON matching LayerCompletionTutorOutput and nothing else.\n"
    "Required keys: praise, layer_summary, transition_framing.\n"
    "There is NO next_question, theory_body, follow_up_question, "
    "status_header, or suggested_action_chips field — do not invent them.\n"
    "transition_framing invites HOW/MECH/Advanced/Deep vs next topic; "
    "it is not a technical checkpoint. Host owns chips and ready_for_transition.\n"
    "User-facing string fields MUST be in natural Russian.\n"
)


@dataclass(frozen=True)
class LayerCompletionSnapshot:
    """Evaluator pre-score snapshot — latch fires only on a False→True flip."""

    coverage_complete: bool
    drill_complete: bool
    how_complete: bool
    mech_complete: bool


def drill_session_of(memory: SessionMemory | None) -> LayerDrillSession | None:
    if memory is None:
        return None
    drill = getattr(memory, "layer_drill", None)
    return drill if isinstance(drill, LayerDrillSession) else None


def is_layer_just_completed(memory: SessionMemory | None) -> bool:
    return bool(memory is not None and getattr(memory, "is_layer_just_completed", False))


def _core_rows(memory: SessionMemory | None) -> list:
    if memory is None:
        return []
    from knowledge_engine.src.node_deep_dive.concept_map_state import (
        core_sub_concepts,
    )

    return list(core_sub_concepts(memory) or [])


def _all_core_flag(memory: SessionMemory | None, attr: str) -> bool:
    rows = _core_rows(memory)
    if not rows:
        return False
    return all(bool(getattr(sc, attr, False)) for sc in rows)


def _drill_just_complete(memory: SessionMemory | None) -> bool:
    drill = drill_session_of(memory)
    if drill is None:
        return False
    return (not drill.is_active) and (drill.status or "") == "DRILL_COMPLETE"


def capture_layer_completion_snapshot(
    memory: SessionMemory | None,
) -> LayerCompletionSnapshot:
    from knowledge_engine.src.node_deep_dive.concept_map_state import (
        sub_concept_coverage_complete,
    )

    return LayerCompletionSnapshot(
        coverage_complete=bool(
            memory is not None and sub_concept_coverage_complete(memory)
        ),
        drill_complete=_drill_just_complete(memory),
        how_complete=_all_core_flag(memory, "how_passed"),
        mech_complete=_all_core_flag(memory, "mechanic_passed"),
    )


def latch_layer_just_completed(
    memory: SessionMemory | None,
    before: LayerCompletionSnapshot,
) -> None:
    """Set the one-shot Evaluator latch when this scored turn closed a layer."""
    if memory is None:
        return
    from knowledge_engine.src.node_deep_dive.star_task_fsm import (
        layer_drill_is_active,
        star_task_blocks_transition,
    )

    if layer_drill_is_active(memory) or star_task_blocks_transition(memory):
        memory.is_layer_just_completed = False
        return
    after = capture_layer_completion_snapshot(memory)
    memory.is_layer_just_completed = bool(
        (not before.coverage_complete and after.coverage_complete)
        or (not before.drill_complete and after.drill_complete)
        or (not before.how_complete and after.how_complete)
        or (not before.mech_complete and after.mech_complete)
    )


def select_drill_response_schema(
    memory: SessionMemory | None,
) -> type[ActiveDrillStepResponse] | type[LayerCompletionTutorOutput] | None:
    """
    Deterministic schema pick BEFORE the LLM call.

    Evaluator latch / layer just closed → LayerCompletionTutorOutput
    (no next_question field). Remaining queued sub-topics →
    ActiveDrillStepResponse / StandardDrillTutorOutput (required next_question).
    """
    if is_layer_just_completed(memory):
        return LayerCompletionTutorOutput
    drill = drill_session_of(memory)
    if drill is None:
        return None
    if drill.has_more_questions():
        return ActiveDrillStepResponse
    if (drill.status or "") == "DRILL_COMPLETE":
        return LayerCompletionTutorOutput
    return None


def json_contract_tail_for_schema(schema: type | None) -> str:
    name = getattr(schema, "__name__", "") or ""
    if name in ("ActiveDrillStepResponse", "StandardDrillTutorOutput"):
        return _ACTIVE_DRILL_JSON_TAIL
    if name == "LayerCompletionTutorOutput":
        return _LAYER_COMPLETION_JSON_TAIL
    return ""


def render_drill_markdown(response_obj: DrillStepResponse) -> str:
    """Host-owned UI markdown — the model never emits this wrapping."""
    if isinstance(response_obj, ActiveDrillStepResponse):
        return (
            f"{response_obj.status_header}\n\n"
            f"{response_obj.feedback_on_previous}\n\n"
            f"{response_obj.theory_body}\n\n"
            f"**Вопрос:** {response_obj.next_question}"
        )
    return (
        f"{response_obj.praise}\n\n"
        f"{response_obj.layer_summary}\n\n"
        f"{response_obj.transition_framing}"
    )


def drill_response_to_llm_output(
    response_obj: DrillStepResponse,
    *,
    memory: SessionMemory | None = None,
    concept_id: str = "",
) -> DeepDiveLLMOutput:
    """Map a validated drill contract onto DeepDiveLLMOutput for the existing pipeline."""
    _ = memory
    if isinstance(response_obj, ActiveDrillStepResponse):
        return DeepDiveLLMOutput(
            feedback_on_answer=(
                f"{response_obj.status_header}\n\n{response_obj.feedback_on_previous}"
            ),
            technical_explanation=response_obj.theory_body,
            follow_up_question=f"**Вопрос:** {response_obj.next_question}",
            question_sub_concept_id=(concept_id or "").strip() or None,
            ready_for_transition=False,
            suggested_next_step=None,
            quick_replies=[],
        )
    return DeepDiveLLMOutput(
        feedback_on_answer=response_obj.praise,
        technical_explanation=response_obj.layer_summary,
        follow_up_question=response_obj.transition_framing,
        ready_for_transition=True,
        suggested_next_step="deep_dive_optional",
        quick_replies=[],
    )


def consume_completed_drill_latch(memory: SessionMemory | None) -> None:
    """One-shot: after a LayerCompletion turn, drop the drill latch and Evaluator flag."""
    if memory is None:
        return
    memory.is_layer_just_completed = False
    drill = drill_session_of(memory)
    if drill is None:
        return
    if drill.is_active or (drill.status or "") != "DRILL_COMPLETE":
        return
    from knowledge_engine.src.node_deep_dive.star_task_fsm import clear_layer_drill

    clear_layer_drill(memory)


def is_drill_structured_schema(schema: type | None) -> bool:
    name = getattr(schema, "__name__", "") or ""
    return name in (
        "ActiveDrillStepResponse",
        "StandardDrillTutorOutput",
        "LayerCompletionTutorOutput",
    )


__all__ = [
    "LayerCompletionSnapshot",
    "StandardDrillTutorOutput",
    "capture_layer_completion_snapshot",
    "consume_completed_drill_latch",
    "drill_response_to_llm_output",
    "drill_session_of",
    "is_drill_structured_schema",
    "is_layer_just_completed",
    "json_contract_tail_for_schema",
    "latch_layer_just_completed",
    "render_drill_markdown",
    "select_drill_response_schema",
]
