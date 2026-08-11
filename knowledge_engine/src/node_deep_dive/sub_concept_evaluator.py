"""Sub-concept layer evaluator (WHY/HOW/MECHANIC) + deterministic Threshold Engine."""

from __future__ import annotations

import json
import logging
from typing import Literal

from knowledge_engine.config import (
    GEMINI_LITE_MAX_OUTPUT_TOKENS,
    GEMINI_LITE_MODEL,
    GEMINI_RPM_PAUSE_SEC,
)
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.services.gemini_stateless import run_gemini_structured_with_chain
from knowledge_engine.src.node_deep_dive.concept_map_state import (
    _touch_sub_concept,
    advance_next_question_after_evaluation,
    advance_sub_concepts_after_user_answer,
    build_evaluator_feedback,
    ensure_sub_concept_map,
    find_sub_concept,
    last_tutor_question_text_for_eval,
    resolve_evaluation_target_id,
    sync_concepts_matrix_from_sub_concepts,
    trace_coverage_update,
)
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    SessionMemory,
    SubConceptRecord,
    SubConceptStatus,
)
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput
from knowledge_engine.ui.run_log import trace

logger = logging.getLogger(__name__)

EvalDirective = Literal[
    "PROBE_NEXT_LAYER:WHY",
    "PROBE_NEXT_LAYER:HOW",
    "PASSED_WITH_GLOSS",
    "PASSED_CLEAN",
]

GAP_EVAL_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "You are a FACT EXTRACTOR for one curriculum sub-topic (Topic Concept Map).\n"
    "You do NOT decide mastery / VERIFIED / PARTIAL. Python owns the pass threshold.\n\n"
    "Input: asked_question_sub_concept_id, evaluation_target, node_goal, "
    "last_tutor_question, user_message.\n\n"
    "Evaluate ONLY evaluation_target / asked_question_sub_concept_id.\n"
    "Score the user's THIS message for three INDEPENDENT layers "
    "(set True only when that layer is substantively present):\n"
    "- why_passed (WHY): problem, motivation, why the approach exists / what fails without it.\n"
    "- how_passed (HOW): architecture, invariants, role split, pipeline stages — "
    "high-level design without requiring formulas/code.\n"
    "- mechanic_passed (MECHANIC): precise math, algorithms, formulas, APIs, or code.\n\n"
    "DEPTH RULES:\n"
    "- If the user covers WHY+HOW+MECHANIC in one answer → set ALL THREE True.\n"
    "- Buzzword lists without substance → all False (or only why_passed if motivation is clear).\n"
    "- Do NOT withhold True on a layer the user actually covered just because another layer is weak.\n"
    "- Off-topic / refusal («не знаю», skip) → all three False; explain in evidence/focus_hint.\n"
    "- Partial relevance still counts: if the answer discusses a related mechanism "
    "(e.g. async pools under hierarchy) set how_passed/mechanic_passed True when earned; "
    "do not return an empty updates list.\n\n"
    "JSON rules:\n"
    "- updates: exactly one entry; id MUST equal active_question_sub_concept_id "
    "(copy the id string exactly from the payload).\n"
    "- Emit why_passed / how_passed / mechanic_passed booleans (required).\n"
    "- evidence: short digest of what THIS answer demonstrated.\n"
    "- focus_hint: optional note on the weakest missing layer in THIS answer.\n"
    "- Do NOT rely on status for mastery; leave status null/omit if possible.\n"
    "- NEVER return updates: [] for a non-empty user answer — always emit one object.\n"
)
"""
RU (пояснение): Evaluator — только булевы слои WHY/HOW/MECHANIC; порог сдачи в Python.
"""

_PROBE_HINTS = {
    "WHY": (
        "Нужно раскрыть WHY: зачем этот подход, какая проблема/мотивация, "
        "что ломается без него."
    ),
    "HOW": (
        "Нужно раскрыть HOW: архитектура, инварианты, разделение ролей/"
        "стадий — без обязательных формул."
    ),
    "MECHANIC": (
        "Опционально для Deep Mastery: точная механика (формула/алгоритм/код)."
    ),
}

_DEGRADED_FOCUS = (
    "Автооценка ответа не завершилась корректно; уточните WHY/HOW по критерию подтемы."
)
_DEGRADED_EVIDENCE = "evaluator_degraded: empty_or_failed_llm_update"


def _is_empty_answer(text: str) -> bool:
    return len((text or "").strip()) < 5


def _normalize_layer(layer: str) -> str:
    raw = (layer or "foundation").strip().lower()
    if raw in ("fundamental", "base", "intro"):
        return "foundation"
    if raw in ("adv",):
        return "advanced"
    if raw in ("sota", "state_of_the_art", "state-of-the-art", "deep_mastery"):
        return "sota"
    if raw in ("foundation", "advanced", "sota"):
        return raw
    return "foundation"


def normalize_node_layer(layer: str) -> str:
    """Public alias for node difficulty layer."""
    return _normalize_layer(layer)


def required_depth_layers(layer: str) -> tuple[str, ...]:
    """Layers required for threshold / ready_for_transition by node difficulty."""
    ly = _normalize_layer(layer)
    if ly == "foundation":
        return ("WHY",)
    if ly == "advanced":
        return ("WHY", "HOW")
    return ("WHY", "HOW", "MECHANIC")


def optional_depth_layers(layer: str) -> tuple[str, ...]:
    """Layers that may stay open after threshold (empty for SotA)."""
    ly = _normalize_layer(layer)
    if ly == "foundation":
        return ("HOW", "MECHANIC")
    if ly == "advanced":
        return ("MECHANIC",)
    return ()


def passes_threshold(
    layer: str,
    why: bool,
    how: bool,
    mechanic: bool,
) -> tuple[bool, EvalDirective]:
    """
    Deterministic pass check + tutor directive.

    Returns ``(threshold_met, directive)``.

    - foundation: WHY required; HOW/MECH optional → PASSED_WITH_GLOSS if either open
    - advanced: WHY+HOW required; MECH optional → PASSED_WITH_GLOSS if MECH open
    - sota: WHY+HOW+MECH all required (no optional gloss path)
    """
    ly = _normalize_layer(layer)
    if ly == "foundation":
        if not why:
            return False, "PROBE_NEXT_LAYER:WHY"
        if how and mechanic:
            return True, "PASSED_CLEAN"
        return True, "PASSED_WITH_GLOSS"

    if ly == "sota":
        if not why:
            return False, "PROBE_NEXT_LAYER:WHY"
        if not how:
            return False, "PROBE_NEXT_LAYER:HOW"
        if not mechanic:
            return False, "PROBE_NEXT_LAYER:MECHANIC"
        return True, "PASSED_CLEAN"

    # advanced
    if not why:
        return False, "PROBE_NEXT_LAYER:WHY"
    if not how:
        return False, "PROBE_NEXT_LAYER:HOW"
    if mechanic:
        return True, "PASSED_CLEAN"
    return True, "PASSED_WITH_GLOSS"


def merge_layer_flags(
    prev_why: bool,
    prev_how: bool,
    prev_mechanic: bool,
    cur_why: bool,
    cur_how: bool,
    cur_mechanic: bool,
) -> tuple[bool, bool, bool]:
    """Cumulative OR-merge — never clear an already-passed layer."""
    return (
        bool(prev_why) or bool(cur_why),
        bool(prev_how) or bool(cur_how),
        bool(prev_mechanic) or bool(cur_mechanic),
    )


def _focus_hint_for_directive(
    directive: EvalDirective,
    llm_hint: str = "",
) -> str:
    if directive.startswith("PROBE_NEXT_LAYER:"):
        layer = directive.split(":", 1)[-1].strip().upper()
        default = _PROBE_HINTS.get(layer, "")
        hint = (llm_hint or "").strip()
        if hint and layer.lower() in hint.lower():
            return hint[:500]
        return default[:500]
    return ""


def apply_threshold_to_sub_concept(
    row: SubConceptRecord,
    *,
    layer: str,
    why: bool,
    how: bool,
    mechanic: bool,
    evidence: str = "",
    llm_focus_hint: str = "",
) -> EvalDirective:
    """
    OR-merge flags onto ``row``, set status from threshold, return tutor directive.
    """
    why_m, how_m, mech_m = merge_layer_flags(
        row.why_passed,
        row.how_passed,
        row.mechanic_passed,
        why,
        how,
        mechanic,
    )
    row.why_passed = why_m
    row.how_passed = how_m
    row.mechanic_passed = mech_m

    ev = (evidence or "").strip()
    if ev:
        row.evidence = ev[:2000]

    passed, directive = passes_threshold(layer, why_m, how_m, mech_m)
    old_st: SubConceptStatus = row.status
    new_st: SubConceptStatus = "verified" if passed else "partial"
    if old_st != new_st:
        trace_coverage_update(row.id, old_st, new_st, source="threshold_engine")
    row.status = new_st
    _touch_sub_concept(row)

    if not passed:
        row.focus_hint = _focus_hint_for_directive(directive, llm_focus_hint)
    elif directive == "PASSED_WITH_GLOSS":
        if not how_m:
            row.focus_hint = _PROBE_HINTS["HOW"][:500]
        else:
            row.focus_hint = _PROBE_HINTS["MECHANIC"][:500]
    else:
        row.focus_hint = ""

    return directive


def apply_degraded_threshold(
    row: SubConceptRecord,
    *,
    layer: str,
    reason: str,
) -> EvalDirective:
    """
    Guaranteed state write when LLM grader fails / returns empty updates.

    Leaves layer flags unchanged (no false credits) but moves unchecked → partial
    so the UI never stays frozen at «Ещё не затронута» without a log trail.
    """
    directive = apply_threshold_to_sub_concept(
        row,
        layer=layer,
        why=False,
        how=False,
        mechanic=False,
        evidence=_DEGRADED_EVIDENCE,
        llm_focus_hint=_DEGRADED_FOCUS,
    )
    if not (row.focus_hint or "").strip():
        row.focus_hint = _DEGRADED_FOCUS[:500]
    else:
        row.focus_hint = f"{_DEGRADED_FOCUS} ({reason})"[:500]
    logger.warning(
        "EVALUATOR_DEGRADED concept=%s reason=%s status=%s directive=%s",
        row.id,
        reason,
        row.status,
        directive,
    )
    trace(
        f"EVALUATOR_DEGRADED | concept={row.id} | reason={reason} | "
        f"status={row.status} directive={directive}"
    )
    return directive


def _gap_eval_payload(
    memory: SessionMemory,
    node: NodeDataInput,
    user_message: str,
    target: SubConceptRecord,
) -> str:
    target_json = {
        "id": target.id,
        "label": target.label,
        "success_criterion": target.success_criterion,
        "why_passed": target.why_passed,
        "how_passed": target.how_passed,
        "mechanic_passed": target.mechanic_passed,
        "status": target.status,
        "evidence": target.evidence,
    }
    tutor_q = (memory.last_tutor_follow_up_question or "").strip()
    if not tutor_q:
        tutor_q = last_tutor_question_text_for_eval(memory)
    return (
        f"### node_title\n{node.title}\n"
        f"### node_layer\n{getattr(node, 'layer', '') or 'foundation'}\n"
        f"### node_goal\n{(memory.node_goal or node.learning_goal or '')[:800]}\n"
        f"### asked_question_sub_concept_id\n{target.id}\n"
        f"### active_question_sub_concept_id\n{target.id}\n"
        "### evaluation_target\n"
        f"{json.dumps(target_json, ensure_ascii=False)}\n"
        "Prior layer flags are informational only — score THIS user_message independently; "
        "Python will OR-merge with history.\n"
        "Evaluate ONLY asked_question_sub_concept_id / evaluation_target. "
        "Do not return updates for other sub_concepts.\n"
        f"Copy id exactly: {target.id}\n"
        f"### last_tutor_question\n{tutor_q[:4000]}\n"
        f"### user_message\n{(user_message or '').strip()[:6000]}\n"
    )


def _select_gap_update(raw_updates: list, target_id: str):
    """Prefer exact id match; soft-accept a single mismatched update with a warning."""
    exact = [
        u
        for u in (raw_updates or [])
        if str(getattr(u, "id", "") or "").strip() == target_id
    ]
    if exact:
        return exact[0], False
    if len(raw_updates or []) == 1:
        u0 = raw_updates[0]
        logger.warning(
            "EVALUATOR_ID_MISMATCH target=%s got=%r — soft-accepting single update",
            target_id,
            getattr(u0, "id", None),
        )
        trace(
            f"EVALUATOR_WARNING | id_mismatch target={target_id} "
            f"got={getattr(u0, 'id', None)!r} — soft-accept"
        )
        return u0, True
    return None, False


def run_sub_concept_gap_eval(
    user_message: str,
    memory: SessionMemory,
    node: NodeDataInput,
    anchor: str,
    *,
    concept_id: str,
    chat_sessions: dict | None = None,
) -> EvalDirective | None:
    """LLM layer extract + Python threshold; returns tutor directive or None."""
    _ = chat_sessions
    text = (user_message or "").strip()
    if _is_empty_answer(text):
        logger.info("sub_concept_gap skip empty/short user_message")
        trace("NODE_DIVE sub_concept_gap skip | empty/short user_message")
        return None
    ensure_sub_concept_map(memory, node)
    target = find_sub_concept(memory, concept_id)
    if target is None:
        logger.warning("sub_concept_gap skip: concept_id=%r not in map", concept_id)
        trace("NODE_DIVE sub_concept_gap skip | no pending evaluation target")
        return None

    from knowledge_engine.schemas.llm_contracts.tutor import SubConceptGapEvalContract

    prior_status = target.status
    payload = _gap_eval_payload(memory, node, text, target)
    layer = str(getattr(node, "layer", "") or "foundation")
    try:
        raw = run_gemini_structured_with_chain(
            GEMINI_LITE_MODEL,
            GAP_EVAL_SYSTEM,
            payload,
            anchor,
            SubConceptGapEvalContract,
            "node_deep_dive / sub_concept_gap",
            rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
            chat_manager=None,
            max_output_tokens=GEMINI_LITE_MAX_OUTPUT_TOKENS,
        )
        u0, soft = _select_gap_update(list(raw.updates or []), target.id)
        if u0 is None:
            logger.error(
                "EVALUATOR_EMPTY_UPDATES concept=%s prior=%s raw_n=%s",
                target.id,
                prior_status,
                len(raw.updates or []),
            )
            trace(
                f"EVALUATOR_ERROR | concept={target.id} | empty_updates "
                f"(prior={prior_status}); applying degraded threshold"
            )
            return apply_degraded_threshold(target, layer=layer, reason="empty_updates")

        why = bool(u0.why_passed)
        how = bool(u0.how_passed)
        mechanic = bool(u0.mechanic_passed)
        if (
            not why
            and not how
            and not mechanic
            and (u0.status or "").upper() == "VERIFIED"
        ):
            why = how = True

        directive = apply_threshold_to_sub_concept(
            target,
            layer=layer,
            why=why,
            how=how,
            mechanic=mechanic,
            evidence=u0.evidence or "",
            llm_focus_hint=u0.focus_hint or "",
        )
        if soft and not (target.focus_hint or "").strip():
            target.focus_hint = _focus_hint_for_directive(
                directive, u0.focus_hint or ""
            )
        trace(
            f"NODE_DIVE sub_concept_gap ✓ | layers="
            f"W{int(target.why_passed)}H{int(target.how_passed)}"
            f"M{int(target.mechanic_passed)} "
            f"status={target.status} directive={directive} "
            f"verified={sum(1 for s in memory.sub_concepts if s.status == 'verified')}"
            f"/{len(memory.sub_concepts)}"
        )
        return directive
    except Exception as exc:
        logger.exception(
            "EVALUATOR_EXCEPTION concept=%s prior=%s", target.id, prior_status
        )
        trace(
            f"EVALUATOR_ERROR | concept={target.id} | "
            f"{type(exc).__name__}: {exc} | applying degraded threshold"
        )
        return apply_degraded_threshold(
            target,
            layer=layer,
            reason=f"{type(exc).__name__}",
        )


def process_sub_concept_user_answer(
    user_message: str,
    memory: SessionMemory,
    node: NodeDataInput,
    anchor: str,
) -> None:
    text = (user_message or "").strip()
    if _is_empty_answer(text):
        logger.info("sub_concept evaluation skip | empty/short user_message")
        trace("NODE_DIVE sub_concept evaluation skip | empty/short user_message")
        return
    from knowledge_engine.src.node_deep_dive.lecture_scope import (
        is_lecture_request_message,
    )

    if is_lecture_request_message(text):
        logger.info("sub_concept evaluation skip | lecture request")
        trace(
            "NODE_DIVE sub_concept evaluation skip | lecture request "
            "(evaluator only scores real answers)"
        )
        return
    from knowledge_engine.src.node_deep_dive.concept_map import (
        is_quick_reply_control_message,
    )

    if is_quick_reply_control_message(text):
        logger.info("sub_concept evaluation skip | quick-reply control chip")
        trace(
            "NODE_DIVE sub_concept evaluation skip | quick-reply control "
            "(Gloss / Дожать / next — not scored)"
        )
        return
    ensure_sub_concept_map(memory, node)
    pending_id = resolve_evaluation_target_id(memory)
    if not pending_id:
        logger.warning(
            "sub_concept evaluation skip | no pending_evaluation_concept_id "
            "(asked=%r pending=%r)",
            memory.asked_question_sub_concept_id,
            memory.pending_evaluation_concept_id,
        )
        trace(
            "NODE_DIVE sub_concept evaluation skip | no pending_evaluation_concept_id"
        )
        return
    trace(
        f"NODE_DIVE sub_concept evaluation ▶ | target={pending_id} "
        f"next_q={memory.next_question_concept_id or '—'} "
        f"layer={getattr(node, 'layer', '') or 'foundation'}"
    )
    row = find_sub_concept(memory, pending_id)
    if row is None:
        logger.error(
            "sub_concept evaluation skip | pending id %r not in map", pending_id
        )
        trace("NODE_DIVE sub_concept evaluation skip | pending id not in map")
        return

    directive = run_sub_concept_gap_eval(
        text,
        memory,
        node,
        anchor,
        concept_id=pending_id,
        chat_sessions=memory.chat_sessions,
    )
    if directive:
        memory.last_eval_directive = directive
    else:
        memory.last_eval_directive = memory.last_eval_directive or ""
        logger.warning(
            "sub_concept evaluation returned no directive | concept=%s status=%s",
            pending_id,
            row.status,
        )
    memory.last_evaluator_feedback = build_evaluator_feedback(row)
    advance_sub_concepts_after_user_answer(
        memory,
        text,
        concept_id=pending_id,
    )
    advance_next_question_after_evaluation(memory, evaluated_id=pending_id)
    memory.pending_evaluation_concept_id = ""
    memory.last_tutor_sub_concept_id = ""
    sync_concepts_matrix_from_sub_concepts(memory)
