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
from knowledge_engine.schemas.drill_schemas import AnswerAccuracyGrade
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
from knowledge_engine.src.node_deep_dive.context_bounded_eval import (
    CONTEXT_BOUNDED_EVAL_RULES,
)
from knowledge_engine.ui.run_log import trace

logger = logging.getLogger(__name__)

EvalDirective = Literal[
    "PROBE_NEXT_LAYER:WHY",
    "PROBE_NEXT_LAYER:HOW",
    "PROBE_NEXT_LAYER:MECHANIC",
    "PASSED_WITH_GLOSS",
    "PASSED_CLEAN",
    "DEEP_MASTERY_EARNED",
    "STAR_TASK_NEEDS_REFINEMENT",
]

GAP_EVAL_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "You are a FACT EXTRACTOR for one curriculum sub-topic (Topic Concept Map).\n"
    "You do NOT decide mastery / VERIFIED / PARTIAL status. Python owns the "
    "pass threshold and only credits a layer when accuracy_grade is "
    "EXACT_AND_CORRECT and detected_errors_or_misconceptions is empty.\n\n"
    "Input: asked_question_sub_concept_id, evaluation_target, node_goal, "
    "last_tutor_question, user_message, scoring_layer (optional).\n\n"
    f"{CONTEXT_BOUNDED_EVAL_RULES}\n"
    "Evaluate ONLY evaluation_target / asked_question_sub_concept_id.\n"
    "Score the user's THIS message for three INDEPENDENT layers "
    "(set True only when that layer is substantively present IN THIS answer):\n"
    "- why_passed (WHY): problem, motivation, why the approach exists / what fails without it.\n"
    "- how_passed (HOW): structure, roles, invariants, relations — "
    "without requiring execution-level detail.\n"
    "- mechanic_passed (MECHANIC): named execution steps, formal detail, or "
    "procedures that THIS answer actually contains. Missing MECHANIC is NOT "
    "a PARTIAL reason unless last_tutor_question / scoring_layer asked for "
    "that depth.\n\n"
    "ACCURACY (independent of the three booleans; bounded by last_tutor_question):\n"
    "- accuracy_grade=EXACT_AND_CORRECT: complete FOR THE ASKED QUESTION's "
    "layer, no hallucinations; detected_errors_or_misconceptions MUST be [].\n"
    "- accuracy_grade=PARTIAL: some theses are right but an aspect THE QUESTION "
    "(or named scoring_layer) asked is missing; fill correct_claims with the "
    "right theses and focus_hint with that single asked-but-missing aspect. "
    "Do NOT use EXACT_AND_CORRECT. Do NOT name unasked deeper-layer terms in "
    "focus_hint.\n"
    "- accuracy_grade=NEEDS_CORRECTION: factual error that is not a full "
    "rewrite of the concept.\n"
    "- accuracy_grade=MISUNDERSTANDING: gross error or hallucination of the "
    "concept (wrong mechanism, invented entity, category error).\n"
    "- correct_claims: technical facts that were already right (no praise). "
    "REQUIRED non-empty when accuracy_grade is PARTIAL.\n"
    "- detected_errors_or_misconceptions: factual errors / mix-ups; empty on EXACT.\n"
    "- focus_hint: on PARTIAL / non-exact, the one missing or wrong fragment "
    "that last_tutor_question actually asked.\n\n"
    "DEPTH RULES:\n"
    "- If the user covers WHY+HOW+MECHANIC in one answer AND the answer is "
    "exact → set ALL THREE True and accuracy_grade=EXACT_AND_CORRECT.\n"
    "- Buzzword lists without substance → all False and not EXACT.\n"
    "- Do NOT withhold True on a layer the user actually covered just because "
    "another layer is weak. Set PARTIAL only if an aspect the question / "
    "scoring_layer asked is missing — not because a deeper unasked layer is empty.\n"
    "- Off-topic / refusal («не знаю», skip) → all three False; "
    "accuracy_grade=MISUNDERSTANDING or NEEDS_CORRECTION; explain in "
    "evidence/focus_hint.\n"
    "- Partial relevance still scores the booleans for substance present; "
    "Python will not OR-merge those flags unless the grade is EXACT.\n\n"
    "JSON rules:\n"
    "- updates: exactly one entry; id MUST equal active_question_sub_concept_id "
    "(copy the id string exactly from the payload).\n"
    "- Emit why_passed / how_passed / mechanic_passed booleans (required).\n"
    "- Emit accuracy_grade (required) and detected_errors_or_misconceptions.\n"
    "- evidence: short digest of what THIS answer demonstrated.\n"
    "- focus_hint: the weakest asked-but-missing aspect in THIS answer "
    "(must be licensed by last_tutor_question / scoring_layer).\n"
    "- Do NOT rely on status for mastery; leave status null/omit if possible.\n"
    "- NEVER return updates: [] for a non-empty user answer — always emit one object.\n"
    "- If the payload has scoring_layer HOW/MECHANIC/WHY, that flag is the primary "
    "credit bit for this turn; still score other layer booleans independently "
    "(bonus True is allowed; missing deeper layers do not force PARTIAL).\n"
)
"""
RU (пояснение): Evaluator — булевы слои + accuracy_grade в рамках last_tutor_question;
Python закрывает слой только на EXACT.
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


def mark_evaluator_skipped(memory: SessionMemory, reason: str) -> None:
    memory.evaluator_skipped = True
    trace(f"NODE_DIVE sub_concept evaluation skip | {reason}")


def mark_evaluator_ran(memory: SessionMemory) -> None:
    memory.evaluator_skipped = False


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


def _parse_accuracy_grade(raw: object) -> AnswerAccuracyGrade | None:
    if raw is None:
        return None
    if isinstance(raw, AnswerAccuracyGrade):
        return raw
    text = str(getattr(raw, "value", raw) or "").strip()
    try:
        return AnswerAccuracyGrade(text)
    except ValueError:
        return None


def _as_str_list(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    try:
        items = list(raw)
    except TypeError:
        return []
    out: list[str] = []
    for item in items:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def turn_is_strict_exact(
    accuracy_grade: AnswerAccuracyGrade | str | None,
    detected_errors: list[str] | None,
) -> bool:
    """True only for EXACT_AND_CORRECT with no listed errors.

    Omitted grade (legacy unit tests / MagicMock extractor) counts as exact
    when the error list is empty so Host-side credits keep working.
    """
    if _as_str_list(detected_errors):
        return False
    parsed = _parse_accuracy_grade(accuracy_grade)
    if parsed is None:
        return True
    return parsed == AnswerAccuracyGrade.EXACT_AND_CORRECT


def _threshold_from_flags(
    layer: str,
    why: bool,
    how: bool,
    mechanic: bool,
) -> tuple[bool, EvalDirective]:
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

    if not why:
        return False, "PROBE_NEXT_LAYER:WHY"
    if not how:
        return False, "PROBE_NEXT_LAYER:HOW"
    if mechanic:
        return True, "PASSED_CLEAN"
    return True, "PASSED_WITH_GLOSS"


def _probe_when_flags_would_pass(how: bool, mechanic: bool) -> EvalDirective:
    if not how:
        return "PROBE_NEXT_LAYER:HOW"
    if not mechanic:
        return "PROBE_NEXT_LAYER:MECHANIC"
    return "PROBE_NEXT_LAYER:WHY"


def passes_threshold(
    layer: str,
    why: bool,
    how: bool,
    mechanic: bool,
    *,
    accuracy_grade: AnswerAccuracyGrade | str | None = None,
    detected_errors: list[str] | None = None,
) -> tuple[bool, EvalDirective]:
    """
    Deterministic pass check + tutor directive.

    Returns ``(threshold_met, directive)``.

    A non-exact accuracy grade never meets the threshold, even if historical
    layer flags would otherwise pass.

    - foundation: WHY required; HOW/MECH optional → PASSED_WITH_GLOSS if either open
    - advanced: WHY+HOW required; MECH optional → PASSED_WITH_GLOSS if MECH open
    - sota: WHY+HOW+MECH all required (no optional gloss path)
    """
    ok, directive = _threshold_from_flags(layer, why, how, mechanic)
    if turn_is_strict_exact(accuracy_grade, detected_errors):
        return ok, directive
    if not ok:
        return False, directive
    return False, _probe_when_flags_would_pass(how, mechanic)


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
    accuracy_grade: AnswerAccuracyGrade | str | None = None,
    detected_errors: list[str] | None = None,
    correct_claims: list[str] | None = None,
    count_attempt: bool = True,
) -> EvalDirective:
    """
    Merge flags onto ``row`` only on a strict-exact turn, set status, return directive.

    PARTIAL never closes a layer. MISUNDERSTANDING on a ``partial`` row soft-regresses
    status to ``gap``. Already ``verified`` rows are not un-verified on PARTIAL.
    """
    errors = _as_str_list(detected_errors)
    claims = _as_str_list(correct_claims)
    parsed_grade = _parse_accuracy_grade(accuracy_grade)
    exact = turn_is_strict_exact(parsed_grade, errors)

    if exact:
        why_m, how_m, mech_m = merge_layer_flags(
            row.why_passed,
            row.how_passed,
            row.mechanic_passed,
            why,
            how,
            mechanic,
        )
    else:
        why_m, how_m, mech_m = (
            bool(row.why_passed),
            bool(row.how_passed),
            bool(row.mechanic_passed),
        )
    row.why_passed = why_m
    row.how_passed = how_m
    row.mechanic_passed = mech_m

    ev = (evidence or "").strip()
    if not ev and claims:
        ev = "; ".join(claims)
    if ev:
        row.merge_evidence(ev)

    if parsed_grade is not None:
        row.last_accuracy_grade = parsed_grade.value
    elif exact:
        row.last_accuracy_grade = AnswerAccuracyGrade.EXACT_AND_CORRECT.value
    else:
        row.last_accuracy_grade = AnswerAccuracyGrade.PARTIAL.value

    if count_attempt:
        if exact:
            row.failed_attempts = 0
        else:
            row.failed_attempts = min(int(row.failed_attempts or 0) + 1, 99)

    passed, directive = passes_threshold(
        layer,
        why_m,
        how_m,
        mech_m,
        accuracy_grade=parsed_grade if parsed_grade is not None else (
            AnswerAccuracyGrade.EXACT_AND_CORRECT if exact else AnswerAccuracyGrade.PARTIAL
        ),
        detected_errors=errors,
    )
    old_st: SubConceptStatus = row.status
    if exact:
        new_st: SubConceptStatus = "verified" if passed else "partial"
    elif old_st == "verified":
        new_st = "verified"
    elif parsed_grade == AnswerAccuracyGrade.MISUNDERSTANDING:
        new_st = "gap"
    else:
        new_st = "partial"
    if old_st != new_st:
        trace_coverage_update(row.id, old_st, new_st, source="threshold_engine")
    row.status = new_st
    _touch_sub_concept(row)

    hint = (llm_focus_hint or "").strip()
    if not passed:
        row.focus_hint = (hint or _focus_hint_for_directive(directive, llm_focus_hint))[:500]
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
    memory: SessionMemory | None = None,
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
        count_attempt=False,
    )
    if not (row.focus_hint or "").strip():
        row.focus_hint = _DEGRADED_FOCUS[:500]
    else:
        row.focus_hint = f"{_DEGRADED_FOCUS} ({reason})"[:500]
    if memory is not None:
        directive = _directive_for_open_core_drill(memory, row, directive)  # type: ignore[assignment]
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


def _active_core_scoring_layer(memory: SessionMemory) -> str:
    from knowledge_engine.src.node_deep_dive.star_task_fsm import (
        active_core_scoring_layer,
    )

    return active_core_scoring_layer(memory)


def _apply_core_drill_turn_flags(
    memory: SessionMemory,
    *,
    why: bool,
    how: bool,
    mechanic: bool,
) -> tuple[bool, bool, bool, bool]:
    """
    Observe the open HOW/MECH/WHY drill bit without auto-credit.

    The extractor booleans are the only credit signal. Host no longer forces
    the active drill layer True on a non-empty unrelated bit.
    """
    scoring = _active_core_scoring_layer(memory)
    turn_miss = not why and not how and not mechanic
    if scoring:
        trace(
            f"drill scoring_layer={scoring} extractor "
            f"W{int(why)}H{int(how)}M{int(mechanic)} miss={int(turn_miss)}"
        )
    return why, how, mechanic, turn_miss


def _directive_for_open_core_drill(
    memory: SessionMemory,
    row: SubConceptRecord,
    directive: EvalDirective | str | None,
) -> EvalDirective | str | None:
    scoring = _active_core_scoring_layer(memory)
    if scoring == "HOW" and not bool(row.how_passed):
        if not (row.focus_hint or "").strip():
            row.focus_hint = _PROBE_HINTS["HOW"][:500]
        return "PROBE_NEXT_LAYER:HOW"
    if scoring == "MECHANIC" and not bool(row.mechanic_passed):
        if not (row.focus_hint or "").strip():
            row.focus_hint = _PROBE_HINTS["MECHANIC"][:500]
        return "PROBE_NEXT_LAYER:MECHANIC"
    if scoring == "WHY" and not bool(row.why_passed):
        if not (row.focus_hint or "").strip():
            row.focus_hint = _PROBE_HINTS["WHY"][:500]
        return "PROBE_NEXT_LAYER:WHY"
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
    scoring = _active_core_scoring_layer(memory)
    scoring_block = ""
    if scoring:
        scoring_block = (
            f"### scoring_layer\n{scoring}\n"
            f"Host is running a {scoring} layer drill on this sub-topic. "
            f"Set {scoring.lower()}_passed from THIS answer; Python ORs with history "
            "only when accuracy_grade is EXACT_AND_CORRECT with empty errors.\n"
        )
    return (
        f"### node_title\n{node.title}\n"
        f"### node_layer\n{getattr(node, 'layer', '') or 'foundation'}\n"
        f"### node_goal\n{(memory.node_goal or node.learning_goal or '')[:800]}\n"
        f"{scoring_block}"
        f"### asked_question_sub_concept_id\n{target.id}\n"
        f"### active_question_sub_concept_id\n{target.id}\n"
        "### evaluation_target\n"
        f"{json.dumps(target_json, ensure_ascii=False)}\n"
        "Prior layer flags are informational only — score THIS user_message independently; "
        "Python credits a layer only on EXACT_AND_CORRECT with empty errors.\n"
        "Evaluate ONLY asked_question_sub_concept_id / evaluation_target. "
        "Do not return updates for other sub_concepts.\n"
        f"Copy id exactly: {target.id}\n"
        "### context_bound\n"
        "Evaluation scope is strictly bounded by last_tutor_question "
        "(asked abstraction layer + explicit constraints in that text). "
        "success_criterion and node_goal name the topic; they do not expand "
        "required terms. Do not fail EXACT for deeper or adjacent-layer jargon "
        "absent from last_tutor_question unless that text (or scoring_layer "
        "MECHANIC with an explicit detail ask) requested that layer.\n"
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
    from knowledge_engine.src.node_deep_dive.deep_analysis_eval_prompt import (
        ADVANCED_ANALYSIS_EVAL_SYSTEM,
        DEEP_DESIGN_EVAL_SYSTEM,
    )
    from knowledge_engine.src.node_deep_dive.eval_result_adapter import (
        critique_to_memory_payload,
        normalize_overlay_target_layer,
    )
    from knowledge_engine.src.node_deep_dive.star_task_fsm import (
        core_layer_drill_blocks_overlay_eval,
        is_overlay_eval_kind,
        overlay_kind_to_target_layer,
    )
    from knowledge_engine.schemas.llm_contracts.evaluator_critique import (
        EvaluatorCritiqueContract,
    )

    prior_status = target.status
    kind = (memory.pending_eval_kind or "").strip().lower()
    use_overlay = is_overlay_eval_kind(kind) and not core_layer_drill_blocks_overlay_eval(
        memory
    )
    payload = _gap_eval_payload(memory, node, text, target)
    layer = str(getattr(node, "layer", "") or "foundation")

    if use_overlay:
        forced_layer = overlay_kind_to_target_layer(kind)
        payload = (
            f"### eval_mode\n{kind}\n"
            f"### overlay_target_layer\n{forced_layer}\n\n{payload}"
        )
        system = (
            ADVANCED_ANALYSIS_EVAL_SYSTEM
            if kind == "advanced_analysis"
            else DEEP_DESIGN_EVAL_SYSTEM
        )
        from knowledge_engine.context_drift_manager import (
            mix_prior_weaknesses_into_eval_system,
            parse_curriculum_id_from_anchor,
        )

        cid = parse_curriculum_id_from_anchor(anchor)
        system = mix_prior_weaknesses_into_eval_system(
            system,
            curriculum_id=cid,
            exclude_node_id=str(getattr(node, "node_id", "") or ""),
        )
        label = f"node_deep_dive / {kind}_eval"
        trace(f"NODE_DIVE overlay_eval ▶ | concept={target.id} | kind={kind}")
        try:
            raw_critique = run_gemini_structured_with_chain(
                GEMINI_LITE_MODEL,
                system,
                payload,
                anchor,
                EvaluatorCritiqueContract,
                label,
                rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
                chat_manager=None,
                max_output_tokens=GEMINI_LITE_MAX_OUTPUT_TOKENS,
            )
            critique = EvaluatorCritiqueContract.model_validate(
                raw_critique.model_dump()
                if hasattr(raw_critique, "model_dump")
                else raw_critique
            )
            # Host-authoritative overlay layer (model must not drift into core flags).
            critique = critique.model_copy(
                update={
                    "target_layer": forced_layer
                    if forced_layer in ("ADVANCED", "DEEP")
                    else normalize_overlay_target_layer(kind)
                }
            )
            return _apply_overlay_critique(
                memory,
                target,
                critique,
                prior_status=prior_status,
                overlay_kind=kind,
                curriculum_id=cid,
                node_id=str(getattr(node, "node_id", "") or ""),
                node_title=str(getattr(node, "title", "") or ""),
            )
        except Exception as exc:
            from knowledge_engine.src.resilience_manager import is_llm_resilience_error

            logger.exception(
                "EVALUATOR_EXCEPTION overlay concept=%s prior=%s",
                target.id,
                prior_status,
            )
            resilient = is_llm_resilience_error(exc)
            trace(
                f"EVALUATOR_ERROR | overlay concept={target.id} | "
                f"{type(exc).__name__}: {exc} | "
                f"resilience={resilient} | asterisk-question needs_refinement "
                f"(core layers untouched; FSM preserved)"
            )
            # Overlay fail-soft: never credit HOW/MECH; never run core threshold;
            # never raise (host must not 500 on 429 / 5xx / timeout).
            memory.last_evaluator_critique = {}
            memory.last_evaluator_feedback = (
                "[EVALUATOR_CRITIQUE]\n"
                "passes_threshold: false\n"
                f"verdict_reason: overlay evaluator failed ({type(exc).__name__})\n"
                "=== UNACCOUNTED_EDGE_CASES ===\n"
                "- Refine design constraints / trade-offs from the asterisk-question.\n"
            )[:1200]
            target.focus_hint = (
                "Нужно закрыть крайние случаи и trade-offs из разбора "
                "(зависимости / точки отказа / матрица компромиссов)."
            )[:500]
            return "STAR_TASK_NEEDS_REFINEMENT"

    system = GAP_EVAL_SYSTEM
    from knowledge_engine.context_drift_manager import (
        mix_prior_weaknesses_into_eval_system,
        parse_curriculum_id_from_anchor,
    )

    cid = parse_curriculum_id_from_anchor(anchor)
    system = mix_prior_weaknesses_into_eval_system(
        system,
        curriculum_id=cid,
        exclude_node_id=str(getattr(node, "node_id", "") or ""),
    )
    label = "node_deep_dive / sub_concept_gap"
    try:
        raw = run_gemini_structured_with_chain(
            GEMINI_LITE_MODEL,
            system,
            payload,
            anchor,
            SubConceptGapEvalContract,
            label,
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
            return apply_degraded_threshold(
                target, layer=layer, reason="empty_updates", memory=memory
            )

        why = bool(u0.why_passed)
        how = bool(u0.how_passed)
        mechanic = bool(u0.mechanic_passed)
        why, how, mechanic, _turn_miss = _apply_core_drill_turn_flags(
            memory, why=why, how=how, mechanic=mechanic
        )

        directive = apply_threshold_to_sub_concept(
            target,
            layer=layer,
            why=why,
            how=how,
            mechanic=mechanic,
            evidence=u0.evidence or "",
            llm_focus_hint=u0.focus_hint or "",
            accuracy_grade=getattr(u0, "accuracy_grade", None),
            detected_errors=_as_str_list(
                getattr(u0, "detected_errors_or_misconceptions", None)
            ),
            correct_claims=_as_str_list(getattr(u0, "correct_claims", None)),
        )
        directive = _directive_for_open_core_drill(memory, target, directive)
        if soft and not (target.focus_hint or "").strip():
            target.focus_hint = _focus_hint_for_directive(
                directive, u0.focus_hint or ""
            )
        _sync_weakness_ledger_after_core_eval(
            memory,
            node,
            target,
            directive=directive or "",
            curriculum_id=cid,
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
        from knowledge_engine.src.resilience_manager import is_llm_resilience_error

        logger.exception(
            "EVALUATOR_EXCEPTION concept=%s prior=%s", target.id, prior_status
        )
        resilient = is_llm_resilience_error(exc)
        trace(
            f"EVALUATOR_ERROR | concept={target.id} | "
            f"{type(exc).__name__}: {exc} | resilience={resilient} | "
            "applying degraded threshold (FSM preserved)"
        )
        return apply_degraded_threshold(
            target,
            layer=layer,
            reason=f"{type(exc).__name__}",
            memory=memory,
        )


def _apply_overlay_critique(
    memory: SessionMemory,
    target: SubConceptRecord,
    critique: EvaluatorCritiqueContract,
    *,
    prior_status: str,
    overlay_kind: str = "deep_design",
    curriculum_id: str = "",
    node_id: str = "",
    node_title: str = "",
) -> EvalDirective:
    """
    Apply overlay critique without mutating core WHY/HOW/MECH flags.

    Success → deep_mastery_concepts + overlay_type only; fail → refinement directive.
    """
    from knowledge_engine.src.node_deep_dive.concept_map_state import (
        register_deep_mastery,
    )
    from knowledge_engine.src.node_deep_dive.eval_result_adapter import (
        critique_to_feedback_text,
        critique_to_memory_payload,
    )
    from knowledge_engine.src.node_deep_dive.star_task_fsm import (
        overlay_type_for_kind,
    )

    # Snapshot core flags to prove isolation (and restore if anything drifts).
    why0, how0, mech0 = target.why_passed, target.how_passed, target.mechanic_passed
    status0 = target.status

    feedback = critique_to_feedback_text(critique)
    memory.last_evaluator_critique = critique_to_memory_payload(critique)
    memory.last_evaluator_feedback = feedback[:1200]

    edges = [
        e.strip()
        for e in (critique.unaccounted_edge_cases or [])
        if (e or "").strip()
    ]
    if edges:
        target.focus_hint = "; ".join(edges)[:500]
    elif not critique.passes_threshold:
        target.focus_hint = (
            (critique.verdict_reason or "").strip()
            or "Close remaining edge cases / trade-offs from the asterisk-question."
        )[:500]

    # Evidence for transparency only — not a core-layer credit signal.
    target.merge_evidence((critique.verdict_reason or "").strip())

    passed = bool(critique.passes_threshold) and bool(critique.bloom_level_matched)
    if passed:
        register_deep_mastery(
            memory,
            target.id,
            overlay_type=overlay_type_for_kind(overlay_kind),
        )
        directive: EvalDirective = "DEEP_MASTERY_EARNED"
        if not edges:
            # Keep prior focus empty on clean pass.
            target.focus_hint = ""
        _sync_weakness_ledger_after_overlay(
            memory,
            critique,
            passed=True,
            overlay_kind=overlay_kind,
            curriculum_id=curriculum_id,
            node_id=node_id,
            node_title=node_title,
        )
        trace(
            f"NODE_DIVE deep_mastery_eval ✓ | concept={target.id} "
            f"overlay={critique.target_layer} "
            f"core_untouched=W{int(why0)}H{int(how0)}M{int(mech0)} "
            f"prior_status={prior_status}"
        )
    else:
        directive = "STAR_TASK_NEEDS_REFINEMENT"
        if not (target.focus_hint or "").strip():
            target.focus_hint = (
                "Нужно закрыть крайние случаи и trade-offs из разбора "
                "(зависимости / точки отказа / матрица компромиссов)."
            )[:500]
        trace(
            f"NODE_DIVE star_task needs_refinement eval | concept={target.id} "
            f"overlay={critique.target_layer}"
        )
        _sync_weakness_ledger_after_overlay(
            memory,
            critique,
            passed=False,
            overlay_kind=overlay_kind,
            curriculum_id=curriculum_id,
            node_id=node_id,
            node_title=node_title,
        )

    # Hard isolation: never let overlay mutate core layer flags / verified status.
    target.why_passed = why0
    target.how_passed = how0
    target.mechanic_passed = mech0
    target.status = status0
    return directive


def _sync_weakness_ledger_after_core_eval(
    memory: SessionMemory,
    node: NodeDataInput,
    target: SubConceptRecord,
    *,
    directive: str,
    curriculum_id: str,
) -> None:
    cid = (curriculum_id or "").strip()
    if not cid:
        return
    from knowledge_engine.context_drift_manager import (
        ContextDriftManager,
        tags_from_focus_and_critique,
    )

    d = (directive or "").strip()
    if d.startswith("PROBE_NEXT_LAYER") or (
        (target.focus_hint or "").strip() and target.status != "verified"
    ):
        tags = tags_from_focus_and_critique(
            focus_hint=target.focus_hint or "",
            directive=d,
        )
        if tags:
            ContextDriftManager(cid).record_weaknesses(
                tags,
                node_id=str(getattr(node, "node_id", "") or ""),
                title=str(getattr(node, "title", "") or ""),
                topic_mastery_score=int(getattr(memory, "topic_mastery_score", 0) or 0),
            )


def _sync_weakness_ledger_after_overlay(
    memory: SessionMemory,
    critique: EvaluatorCritiqueContract,
    *,
    passed: bool,
    overlay_kind: str,
    curriculum_id: str,
    node_id: str,
    node_title: str,
) -> None:
    cid = (curriculum_id or "").strip()
    if not cid:
        return
    from knowledge_engine.context_drift_manager import (
        ContextDriftManager,
        tags_from_focus_and_critique,
    )
    from knowledge_engine.schemas.llm_contracts.evaluator_critique import IdeaStatus
    from knowledge_engine.src.node_deep_dive.star_task_fsm import overlay_type_for_kind

    mgr = ContextDriftManager(cid)
    nid = (node_id or "").strip()
    if passed:
        cleared = [
            str(t).strip()
            for t in (critique.cleared_weakness_tags or [])
            if str(t).strip()
        ]
        mgr.clear_weaknesses(
            cleared or None,
            node_id=nid,
            overlay_type=overlay_type_for_kind(overlay_kind),
        )
        return
    weak_concepts = [
        (idea.idea_concept or "").strip()
        for idea in (critique.analyzed_ideas or [])
        if str(getattr(idea, "status", "") or "")
        in (IdeaStatus.WEAK.value, IdeaStatus.RISK.value, "WEAK", "RISK")
    ]
    tags = tags_from_focus_and_critique(
        unaccounted_edge_cases=list(critique.unaccounted_edge_cases or []),
        weak_or_risk_concepts=weak_concepts,
        directive="STAR_TASK_NEEDS_REFINEMENT",
    )
    if tags:
        mgr.record_weaknesses(
            tags,
            node_id=nid,
            title=node_title,
            topic_mastery_score=int(getattr(memory, "topic_mastery_score", 0) or 0),
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
        mark_evaluator_skipped(memory, "empty/short user_message")
        return
    from knowledge_engine.src.node_deep_dive.lecture_scope import (
        is_lecture_request_message,
    )

    if is_lecture_request_message(text):
        logger.info("sub_concept evaluation skip | lecture request")
        mark_evaluator_skipped(
            memory, "lecture request (evaluator only scores real answers)"
        )
        return
    from knowledge_engine.src.node_deep_dive.concept_map import (
        is_quick_reply_control_message,
    )

    if is_quick_reply_control_message(text, memory):
        logger.info("sub_concept evaluation skip | quick-reply control chip")
        mark_evaluator_skipped(
            memory, "quick-reply control (Gloss / Дожать / next — not scored)"
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
        mark_evaluator_skipped(memory, "no pending_evaluation_concept_id")
        return
    from knowledge_engine.src.node_deep_dive.star_task_fsm import (
        apply_star_task_eval_outcome,
        continue_overlay_push_or_resolve,
        core_layer_drill_blocks_overlay_eval,
        get_star_task_status,
        is_overlay_eval_kind,
        star_task_blocks_transition,
    )

    kind_before = (memory.pending_eval_kind or "").strip().lower()
    scoring_star = (
        not core_layer_drill_blocks_overlay_eval(memory)
        and (
            is_overlay_eval_kind(kind_before) or star_task_blocks_transition(memory)
        )
    )
    trace(
        f"NODE_DIVE sub_concept evaluation ▶ | target={pending_id} "
        f"next_q={memory.next_question_concept_id or '—'} "
        f"layer={getattr(node, 'layer', '') or 'foundation'} "
        f"star={get_star_task_status(memory)} kind={kind_before or '—'}"
    )
    row = find_sub_concept(memory, pending_id)
    if row is None:
        logger.error(
            "sub_concept evaluation skip | pending id %r not in map", pending_id
        )
        mark_evaluator_skipped(memory, "pending id not in map")
        return

    mark_evaluator_ran(memory)

    from knowledge_engine.src.node_deep_dive.drill_orchestrator import (
        capture_layer_completion_snapshot,
        latch_layer_just_completed,
    )

    before = capture_layer_completion_snapshot(memory)
    try:
        directive = run_sub_concept_gap_eval(
            text,
            memory,
            node,
            anchor,
            concept_id=pending_id,
            chat_sessions=memory.chat_sessions,
        )
    except Exception as exc:
        from knowledge_engine.src.resilience_manager import is_llm_resilience_error

        logger.exception(
            "EVALUATOR_EXCEPTION process_answer concept=%s", pending_id
        )
        trace(
            f"EVALUATOR_ERROR | process_answer | {type(exc).__name__}: {exc} | "
            f"resilience={is_llm_resilience_error(exc)} | FSM preserved"
        )
        if scoring_star:
            memory.last_eval_directive = "STAR_TASK_NEEDS_REFINEMENT"
            if not (memory.last_evaluator_feedback or "").strip():
                memory.last_evaluator_feedback = (
                    "Проверка задачки со звёздочкой временно недоступна. "
                    "Контекст сохранён — повторите ответ."
                )[:1200]
            apply_star_task_eval_outcome(
                memory, concept_id=pending_id, resolved=False
            )
            sync_concepts_matrix_from_sub_concepts(memory)
            latch_layer_just_completed(memory, before)
            return
        layer = str(getattr(node, "layer", "") or "foundation")
        directive = apply_degraded_threshold(
            row, layer=layer, reason=f"{type(exc).__name__}", memory=memory
        )
        memory.last_eval_directive = directive or ""
        memory.last_evaluator_feedback = build_evaluator_feedback(row)
        sync_concepts_matrix_from_sub_concepts(memory)
        latch_layer_just_completed(memory, before)
        return
    if directive:
        memory.last_eval_directive = directive
    else:
        memory.last_eval_directive = memory.last_eval_directive or ""
        logger.warning(
            "sub_concept evaluation returned no directive | concept=%s status=%s",
            pending_id,
            row.status,
        )
    # Overlay path writes critique-backed feedback; keep it for Tutor.
    if scoring_star and (memory.last_eval_directive or "") in (
        "DEEP_MASTERY_EARNED",
        "STAR_TASK_NEEDS_REFINEMENT",
    ):
        if not (memory.last_evaluator_feedback or "").strip():
            memory.last_evaluator_feedback = build_evaluator_feedback(row)
    else:
        memory.last_evaluator_feedback = build_evaluator_feedback(row)

    if scoring_star:
        resolved = (directive or "").strip() == "DEEP_MASTERY_EARNED"
        if not resolved:
            apply_star_task_eval_outcome(
                memory,
                concept_id=pending_id,
                resolved=False,
            )
            sync_concepts_matrix_from_sub_concepts(memory)
            latch_layer_just_completed(memory, before)
            return
        continue_overlay_push_or_resolve(
            memory,
            concept_id=pending_id,
            resolved=True,
            overlay_kind=kind_before,
        )
        sync_concepts_matrix_from_sub_concepts(memory)
        latch_layer_just_completed(memory, before)
        return

    advance_sub_concepts_after_user_answer(
        memory,
        text,
        concept_id=pending_id,
    )
    advance_next_question_after_evaluation(memory, evaluated_id=pending_id)
    memory.pending_evaluation_concept_id = ""
    memory.pending_eval_kind = ""
    memory.last_tutor_sub_concept_id = ""
    sync_concepts_matrix_from_sub_concepts(memory)
    latch_layer_just_completed(memory, before)
