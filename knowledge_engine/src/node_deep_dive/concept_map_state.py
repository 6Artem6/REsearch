"""Topic Concept Map: SessionMemory sub_concepts (pure state, no text heuristics)."""

from __future__ import annotations

import re

from knowledge_engine.schemas.global_knowledge import utc_now_iso
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    OverlayMasteryRecord,
    OverlayType,
    SessionMemory,
    SubConceptRecord,
    SubConceptStatus,
)
from knowledge_engine.src.node_deep_dive.schemas import (
    CoverageItem,
    CoverageLayerProgress,
    CoverageLayers,
    CoverageSummary,
    NodeDataInput,
)
from knowledge_engine.ui.run_log import trace

_STATUS_RANK: dict[SubConceptStatus, int] = {
    "verified": 4,
    "partial": 3,
    "gap": 2,
    "unchecked": 1,
}

_UI_STATE: dict[SubConceptStatus, str] = {
    "verified": "verified",
    "partial": "in_progress",
    "gap": "gap",
    "unchecked": "unchecked",
}


def _touch_sub_concept(row: SubConceptRecord) -> None:
    row.updated_at = utc_now_iso()


def is_core_sub_concept(sc: SubConceptRecord) -> bool:
    """Core map row — overlay extensions never enter Core Progress %."""
    return not bool(getattr(sc, "is_extension", False))


def core_sub_concepts(memory: SessionMemory) -> list[SubConceptRecord]:
    """Sub-concepts that form the node core (denominator of topic_mastery_score)."""
    return [sc for sc in (memory.sub_concepts or []) if is_core_sub_concept(sc)]


def optional_teaching_layer(memory: SessionMemory) -> str:
    """HOW | MECHANIC while a Layer Drill Session (or compat flag) is live."""
    drill = getattr(memory, "layer_drill", None)
    if drill is not None and bool(getattr(drill, "is_active", False)):
        layer = (getattr(drill, "target_layer", None) or "").strip().upper()
        if layer == "HOW":
            return "HOW"
        if layer == "MECH":
            return "MECHANIC"
    return (getattr(memory, "active_optional_layer", "") or "").strip().upper()


def overlay_push_kind(memory: SessionMemory) -> str:
    """Factory overlay kind while an asterisk-question push session is live."""
    from knowledge_engine.src.node_deep_dive.star_task_fsm import is_overlay_eval_kind

    kind = (getattr(memory, "pending_eval_kind", None) or "").strip().lower()
    if is_overlay_eval_kind(kind):
        return kind
    return ""


def row_open_for_optional_depth_eval(
    row: SubConceptRecord,
    *,
    layer_name: str = "",
) -> bool:
    """True when a (often VERIFIED) core row still has HOW/MECH to score."""
    name = (layer_name or "").strip().upper()
    why = bool(row.why_passed)
    if name == "HOW":
        return why and not bool(row.how_passed)
    if name in ("MECHANIC", "MECH"):
        return why and not bool(row.mechanic_passed)
    if not why:
        return False
    return (not bool(row.how_passed)) or (not bool(row.mechanic_passed))


def row_is_fully_mastered(row: SubConceptRecord) -> bool:
    """Verified sub-topic with WHY+HOW+MECH all closed — never quiz again."""
    return (
        row.status == "verified"
        and bool(row.why_passed)
        and bool(row.how_passed)
        and bool(row.mechanic_passed)
    )


_ATTRACT_FAILED_ATTEMPTS = 2


def first_open_optional_layer_row(
    memory: SessionMemory,
    layer_name: str,
) -> SubConceptRecord | None:
    """First core row still missing ``layer_name`` (HOW or MECHANIC)."""
    name = (layer_name or "").strip().upper()
    if name not in ("HOW", "MECHANIC"):
        return None
    for sc in core_sub_concepts(memory):
        if row_open_for_optional_depth_eval(sc, layer_name=name):
            return sc
    return None


def _overlay_pending_open(memory: SessionMemory) -> bool:
    from knowledge_engine.src.node_deep_dive.star_task_fsm import (
        is_overlay_eval_kind,
        star_task_blocks_transition,
    )

    return is_overlay_eval_kind(
        getattr(memory, "pending_eval_kind", None)
    ) or star_task_blocks_transition(memory)


def list_verified_sub_concept_ids(memory: SessionMemory) -> list[str]:
    return [sc.id for sc in core_sub_concepts(memory) if sc.status == "verified"]


def slug_sub_concept_id(label: str, index: int = 0) -> str:
    t = (label or "").strip().lower()
    t = re.sub(r"[^\w\s-]", "", t, flags=re.UNICODE)
    t = re.sub(r"[\s-]+", "_", t).strip("_")
    if not t:
        return f"sub_{index + 1}"
    return t[:48]


def _heuristic_sub_concepts(node: NodeDataInput) -> list[SubConceptRecord]:
    concepts = [str(c).strip() for c in (node.core_concepts or []) if str(c).strip()]
    if not concepts:
        concepts = [node.title.strip()] if node.title else ["Core topic"]
    out: list[SubConceptRecord] = []
    for i, c in enumerate(concepts[:6]):
        out.append(
            SubConceptRecord(
                id=slug_sub_concept_id(c, i),
                label=c[:200],
                success_criterion=f"Практическое понимание: {c[:300]}",
                status="unchecked",
            )
        )
    return out


def ensure_sub_concept_map(memory: SessionMemory, node: NodeDataInput) -> None:
    if memory.sub_concepts:
        return
    memory.node_goal = (
        (node.learning_goal or "").strip()
        or (node.brief_summary or "").strip()
        or (node.title or "").strip()
    )[:800]
    memory.sub_concepts = _heuristic_sub_concepts(node)
    trace(
        f"NODE_DIVE concept_map ✓ | {node.node_id} "
        f"sub_concepts={len(memory.sub_concepts)}"
    )


def find_sub_concept(
    memory: SessionMemory,
    ref: str,
) -> SubConceptRecord | None:
    key = (ref or "").strip().lower()
    if not key:
        return None
    for sc in memory.sub_concepts:
        if sc.id.lower() == key or sc.label.lower() == key:
            return sc
    for sc in memory.sub_concepts:
        if key in sc.label.lower() or key in sc.id.lower():
            return sc
    return None


def _normalize_patch_status(raw: str) -> SubConceptStatus | None:
    t = (raw or "").strip().upper()
    if t == "VERIFIED":
        return "verified"
    if t in ("PARTIAL", "IN_PROGRESS"):
        return "partial"
    if t == "GAP":
        return "gap"
    if t in ("UNCHECKED", "UN_CHECKED"):
        return "unchecked"
    return None


def trace_coverage_update(
    concept_id: str,
    old: SubConceptStatus,
    new: SubConceptStatus,
    *,
    source: str,
) -> None:
    if old == new:
        return
    trace(f"COVERAGE_UPDATE | id={concept_id} {old}→{new} | source={source}")


def build_evaluator_feedback(row: SubConceptRecord) -> str:
    label = (row.label or row.id).strip()
    crit = (row.success_criterion or "").strip()
    hint = (row.focus_hint or "").strip()
    ev = (row.evidence or "").strip()
    layers = (
        f"слои WHY={int(row.why_passed)} HOW={int(row.how_passed)} "
        f"MECHANIC={int(row.mechanic_passed)}"
    )
    if row.status == "verified":
        if row.why_passed and row.how_passed and not row.mechanic_passed:
            return (
                f"Подтема «{label}» засчитана (VERIFIED, порог закрыт; {layers}). "
                "MECHANIC опционален — тьютор может кратко пояснить сам."
            )
        return f"Подтема «{label}» засчитана (VERIFIED; {layers})."
    if row.status == "gap":
        base = f"Подтема «{label}» не закрыта (GAP; {layers})."
        if hint:
            return f"{base} Не хватило: {hint}"
        if crit:
            return f"{base} Критерий зачёта: {crit}"
        return base
    if row.status == "partial":
        base = f"Подтема «{label}» частично (PARTIAL; {layers})."
        if hint:
            return f"{base} Для порога сдачи не хватило: {hint}"
        if crit:
            return f"{base} Нужно раскрыть: {crit}"
        if ev:
            return f"{base} Уже есть: {ev[:200]}"
        return base
    return ""


def resolve_transparency_row(
    memory: SessionMemory,
    *,
    prefer_id: str = "",
) -> SubConceptRecord | None:
    """Deterministic sub-topic row for evaluator transparency / UI plaque."""
    candidates: list[str] = []
    for raw in (
        prefer_id,
        memory.asked_question_sub_concept_id,
        memory.pending_evaluation_concept_id,
        memory.next_question_concept_id,
    ):
        cid = (raw or "").strip()
        if cid and cid not in candidates:
            candidates.append(cid)
    for cid in candidates:
        found = find_sub_concept(memory, cid)
        if found is not None and found.status in ("partial", "gap"):
            return found
    for sc in memory.sub_concepts or []:
        if sc.status in ("partial", "gap") and (sc.focus_hint or "").strip():
            return sc
    return None


def format_evaluator_transparency_payload(
    memory: SessionMemory,
    *,
    prefer_id: str = "",
) -> str:
    """
    Structured evidence + focus_hint for tutor prompt (PARTIAL/GAP only).

    Ensures focus_hint / evidence are explicit fields, not only prose
    in last_evaluator_feedback.
    """
    status_label = {
        "verified": "VERIFIED",
        "partial": "PARTIAL",
        "gap": "GAP",
        "unchecked": "UNCHECKED",
    }
    row = resolve_transparency_row(memory, prefer_id=prefer_id)
    if row is None:
        return ""
    hint = (row.focus_hint or "").strip()
    if not hint:
        return ""
    directive = (memory.last_eval_directive or "").strip()
    lines = [
        "[EVALUATOR_TRANSPARENCY]",
        f"last_evaluator_sub_concept_id: {row.id}",
        f"last_evaluator_status: {status_label.get(row.status, row.status.upper())}",
        (
            f"last_evaluator_layers: WHY={int(row.why_passed)} "
            f"HOW={int(row.how_passed)} MECHANIC={int(row.mechanic_passed)}"
        ),
        f"last_evaluator_evidence: {(row.evidence or '').strip()[:800] or '(none)'}",
        f"last_evaluator_focus_hint: {hint[:500]}",
        "focus_hint language: Russian learner-facing — copy as-is, do not re-translate.",
    ]
    if directive:
        lines.append(f"last_eval_directive: {directive}")
    lines.append(
        "TRANSPARENCY: Host prepends the credited/missing plaque in Python. "
        "Do not emit 📋/🎯. Use last_evaluator_focus_hint in "
        "correction_breakdown as technical Russian prose."
    )
    return "\n".join(lines)


def probe_layer_from_directive(directive: str) -> str:
    """Return WHY/HOW/MECHANIC from ``PROBE_NEXT_LAYER:X``, else empty."""
    text = (directive or "").strip()
    if not text.startswith("PROBE_NEXT_LAYER:"):
        return ""
    return text.split(":", 1)[-1].strip().upper()


def format_lecture_target_focus_and_gaps(
    memory: SessionMemory,
    *,
    prefer_id: str = "",
) -> str:
    """Lecture-only steering block: open layer + focus_hint + credited evidence.

    Distinct from ``[EVALUATOR_TRANSPARENCY]`` (dialogue plaque contract):
    this block tells dense lecture how to spend ``lecture_body`` budget and
    what ``checkpoint_prompt`` must test.
    """
    directive = (memory.last_eval_directive or "").strip()
    probe_layer = probe_layer_from_directive(directive)
    row = resolve_transparency_row(memory, prefer_id=prefer_id)
    hint = (row.focus_hint or "").strip() if row else ""
    evidence = (row.evidence or "").strip() if row else ""
    if not hint and not probe_layer:
        return ""
    lines = [
        "[TARGET_FOCUS_AND_GAPS]",
        f"last_eval_directive: {directive or '(none)'}",
        f"probe_layer: {probe_layer or '(none)'}",
        f"last_evaluator_focus_hint: {hint or '(none)'}",
        f"last_evaluator_evidence: {evidence[:800] or '(none)'}",
        "BUDGET: spend ≥80% of lecture_body on probe_layer / focus_hint "
        "mechanics (C-structures, memory, invariants, race conditions). "
        "Credited last_evaluator_evidence is brief context only — do not "
        "re-teach already passed layers.",
        "CHECKPOINT: checkpoint_prompt MUST test this gap / probe_layer. "
        "FORBIDDEN: blind RE-STATE of [OPEN_NODE_QUESTION] if it belongs "
        "to an already-passed layer.",
        "Do not copy 📋/🎯 scoreboard strings into lecture_body.",
    ]
    return "\n".join(lines)


def stream_host_transparency_plaque(
    stream_callback,
    memory: SessionMemory | None,
) -> str:
    """SSE the Host plaque as the first tutor tokens, then return the same text.

    Persistence still merges this into ``feedback_on_answer`` after the LLM
    returns. Streaming it first keeps the frontend prefix-stable: the plaque
    types with the message instead of jumping in at complete on top.
    """
    plaque = compose_host_transparency_plaque(memory)
    if plaque and stream_callback is not None:
        stream_callback(plaque + "\n\n")
    return plaque


def compose_host_transparency_plaque(memory: SessionMemory | None) -> str:
    """UI plaque assembled by Host — never generated by the Tutor LLM."""
    if memory is None or bool(getattr(memory, "evaluator_skipped", False)):
        return ""
    directive = (memory.last_eval_directive or "").strip()
    if not (
        directive.startswith("PROBE_NEXT_LAYER:")
        or directive == "STAR_TASK_NEEDS_REFINEMENT"
    ):
        return ""
    row = resolve_transparency_row(memory)
    if row is None:
        return ""
    hint = (row.focus_hint or "").strip()
    evidence = (row.evidence or "").strip()
    if not hint and not evidence:
        return ""
    lines = ["---"]
    lines.append(f"**📋 Что уже зачтено:** {evidence or '—'}")
    if hint:
        lines.append(f"**🎯 Чего не хватило для полного зачёта:** {hint}")
    lines.append("---")
    return "\n".join(lines)


def apply_sub_concept_updates(
    memory: SessionMemory,
    updates: list[dict],
    *,
    restrict_to_id: str = "",
) -> None:
    """
    Legacy/compat patch path.

    Prefer Threshold Engine (`apply_threshold_to_sub_concept`) for mastery.
    When layer flags are present, OR-merge them; status still accepted if provided.
    """
    only = (restrict_to_id or "").strip()
    for item in updates or []:
        sid = str(item.get("id") or item.get("sub_concept_id") or "").strip()
        if not sid:
            continue
        if only and sid != only:
            continue
        row = find_sub_concept(memory, sid)
        if row is None:
            continue

        if any(k in item for k in ("why_passed", "how_passed", "mechanic_passed")):
            row.why_passed = bool(row.why_passed) or bool(item.get("why_passed"))
            row.how_passed = bool(row.how_passed) or bool(item.get("how_passed"))
            row.mechanic_passed = bool(row.mechanic_passed) or bool(
                item.get("mechanic_passed")
            )
            _touch_sub_concept(row)

        new_st = _normalize_patch_status(str(item.get("status") or ""))
        if new_st is not None:
            old_st = row.status
            if old_st == "verified" and new_st in ("unchecked", "gap", "partial"):
                regress = (item.get("evidence") or "").strip()
                if not regress or new_st != "gap":
                    new_st = None
            if new_st is not None:
                if _STATUS_RANK.get(new_st, 0) >= _STATUS_RANK.get(old_st, 0):
                    if old_st != new_st:
                        trace_coverage_update(
                            sid, old_st, new_st, source="gap_eval_patch"
                        )
                    row.status = new_st
                    _touch_sub_concept(row)
                elif new_st == "gap" and old_st == "verified":
                    trace_coverage_update(sid, old_st, "gap", source="gap_eval_regress")
                    row.status = "gap"
                    _touch_sub_concept(row)

        ev = (item.get("evidence") or "").strip()
        if ev:
            row.merge_evidence(ev)
        hint = (item.get("focus_hint") or "").strip()
        if hint:
            row.focus_hint = hint[:500]


def _last_tutor_message_text(memory: SessionMemory) -> str:
    for item in reversed(memory.active_window or []):
        if (item.get("role") or "").strip() == "tutor":
            return (item.get("content") or "").strip()
    at = memory.anchor_turn
    if isinstance(at, dict) and (at.get("role") or "tutor").strip() == "tutor":
        return (at.get("content") or "").strip()
    return ""


def last_tutor_question_text_for_eval(memory: SessionMemory) -> str:
    window = _last_tutor_message_text(memory)
    follow = (memory.last_tutor_follow_up_question or "").strip()
    if follow and follow not in window:
        if window:
            return f"{window}\n\n{follow}"
        return follow
    return window


def stored_pending_evaluation_id(memory: SessionMemory) -> str:
    """
    Outstanding question id for the evaluator.

    Requires pending_evaluation_concept_id (set when tutor asked). Prefers
    asked_question_sub_concept_id when both are set (must match).
    """
    pending = (memory.pending_evaluation_concept_id or "").strip()
    asked = (memory.asked_question_sub_concept_id or "").strip()
    if not pending:
        # Legacy fallback only
        legacy = (memory.last_tutor_sub_concept_id or "").strip()
        if legacy and find_sub_concept(memory, legacy):
            return legacy
        return ""
    if asked and asked != pending:
        trace(
            f"WARN asked/pending desync | asked={asked} pending={pending} — prefer asked"
        )
        if find_sub_concept(memory, asked):
            return asked
    if find_sub_concept(memory, pending):
        return pending
    if asked and find_sub_concept(memory, asked):
        return asked
    return ""


def active_question_sub_concept_id(memory: SessionMemory) -> str:
    """Alias: asked question id for evaluator (not next_question generation focus)."""
    return stored_pending_evaluation_id(memory)


def resolve_evaluation_target_id(memory: SessionMemory) -> str:
    stored = stored_pending_evaluation_id(memory)
    if not stored:
        return ""
    row = find_sub_concept(memory, stored)
    overlay_open = _overlay_pending_open(memory)
    teaching = optional_teaching_layer(memory)
    optional_open = (
        teaching in ("HOW", "MECHANIC")
        and row is not None
        and row_open_for_optional_depth_eval(row, layer_name=teaching)
    )
    if (
        row is not None
        and row.status == "verified"
        and not overlay_open
        and not optional_open
    ):
        trace(f"EVALUATOR_SKIP | stored asked/pending is VERIFIED id={stored}")
        return ""
    return stored


def resolve_pending_evaluation_id(
    memory: SessionMemory,
    user_message: str = "",
) -> str:
    _ = user_message
    return resolve_evaluation_target_id(memory)


def set_pending_evaluation_for_tutor_turn(
    memory: SessionMemory,
    focus_sub_concept_id: str,
) -> str:
    cid = (focus_sub_concept_id or "").strip()
    if not cid:
        trace("WARN pending not set | empty focus_sub_concept_id")
        return ""
    row = find_sub_concept(memory, cid)
    if row is None:
        trace(f"WARN pending not set | unknown sub_concept id={cid}")
        return ""
    if row.status == "verified":
        from knowledge_engine.src.node_deep_dive.star_task_fsm import (
            layer_drill_is_active,
            row_open_for_drill_layer,
        )

        teaching = optional_teaching_layer(memory)
        drill_ok = layer_drill_is_active(memory) and row_open_for_drill_layer(
            memory, row, getattr(memory.layer_drill, "target_layer", None)
        )
        allowed = (
            _overlay_pending_open(memory)
            or drill_ok
            or (
                teaching in ("HOW", "MECHANIC")
                and row_open_for_optional_depth_eval(row, layer_name=teaching)
            )
        )
        if not allowed:
            trace(f"WARN pending not set | focus already VERIFIED id={cid}")
            return ""
    memory.asked_question_sub_concept_id = cid
    memory.pending_evaluation_concept_id = cid
    memory.last_tutor_sub_concept_id = cid
    # Generation focus stays on the asked id until VERIFIED
    memory.next_question_concept_id = cid
    return cid


def select_next_sub_concept(
    memory: SessionMemory,
    *,
    skip_id: str = "",
) -> SubConceptRecord | None:
    skip = (skip_id or "").strip().lower()
    from knowledge_engine.src.node_deep_dive.star_task_fsm import (
        current_layer_drill_concept_id,
        layer_drill_is_active,
        row_open_for_drill_layer,
    )

    if layer_drill_is_active(memory):
        current = current_layer_drill_concept_id(memory)
        if current and current.lower() != skip:
            row = find_sub_concept(memory, current)
            if row is not None and row_open_for_drill_layer(
                memory, row, getattr(memory.layer_drill, "target_layer", None)
            ):
                return row
        teaching = optional_teaching_layer(memory)
        if teaching in ("HOW", "MECHANIC"):
            for sc in core_sub_concepts(memory):
                if skip and sc.id.lower() == skip:
                    continue
                if row_open_for_optional_depth_eval(sc, layer_name=teaching):
                    return sc
            return None
        overlay_kind = overlay_push_kind(memory)
        if overlay_kind:
            return first_open_overlay_row(memory, overlay_kind, skip_id=skip_id)
        return None
    teaching = optional_teaching_layer(memory)
    if teaching in ("HOW", "MECHANIC"):
        asked = (memory.asked_question_sub_concept_id or "").strip() or (
            memory.pending_evaluation_concept_id or ""
        ).strip()
        if asked and asked.lower() != skip:
            row = find_sub_concept(memory, asked)
            if row is not None and row_open_for_optional_depth_eval(
                row, layer_name=teaching
            ):
                return row
        for sc in core_sub_concepts(memory):
            if skip and sc.id.lower() == skip:
                continue
            if row_open_for_optional_depth_eval(sc, layer_name=teaching):
                return sc
        return None
    overlay_kind = overlay_push_kind(memory)
    if overlay_kind:
        from knowledge_engine.src.node_deep_dive.star_task_fsm import (
            overlay_type_for_kind,
        )

        otype = overlay_type_for_kind(overlay_kind)
        asked = (memory.asked_question_sub_concept_id or "").strip() or (
            memory.pending_evaluation_concept_id or ""
        ).strip()
        if asked and asked.lower() != skip:
            row = find_sub_concept(memory, asked)
            if row is not None and not has_overlay_award(memory, row.id, otype):
                return row
        return first_open_overlay_row(memory, overlay_kind, skip_id=skip_id)
    verified_ids = {sc.id for sc in memory.sub_concepts if sc.status == "verified"}
    # Prefer asked/pending until VERIFIED — do not jump active ahead of evaluation
    asked = (memory.asked_question_sub_concept_id or "").strip() or (
        memory.pending_evaluation_concept_id or ""
    ).strip()
    if asked and asked.lower() not in verified_ids:
        row = find_sub_concept(memory, asked)
        if row is not None and row.status != "verified":
            return row
    pending = (memory.pending_evaluation_concept_id or "").strip()
    if pending and pending.lower() not in verified_ids:
        row = find_sub_concept(memory, pending)
        if row is not None and row.status != "verified":
            return row
    candidates = [
        sc
        for sc in core_sub_concepts(memory)
        if sc.status != "verified"
        and sc.id not in verified_ids
        and (not skip or sc.id.lower() != skip)
    ]
    if not candidates:
        return None

    def sort_key(sc: SubConceptRecord) -> tuple[int, int]:
        pri = {"gap": 0, "unchecked": 1, "partial": 2}.get(sc.status, 3)
        grade = (sc.last_accuracy_grade or "").strip()
        attract = (
            grade == "MISUNDERSTANDING"
            or int(sc.failed_attempts or 0) >= _ATTRACT_FAILED_ATTEMPTS
        )
        if attract:
            pri = max(0, pri - 1)
        return (pri, memory.sub_concepts.index(sc))

    return sorted(candidates, key=sort_key)[0]


def advance_next_question_after_evaluation(
    memory: SessionMemory,
    *,
    evaluated_id: str = "",
) -> str:
    """
    Advance active generation focus ONLY after asked id is VERIFIED.

    PARTIAL/GAP → keep next_question on evaluated_id.
    Optional HOW/MECH session → stay until that flag is set, then the next
    core row still missing the layer (even if status is already verified).
    """
    ev_id = (evaluated_id or "").strip()
    from knowledge_engine.src.node_deep_dive.star_task_fsm import (
        advance_layer_drill_after_pass,
        current_layer_drill_concept_id,
        layer_drill_is_active,
    )

    if layer_drill_is_active(memory):
        advance_layer_drill_after_pass(memory, evaluated_id=ev_id)
        cid = current_layer_drill_concept_id(memory)
        memory.next_question_concept_id = cid
        if not cid:
            memory.asked_question_sub_concept_id = ""
        return cid
    teaching = optional_teaching_layer(memory)
    if teaching in ("HOW", "MECHANIC"):
        if ev_id:
            row = find_sub_concept(memory, ev_id)
            if row is not None and row_open_for_optional_depth_eval(
                row, layer_name=teaching
            ):
                memory.next_question_concept_id = ev_id
                memory.asked_question_sub_concept_id = ev_id
                return ev_id
            memory.asked_question_sub_concept_id = ""
        nxt = select_next_sub_concept(memory, skip_id=ev_id)
        cid = (nxt.id if nxt else "").strip()
        memory.next_question_concept_id = cid
        if not cid:
            memory.active_optional_layer = ""
        return cid
    if ev_id:
        row = find_sub_concept(memory, ev_id)
        if row is not None and row.status != "verified":
            memory.next_question_concept_id = ev_id
            memory.asked_question_sub_concept_id = ev_id
            return ev_id
        if row is not None and row.status == "verified":
            # Asked closed — clear asked pointer; pick next unchecked
            memory.asked_question_sub_concept_id = ""
    nxt = select_next_sub_concept(memory, skip_id=ev_id)
    cid = (nxt.id if nxt else "").strip()
    memory.next_question_concept_id = cid
    return cid


def advance_sub_concepts_after_user_answer(
    memory: SessionMemory,
    user_message: str,
    *,
    concept_id: str,
) -> None:
    _ = (memory, user_message, concept_id)


def promote_sub_concepts_after_tutor_message(
    memory: SessionMemory,
    tutor_message: str,
) -> None:
    _ = (memory, tutor_message)


def sync_concepts_matrix_from_sub_concepts(memory: SessionMemory) -> None:
    if not memory.sub_concepts or not memory.concepts_matrix:
        return
    from knowledge_engine.src.node_deep_dive.tiered_memory import find_concept_record

    for sc in memory.sub_concepts:
        if sc.status != "verified":
            continue
        row = find_concept_record(memory.concepts_matrix, sc.label)
        if row is None:
            continue
        row.status = "verified"
        row.mastery_score = max(int(row.mastery_score), 85)
        if not (row.evidence or "").strip() and (sc.evidence or "").strip():
            row.evidence = sc.evidence[:2000]


def sub_concept_coverage_complete(memory: SessionMemory) -> bool:
    core = core_sub_concepts(memory)
    if not core:
        return False
    return all(sc.status == "verified" for sc in core)


def _active_layer_from_directive(directive: str) -> str | None:
    raw = (directive or "").strip().upper()
    if raw.startswith("PROBE_NEXT_LAYER:"):
        layer = raw.split(":", 1)[-1].strip()
        if layer in ("WHY", "HOW", "MECHANIC"):
            return layer
    if raw == "PASSED_WITH_GLOSS":
        return "MECHANIC"
    return None


def _layer_progress(
    flags: list[bool],
    *,
    is_active: bool,
    gloss: bool = False,
) -> CoverageLayerProgress:
    n = len(flags)
    if n <= 0:
        return CoverageLayerProgress(status="locked", score=0.0)
    score = sum(1.0 for f in flags if f) / float(n)
    if score >= 1.0 - 1e-9:
        return CoverageLayerProgress(status="passed", score=1.0)
    if gloss and score < 1.0:
        return CoverageLayerProgress(status="gloss", score=round(score, 3))
    if is_active or score > 0:
        return CoverageLayerProgress(
            status="in_progress",
            score=round(score, 3),
        )
    return CoverageLayerProgress(status="locked", score=0.0)


def _status_hint_for_sub_concept(sc: SubConceptRecord) -> str:
    """Short UI line — never repeats the subtopic label."""
    why = bool(sc.why_passed)
    how = bool(sc.how_passed)
    mech = bool(sc.mechanic_passed)
    raw_focus = (sc.focus_hint or "").strip()
    # Prefer evaluator focus_hint if it does not echo the label.
    label = (sc.label or "").strip()
    if raw_focus:
        low = raw_focus.lower()
        if label and label.lower() in low:
            # Strip «Подтема «X»: …» / label prefix when present
            cleaned = raw_focus
            for prefix in (
                f"Подтема «{label}»:",
                f"Подтема «{label}»",
                f"«{label}»:",
                f"{label}:",
            ):
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix) :].strip()
                    break
            if cleaned and label.lower() not in cleaned.lower():
                return cleaned[:240]
        else:
            return raw_focus[:240]
    if why and how and mech:
        return ""
    if why and how and not mech:
        return "Не хватает механик реализации"
    if why and not how:
        return "Не хватает архитектуры (HOW)"
    if not why and (how or mech):
        return "Концепция (WHY) не раскрыта"
    if not why and not how and not mech:
        return "Ещё не затронута"
    return ""


def _overlay_record_concept_id(raw: object) -> str:
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        return str(raw.get("concept_id") or "").strip()
    return str(getattr(raw, "concept_id", "") or "").strip()


def _overlay_record_type(raw: object) -> OverlayType:
    if isinstance(raw, str):
        return "DEEP_ASTERISK"
    val = ""
    if isinstance(raw, dict):
        val = str(raw.get("overlay_type") or "").strip().upper()
    else:
        val = str(getattr(raw, "overlay_type", "") or "").strip().upper()
    if val == "ADVANCED_ASTERISK":
        return "ADVANCED_ASTERISK"
    return "DEEP_ASTERISK"


def list_overlay_mastery_records(memory: SessionMemory) -> list[OverlayMasteryRecord]:
    """Typed overlay awards (concept_id + ADVANCED_ASTERISK | DEEP_ASTERISK)."""
    out: list[OverlayMasteryRecord] = []
    seen: set[tuple[str, str]] = set()
    for raw in memory.deep_mastery_concepts or []:
        if isinstance(raw, OverlayMasteryRecord):
            rec = raw
        else:
            cid = _overlay_record_concept_id(raw)
            if not cid:
                continue
            rec = OverlayMasteryRecord(
                concept_id=cid,
                overlay_type=_overlay_record_type(raw),
            )
        key = (rec.concept_id, rec.overlay_type)
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out[:16]


def list_deep_mastery_concept_ids(memory: SessionMemory) -> list[str]:
    """Stable unique ids with Asterisk-question Deep Mastery (any overlay type)."""
    out: list[str] = []
    seen: set[str] = set()
    for rec in list_overlay_mastery_records(memory):
        cid = rec.concept_id
        if not cid or cid in seen:
            continue
        seen.add(cid)
        out.append(cid)
    return out[:16]


def has_overlay_award(
    memory: SessionMemory,
    concept_id: str,
    overlay_type: OverlayType | str | None = None,
) -> bool:
    cid = (concept_id or "").strip()
    if not cid:
        return False
    want = None
    if overlay_type:
        raw = str(overlay_type).strip().upper()
        want = (
            "ADVANCED_ASTERISK"
            if raw in ("ADVANCED_ASTERISK", "ADVANCED", "ADVANCED_ANALYSIS")
            else "DEEP_ASTERISK"
        )
    for rec in list_overlay_mastery_records(memory):
        if rec.concept_id != cid:
            continue
        if want is None or rec.overlay_type == want:
            return True
    return False


def has_deep_mastery(memory: SessionMemory, concept_id: str = "") -> bool:
    ids = list_deep_mastery_concept_ids(memory)
    want = (concept_id or "").strip()
    if want:
        return want in ids
    return bool(ids)


def first_open_overlay_row(
    memory: SessionMemory,
    overlay_kind: str,
    *,
    skip_id: str = "",
) -> SubConceptRecord | None:
    """First core row still missing this overlay award (ADVANCED or DEEP asterisk)."""
    from knowledge_engine.src.node_deep_dive.star_task_fsm import overlay_type_for_kind

    otype = overlay_type_for_kind(overlay_kind)
    skip = (skip_id or "").strip().lower()
    for sc in core_sub_concepts(memory):
        if skip and sc.id.lower() == skip:
            continue
        if not has_overlay_award(memory, sc.id, otype):
            return sc
    return None


def register_deep_mastery(
    memory: SessionMemory,
    concept_id: str,
    overlay_type: OverlayType | str = "DEEP_ASTERISK",
) -> bool:
    """
    Record Asterisk-question overlay award for a sub-concept.

    Parallel to WHY/HOW/MECH — does not change base depth %.
    Same concept may earn both ADVANCED_ASTERISK and DEEP_ASTERISK.
    Returns True when newly added.
    """
    cid = (concept_id or "").strip()
    if not cid:
        return False
    raw = str(overlay_type or "DEEP_ASTERISK").strip().upper()
    otype: OverlayType = (
        "ADVANCED_ASTERISK"
        if raw in ("ADVANCED_ASTERISK", "ADVANCED", "ADVANCED_ANALYSIS")
        else "DEEP_ASTERISK"
    )
    records = list_overlay_mastery_records(memory)
    if any(r.concept_id == cid and r.overlay_type == otype for r in records):
        memory.deep_mastery_concepts = records
        return False
    records.append(OverlayMasteryRecord(concept_id=cid, overlay_type=otype))
    memory.deep_mastery_concepts = records[:16]
    trace(
        f"NODE_DIVE asterisk_question_deep_mastery | concept={cid} | "
        f"overlay_type={otype} | n={len(records)}"
    )
    return True


def build_coverage_summary(memory: SessionMemory) -> CoverageSummary | None:
    if not memory.sub_concepts:
        return None
    items: list[CoverageItem] = []
    verified = 0
    why_flags: list[bool] = []
    how_flags: list[bool] = []
    mech_flags: list[bool] = []
    core_rows = core_sub_concepts(memory)
    for sc in memory.sub_concepts:
        ui = _UI_STATE.get(sc.status, "unchecked")
        items.append(
            CoverageItem(
                id=sc.id,
                label=sc.label,
                state=ui,  # type: ignore[arg-type]
                why_passed=bool(sc.why_passed),
                how_passed=bool(sc.how_passed),
                mechanic_passed=bool(sc.mechanic_passed),
                is_extension=bool(getattr(sc, "is_extension", False)),
                last_accuracy_grade=(sc.last_accuracy_grade or "").strip(),
                status_hint=_status_hint_for_sub_concept(sc),
            )
        )
    for sc in core_rows:
        ui = _UI_STATE.get(sc.status, "unchecked")
        if ui == "verified":
            verified += 1
        why_flags.append(bool(sc.why_passed))
        how_flags.append(bool(sc.how_passed))
        mech_flags.append(bool(sc.mechanic_passed))

    directive = (memory.last_eval_directive or "").strip()
    active = _active_layer_from_directive(directive)
    why_all = all(why_flags) if why_flags else False
    how_all = all(how_flags) if how_flags else False
    mech_all = all(mech_flags) if mech_flags else False
    gloss_mode = (directive.upper() == "PASSED_WITH_GLOSS") or (
        why_all and how_all and not mech_all
    )
    layers = CoverageLayers(
        why=_layer_progress(why_flags, is_active=active == "WHY"),
        how=_layer_progress(how_flags, is_active=active == "HOW"),
        mechanic=_layer_progress(
            mech_flags,
            is_active=active == "MECHANIC",
            gloss=gloss_mode and not mech_all,
        ),
    )
    overall = int(
        round(
            100.0 * (layers.why.score + layers.how.score + layers.mechanic.score) / 3.0
        )
    )
    gloss_hint = ""
    if gloss_mode and not mech_all:
        gloss_hint = (
            "Концепция зачтена. Дополните механики реализации или изучите Gloss"
        )
    mastery_ids = list_deep_mastery_concept_ids(memory)
    overlay_awards = list_overlay_mastery_records(memory)
    return CoverageSummary(
        total=len(core_rows),
        verified=verified,
        items=items,
        layers=layers,
        overall_score=min(100, max(0, overall)),
        active_layer=active,  # type: ignore[arg-type]
        gloss_hint=gloss_hint,
        deep_mastery_ids=mastery_ids,
        deep_mastery_count=len(mastery_ids),
        overlay_awards=overlay_awards,
    )


def format_concept_map_for_tutor(
    memory: SessionMemory,
    *,
    focus_id: str = "",
    include_evaluator_transparency: bool = True,
    suppress_topic_completion: bool = False,
    node_layer: str = "",
) -> str:
    if not memory.sub_concepts:
        return ""
    lines = ["[CURRENT_CONCEPT_MAP]"]
    if (memory.node_goal or "").strip():
        lines.append(f"node_goal: {memory.node_goal.strip()[:600]}")
    directive = (memory.last_eval_directive or "").strip()
    if directive:
        lines.append(f"last_eval_directive: {directive}")
        if directive.startswith("PROBE_NEXT_LAYER:"):
            layer = directive.split(":", 1)[-1].strip().upper()
            lines.append(
                f"THRESHOLD_DIRECTIVE: ask ONLY about layer {layer}; "
                "do not demand deeper layers in follow_up_question."
            )
        elif directive == "PASSED_WITH_GLOSS":
            if suppress_topic_completion:
                lines.append(
                    "THRESHOLD_DIRECTIVE: PASSED_WITH_GLOSS (informational only) — "
                    "Star Task / deep_analysis is active; ignore optional-layer forks."
                )
            else:
                lines.append(
                    "THRESHOLD_DIRECTIVE: PASSED_WITH_GLOSS — threshold met with optional "
                    "depth open (foundation: HOW/MECH; advanced: MECH; never SotA). "
                    "Mid-map: gloss optional depth yourself then advance. "
                    "If coverage complete: OPTIONAL_LAYER FORK + quick_replies."
                )
        elif directive == "PASSED_CLEAN":
            lines.append(
                "THRESHOLD_DIRECTIVE: PASSED_CLEAN — brief credit and advance; "
                "no extra grilling on this id."
            )
        elif directive == "DEEP_MASTERY_EARNED":
            if not suppress_topic_completion:
                lines.append(
                    "THRESHOLD_DIRECTIVE: DEEP_MASTERY_EARNED — parallel Asterisk-question "
                    "Deep Mastery credit; celebrate deep material analysis; offer the next "
                    "node or another Asterisk question."
                )
    if include_evaluator_transparency:
        from knowledge_engine.src.node_deep_dive.eval_result_adapter import (
            format_evaluator_critique_for_tutor,
            should_inject_evaluator_critique,
        )

        if should_inject_evaluator_critique(memory):
            critique_block = format_evaluator_critique_for_tutor(
                getattr(memory, "last_evaluator_critique", None)
            )
            if critique_block:
                lines.append(critique_block)
            else:
                feedback = (memory.last_evaluator_feedback or "").strip()
                if feedback:
                    lines.append(f"last_evaluator_feedback: {feedback[:900]}")
        else:
            feedback = (memory.last_evaluator_feedback or "").strip()
            if feedback:
                lines.append(f"last_evaluator_feedback: {feedback[:900]}")
        transparency = format_evaluator_transparency_payload(memory, prefer_id=focus_id)
        if transparency:
            lines.append(transparency)
    status_label = {
        "verified": "VERIFIED",
        "partial": "PARTIAL",
        "gap": "GAP",
        "unchecked": "UNCHECKED",
    }
    for sc in memory.sub_concepts:
        st = status_label.get(sc.status, sc.status.upper())
        layers = f"W{int(sc.why_passed)}H{int(sc.how_passed)}M{int(sc.mechanic_passed)}"
        ev = (sc.evidence or "").strip()
        hint = (sc.focus_hint or "").strip()
        tail = f" ({ev[:120]})" if ev else ""
        line = f"- {sc.id} | {sc.label}: {st} [{layers}]{tail}"
        if sc.status in ("partial", "gap") and hint:
            line += f" | focus_hint: {hint[:220]}"
        lines.append(line)
    active = active_question_sub_concept_id(memory)
    if active:
        lines.append(
            f"asked_question_sub_concept_id: {active} "
            "(evaluate the user answer ONLY against this id)"
        )
    verified_ids = list_verified_sub_concept_ids(memory)
    if verified_ids:
        lines.append(
            "verified_sub_concept_ids (do not quiz these): "
            + ", ".join(verified_ids[:12])
        )
    fid = (focus_id or memory.next_question_concept_id or "").strip()
    focus = find_sub_concept(memory, fid) if fid else None
    if focus is None or focus.status == "verified" or focus.id in verified_ids:
        focus = select_next_sub_concept(memory)
    pending = (memory.pending_evaluation_concept_id or "").strip()
    if pending:
        lines.append(
            f"pending_evaluation_concept_id: {pending} "
            "(outstanding tutor question awaiting user answer)"
        )
    if focus:
        lines.append(
            f"active_subconcept_id (generation focus): {focus.id} "
            f"— «{focus.label}» [{status_label.get(focus.status, focus.status.upper())}]"
        )
        lines.append(
            "HARD ANCHOR: lecture / follow_up_question / question_sub_concept_id "
            f"MUST target id={focus.id}. Do NOT pick the topic from chat_history."
        )
    lines.append("")
    lines.append("NEXT-STEP INSTRUCTIONS:")
    lines.append(
        "1. FORBIDDEN: ask about VERIFIED sub-concepts or ids in verified_sub_concept_ids."
    )
    lines.append(
        "2. Do not mark a new sub-topic as completed in this turn: "
        "the user has not answered its question yet."
    )
    lines.append(
        "3. If user_message closed the previous micro-topic — brief summary "
        "(1–2 sentences) and SWITCH sub-topic; no third consecutive question "
        "on the same diagram/mechanic."
    )
    if focus:
        if focus.status in ("partial", "gap"):
            hint = (focus.focus_hint or "").strip()
            lines.append(
                f"4. CURRENT sub-topic is NOT closed ({focus.status.upper()}, "
                f"id={focus.id}). FORBIDDEN to move to a new sub-topic in "
                "technical_explanation / follow_up_question until VERIFIED or "
                "the user explicitly asks to skip."
            )
            if hint:
                lines.append(
                    f"   focus_hint / last_evaluator_focus_hint "
                    f"(Russian; Host plaque uses this; technical prose in "
                    f"correction_breakdown): {hint[:400]}"
                )
            probe = (memory.last_eval_directive or "").strip()
            layer_note = ""
            if probe.startswith("PROBE_NEXT_LAYER:"):
                layer_note = (
                    f" Probe ONLY layer {probe.split(':', 1)[-1].strip().upper()}."
                )
            lines.append(
                "5. Host prepends the credited/missing plaque in Python. "
                "Do not emit 📋/🎯 or feedback_on_answer. "
                "follow_up_question — one «?» on THIS same id." + layer_note
            )
        else:
            lines.append(
                f"4. Next focus (active_subconcept_id): «{focus.label}» "
                f"(id={focus.id}, status {focus.status.upper()}). "
                "Do not mix with VERIFIED ids."
            )
            lines.append(
                "5. Deliver a production-depth block on this sub-topic and exactly "
                "one ending «?» question (NO-DEAD-END). "
                f"question_sub_concept_id MUST be {focus.id}."
            )
        upcoming = [
            sc
            for sc in memory.sub_concepts
            if sc.status != "verified"
            and sc.id not in verified_ids
            and sc.id != focus.id
        ][:2]
        if upcoming:
            labels = ", ".join(f"{sc.label} ({sc.id})" for sc in upcoming)
            lines.append(f"6. Later in the program: {labels}")
    else:
        mastery_ids = list_deep_mastery_concept_ids(memory)
        mastery_note = (
            f" Asterisk-question Deep Mastery already earned for: {', '.join(mastery_ids[:8])}."
            if mastery_ids
            else ""
        )
        if suppress_topic_completion:
            # Do NOT mention topic completion / base coverage — those leak into
            # deep_analysis and cause the model to emit transition decrees.
            lines.append(
                "4. SESSION FLAGS: node_completed=false "
                "(Deep Analysis / Star Task active)."
                f"{mastery_note} "
                "Produce technical_explanation + REQUIRED non-empty "
                "follow_up_question only. No transition menus or pathway chips."
            )
        else:
            from knowledge_engine.src.node_deep_dive.concept_map import (
                is_full_depth_closure,
                open_optional_layers,
            )
            from knowledge_engine.src.node_deep_dive.sub_concept_evaluator import (
                normalize_node_layer,
            )

            ly = normalize_node_layer(node_layer or "foundation")
            open_layers = open_optional_layers(memory, ly)
            if open_layers and not is_full_depth_closure(memory, ly) and ly != "sota":
                lines.append(
                    "4. Host pathway=optional_fork — threshold met; "
                    f"open_optional_layers=[{'/'.join(open_layers)}] "
                    "(Host will set quick_replies / ready_for_transition)."
                    f"{mastery_note} "
                    "Write natural peer commentary for this pathway; "
                    "NON-EMPTY follow_up_question = mode-choice CTA only "
                    "(DO NOT generate a new technical/evaluative question); "
                    "do not invent next node titles "
                    "or chip menus. FORBIDDEN clichés: base-theory-closed / "
                    "optional-layer scripts."
                )
            else:
                lines.append(
                    "4. Host pathway=base_complete — WHY/HOW/MECH threshold met "
                    "(BASE theory closure, NOT absolute node mastery)."
                    f"{mastery_note} "
                    "Host owns next-node + Asterisk chips and orchestration flags. "
                    "Write natural peer commentary; NON-EMPTY follow_up_question = "
                    "mode-choice CTA only (DO NOT generate a new technical/"
                    "evaluative question); "
                    "do not invent next node titles or chip menus. "
                    "FORBIDDEN clichés: base-theory-closed / optional-MECH / "
                    "absolute 100%-node decrees."
                )
    return "\n".join(lines)
