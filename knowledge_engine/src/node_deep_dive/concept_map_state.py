"""Topic Concept Map: SessionMemory sub_concepts (pure state, no text heuristics)."""

from __future__ import annotations

import re

from knowledge_engine.schemas.global_knowledge import utc_now_iso
from knowledge_engine.src.node_deep_dive.memory_schemas import (
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


def list_verified_sub_concept_ids(memory: SessionMemory) -> list[str]:
    return [sc.id for sc in (memory.sub_concepts or []) if sc.status == "verified"]


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
    row: SubConceptRecord | None = None
    for cid in candidates:
        found = find_sub_concept(memory, cid)
        if found is not None and found.status in ("partial", "gap"):
            row = found
            break
    if row is None:
        for sc in memory.sub_concepts or []:
            if sc.status in ("partial", "gap") and (sc.focus_hint or "").strip():
                row = sc
                break
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
    ]
    if directive:
        lines.append(f"last_eval_directive: {directive}")
    lines.append(
        "TRANSPARENCY: copy last_evaluator_focus_hint into feedback_on_answer "
        "«Чего не хватило…» almost verbatim."
    )
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
            row.evidence = ev[:2000]
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
    if row is not None and row.status == "verified":
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
        for sc in memory.sub_concepts
        if sc.status != "verified"
        and sc.id not in verified_ids
        and (not skip or sc.id.lower() != skip)
    ]
    if not candidates:
        return None

    def sort_key(sc: SubConceptRecord) -> tuple[int, int]:
        pri = {"gap": 0, "unchecked": 1, "partial": 2}.get(sc.status, 3)
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
    """
    ev_id = (evaluated_id or "").strip()
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
    if not memory.sub_concepts:
        return False
    return all(sc.status == "verified" for sc in memory.sub_concepts)


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


def build_coverage_summary(memory: SessionMemory) -> CoverageSummary | None:
    if not memory.sub_concepts:
        return None
    items: list[CoverageItem] = []
    verified = 0
    why_flags: list[bool] = []
    how_flags: list[bool] = []
    mech_flags: list[bool] = []
    for sc in memory.sub_concepts:
        ui = _UI_STATE.get(sc.status, "unchecked")
        if ui == "verified":
            verified += 1
        why_flags.append(bool(sc.why_passed))
        how_flags.append(bool(sc.how_passed))
        mech_flags.append(bool(sc.mechanic_passed))
        items.append(
            CoverageItem(
                id=sc.id,
                label=sc.label,
                state=ui,  # type: ignore[arg-type]
                why_passed=bool(sc.why_passed),
                how_passed=bool(sc.how_passed),
                mechanic_passed=bool(sc.mechanic_passed),
                status_hint=_status_hint_for_sub_concept(sc),
            )
        )

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
    return CoverageSummary(
        total=len(items),
        verified=verified,
        items=items,
        layers=layers,
        overall_score=min(100, max(0, overall)),
        active_layer=active,  # type: ignore[arg-type]
        gloss_hint=gloss_hint,
    )


def format_concept_map_for_tutor(
    memory: SessionMemory,
    *,
    focus_id: str = "",
    include_evaluator_transparency: bool = True,
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
    if include_evaluator_transparency:
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
                    f"(REQUIRED verbatim in transparency block): {hint[:400]}"
                )
            probe = (memory.last_eval_directive or "").strip()
            layer_note = ""
            if probe.startswith("PROBE_NEXT_LAYER:"):
                layer_note = (
                    f" Probe ONLY layer {probe.split(':', 1)[-1].strip().upper()}."
                )
            lines.append(
                "5. feedback_on_answer: MUST open with CRITICAL TRANSPARENCY block "
                "(📋 credited / 🎯 missing = focus_hint); then optional prose. "
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
        lines.append(
            "4. COVERAGE COMPLETE — threshold met for this node layer. "
            "TOPIC COMPLETION: ready_for_transition=true; "
            "follow_up_question NON-EMPTY; do not invent next node titles. "
            "If optional depth still open (foundation: HOW/MECH; advanced: MECH): "
            "OPTIONAL_LAYER FORK with quick_replies "
            "[Хочу Gloss | Дожать HOW|MECH | Идем дальше]. "
            "If full depth / SotA: 100% CTA only (next-node UI chips). "
            "SotA never offers skipping MECHANIC."
        )
    return "\n".join(lines)
