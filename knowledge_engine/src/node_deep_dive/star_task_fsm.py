"""Star Task FSM — asterisk-question overlay lifecycle (ADVANCED L4 / DEEP L5–L6)."""

from __future__ import annotations

from typing import Iterable, Literal, Sequence

from knowledge_engine.src.node_deep_dive.intent_definitions import (
    CHIP_ADVANCED_ANALYSIS,
    CHIP_DEEP_DESIGN,
    CHIP_OVERLAY_NEXT,
)
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    LayerDrillLayer,
    LayerDrillSession,
    OverlayKind,
    OverlayType,
    SessionMemory,
)
from knowledge_engine.ui.run_log import trace

StarTaskStatus = Literal[
    "not_started",
    "in_progress",
    "needs_refinement",
    "resolved",
]

STAR_TASK_ACTIVE = frozenset({"in_progress", "needs_refinement"})

# Canonical overlay kinds (Bloom L4 vs L5/L6). Legacy ``deep_analysis`` aliases deep_design.
OVERLAY_EVAL_KINDS: frozenset[str] = frozenset(
    {"advanced_analysis", "deep_design", "deep_analysis"}
)


def overlay_offer_quick_replies(
    *,
    weakness_tags: Sequence[str] | Iterable[str] | None = None,
) -> list[str]:
    """
    Host chips after core close (pathway overlay_offer / base_complete).

    Clean core (no open weakness_tags) → prioritize DEEP_ASTERISK.
    Open weakness_tags → targeted ADVANCED_ASTERISK (Bloom L4).
    Always keep the next-node fallback.
    """
    tags = [
        str(t).strip()
        for t in (weakness_tags or [])
        if str(t).strip()
    ]
    if tags:
        return [CHIP_ADVANCED_ANALYSIS, CHIP_OVERLAY_NEXT]
    return [CHIP_DEEP_DESIGN, CHIP_OVERLAY_NEXT]


def is_overlay_eval_kind(kind: str | None) -> bool:
    return (kind or "").strip().lower() in OVERLAY_EVAL_KINDS


def canonical_overlay_kind(kind: str | None) -> OverlayKind:
    """Map pending_eval_kind / factory mode → advanced_analysis | deep_design."""
    k = (kind or "").strip().lower()
    if k in ("advanced_analysis", "advanced"):
        return "advanced_analysis"
    return "deep_design"


def overlay_type_for_kind(kind: str | None) -> OverlayType:
    if canonical_overlay_kind(kind) == "advanced_analysis":
        return "ADVANCED_ASTERISK"
    return "DEEP_ASTERISK"


def overlay_kind_to_target_layer(kind: str | None) -> str:
    if canonical_overlay_kind(kind) == "advanced_analysis":
        return "ADVANCED"
    return "DEEP"


def overlay_factory_mode_tag(
    memory: SessionMemory | None,
    user_mode: str = "",
) -> str:
    """``[mode:…]`` tag for isolated overlay system (preserve legacy deep_analysis)."""
    user = (user_mode or "").strip().lower()
    if user in OVERLAY_EVAL_KINDS:
        return user
    kind = ""
    if memory is not None:
        kind = (getattr(memory, "pending_eval_kind", None) or "").strip().lower()
    if kind in OVERLAY_EVAL_KINDS:
        return kind
    return "deep_design"


def normalize_star_task_status(raw: str | None) -> StarTaskStatus:
    val = (raw or "").strip().lower()
    if val in ("not_started", "in_progress", "needs_refinement", "resolved"):
        return val  # type: ignore[return-value]
    return "not_started"


def get_star_task_status(memory: SessionMemory) -> StarTaskStatus:
    return normalize_star_task_status(getattr(memory, "star_task_status", None))


def star_task_blocks_transition(memory: SessionMemory) -> bool:
    """True while the asterisk-question overlay is open — node must not finalize."""
    return get_star_task_status(memory) in STAR_TASK_ACTIVE


def set_star_task_status(memory: SessionMemory, status: StarTaskStatus) -> None:
    prev = get_star_task_status(memory)
    if prev != status:
        from knowledge_engine.src.resilience_manager import note_asterisk_fsm_hop

        if not note_asterisk_fsm_hop(memory):
            return
    memory.star_task_status = status
    if prev != status:
        trace(
            f"NODE_DIVE asterisk_question status | {prev} → {status}"
        )


def mark_star_task_in_progress(
    memory: SessionMemory,
    *,
    concept_id: str = "",
    overlay_kind: str = "",
) -> None:
    """Tutor issued an asterisk-question — await the learner's overlay answer."""
    set_star_task_status(memory, "in_progress")
    cid = (concept_id or "").strip() or (
        memory.pending_evaluation_concept_id or ""
    ).strip()
    requested = (overlay_kind or "").strip().lower()
    current = (memory.pending_eval_kind or "").strip().lower()
    if is_overlay_eval_kind(requested):
        memory.pending_eval_kind = requested  # type: ignore[assignment]
    elif is_overlay_eval_kind(current):
        memory.pending_eval_kind = current  # type: ignore[assignment]
    else:
        memory.pending_eval_kind = "deep_analysis"
    if cid:
        trace(
            f"NODE_DIVE star_task in_progress | concept={cid} | "
            f"kind={memory.pending_eval_kind}"
        )


def apply_star_task_eval_outcome(
    memory: SessionMemory,
    *,
    concept_id: str,
    resolved: bool,
) -> None:
    """
    Evaluator outcome for an open asterisk-question.

    resolved=True  → overlay award; transition may open.
    resolved=False → needs_refinement; keep the same overlay kind for review.
    """
    cid = (concept_id or "").strip()
    kind = (memory.pending_eval_kind or "").strip().lower()
    if resolved:
        set_star_task_status(memory, "resolved")
        memory.pending_eval_kind = ""
        trace(f"NODE_DIVE asterisk_question resolved | concept={cid} | kind={kind}")
        return
    set_star_task_status(memory, "needs_refinement")
    if cid:
        memory.pending_evaluation_concept_id = cid
        memory.asked_question_sub_concept_id = cid
        memory.next_question_concept_id = cid
        memory.last_tutor_sub_concept_id = cid
    memory.pending_eval_kind = (  # type: ignore[assignment]
        kind if is_overlay_eval_kind(kind) else "deep_analysis"
    )
    memory.last_eval_directive = "STAR_TASK_NEEDS_REFINEMENT"
    trace(
        f"NODE_DIVE star_task needs_refinement | concept={cid} | "
        f"kind={memory.pending_eval_kind}"
    )


def continue_overlay_push_or_resolve(
    memory: SessionMemory,
    *,
    concept_id: str,
    resolved: bool,
    overlay_kind: str = "",
) -> bool:
    """
    Apply overlay eval outcome; keep the push session if another core row
    still lacks this overlay award.

    Returns True when teaching continues on a new concept (caller must not
    clear pending_eval_kind / close the node).
    """
    kind = (overlay_kind or memory.pending_eval_kind or "").strip().lower()
    if not resolved:
        apply_star_task_eval_outcome(
            memory, concept_id=concept_id, resolved=False
        )
        return False
    from knowledge_engine.src.node_deep_dive.concept_map_state import (
        first_open_overlay_row,
    )

    if layer_drill_is_active(memory):
        done = advance_layer_drill_after_pass(memory, evaluated_id=concept_id)
        if done:
            apply_star_task_eval_outcome(
                memory, concept_id=concept_id, resolved=True
            )
            return False
        nxt_id = current_layer_drill_concept_id(memory)
        if not nxt_id:
            apply_star_task_eval_outcome(
                memory, concept_id=concept_id, resolved=True
            )
            return False
        mark_star_task_in_progress(
            memory, concept_id=nxt_id, overlay_kind=kind or "deep_design"
        )
        memory.next_question_concept_id = nxt_id
        memory.asked_question_sub_concept_id = ""
        memory.pending_evaluation_concept_id = ""
        memory.last_tutor_sub_concept_id = ""
        trace(
            f"NODE_DIVE overlay push continue | next={nxt_id} | kind={kind}"
        )
        return True

    nxt = first_open_overlay_row(memory, kind or "deep_design")
    if nxt is None:
        apply_star_task_eval_outcome(
            memory, concept_id=concept_id, resolved=True
        )
        return False
    mark_star_task_in_progress(
        memory, concept_id=nxt.id, overlay_kind=kind or "deep_design"
    )
    memory.next_question_concept_id = nxt.id
    memory.asked_question_sub_concept_id = ""
    memory.pending_evaluation_concept_id = ""
    memory.last_tutor_sub_concept_id = ""
    trace(
        f"NODE_DIVE overlay push continue | next={nxt.id} | kind={kind}"
    )
    return True


# ---------------------------------------------------------------------------
# Layer Drill Session — HOW / MECH / overlay asterisk walk
# ---------------------------------------------------------------------------

_DRILL_LAYER_ALIASES: dict[str, LayerDrillLayer] = {
    "why": "WHY",
    "how": "HOW",
    "deep_dive_how": "HOW",
    "mech": "MECH",
    "mechanic": "MECH",
    "mechanics": "MECH",
    "deep_dive_mech": "MECH",
    "advanced_analysis": "ADVANCED_ASTERISK",
    "advanced_asterisk": "ADVANCED_ASTERISK",
    "advanced": "ADVANCED_ASTERISK",
    "deep_design": "DEEP_ASTERISK",
    "deep_analysis": "DEEP_ASTERISK",
    "deep_asterisk": "DEEP_ASTERISK",
    "deep": "DEEP_ASTERISK",
}


def normalize_drill_layer(raw: str | None) -> LayerDrillLayer | None:
    key = (raw or "").strip().upper().replace("-", "_")
    if key in ("WHY", "HOW", "MECH", "ADVANCED_ASTERISK", "DEEP_ASTERISK"):
        return key  # type: ignore[return-value]
    return _DRILL_LAYER_ALIASES.get((raw or "").strip().lower())


def _layer_drill(memory: SessionMemory) -> LayerDrillSession:
    drill = getattr(memory, "layer_drill", None)
    if isinstance(drill, LayerDrillSession):
        return drill
    memory.layer_drill = LayerDrillSession()
    return memory.layer_drill


def layer_drill_is_active(memory: SessionMemory | None) -> bool:
    if memory is None:
        return False
    drill = getattr(memory, "layer_drill", None)
    return bool(getattr(drill, "is_active", False))


def active_core_scoring_layer(memory: SessionMemory | None) -> str:
    """HOW | MECHANIC | WHY while a core (non-overlay) drill is live, else ''."""
    if memory is None:
        return ""
    drill = getattr(memory, "layer_drill", None)
    if drill is not None and bool(getattr(drill, "is_active", False)):
        name = normalize_drill_layer(getattr(drill, "target_layer", None))
        if name == "HOW":
            return "HOW"
        if name == "MECH":
            return "MECHANIC"
        if name == "WHY":
            return "WHY"
    teaching = (getattr(memory, "active_optional_layer", "") or "").strip().upper()
    if teaching in ("HOW", "MECHANIC"):
        return teaching
    return ""


def core_layer_drill_blocks_overlay_eval(memory: SessionMemory | None) -> bool:
    """True when HOW/MECH/WHY drill owns scoring — overlay evaluator must not run."""
    return active_core_scoring_layer(memory) in ("HOW", "MECHANIC", "WHY")


def layer_drill_blocks_transition(memory: SessionMemory | None) -> bool:
    return layer_drill_is_active(memory)


def current_layer_drill_concept_id(memory: SessionMemory) -> str:
    drill = _layer_drill(memory)
    return (drill.get_current_sub_concept_id() or "").strip()


def _sync_compat_optional_layer(memory: SessionMemory) -> None:
    drill = _layer_drill(memory)
    if drill.is_active and drill.target_layer == "HOW":
        memory.active_optional_layer = "HOW"
    elif drill.is_active and drill.target_layer == "MECH":
        memory.active_optional_layer = "MECHANIC"
    elif (memory.active_optional_layer or "") in ("HOW", "MECHANIC"):
        if not drill.is_active:
            memory.active_optional_layer = ""


def clear_layer_drill(memory: SessionMemory) -> None:
    prev = _layer_drill(memory)
    layer = prev.target_layer or ""
    memory.layer_drill = LayerDrillSession()
    if (memory.active_optional_layer or "") in ("HOW", "MECHANIC"):
        memory.active_optional_layer = ""
    if layer:
        trace(f"NODE_DIVE layer_drill clear | layer={layer}")


def row_open_for_drill_layer(
    memory: SessionMemory,
    row: object,
    layer: str | None,
) -> bool:
    name = normalize_drill_layer(layer) or ""
    if name == "WHY":
        return not bool(getattr(row, "why_passed", False))
    if name == "HOW":
        return not bool(getattr(row, "how_passed", False))
    if name == "MECH":
        return not bool(getattr(row, "mechanic_passed", False))
    if name in ("ADVANCED_ASTERISK", "DEEP_ASTERISK"):
        from knowledge_engine.src.node_deep_dive.concept_map_state import (
            has_overlay_award,
        )

        return not has_overlay_award(memory, getattr(row, "id", "") or "", name)
    return False


def collect_drill_target_ids(
    memory: SessionMemory,
    layer: str | None,
) -> list[str]:
    from knowledge_engine.src.node_deep_dive.concept_map_state import core_sub_concepts

    name = normalize_drill_layer(layer)
    if not name:
        return []
    out: list[str] = []
    for sc in core_sub_concepts(memory):
        if row_open_for_drill_layer(memory, sc, name):
            cid = (sc.id or "").strip()
            if cid:
                out.append(cid)
    return out[:8]


def start_layer_drill(
    memory: SessionMemory,
    layer: str,
    *,
    concept_id: str = "",
) -> str:
    """Open DRILL_IN_PROGRESS and queue every core row still missing this layer."""
    name = normalize_drill_layer(layer)
    if not name:
        return ""
    memory.is_layer_just_completed = False
    ids = collect_drill_target_ids(memory, name)
    if not ids:
        clear_layer_drill(memory)
        return ""
    pin = (concept_id or "").strip()
    idx = ids.index(pin) if pin and pin in ids else 0
    memory.layer_drill = LayerDrillSession(
        is_active=True,
        target_layer=name,
        target_sub_concept_ids=ids,
        current_index=idx,
        status="DRILL_IN_PROGRESS",
    )
    _sync_compat_optional_layer(memory)
    current = current_layer_drill_concept_id(memory)
    if current:
        memory.next_question_concept_id = current
    if name in ("HOW", "MECH", "WHY"):
        # Core drill must not inherit a leftover asterisk evaluator / FSM.
        if is_overlay_eval_kind(memory.pending_eval_kind):
            memory.pending_eval_kind = ""
        if get_star_task_status(memory) in STAR_TASK_ACTIVE:
            memory.star_task_status = "not_started"
    if name in ("ADVANCED_ASTERISK", "DEEP_ASTERISK"):
        requested = (layer or "").strip().lower()
        current_kind = (memory.pending_eval_kind or "").strip().lower()
        overlay_kind = requested if is_overlay_eval_kind(requested) else current_kind
        if not is_overlay_eval_kind(overlay_kind):
            overlay_kind = (
                "advanced_analysis" if name == "ADVANCED_ASTERISK" else "deep_design"
            )
        mark_star_task_in_progress(
            memory, concept_id=current, overlay_kind=overlay_kind
        )
    trace(
        f"NODE_DIVE layer_drill start | layer={name} | n={len(ids)} | "
        f"current={current or '—'}"
    )
    return current


def advance_layer_drill_after_pass(
    memory: SessionMemory,
    *,
    evaluated_id: str = "",
) -> bool:
    """
    After a scored answer: stay if the current row is still open, else advance.

    Returns True when the drill session has fully completed.
    """
    drill = _layer_drill(memory)
    if not drill.is_active:
        return True
    from knowledge_engine.src.node_deep_dive.concept_map_state import find_sub_concept

    ev_id = (evaluated_id or "").strip()
    current = current_layer_drill_concept_id(memory)
    check_id = ev_id or current
    row = find_sub_concept(memory, check_id) if check_id else None
    if row is not None and row_open_for_drill_layer(
        memory, row, drill.target_layer
    ):
        memory.next_question_concept_id = check_id
        memory.asked_question_sub_concept_id = check_id
        return False
    while True:
        done = drill.advance_or_complete()
        if done:
            _sync_compat_optional_layer(memory)
            memory.next_question_concept_id = ""
            trace(
                f"NODE_DIVE layer_drill complete | layer={drill.target_layer}"
            )
            return True
        nxt = current_layer_drill_concept_id(memory)
        nxt_row = find_sub_concept(memory, nxt) if nxt else None
        if nxt_row is None or not row_open_for_drill_layer(
            memory, nxt_row, drill.target_layer
        ):
            continue
        _sync_compat_optional_layer(memory)
        memory.next_question_concept_id = nxt
        memory.asked_question_sub_concept_id = ""
        trace(
            f"NODE_DIVE layer_drill advance | layer={drill.target_layer} | "
            f"idx={drill.current_index}/{len(drill.target_sub_concept_ids)} | "
            f"next={nxt}"
        )
        return False


def layer_drill_progress(memory: SessionMemory | None) -> dict[str, object]:
    """Snapshot for prompts/tests: status, Progress n/N, current/next titles."""
    empty: dict[str, object] = {
        "status": "DRILL_INACTIVE",
        "target_layer": "",
        "completed": 0,
        "total": 0,
        "progress": "0/0",
        "current_sub_concept_id": "",
        "current_sub_concept_title": "",
        "next_sub_concept_id": "",
        "next_sub_concept_title": "",
    }
    if memory is None or not layer_drill_is_active(memory):
        return empty
    from knowledge_engine.src.node_deep_dive.concept_map_state import find_sub_concept

    drill = _layer_drill(memory)
    ids = list(drill.target_sub_concept_ids or [])
    total = len(ids)
    completed = min(int(drill.current_index), total)
    current_id = current_layer_drill_concept_id(memory)
    next_id = ""
    if drill.current_index + 1 < total:
        next_id = ids[drill.current_index + 1]
    cur_row = find_sub_concept(memory, current_id) if current_id else None
    nxt_row = find_sub_concept(memory, next_id) if next_id else None
    return {
        "status": "DRILL_ACTIVE",
        "target_layer": drill.target_layer or "",
        "completed": completed,
        "total": total,
        "progress": f"{completed}/{total}",
        "current_sub_concept_id": current_id,
        "current_sub_concept_title": ((cur_row.label if cur_row else "") or current_id),
        "next_sub_concept_id": next_id,
        "next_sub_concept_title": ((nxt_row.label if nxt_row else "") or next_id),
    }


def format_layer_drill_invariants(memory: SessionMemory | None) -> str:
    """English system-prompt block. Empty when no drill session is active."""
    snap = layer_drill_progress(memory)
    if snap["status"] != "DRILL_ACTIVE":
        return ""
    layer = snap["target_layer"]
    total = int(snap["total"] or 0)
    completed = int(snap["completed"] or 0)
    ordinal = completed + 1
    title = str(snap["current_sub_concept_title"] or "")
    nxt_title = str(snap["next_sub_concept_title"] or "")
    nxt_line = (
        f'3. If the answer is correct: credit this sub-concept, then IMMEDIATELY '
        f'teach the next sub-concept "{nxt_title}" in layer {layer}. '
        "Do not offer pathway chips.\n"
        if nxt_title
        else "3. If the answer is correct and this is the LAST queued sub-concept, "
        "you may then allow Host to close this layer.\n"
    )
    ru_status = (
        f"[Слой {layer}: Проверено {completed}/{total} подтем. "
        f"Переходим к подтеме №{ordinal}: «{title}»]"
    )
    return (
        "[MANDATORY DRILL SESSION INVARIANTS]\n"
        "Current Session Status: DRILL_ACTIVE\n"
        f"Target Layer: {layer}\n"
        f"Progress: Sub-concept {ordinal} of {total} ({title}) "
        f"— checked {completed}/{total}.\n"
        "\n"
        "INSTRUCTIONS:\n"
        "1. DO NOT declare the node or layer complete. "
        "FORBIDDEN: «базовая теория закрыта», «всё успешно закрыто», "
        "base-theory-closed, 100%-node.\n"
        f"2. Evaluate the learner ONLY for the current sub-concept ({title}).\n"
        f"{nxt_line}"
        f"4. ONLY when Progress reaches {total} of {total} and every queued "
        "sub-concept of this layer is passed may Host close the layer.\n"
        "5. User-facing Russian feedback MUST include this one-line status:\n"
        f"   {ru_status}\n"
    )
