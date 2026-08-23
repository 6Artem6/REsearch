"""Topic Concept Map: facade (state + evaluator + tutor orchestration)."""

from __future__ import annotations

from knowledge_engine.src.node_deep_dive.concept_map_state import (  # noqa: F401
    active_question_sub_concept_id,
    advance_next_question_after_evaluation,
    advance_sub_concepts_after_user_answer,
    apply_sub_concept_updates,
    build_coverage_summary,
    core_sub_concepts,
    ensure_sub_concept_map,
    find_sub_concept,
    first_open_optional_layer_row,
    first_open_overlay_row,
    format_concept_map_for_tutor,
    has_deep_mastery,
    list_deep_mastery_concept_ids,
    list_verified_sub_concept_ids,
    optional_teaching_layer,
    overlay_push_kind,
    promote_sub_concepts_after_tutor_message,
    register_deep_mastery,
    resolve_evaluation_target_id,
    resolve_pending_evaluation_id,
    row_open_for_optional_depth_eval,
    select_next_sub_concept,
    set_pending_evaluation_for_tutor_turn,
    slug_sub_concept_id,
    stored_pending_evaluation_id,
    sub_concept_coverage_complete,
    sync_concepts_matrix_from_sub_concepts,
)
from knowledge_engine.src.node_deep_dive.memory_schemas import SessionMemory
from knowledge_engine.src.node_deep_dive.schemas import DeepDiveLLMOutput
from knowledge_engine.src.node_deep_dive.sub_concept_evaluator import (  # noqa: F401
    GAP_EVAL_SYSTEM,
    apply_degraded_threshold,
    apply_threshold_to_sub_concept,
    process_sub_concept_user_answer,
    run_sub_concept_gap_eval,
)

# Back-compat alias for tests/docs
_GAP_EVAL_SYSTEM = GAP_EVAL_SYSTEM


def format_diagram_repeat_guard(memory: SessionMemory) -> str:
    """Запрет повторной ссылки на diagram-1, если уже было в последних репликах тьютора."""
    tutor_bodies: list[str] = []
    for item in memory.active_window or []:
        if (item.get("role") or "").strip() != "tutor":
            continue
        tutor_bodies.append((item.get("content") or "").lower())
    recent = tutor_bodies[-3:]
    diagram_hits = 0
    for body in recent:
        if any(
            m in body
            for m in (
                "diagram-1",
                "[diagram 1",
                "диаграмму 1",
                "диаграмма 1",
                "diagram 1:",
            )
        ):
            diagram_hits += 1
    if diagram_hits < 2:
        return ""
    return (
        "### diagram_repeat_guard\n"
        "В последних репликах ты уже разбирал [diagram-1] / Diagram 1. "
        "ЗАПРЕЩЕНО снова ссылаться на эту же диаграмму в этом ответе. "
        "Используй code assets, другие diagram id, URL материалов или следующую подтему "
        "из concept_map.\n"
    )


def user_accepted_optional_deep_dive(user_message: str) -> bool:
    """Явное согласие пользователя на углубление (soft pitch) — только короткие фразы."""
    from knowledge_engine.src.node_deep_dive.control_intent import (
        is_short_accept_deep_dive,
    )

    return is_short_accept_deep_dive(user_message)


GLOSS_FORK_QUICK_REPLIES: tuple[str, ...] = (
    "Хочу Gloss",
    "Дожать HOW",
    "Дожать MECH",
    "Идем дальше",
)


def aggregate_depth_flags(memory: SessionMemory) -> tuple[bool, bool, bool]:
    from knowledge_engine.src.node_deep_dive.concept_map_state import core_sub_concepts

    rows = core_sub_concepts(memory)
    if not rows:
        return False, False, False
    return (
        all(bool(sc.why_passed) for sc in rows),
        all(bool(sc.how_passed) for sc in rows),
        all(bool(sc.mechanic_passed) for sc in rows),
    )


def open_optional_layers(memory: SessionMemory, node_layer: str) -> list[str]:
    """Optional depth layers still open for this node difficulty."""
    from knowledge_engine.src.node_deep_dive.sub_concept_evaluator import (
        optional_depth_layers,
    )

    why, how, mech = aggregate_depth_flags(memory)
    open_layers: list[str] = []
    for name in optional_depth_layers(node_layer):
        if name == "HOW" and not how:
            open_layers.append("HOW")
        elif name == "MECHANIC" and not mech:
            open_layers.append("MECHANIC")
    # Also catch per-row gaps when aggregates are mixed
    if not open_layers:
        opts = set(optional_depth_layers(node_layer))
        for sc in core_sub_concepts(memory):
            if "HOW" in opts and sc.why_passed and not sc.how_passed:
                open_layers.append("HOW")
            if (
                "MECHANIC" in opts
                and sc.why_passed
                and sc.how_passed
                and not sc.mechanic_passed
            ):
                open_layers.append("MECHANIC")
        # dedupe preserving order
        seen: set[str] = set()
        uniq: list[str] = []
        for x in open_layers:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq
    return open_layers


def has_optional_mech_open(memory: SessionMemory, node_layer: str = "advanced") -> bool:
    """Back-compat: True when any optional depth layer is still open."""
    return bool(open_optional_layers(memory, node_layer))


def is_full_depth_closure(memory: SessionMemory, node_layer: str) -> bool:
    why, how, mech = aggregate_depth_flags(memory)
    return bool(why and how and mech)


def _optional_push_chip_label(layer: str) -> str:
    name = (layer or "").strip().upper()
    if name in ("MECHANIC", "MECH"):
        return "Дожать MECH"
    if name == "HOW":
        return "Дожать HOW"
    return f"Дожать {layer}"


def gloss_fork_quick_replies(open_layers: list[str]) -> list[str]:
    """Chips only for *open* optional layers — never invent MECH/HOW if already closed."""
    if not open_layers:
        return []
    # Prefer shallowest open optional for the push chip
    push = open_layers[0]
    return ["Хочу Gloss", _optional_push_chip_label(push), "Идем дальше"]


def classify_gloss_fork_choice(user_message: str) -> str:
    """
    Map user chip / free-text to gloss-fork action.

    Returns ``gloss`` | ``how`` | ``mech`` | ``deep_analysis`` |
    ``advanced_analysis`` | ``deep_design`` | ``next`` | ``\"\"``.

    Delegates to ``control_intent.classify_control_chip`` (explicit tags,
    exact labels, length-gated fuzzy). Broad technical stems are forbidden.
    """
    from knowledge_engine.src.node_deep_dive.control_intent import (
        classify_control_chip,
    )

    chip = classify_control_chip(user_message)
    if chip in (
        "gloss",
        "how",
        "mech",
        "next",
        "deep_analysis",
        "advanced_analysis",
        "deep_design",
    ):
        return chip
    return ""


def first_optional_layer_concept_id(memory: SessionMemory, layer_name: str) -> str:
    row = first_open_optional_layer_row(memory, layer_name)
    return (row.id if row else "").strip()


def first_optional_mech_concept_id(memory: SessionMemory) -> str:
    return first_optional_layer_concept_id(memory, "MECHANIC")


def start_overlay_push(
    memory: SessionMemory,
    overlay_kind: str,
    *,
    concept_id: str = "",
) -> str:
    """
    Open an asterisk-question push session (same Host contract as «Дожать HOW/MECH»).

    Pins generation focus to the first core row still missing this overlay award.
    """
    from knowledge_engine.src.node_deep_dive.star_task_fsm import (
        start_layer_drill,
    )

    return start_layer_drill(memory, overlay_kind, concept_id=concept_id)


def is_quick_reply_control_message(
    user_message: str,
    memory: SessionMemory | None = None,
) -> bool:
    """True for UI Quick Reply chips — not a substantive answer for the Evaluator."""
    from knowledge_engine.src.node_deep_dive.control_intent import (
        is_control_chip_message,
    )

    return is_control_chip_message(user_message, memory=memory)


def credit_open_optional_layers(
    memory: SessionMemory,
    node_layer: str,
    *,
    evidence: str = "gloss_credit: optional layer marked via Gloss chip",
) -> list[str]:
    """
    Auto-credit open optional HOW/MECH flags (Gloss path) without LLM Evaluator.

    Returns names of layers credited.
    """
    from knowledge_engine.src.node_deep_dive.sub_concept_evaluator import (
        optional_depth_layers,
    )

    open_before = open_optional_layers(memory, node_layer)
    if not open_before:
        return []
    opts = set(optional_depth_layers(node_layer))
    credited: list[str] = []
    for sc in core_sub_concepts(memory):
        why = bool(sc.why_passed)
        how = bool(sc.how_passed)
        mech = bool(sc.mechanic_passed)
        if "HOW" in opts and why and not how:
            how = True
            if "HOW" not in credited:
                credited.append("HOW")
        if "MECHANIC" in opts and why and not mech:
            # foundation may credit MECH even if HOW was just filled
            mech = True
            if "MECHANIC" not in credited:
                credited.append("MECHANIC")
        if how == sc.how_passed and mech == sc.mechanic_passed:
            continue
        apply_threshold_to_sub_concept(
            sc,
            layer=node_layer,
            why=why,
            how=how,
            mechanic=mech,
            evidence=evidence,
            llm_focus_hint="",
        )
    sync_concepts_matrix_from_sub_concepts(memory)
    return credited


def _host_hold_orchestration() -> dict[str, object]:
    """Force Host-owned flags off — never copy LLM chip/transition guesses."""
    return {
        "ready_for_transition": False,
        "suggested_next_step": None,
        "quick_replies": [],
    }


def _activate_optional_teaching(memory: SessionMemory, choice: str) -> None:
    from knowledge_engine.src.node_deep_dive.star_task_fsm import start_layer_drill

    if choice == "how":
        start_layer_drill(memory, "HOW")
    elif choice == "mech":
        start_layer_drill(memory, "MECH")


def _clear_optional_teaching(memory: SessionMemory) -> None:
    from knowledge_engine.src.node_deep_dive.star_task_fsm import clear_layer_drill

    clear_layer_drill(memory)


def optional_teaching_blocks_transition(
    memory: SessionMemory,
    *,
    choice: str = "",
) -> bool:
    """
    Hold node-close while a Layer Drill Session still has queued gaps.

    Idle optional_fork (chips after WHY threshold, no teaching session) must
    still allow ready_for_transition so Gloss/HOW/Next chips stay visible.
    """
    from knowledge_engine.src.node_deep_dive.star_task_fsm import (
        layer_drill_blocks_transition,
    )

    if choice in ("gloss", "next"):
        return False
    if choice in ("how", "mech"):
        return True
    if layer_drill_blocks_transition(memory):
        return True
    teaching = optional_teaching_layer(memory)
    if teaching not in ("HOW", "MECHANIC"):
        return False
    if first_optional_layer_concept_id(memory, teaching):
        return True
    _clear_optional_teaching(memory)
    return False


def host_ready_for_transition(
    memory: SessionMemory,
    *,
    user_message: str = "",
    node_layer: str = "",
) -> bool:
    """
    Deterministic transition flag from chips + FSM.

    Ignores any LLM-emitted ``ready_for_transition`` / ``quick_replies``.
    ``node_layer`` is accepted for call-site symmetry with orchestrate.
    """
    from knowledge_engine.src.node_deep_dive.star_task_fsm import (
        get_star_task_status,
        layer_drill_blocks_transition,
        star_task_blocks_transition,
    )

    _ = node_layer
    choice = classify_gloss_fork_choice(user_message)
    if choice in ("how", "mech", "deep_analysis", "advanced_analysis", "deep_design"):
        return False
    if star_task_blocks_transition(memory):
        return False
    if layer_drill_blocks_transition(memory):
        return False
    if get_star_task_status(memory) not in ("not_started", "resolved"):
        return False
    if optional_teaching_blocks_transition(memory, choice=choice):
        return False
    if choice in ("gloss", "next"):
        return True
    return bool(sub_concept_coverage_complete(memory))


def _host_suggested_next_step(
    memory: SessionMemory,
    *,
    node_layer: str,
    choice: str,
) -> str:
    from knowledge_engine.src.node_deep_dive.sub_concept_evaluator import (
        normalize_node_layer,
    )

    ly = normalize_node_layer(node_layer)
    open_layers = open_optional_layers(memory, ly)
    if (
        choice not in ("gloss", "next")
        and open_layers
        and not is_full_depth_closure(memory, ly)
        and ly != "sota"
    ):
        return "deep_dive_optional"
    return "next_node"


def orchestrate_tutor_llm_output(
    memory: SessionMemory,
    llm_out: DeepDiveLLMOutput,
    *,
    user_message: str = "",
    node_layer: str = "",
) -> DeepDiveLLMOutput:
    from knowledge_engine.src.node_deep_dive.star_task_fsm import (
        get_star_task_status,
        layer_drill_blocks_transition,
        star_task_blocks_transition,
        start_layer_drill,
    )

    choice = classify_gloss_fork_choice(user_message)
    # Active teaching after factory control chips — never force transition / clear pending.
    if choice in ("how", "mech", "deep_analysis", "advanced_analysis", "deep_design"):
        if choice in ("how", "mech"):
            _activate_optional_teaching(memory, choice)
        else:
            start_layer_drill(memory, choice)
        return llm_out.model_copy(update=_host_hold_orchestration())

    # Star Task FSM open → hard hold (discussion / refinement), no node-close menu.
    if star_task_blocks_transition(memory) or layer_drill_blocks_transition(memory):
        return llm_out.model_copy(update=_host_hold_orchestration())

    if choice == "gloss":
        credit_open_optional_layers(memory, node_layer or "foundation")
        _clear_optional_teaching(memory)

    if choice == "next":
        _clear_optional_teaching(memory)

    if optional_teaching_blocks_transition(memory, choice=choice):
        return llm_out.model_copy(update=_host_hold_orchestration())

    if get_star_task_status(memory) not in ("not_started", "resolved"):
        return llm_out.model_copy(update=_host_hold_orchestration())

    complete = sub_concept_coverage_complete(memory)
    from knowledge_engine.src.node_deep_dive.tutor_behavior_state import (
        is_layer_completion_turn,
    )

    layer_done = is_layer_completion_turn(memory)
    # Host FSM: chips + coverage. Never trust LLM ready_for_transition.
    # Pass through tutor fields as generated — no follow_up coerce / ? stripping.
    transition = choice in ("gloss", "next") or complete or layer_done
    if not transition:
        return llm_out.model_copy(update=_host_hold_orchestration())
    step = _host_suggested_next_step(
        memory, node_layer=node_layer, choice=choice
    )
    memory.pending_evaluation_concept_id = ""
    memory.pending_eval_kind = ""
    memory.next_question_concept_id = ""
    memory.last_tutor_sub_concept_id = ""
    memory.asked_question_sub_concept_id = ""
    _clear_optional_teaching(memory)
    if (complete or layer_done) and memory.learning_phase in (
        "checkpoint",
        "dense_material",
        "intro_assessment",
    ):
        memory.learning_phase = "pathway_decision"
    return llm_out.model_copy(
        update={
            "ready_for_transition": True,
            "suggested_next_step": step,
            "quick_replies": [],
        }
    )
