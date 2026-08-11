"""Topic Concept Map: facade (state + evaluator + tutor orchestration)."""

from __future__ import annotations

from knowledge_engine.src.node_deep_dive.concept_map_state import (  # noqa: F401
    active_question_sub_concept_id,
    advance_next_question_after_evaluation,
    advance_sub_concepts_after_user_answer,
    apply_sub_concept_updates,
    build_coverage_summary,
    ensure_sub_concept_map,
    find_sub_concept,
    format_concept_map_for_tutor,
    list_verified_sub_concept_ids,
    promote_sub_concepts_after_tutor_message,
    resolve_evaluation_target_id,
    resolve_pending_evaluation_id,
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
    """Явное согласие пользователя на углубление (soft pitch)."""
    t = (user_message or "").strip().lower()
    if not t:
        return False
    markers = (
        "да, давай",
        "давай разбер",
        "хочу углуб",
        "углубиться",
        "deep dive",
        "разберем",
        "разберём",
        "да, хочу",
        "давай углуб",
        "погрузимся",
    )
    return any(m in t for m in markers)


GLOSS_FORK_QUICK_REPLIES: tuple[str, ...] = (
    "Хочу Gloss",
    "Дожать HOW",
    "Дожать MECH",
    "Идем дальше",
)


def aggregate_depth_flags(memory: SessionMemory) -> tuple[bool, bool, bool]:
    rows = list(memory.sub_concepts or [])
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
        for sc in memory.sub_concepts or []:
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


def gloss_fork_quick_replies(open_layers: list[str]) -> list[str]:
    if not open_layers:
        return []
    # Prefer shallowest open optional for the push chip
    push = open_layers[0]
    return ["Хочу Gloss", f"Дожать {push}", "Идем дальше"]


def classify_gloss_fork_choice(user_message: str) -> str:
    """
    Map user chip / free-text to gloss-fork action.

    Returns ``gloss`` | ``how`` | ``mech`` | ``next`` | ``\"\"``.

    Recognizes Prompt Factory prefixes ``[mode:deep_dive_mech|deep_dive_how|gloss]``
    as well as legacy plain chip labels.
    """
    raw = (user_message or "").strip()
    if not raw:
        return ""
    from knowledge_engine.src.node_deep_dive.prompt_factory import (
        factory_mode_to_gloss_choice,
        parse_tutor_mode_prefix,
    )

    cleaned, factory_mode = parse_tutor_mode_prefix(raw)
    mapped = factory_mode_to_gloss_choice(factory_mode)
    if mapped:
        return mapped
    # Match chip label on the body after a stripped unknown/default prefix.
    label = (cleaned or raw).strip()
    exact = {
        "Хочу Gloss": "gloss",
        "Дожать HOW": "how",
        "Дожать MECH": "mech",
        "Идем дальше": "next",
    }
    if label in exact:
        return exact[label]
    if raw in exact:
        return exact[raw]
    t = label.lower()
    if any(
        m in t
        for m in (
            "хочу gloss",
            "хочу глосс",
            "дай gloss",
            "дай глосс",
            "выжимк",
            "краткий gloss",
        )
    ):
        return "gloss"
    if any(
        m in t
        for m in ("дожать how", "дожать хоу", "хочу how", "слой how", "архитектур")
    ):
        return "how"
    if any(
        m in t
        for m in (
            "дожать mech",
            "дожать мех",
            "хочу mech",
            "хочу мех",
            "по формуле",
            "по коду",
            "механик",
        )
    ):
        return "mech"
    if any(
        m in t
        for m in (
            "идем дальше",
            "идём дальше",
            "к следующей нод",
            "next node",
            "перейти дальше",
        )
    ):
        return "next"
    return ""


def first_optional_layer_concept_id(memory: SessionMemory, layer_name: str) -> str:
    name = (layer_name or "").strip().upper()
    for sc in memory.sub_concepts or []:
        if name == "HOW" and sc.why_passed and not sc.how_passed:
            return (sc.id or "").strip()
        if name == "MECHANIC" and (
            (sc.why_passed and sc.how_passed and not sc.mechanic_passed)
            or (sc.why_passed and not sc.mechanic_passed)
        ):
            return (sc.id or "").strip()
    return ""


def first_optional_mech_concept_id(memory: SessionMemory) -> str:
    return first_optional_layer_concept_id(memory, "MECHANIC")


def is_quick_reply_control_message(user_message: str) -> bool:
    """True for UI Quick Reply chips — not a substantive answer for the Evaluator."""
    return classify_gloss_fork_choice(user_message) in (
        "gloss",
        "how",
        "mech",
        "next",
    )


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
    for sc in memory.sub_concepts or []:
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


def orchestrate_tutor_llm_output(
    memory: SessionMemory,
    llm_out: DeepDiveLLMOutput,
    *,
    user_message: str = "",
    node_layer: str = "",
) -> DeepDiveLLMOutput:
    choice = classify_gloss_fork_choice(user_message)
    # Active teaching after «Дожать HOW|MECH» — never force transition / clear pending.
    if choice in ("how", "mech"):
        return llm_out.model_copy(
            update={
                "ready_for_transition": False,
                "suggested_next_step": None,
            }
        )

    if choice == "gloss":
        credit_open_optional_layers(memory, node_layer or "foundation")

    complete = sub_concept_coverage_complete(memory)
    # Awaiting answer to a deep-dive control question — do not force transition.
    if (memory.pending_evaluation_concept_id or "").strip() and not bool(
        llm_out.ready_for_transition
    ):
        return llm_out

    transition = bool(llm_out.ready_for_transition) or (
        complete and choice != "how" and choice != "mech"
    )
    if choice == "gloss":
        transition = True
    if not transition:
        return llm_out
    step = (llm_out.suggested_next_step or "").strip()
    if step not in ("next_node", "deep_dive_optional"):
        step = "next_node"
    memory.pending_evaluation_concept_id = ""
    memory.next_question_concept_id = ""
    memory.last_tutor_sub_concept_id = ""
    memory.asked_question_sub_concept_id = ""
    if complete and memory.learning_phase in (
        "checkpoint",
        "dense_material",
        "intro_assessment",
    ):
        memory.learning_phase = "pathway_decision"
    from knowledge_engine.src.node_deep_dive.tutor_dialogue import (
        compose_tutor_dialogue_from_output,
        deep_dive_llm_output_from_chat_text,
    )
    from knowledge_engine.src.node_deep_dive.tutor_reply_sanitize import (
        sanitize_tutor_message_for_transition,
    )

    tutor_msg = sanitize_tutor_message_for_transition(
        compose_tutor_dialogue_from_output(llm_out)
    )
    repacked = deep_dive_llm_output_from_chat_text(
        tutor_msg,
        node_status=llm_out.node_status,
    )
    return llm_out.model_copy(
        update={
            "ready_for_transition": True,
            "suggested_next_step": step,
            "feedback_on_answer": repacked.feedback_on_answer,
            "technical_explanation": repacked.technical_explanation,
            "follow_up_question": repacked.follow_up_question,
            "quick_replies": list(getattr(llm_out, "quick_replies", None) or []),
        }
    )
