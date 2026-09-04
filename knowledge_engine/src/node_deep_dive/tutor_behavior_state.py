"""State Vector для tutor_behavior — только динамика, без дублирования system rules."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal

from knowledge_engine.src.node_deep_dive.memory_schemas import UserIntent

if TYPE_CHECKING:
    from knowledge_engine.src.node_deep_dive.memory_schemas import SessionMemory

TutorMode = str  # dialogue_feedback | lecture_dense | verify | socratic | finalize
PathwayFlag = Literal["", "base_complete", "optional_fork", "overlay_offer"]

LAYER_COMPLETION_DIRECTIVES = frozenset(
    {
        "DEEP_MASTERY_EARNED",
        "PASSED_LAYER",
    }
)

LAYER_COMPLETION_NO_QUIZ_RULE = (
    "If the current layer is fully completed, DO NOT generate a new "
    "technical/evaluative question in next_question / follow_up_question. "
    "Instead, congratulate the learner, summarize the layer, and prompt them "
    "to choose whether to dive into HOW/MECH/Advanced/Deep mode or proceed "
    "to the next topic. Host owns the chips."
)


def is_layer_completion_turn(memory: SessionMemory | None) -> bool:
    """True when this turn closed a layer — Host offers pathway chips."""
    if memory is None:
        return False
    if bool(getattr(memory, "is_layer_just_completed", False)):
        return True
    from knowledge_engine.src.node_deep_dive.star_task_fsm import (
        layer_drill_is_active,
        star_task_blocks_transition,
    )

    if layer_drill_is_active(memory) or star_task_blocks_transition(memory):
        return False
    drill = getattr(memory, "layer_drill", None)
    if drill is not None and (getattr(drill, "status", "") or "") == "DRILL_COMPLETE":
        return True
    directive = (memory.last_eval_directive or "").strip()
    if directive in LAYER_COMPLETION_DIRECTIVES:
        return True
    from knowledge_engine.src.node_deep_dive.concept_map_state import (
        sub_concept_coverage_complete,
    )

    # Coverage closed even if Host still holds the last credited id as next_question.
    if sub_concept_coverage_complete(memory):
        return True
    nxt = (memory.next_question_concept_id or "").strip()
    if nxt:
        return False
    return directive in ("PASSED_CLEAN", "PASSED_WITH_GLOSS")


def resolve_tutor_mode(
    intent: UserIntent,
    action: str,
    learning_mode: str,
    learning_phase: str,
    user_message: str,
) -> TutorMode:
    if action == "verify":
        return "verify"
    if learning_mode == "socratic_point" or learning_phase == "socratic_focus":
        return "socratic"
    if intent == "INTENT_FINALIZE":
        return "finalize"
    if intent == "INTENT_EXPLAIN":
        return "lecture_dense"
    if learning_mode == "lecture":
        from knowledge_engine.src.node_deep_dive.control_intent import (
            is_short_lecture_request,
        )

        if is_short_lecture_request(user_message):
            return "lecture_dense"
    if intent == "INTENT_SHIFT_FOCUS":
        return "dialogue_feedback"
    return "dialogue_feedback"


def build_tutor_behavior_state(
    intent: UserIntent,
    action: str,
    learning_mode: str,
    learning_phase: str,
    user_message: str,
    *,
    has_user_focus: bool = False,
    memory: SessionMemory | None = None,
    node_layer: str = "",
) -> dict[str, Any]:
    mode = resolve_tutor_mode(
        intent, action, learning_mode, learning_phase, user_message
    )
    focus_restriction = (
        "Узкий фокус текущей подтемы; не разворачивать всю ноду."
        if has_user_focus or mode == "dialogue_feedback"
        else ""
    )
    next_action, pathway = _next_action_and_pathway_for_mode(
        mode,
        intent,
        action,
        learning_phase,
        memory=memory,
        user_message=user_message,
        node_layer=node_layer,
    )
    drill_snap: dict[str, Any] = {}
    if memory is not None:
        from knowledge_engine.src.node_deep_dive.star_task_fsm import (
            layer_drill_progress,
        )

        drill_snap = layer_drill_progress(memory)
    return {
        "current_mode": mode,
        "step_intent": intent,
        "learning_phase": learning_phase,
        "learning_mode": learning_mode,
        "node_layer": (node_layer or "").strip() or "foundation",
        "pathway": pathway,
        "focus_restriction": focus_restriction,
        "next_action": next_action,
        "layer_drill_session": drill_snap,
        "last_eval_directive": (
            (memory.last_eval_directive or "").strip() if memory is not None else ""
        ),
    }


def _next_action_for_mode(
    mode: TutorMode,
    intent: UserIntent,
    action: str,
    learning_phase: str = "",
    *,
    memory: SessionMemory | None = None,
    user_message: str = "",
    node_layer: str = "",
) -> str:
    """Back-compat wrapper — returns next_action text only."""
    text, _pathway = _next_action_and_pathway_for_mode(
        mode,
        intent,
        action,
        learning_phase,
        memory=memory,
        user_message=user_message,
        node_layer=node_layer,
    )
    return text


def _layer_drill_teaching_action(memory: SessionMemory) -> tuple[str, PathwayFlag]:
    from knowledge_engine.src.node_deep_dive.star_task_fsm import (
        current_layer_drill_concept_id,
        format_layer_drill_invariants,
        layer_drill_progress,
    )

    snap = layer_drill_progress(memory)
    cid = current_layer_drill_concept_id(memory) or "open sub-topic"
    layer = str(snap.get("target_layer") or "")
    progress = str(snap.get("progress") or "0/0")
    title = str(snap.get("current_sub_concept_title") or cid)
    invariants = format_layer_drill_invariants(memory).strip()
    if layer in ("HOW", "MECH"):
        teach_layer = "HOW" if layer == "HOW" else "MECHANIC"
        text, pathway = _optional_layer_teaching_action(teach_layer, cid)
        extra = (
            f" DRILL_ACTIVE Progress={progress} current=`{title}`. "
            "DO NOT declare the node or layer complete."
        )
        return (
            f"{invariants}\n\n{text}{extra}" if invariants else f"{text}{extra}",
            pathway,
        )
    extra = (
        f"DRILL_ACTIVE overlay layer={layer} Progress={progress} "
        f"current=`{title}` id=`{cid}`. node_completed=false. "
        "DO NOT declare the node or layer complete."
    )
    return ((f"{invariants}\n\n{extra}" if invariants else extra), "")


def _optional_layer_teaching_action(
    layer_name: str, cid: str
) -> tuple[str, PathwayFlag]:
    name = (layer_name or "").strip().upper()
    if name == "MECHANIC":
        return (
            f"GLOSS_FORK_CHOICE=mech DEEP_DIVE: Active Teaching on `{cid}`. "
            "ready_for_transition=false; suggested_next_step=null; "
            f"question_sub_concept_id={cid}. "
            "HARD MECH RULES: in technical_explanation you MUST include at least one "
            "hands-on artifact — Python/Pydantic code, asyncio architecture snippet, "
            "OR explicit mathematical consensus/weighting formula in $LaTeX$. "
            "FORBIDDEN: abstract summary without code/math. "
            "Then follow_up_question = ONE short edge-case question about THAT "
            "code/formula (verify mastery). Do NOT close the node; wait for the "
            "user answer (Evaluator scores the next turn).",
            "",
        )
    return (
        f"GLOSS_FORK_CHOICE=how DEEP_DIVE: Active Teaching on `{cid}`. "
        "ready_for_transition=false; suggested_next_step=null; "
        f"question_sub_concept_id={cid}. "
        "Deliver concrete architecture/invariants (diagrams/pipelines OK); "
        "follow_up_question = ONE targeted HOW question. "
        "Do NOT set ready_for_transition=true; wait for user answer.",
        "",
    )


def _next_action_and_pathway_for_mode(
    mode: TutorMode,
    intent: UserIntent,
    action: str,
    learning_phase: str = "",
    *,
    memory: SessionMemory | None = None,
    user_message: str = "",
    node_layer: str = "",
) -> tuple[str, PathwayFlag]:
    from knowledge_engine.src.node_deep_dive.concept_map import (
        classify_gloss_fork_choice,
        first_optional_layer_concept_id,
        is_full_depth_closure,
        open_optional_layers,
        sub_concept_coverage_complete,
        user_accepted_optional_deep_dive,
    )
    from knowledge_engine.src.node_deep_dive.sub_concept_evaluator import (
        normalize_node_layer,
    )

    phase = (learning_phase or "").strip()
    ly = normalize_node_layer(node_layer)
    if memory is not None:
        from knowledge_engine.src.node_deep_dive.star_task_fsm import (
            layer_drill_is_active,
        )

        early_choice = classify_gloss_fork_choice(user_message)
        if layer_drill_is_active(memory) and early_choice not in ("gloss", "next"):
            return _layer_drill_teaching_action(memory)
    if memory is not None and (
        sub_concept_coverage_complete(memory) or is_layer_completion_turn(memory)
    ):
        from knowledge_engine.src.node_deep_dive.star_task_fsm import (
            get_star_task_status,
            is_overlay_eval_kind,
            star_task_blocks_transition,
        )

        choice = classify_gloss_fork_choice(user_message)
        open_layers = open_optional_layers(memory, ly)
        pending_kind = (memory.pending_eval_kind or "").strip().lower()
        last_dir = (memory.last_eval_directive or "").strip()
        star = get_star_task_status(memory)

        if choice == "gloss":
            named = "/".join(open_layers) or "optional"
            return (
                f"GLOSS_FORK_CHOICE=gloss: give a short Glossary digest of open optional "
                f"layer(s) [{named}] with key formulas/patterns (2–8 sentences); NO quiz; "
                "system will auto-credit those optional layers; "
                "Host owns ready_for_transition / chips. "
                "question_sub_concept_id=null; "
                "invite UI next-node choice; do not invent next node titles.",
                "",
            )
        if choice in ("how", "mech"):
            layer_name = "HOW" if choice == "how" else "MECHANIC"
            cid = (
                first_optional_layer_concept_id(memory, layer_name) or "open sub-topic"
            )
            return _optional_layer_teaching_action(layer_name, cid)
        if (
            is_overlay_eval_kind(choice)
            or is_overlay_eval_kind(pending_kind)
            or star_task_blocks_transition(memory)
        ):
            from knowledge_engine.src.node_deep_dive.concept_map_state import (
                list_deep_mastery_concept_ids,
            )
            from knowledge_engine.src.node_deep_dive.star_task_fsm import (
                canonical_overlay_kind,
            )

            focus = (
                (memory.next_question_concept_id or "").strip()
                or (memory.pending_evaluation_concept_id or "").strip()
                or "active sub-topic"
            )
            earned = list_deep_mastery_concept_ids(memory)
            earned_note = (
                f" Asterisk-question Deep Mastery already earned for: {', '.join(earned[:6])}."
                if earned
                else ""
            )
            kind = canonical_overlay_kind(choice or pending_kind)
            if last_dir == "DEEP_MASTERY_EARNED" and star == "in_progress":
                return (
                    f"DEEP_MASTERY_EARNED continue overlay on `{focus}` "
                    f"(overlay_kind={kind}; star_task_status={star}; "
                    "node_completed=false)."
                    f"{earned_note} "
                    "Brief credit (1–3 sentences) for the prior overlay pass; "
                    "then Active Teaching on THIS new sub-topic — same overlay track. "
                    f"question_sub_concept_id={focus}. "
                    "Do NOT close the node, do not offer pathway chips, "
                    "do not re-decree base-theory completion.",
                    "",
                )
            if star == "needs_refinement" or last_dir == "STAR_TASK_NEEDS_REFINEMENT":
                hint = ""
                from knowledge_engine.src.node_deep_dive.concept_map import (
                    find_sub_concept,
                )

                row = find_sub_concept(memory, focus) if focus else None
                if row is not None and (row.focus_hint or "").strip():
                    hint = (row.focus_hint or "").strip()[:280]
                return (
                    f"STAR_TASK_FSM=needs_refinement on `{focus}` "
                    f"(star_task_status={star}; node_completed=false; "
                    f"overlay_kind={kind})."
                    f"{earned_note} "
                    "DISCUSSION / REVIEW MODE. "
                    f"question_sub_concept_id={focus}. "
                    "If [EVALUATOR_CRITIQUE_JSON] is present: pointwise STRONG/RISK/WEAK "
                    "review + unaccounted_edge_cases (see critique review rules). "
                    "Otherwise feedback_on_answer: brief review of what is still open "
                    "(edge cases / couplings from prior overlay analysis)"
                    + (f"; focus_hint=«{hint}»" if hint else "")
                    + ". "
                    "follow_up_question: ONE concrete refinement prompt that forces "
                    "the learner to close those gaps — no new unrelated homework, "
                    "no synthetic numbers, no transition menus. quick_replies=[].",
                    "",
                )
            if kind == "advanced_analysis":
                return (
                    f"ADVANCED_ANALYSIS / ADVANCED_ASTERISK PATH on `{focus}` "
                    f"(pending_eval_kind={pending_kind or 'requested'}; "
                    f"star_task_status={star}; node_completed=false)."
                    f"{earned_note} "
                    "PARALLEL Bloom L4 Analyze track (vulnerabilities, races, "
                    "edge-cases, resource cost, P99, extreme correctness). "
                    f"question_sub_concept_id={focus}. "
                    "Follow Advanced Analysis isolated prompt: LONG multi-section "
                    "technical_explanation grounded in SOURCE REGISTRY [Sx] and "
                    "RAG [Rx] (cite every major claim; no invented OS/network guts). "
                    "If [EVALUATOR_CRITIQUE_JSON] is present after a learner answer: "
                    "pointwise critique review first, then deepen L4 analysis. "
                    "Then ONE analysis follow_up_question that tests those "
                    "vulnerabilities — no green-field redesign, no synthetic "
                    "counts/timers absent from sources. Wait for the user's answer. "
                    "No transition menus or pathway chips (host sets orchestration).",
                    "",
                )
            return (
                f"DEEP_ANALYSIS / DEEP_DESIGN / DEEP_ASTERISK PATH on `{focus}` "
                f"(pending_eval_kind={pending_kind or 'requested'}; "
                f"star_task_status={star}; node_completed=false)."
                f"{earned_note} "
                "PARALLEL Bloom L5/L6 Evaluate/Create track (system design from "
                "scratch, component synthesis, trade-offs, key decisions). "
                f"question_sub_concept_id={focus}. "
                "Follow Deep Design isolated prompt: LONG multi-section "
                "technical_explanation grounded in SOURCE REGISTRY [Sx] and "
                "RAG [Rx] (cite every major claim; no invented OS/network guts). "
                "If [EVALUATOR_CRITIQUE_JSON] is present after a learner answer: "
                "pointwise critique review first, then deepen design analysis. "
                "Then ONE design follow_up_question that tests couplings/invariants "
                "from that analysis — no synthetic counts/timers absent from "
                "sources. Wait for the user's design answer. "
                "No transition menus or pathway chips (host sets orchestration).",
                "",
            )
        if choice == "next":
            return (
                "GLOSS_FORK_CHOICE=next: confirm readiness; no quiz; "
                "Host owns ready_for_transition / chips. "
                "question_sub_concept_id=null; "
                "do not invent next node titles — UI picks the next node.",
                "base_complete",
            )
        teaching = (getattr(memory, "active_optional_layer", "") or "").strip().upper()
        if teaching in ("HOW", "MECHANIC"):
            cid = first_optional_layer_concept_id(memory, teaching)
            if cid:
                return _optional_layer_teaching_action(teaching, cid)
        if last_dir == "DEEP_MASTERY_EARNED" or star == "resolved":
            from knowledge_engine.src.node_deep_dive.concept_map_state import (
                list_deep_mastery_concept_ids,
            )

            earned = list_deep_mastery_concept_ids(memory)
            starred = ", ".join(earned[-3:]) if earned else "active sub-topic"
            return (
                "pathway=overlay_offer; DEEP_MASTERY_EARNED / star_task_status=resolved: "
                f"briefly celebrate Deep Mastery for [{starred}] as a PARALLEL track "
                "(not a base-theory re-decree). "
                "Host owns ready_for_transition / chips (ADVANCED_ASTERISK vs "
                "DEEP_ASTERISK from the weakness ledger; always a next-node fallback). "
                f"{LAYER_COMPLETION_NO_QUIZ_RULE} "
                "Write natural peer commentary; "
                "no technical quiz; do not invent next node titles. "
                "FORBIDDEN clichés: base-theory-closed / optional-MECH / 100%-node.",
                "overlay_offer",
            )
        if open_layers and not is_full_depth_closure(memory, ly) and ly != "sota":
            named = "/".join(open_layers)
            return (
                f"pathway=optional_fork (node_layer={ly}): threshold met; "
                f"open_optional_layers=[{named}]. "
                "Host will set ready_for_transition and quick_replies from open layers only. "
                f"{LAYER_COMPLETION_NO_QUIZ_RULE} "
                "Write natural peer commentary introducing the Host next-step options; "
                "follow_up_question = mode-choice CTA only (not a technical quiz); "
                "do not invent chip labels or closed layers; "
                "FORBIDDEN clichés: base-theory-closed / conceptual-minimum scripts.",
                "optional_fork",
            )
        return (
            f"pathway=base_complete (node_layer={ly}): base WHY/HOW/MECH coverage closed "
            "(BASE theory mastery — NOT absolute node death). "
            "Host will set ready_for_transition / suggested_next_step and UI chips "
            "from the cross-node weakness ledger: DEEP_ASTERISK after a clean core, "
            "ADVANCED_ASTERISK when weakness_tags are open; always a next-node fallback. "
            f"{LAYER_COMPLETION_NO_QUIZ_RULE} "
            "Write natural peer commentary for this pathway; "
            "follow_up_question = mode-choice CTA only (not a technical quiz); "
            "do not invent next node titles or chip menus. "
            "FORBIDDEN clichés: base-theory-closed scripts, optional-MECH scripts, "
            "absolute 100%-node decrees.",
            "base_complete",
        )
    if phase == "pathway_decision":
        return (
            "PATHWAY: node wrap-up; ready_for_transition=true; "
            "suggested_next_step=next_node or deep_dive_optional; "
            f"{LAYER_COMPLETION_NO_QUIZ_RULE} "
            "follow_up_question = mode-choice CTA only; no automatic technical quiz; "
            "do not invent next node titles.",
            "base_complete",
        )
    if user_accepted_optional_deep_dive(user_message):
        return (
            "Пользователь согласился на углубление — один edge-case вопрос по выбранной "
            "подтеме; ready_for_transition=false.",
            "",
        )
    if mode == "lecture_dense":
        pending_q = ""
        directive = ""
        hint = ""
        evidence = ""
        probe_layer = ""
        if memory is not None:
            pending_q = (memory.last_tutor_follow_up_question or "").strip()
            directive = (memory.last_eval_directive or "").strip()
            from knowledge_engine.src.node_deep_dive.concept_map_state import (
                probe_layer_from_directive,
                resolve_transparency_row,
            )

            probe_layer = probe_layer_from_directive(directive)
            row = resolve_transparency_row(memory)
            if row is not None:
                hint = (row.focus_hint or "").strip()
                evidence = (row.evidence or "").strip()
        gap_open = bool(hint or probe_layer)
        part1 = (
            "LECTURE_DENSE 2-part sequence: PART 1 (lecture_body) = dense "
            "theory/architecture/code. When [TARGET_FOCUS_AND_GAPS] is present, "
            "allocate ~80% of volume to facts and cause-effect required by "
            "last_evaluator_focus_hint for the uncredited probe_layer"
            f"{f' ({probe_layer})' if probe_layer else ''}. "
            "Credited theses from last_evaluator_evidence are brief context only — "
            "do not re-teach already passed layers. "
        )
        part2 = (
            "PART 2 = exactly ONE closing technical question exclusively in "
            "checkpoint_prompt. "
        )
        if gap_open:
            part2 += (
                "checkpoint_prompt MUST test last_evaluator_focus_hint and "
                f"probe_layer{f'={probe_layer}' if probe_layer else ''} "
                "STRICTLY. FORBIDDEN: blind RE-STATE of [OPEN_NODE_QUESTION] "
                "if that question belongs to an already-passed layer. "
            )
            if hint:
                part2 += f"focus_hint: «{hint[:240]}». "
            if evidence:
                part2 += f"already credited (context only): «{evidence[:160]}». "
            if pending_q:
                part2 += (
                    f"Pending question «{pending_q[:160]}» may be refined ONLY "
                    "if it already tests this focus_hint / probe_layer; "
                    "otherwise write a new checkpoint. "
                )
        elif pending_q:
            part2 += (
                "RE-STATE or REFINE "
                f"«{pending_q[:240]}» so it tests the lecture just given. "
            )
        part2 += (
            "NEVER include self-check questions or 'Самопроверка' headers "
            "inside lecture_body."
        )
        return (part1 + part2, "")
    if mode == "verify":
        return (
            "Финальная проверка по матрице концептов; без бесконечного допроса.",
            "",
        )
    if mode == "socratic":
        return ("Один контрвопрос или edge-case; без лекции.", "")
    if mode == "finalize":
        return ("Итог по матрице и mastery; честно назвать пробелы.", "")
    if intent == "INTENT_SHIFT_FOCUS":
        return (
            "Сменить ракурс в рамках ответа пользователя; разбор его паттернов; "
            "без сброса на базовую методичку.",
            "",
        )
    if memory is not None:
        from knowledge_engine.src.node_deep_dive.concept_map import find_sub_concept

        directive = (memory.last_eval_directive or "").strip()
        if directive == "PASSED_WITH_GLOSS":
            return (
                "THRESHOLD PASSED_WITH_GLOSS (mid-map): credit required layers, gloss "
                "open optional depth yourself, then advance to next sub_concept; "
                "FORBIDDEN to quiz optional depth unless user asks; "
                "ready_for_transition stays false unless map is fully VERIFIED "
                "(then use pathway optional_fork / base_complete by node_layer).",
                "",
            )
        if directive == "PASSED_CLEAN":
            return (
                "THRESHOLD PASSED_CLEAN: brief credit and move to next UNCHECKED "
                "sub_concept; no extra grilling on the closed id.",
                "",
            )
        if directive == "DEEP_MASTERY_EARNED":
            return (
                "THRESHOLD DEEP_MASTERY_EARNED: credit Asterisk-question Deep Mastery as a "
                "parallel track; brief celebration; "
                f"{LAYER_COMPLETION_NO_QUIZ_RULE} "
                "FORBIDDEN absolute node-closure decrees / clichés.",
                "overlay_offer",
            )
        if directive == "STAR_TASK_NEEDS_REFINEMENT":
            return (
                "STAR_TASK_FSM=needs_refinement: DISCUSSION / REVIEW mode. "
                "ready_for_transition=false; quick_replies=[]. "
                "If [EVALUATOR_CRITIQUE_JSON] present: pointwise STRONG/RISK/WEAK + edges. "
                "Otherwise point to unclosed edge cases from prior deep analysis; "
                "one refinement follow_up_question; FORBIDDEN node-closure decrees.",
                "",
            )
        if directive.startswith("PROBE_NEXT_LAYER:"):
            layer = directive.split(":", 1)[-1].strip().upper() or "WHY"
            return (
                f"THRESHOLD {directive}: stay on the SAME sub_concept; "
                f"follow_up_question MUST probe ONLY layer {layer} "
                "(do not demand deeper layers). "
                "feedback_kind MUST be NEEDS_CORRECTION (Host already marked "
                "the layer open). Host prepends the credited/missing plaque. "
                "ready_for_transition=false.",
                "",
            )
        nid = (memory.next_question_concept_id or "").strip()
        row = find_sub_concept(memory, nid) if nid else None
        if row is not None and row.status in ("partial", "gap"):
            hint = (row.focus_hint or "").strip()
            return (
                "PARTIAL/GAP on current sub-topic: do NOT switch to a new sub_concept. "
                "feedback_kind MUST be NEEDS_CORRECTION; Host prepends the "
                "credited/missing plaque from focus_hint"
                + (f": «{hint[:200]}»" if hint else "")
                + "; follow_up_question — probe same id per last_eval_directive "
                "(WHY/HOW only as directed); ready_for_transition=false.",
                "",
            )
    if phase in ("checkpoint", "pathway_decision"):
        return (
            "Peer-разбор ответа на «ты»; затем следующая подтема concept_map; "
            "без отчётных заголовков и без третьего вопроса по той же теме.",
            "",
        )
    return (
        "Peer-to-peer на «ты» (без «пользователь указал»); развернутый инженерный разбор "
        "(~180+ слов минимум, без искусственного потолка длины); "
        "разбери паттерны из user_message; obey last_eval_directive from Threshold Engine; "
        "после VERIFIED — переход к следующей sub_concept; при PARTIAL — probe only the "
        "directed layer; один новый вопрос максимум.",
        "",
    )


def overlay_offer_host_chips(
    memory: SessionMemory | None = None,
    *,
    curriculum_id: str = "",
    persist: bool = True,
) -> list[str]:
    """
    Adaptive overlay chips after core close.

    Reads the cross-node weakness ledger: open tags → ADVANCED_ASTERISK chip;
    clean history → DEEP_ASTERISK chip; always include next-node fallback.
    Requires 100% core mastery before any overlay chip is offered.
    """
    from knowledge_engine.src.node_deep_dive.star_task_fsm import (
        overlay_offer_quick_replies,
    )
    from knowledge_engine.src.resilience_manager import core_ready_for_overlay

    if memory is not None and not core_ready_for_overlay(memory):
        return []
    cid = (curriculum_id or "").strip()
    tags: list[str] = []
    if cid:
        from knowledge_engine.context_drift_manager import ContextDriftManager

        tags = ContextDriftManager(cid, persist=persist).open_weakness_tags()
    return overlay_offer_quick_replies(weakness_tags=tags)


def format_tutor_behavior_state_block(state: dict[str, Any]) -> str:
    return "### tutor_behavior_state\n" + json.dumps(
        state, ensure_ascii=False, indent=0
    )
