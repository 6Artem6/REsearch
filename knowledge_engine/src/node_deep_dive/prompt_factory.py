"""Prompt Factory: select isolated system prompts from UI [mode:…] prefixes."""

from __future__ import annotations

import re
from typing import Any, Literal

from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.src.node_deep_dive.advanced_analysis_prompt import (
    ADVANCED_ANALYSIS_PROMPT,
)
from knowledge_engine.src.node_deep_dive.blitz_mode_prompt import BLITZ_MODE_PROMPT
from knowledge_engine.src.node_deep_dive.context_bounded_eval import (
    CONTEXT_BOUNDED_QUESTION_RULES,
)
from knowledge_engine.src.node_deep_dive.deep_analysis_prompt import (
    DEEP_ANALYSIS_PROMPT,
)
from knowledge_engine.src.node_deep_dive.deep_design_prompt import (
    DEEP_DESIGN_PROMPT,
)
from knowledge_engine.src.node_deep_dive.deep_dive_how_prompt import (
    DEEP_DIVE_HOW_PROMPT,
)
from knowledge_engine.src.node_deep_dive.deep_dive_mech_prompt import (
    DEEP_DIVE_MECH_PROMPT,
)
from knowledge_engine.src.node_deep_dive.gloss_summary_prompt import (
    GLOSS_SUMMARY_PROMPT,
)
from knowledge_engine.src.node_deep_dive.intent_definitions import (
    FACTORY_MODE_TO_INTENT,
)
from knowledge_engine.src.node_deep_dive.lecture_prompt_en import (
    LECTURE_MODE_STRUCTURE_RULES,
)
from knowledge_engine.src.node_deep_dive.memory_schemas import LayerDrillSession
from knowledge_engine.src.node_deep_dive.next_module_prompt import NEXT_MODULE_PROMPT
from knowledge_engine.src.node_deep_dive.self_check_mode_prompt import (
    SELF_CHECK_MODE_PROMPT,
)
from knowledge_engine.src.node_deep_dive.socratic_mode_prompt import (
    SOCRATIC_MODE_PROMPT,
)
from knowledge_engine.src.node_deep_dive.socratic_poles import (
    SOCRATIC_POLES_STATIC_RULES,
)
from knowledge_engine.src.node_deep_dive.star_task_fsm import OVERLAY_EVAL_KINDS
from knowledge_engine.src.node_deep_dive.tutor_critique_prompt import (
    ANTI_SYCOPHANCY_INVARIANTS,
    TUTOR_CRITIQUE_REVIEW_RULES,
)

TutorFactoryMode = Literal[
    "default",
    "deep_dive_mech",
    "deep_dive_how",
    "deep_analysis",
    "advanced_analysis",
    "deep_design",
    "gloss",
    "lecture",
    "blitz",
    "socratic",
    "self_check",
    "next_module",
]

_MODE_PREFIX_RE = re.compile(
    r"^\[mode:(deep_dive_mech|deep_dive_how|deep_analysis|"
    r"advanced_analysis|deep_design|gloss|lecture|blitz|socratic|"
    r"self_check|next_module)\]\s*",
    re.I,
)

# Free-text vector matches (no literal [mode:...] tag) promoted to these
# factory modes when classify_control_chip resolves them — see
# _promote_vector_chip_to_mode(). Scoped to modes added for the Intent
# Routing & Evaluator Bypass refactor only; existing gloss/how/mech/etc.
# resolution via exact chip labels is untouched.
_VECTOR_PROMOTABLE_FACTORY_MODES = frozenset(
    {"blitz", "socratic", "self_check", "next_module"}
)

_CHIP_MODE_TO_CHOICE = dict(FACTORY_MODE_TO_INTENT)
# intent (VectorIntentRouter/classify_control_chip output) -> factory_mode.
_CHIP_MODE_TO_CHOICE_REVERSE = {v: k for k, v in _CHIP_MODE_TO_CHOICE.items()}

_JSON_CONTRACT_TAIL = (
    "=== JSON OUTPUT (DeepDiveTutorContract) ===\n"
    "Strictly output valid JSON matching DeepDiveTutorContract. "
    "No tutor_message field. "
    "Generation order: audit (single flat TechnicalConceptAudit, not oneOf) "
    "FIRST, then technical_explanation, then follow_up_question. "
    "Learner-facing review lives inside audit: confirmation (EXACT) or "
    "praise_points + correction_breakdown (PARTIAL / NEEDS_CORRECTION); "
    "unused branch is empty string. "
    "Do not emit feedback_on_answer or 📋/🎯 — Host assembles those. "
    "User-facing text fields (audit confirmation/correction_breakdown, "
    "technical_explanation, follow_up_question) MUST be in natural Russian.\n"
    + ANTI_SYCOPHANCY_INVARIANTS
)

_EXPLAIN_JSON_TAIL = (
    "=== JSON OUTPUT (DeepDiveExplainContract) — HARD ===\n"
    "Host skipped Evaluator this turn. Return DeepDiveExplainContract only. "
    "FORBIDDEN keys: audit, confirmation, correction_breakdown, "
    "feedback_on_answer. Fill technical_explanation and optional "
    "follow_up_question. User-facing strings in natural Russian.\n"
)

_DEEP_ANALYSIS_JSON_TAIL = (
    "=== JSON OUTPUT (DeepDiveDeepAnalysisContract) ===\n"
    "Strictly output valid JSON matching DeepDiveDeepAnalysisContract. "
    "No tutor_message field. "
    "Generation order: audit (single flat TechnicalConceptAudit) FIRST, "
    "then learner-facing fields. "
    "follow_up_question is REQUIRED (non-empty) — exactly ONE engineering "
    "design question from the Problem / Edge / Trade-off analysis. "
    "Do not emit feedback_on_answer or 📋/🎯 — Host assembles those. "
    "User-facing text fields MUST be in natural Russian.\n"
    "Host sets orchestration flags after generation — focus on analysis + question.\n"
    "If SOURCE REGISTRY is empty: references MUST be [].\n" + ANTI_SYCOPHANCY_INVARIANTS
)

_FACTORY_CONTROL_MODES = frozenset(
    {
        "deep_dive_mech",
        "deep_dive_how",
        "deep_analysis",
        "advanced_analysis",
        "deep_design",
        "gloss",
        "blitz",
        "socratic",
        "self_check",
        "next_module",
    }
)

# Affirmative session flags only — never mention topic completion / node closure.
_STAR_TASK_SESSION_FLAGS = (
    "=== SESSION FLAGS (host-authoritative) ===\n"
    "node_completed: false\n"
    "Mode: Asterisk-question overlay — produce the multi-section "
    "technical_explanation and a non-empty follow_up_question.\n"
    "Do not emit transition menus, next-node CTAs, or pathway chips.\n"
)


def requires_deep_analysis_guard(
    factory_mode: TutorFactoryMode | str = "",
    *,
    star_task_status: str = "",
) -> bool:
    """True when generation must use overlay asterisk-question context + isolated system."""
    mode = (factory_mode or "").strip().lower()
    if mode in OVERLAY_EVAL_KINDS:
        return True
    star = (star_task_status or "").strip().lower()
    return star in ("in_progress", "needs_refinement")


def deep_analysis_hard_guard_block() -> str:
    """Session flags injected into movable context under deep_analysis / star task."""
    return _STAR_TASK_SESSION_FLAGS


def format_deep_analysis_novelty_block(
    memory: Any,
    *,
    rag_exhausted: bool = False,
    poles_block: str = "",
    attraction_summary: str = "",
    registry_empty: bool = False,
    atoms_empty: bool = False,
) -> str:
    """
    Dynamic bottom-of-payload for isolated deep_analysis prompts.

    Order (cache-friendly: static system stays stable; this block changes):
    1. [SOCRATIC_POLES_STATE]
    2. [RAG_STATUS: EXHAUSTED] / [CITATION_POLICY] when applicable
    3. [PRIOR_ASTERISK_QUESTION_THESIS_DIGESTS]
    4. [RAG_COVERAGE_STATE]
    """
    from knowledge_engine.src.node_deep_dive.deep_analysis_coverage import (
        format_citation_policy_block,
        format_rag_coverage_state_block,
        format_rag_exhausted_directive,
        format_thesis_digests_block,
    )

    parts: list[str] = []
    poles = (poles_block or "").strip()
    if poles:
        parts.append(poles)
    exhausted = bool(rag_exhausted) or bool(atoms_empty)
    if exhausted:
        parts.append(
            format_rag_exhausted_directive(attraction_summary=attraction_summary)
        )
    cite_pol = format_citation_policy_block(
        registry_empty=registry_empty,
        atoms_empty=exhausted,
    )
    if cite_pol:
        parts.append(cite_pol)
    parts.append(format_thesis_digests_block(memory))
    parts.append(format_rag_coverage_state_block(memory, rag_exhausted=exhausted))
    return "\n\n".join(parts)


def deep_analysis_context_policy() -> dict[str, bool | str]:
    """
    Deterministic host flags for deep_analysis / open Star Task turns.

    Callers must pass these into prompt assembly and force them on the
    LLM response in Python — the model must not invent orchestration.
    """
    return {
        "node_completed": False,
        "ready_for_transition": False,
        "suppress_topic_completion": True,
        "use_isolated_deep_analysis_system": True,
    }


# Visible chip bodies when the client sends a tag-only ``[mode:…]`` payload.
# Must match ``OVERLAY_CHIP_DISPLAY`` in skill-tree ActionChips.js.
_MODE_VISIBLE_FALLBACK: dict[str, str] = {
    "advanced_analysis": "Анализ уязвимостей",
    "deep_design": "Архитектурный дизайн",
    "deep_analysis": "Задачка со звёздочкой",
}


def parse_tutor_mode_prefix(user_message: str) -> tuple[str, TutorFactoryMode]:
    """
    Strip a leading ``[mode:…]`` prefix if present.

    Returns ``(cleaned_user_message, factory_mode)``.
    """
    raw = (user_message or "").strip()
    if not raw:
        return "", "default"
    m = _MODE_PREFIX_RE.match(raw)
    if not m:
        return raw, "default"
    mode = m.group(1).strip().lower()
    body = raw[m.end() :].strip()
    if mode in (
        "deep_dive_mech",
        "deep_dive_how",
        "deep_analysis",
        "advanced_analysis",
        "deep_design",
        "gloss",
        "lecture",
        "blitz",
        "socratic",
        "self_check",
        "next_module",
    ):
        return body, mode  # type: ignore[return-value]
    return body or raw, "default"


def display_user_after_mode_prefix(
    user_message: str,
) -> tuple[str, TutorFactoryMode]:
    """Strip ``[mode:…]`` and keep the visible body for history / LLM.

    Tag-only overlay chips fall back to a short label so the user turn is
    not stored as the raw tag (which the UI then strips to an empty bubble).
    """
    body, mode = parse_tutor_mode_prefix(user_message)
    if body:
        return body, mode
    fallback = _MODE_VISIBLE_FALLBACK.get(mode, "")
    if fallback:
        return fallback, mode
    return body, mode


# ---------------------------------------------------------------------------
# Layer Drill Session — specialized per-layer prompt generators
# ---------------------------------------------------------------------------

_DRILL_DEPTH_INVARIANT = (
    "=== GENERAL DEPTH INVARIANT (HARD — overrides any shorter-form instructions) ===\n"
    "The volume and depth of the theoretical opening block in this drill MUST NOT "
    "fall short of a normal lecture / ordinary teaching turn.\n"
    "FORBIDDEN: answering with 1–2 dry introductory sentences and then a question.\n"
    "REQUIRED sequence:\n"
    "1) First deliver a full, deep theoretical treatment of the current sub-topic "
    "(target ~300 Russian words in theory_body, never fewer than 150: details, "
    "worked examples, "
    "code listings and/or diagrams/schemes).\n"
    "2) Only THEN ask exactly ONE checkpoint question in next_question.\n"
    "This invariant OVERRIDES any «keep it to 1–2 sentences» guidance elsewhere "
    "in the system prompt.\n"
)


def _drill_progress_view(
    drill_session: LayerDrillSession,
    memory: Any = None,
) -> dict[str, Any]:
    """Progress + titles for specialized drill prompts."""
    ids = list(drill_session.target_sub_concept_ids or [])
    total = len(ids)
    completed = min(int(drill_session.current_index or 0), total)
    ordinal = completed + 1 if total else 0
    current_id = (drill_session.get_current_sub_concept_id() or "").strip()
    next_id = ""
    if drill_session.current_index + 1 < total:
        next_id = ids[drill_session.current_index + 1]
    title = current_id
    next_title = next_id
    if memory is not None:
        from knowledge_engine.src.node_deep_dive.star_task_fsm import (
            layer_drill_progress,
        )

        snap = layer_drill_progress(memory)
        if snap.get("status") == "DRILL_ACTIVE":
            completed = int(snap.get("completed") or completed)
            total = int(snap.get("total") or total)
            ordinal = completed + 1 if total else 0
            title = str(snap.get("current_sub_concept_title") or title)
            next_title = str(snap.get("next_sub_concept_title") or next_title)
            current_id = str(snap.get("current_sub_concept_id") or current_id)
        else:
            from knowledge_engine.src.node_deep_dive.concept_map_state import (
                find_sub_concept,
            )

            row = find_sub_concept(memory, current_id) if current_id else None
            if row is not None:
                title = (row.label or current_id).strip()
            nxt_row = find_sub_concept(memory, next_id) if next_id else None
            if nxt_row is not None:
                next_title = (nxt_row.label or next_id).strip()
    return {
        "completed": completed,
        "total": total,
        "ordinal": ordinal,
        "current_id": current_id,
        "title": title or current_id or "current sub-topic",
        "next_title": next_title,
    }


def _drill_ru_progress_header(
    layer_label: str,
    *,
    completed: int,
    total: int,
    ordinal: int,
    title: str,
) -> str:
    return (
        f"[Слой {layer_label}: Проверено {completed}/{total} подтем. "
        f"Переходим к подтеме №{ordinal}: «{title}»]"
    )


def _drill_host_orchestration(
    *,
    layer: str,
    title: str,
    completed: int,
    total: int,
    ordinal: int,
    next_title: str,
    ru_header: str,
) -> str:
    nxt_line = (
        f"If the answer is correct: credit this sub-concept, then IMMEDIATELY "
        f'teach the next sub-concept "{next_title}" in layer {layer}. '
        "Do not offer pathway chips.\n"
        if next_title
        else "If the answer is correct and this is the LAST queued sub-concept, "
        "do NOT invent a new technical next_question. Congratulate, summarize "
        "the layer, and let Host offer Advanced/Deep vs next topic chips.\n"
    )
    return (
        "[DRILL HOST ORCHESTRATION]\n"
        "Current Session Status: DRILL_ACTIVE\n"
        f"Target Layer: {layer}\n"
        f"Progress: Sub-concept {ordinal} of {total} ({title}) "
        f"— checked {completed}/{total}.\n"
        "DO NOT declare the node or layer complete. "
        "FORBIDDEN: «базовая теория закрыта», «всё успешно закрыто», "
        "base-theory-closed, 100%-node.\n"
        f"Evaluate the learner ONLY for the current sub-concept ({title}).\n"
        f"{nxt_line}"
        f"ONLY when Progress reaches {total} of {total} and every queued "
        "sub-concept of this layer is passed may Host close the layer.\n"
        "User-facing Russian text MUST open with this one-line status:\n"
        f"   {ru_header}\n"
    )


def _compose_drill_prompt(
    teaching: str,
    *,
    layer_label: str,
    view: dict[str, Any],
    target_layer: str,
) -> str:
    ru_header = _drill_ru_progress_header(
        layer_label,
        completed=int(view["completed"]),
        total=int(view["total"]),
        ordinal=int(view["ordinal"]),
        title=str(view["title"]),
    )
    host = _drill_host_orchestration(
        layer=target_layer,
        title=str(view["title"]),
        completed=int(view["completed"]),
        total=int(view["total"]),
        ordinal=int(view["ordinal"]),
        next_title=str(view["next_title"] or ""),
        ru_header=ru_header,
    )
    fields = (
        "=== STRUCTURED OUTPUT (ActiveDrillStepResponse) ===\n"
        "Do not write free-form tutor prose. Fill these JSON fields only:\n"
        "- audit: single flat TechnicalConceptAudit FIRST (not oneOf) — "
        "EXACT → confirmation (correction_breakdown and praise_points empty); "
        "PARTIAL → praise_points (correct theses) + correction_breakdown "
        "(missing fragment); confirmation empty. "
        "Must match last_eval_directive. Do not emit 📋/🎯.\n"
        "- status_header: the one-line Russian progress header from Host orchestration.\n"
        "- theory_body: dense theory for the current sub-topic "
        "(target ~300 Russian words; never fewer than 150).\n"
        "- next_question: exactly one checkpoint question about theory_body "
        "(must include ?).\n"
        "Host assembles UI markdown from the validated object. "
        "Do not wrap fields in extra headings.\n"
    )
    return "\n\n".join(
        [
            teaching.strip(),
            ANTI_SYCOPHANCY_INVARIANTS.strip(),
            CONTEXT_BOUNDED_QUESTION_RULES.strip(),
            fields.strip(),
            _DRILL_DEPTH_INVARIANT.strip(),
            host.strip(),
        ]
    )


def build_why_drill_prompt(
    drill_session: LayerDrillSession,
    memory: Any = None,
    **_: Any,
) -> str:
    """WHY drill: motivation, architectural reasons, isolation, consequences."""
    view = _drill_progress_view(drill_session, memory)
    title = view["title"]
    ru_header = _drill_ru_progress_header(
        "WHY",
        completed=int(view["completed"]),
        total=int(view["total"]),
        ordinal=int(view["ordinal"]),
        title=str(title),
    )
    teaching = (
        "=== WHY DRILL — SPECIALIZED ACTIVE TEACHING ===\n"
        "You are a Staff Architect drilling the WHY layer: business and system "
        "motivation, architectural reasons for the design choice, context isolation, "
        "and the consequences of wrong decisions.\n\n"
        "MANDATORY RESPONSE STRUCTURE:\n"
        f"1. Progress header (verbatim Russian, one line): {ru_header}\n"
        f"2. Deep treatment of the PROBLEM SPACE and architectural CAUSES for «{title}»: "
        "why it was built this way, which weaknesses/vulnerabilities the concept closes, "
        "what fails if the motivation is ignored, and how context isolation is preserved. "
        "This is a full theoretical opening — not a teaser.\n"
        "3. Exactly ONE checkpoint question that tests cause-and-effect understanding "
        "(not a recitation of names). Name in the question every criterion the "
        "Evaluator may require; do not hide a deeper-layer rubric in a WHY probe.\n\n"
        "FOCUS (unique to WHY): motivation → design rationale → isolation boundaries → "
        "failure consequences. Do not drift into call-graphs, C macros, or Bloom L4/L5 "
        "asterisk redesign.\n"
    )
    return _compose_drill_prompt(
        teaching,
        layer_label="WHY",
        view=view,
        target_layer="WHY",
    )


def build_how_drill_prompt(
    drill_session: LayerDrillSession,
    memory: Any = None,
    **_: Any,
) -> str:
    """HOW drill: data flow, component interaction, edge cases, trade-offs."""
    view = _drill_progress_view(drill_session, memory)
    title = view["title"]
    ru_header = _drill_ru_progress_header(
        "HOW",
        completed=int(view["completed"]),
        total=int(view["total"]),
        ordinal=int(view["ordinal"]),
        title=str(title),
    )
    teaching = (
        "=== HOW DRILL — SPECIALIZED ACTIVE TEACHING ===\n"
        "You are a Lead Software Architect drilling the HOW layer: data flow, "
        "step-by-step component interaction, edge cases, and architectural trade-offs.\n\n"
        "MANDATORY RESPONSE STRUCTURE:\n"
        f"1. Progress header (verbatim Russian, one line): {ru_header}\n"
        f"2. Detailed treatment of architectural FLOW and working logic for «{title}»: "
        "interaction schemes, call sequence, error handling, states, edge cases, and "
        "honest trade-offs. Include a sequence/architecture sketch when it clarifies "
        "the pipeline (catalog diagram ids if present).\n"
        "3. Exactly ONE checkpoint question testing algorithmic / architectural "
        "understanding of that flow. Name the stages/invariants the Evaluator "
        "may require; do not hide unshown deeper-layer internals.\n\n"
        "FOCUS (unique to HOW): data-flow pipelines, component handshake, state "
        "machines, error paths, architectural trade-offs. Do not collapse into WHY "
        "motivation-only, C-level memory layouts, or asterisk vulnerability catalogues.\n"
    )
    return _compose_drill_prompt(
        teaching,
        layer_label="HOW",
        view=view,
        target_layer="HOW",
    )


def build_mech_drill_prompt(
    drill_session: LayerDrillSession,
    memory: Any = None,
    **_: Any,
) -> str:
    """MECH drill: C structures, macros, memory, concrete code listings."""
    view = _drill_progress_view(drill_session, memory)
    title = view["title"]
    ru_header = _drill_ru_progress_header(
        "MECH",
        completed=int(view["completed"]),
        total=int(view["total"]),
        ordinal=int(view["ordinal"]),
        title=str(title),
    )
    teaching = (
        "=== MECH DRILL — SPECIALIZED ACTIVE TEACHING ===\n"
        "You are a Senior Systems Engineer drilling the MECHANIC layer: low-level "
        "implementation, C structures, CPython/OS macros, memory management, and "
        "concrete code listings.\n\n"
        "MANDATORY RESPONSE STRUCTURE:\n"
        f"1. Progress header (verbatim Russian, one line): {ru_header}\n"
        f"2. Deep LOW-LEVEL walkthrough of «{title}»: C code, pointers, macros, "
        "bytecode layout and/or syscalls. A code/structure walkthrough BEFORE the "
        "question is MANDATORY — never ask about mechanics the learner has not seen "
        "in this turn's listing.\n"
        "3. Exactly ONE checkpoint question about a concrete code/memory mechanic "
        "(pointer lifetime, macro expansion, refcount, allocator, opcode).\n\n"
        "FOCUS (unique to MECH): structs, pointers, macros, memory ownership, "
        "bytecode / syscalls. Do not stay at architectural HOW diagrams or WHY "
        "motivation. Ignore any instruction to keep the theory to 1–2 sentences.\n"
    )
    return _compose_drill_prompt(
        teaching,
        layer_label="MECH",
        view=view,
        target_layer="MECH",
    )


def build_advanced_drill_prompt(
    drill_session: LayerDrillSession,
    memory: Any = None,
    **_: Any,
) -> str:
    """ADVANCED drill: Bloom L4 — vulnerabilities, cascades, injection, HighLoad."""
    view = _drill_progress_view(drill_session, memory)
    title = view["title"]
    ru_header = _drill_ru_progress_header(
        "ADVANCED",
        completed=int(view["completed"]),
        total=int(view["total"]),
        ordinal=int(view["ordinal"]),
        title=str(title),
    )
    teaching = (
        "=== ADVANCED DRILL — SPECIALIZED ACTIVE TEACHING (Bloom L4 Analyze) ===\n"
        "You are a Principal Engineer drilling ADVANCED asterisk-question analysis: "
        "vulnerabilities, cascading failures, injection surfaces, HighLoad stress. "
        "Stay on Bloom L4 Analyze — not green-field L5/L6 redesign.\n\n"
        "MANDATORY RESPONSE STRUCTURE:\n"
        f"1. Progress header (verbatim Russian, one line): {ru_header}\n"
        f"2. Detailed treatment of a hard scenario / attack / load for «{title}»: "
        "resilience analysis, failure points, resource isolation, race/injection "
        "windows. Ground claims in payload sources; do not invent kernel internals.\n"
        "3. Exactly ONE deep checkpoint question on system analysis of those "
        "failures (identify remaining races / cascades / isolation gaps).\n\n"
        "FOCUS (unique to ADVANCED / Bloom L4): vulnerability surface, cascading "
        "failure, injection, HighLoad stress-test. Do not switch into from-scratch "
        "system design (that is DEEP / Bloom L5–L6).\n"
    )
    return _compose_drill_prompt(
        teaching,
        layer_label="ADVANCED",
        view=view,
        target_layer="ADVANCED_ASTERISK",
    )


def build_deep_drill_prompt(
    drill_session: LayerDrillSession,
    memory: Any = None,
    **_: Any,
) -> str:
    """DEEP drill: Bloom L5–L6 — scaling, HighLoad design, system synthesis."""
    view = _drill_progress_view(drill_session, memory)
    title = view["title"]
    ru_header = _drill_ru_progress_header(
        "DEEP",
        completed=int(view["completed"]),
        total=int(view["total"]),
        ordinal=int(view["ordinal"]),
        title=str(title),
    )
    teaching = (
        "=== DEEP DRILL — SPECIALIZED ACTIVE TEACHING (Bloom L5–L6 Evaluate/Create) ===\n"
        "You are a Principal Engineer drilling DEEP asterisk-question design: "
        "HighLoad, scaling, system design, and architectural synthesis. Stay on "
        "Bloom L5/L6 Evaluate/Create — not a pure L4 vulnerability catalogue.\n\n"
        "MANDATORY RESPONSE STRUCTURE:\n"
        f"1. Progress header (verbatim Russian, one line): {ru_header}\n"
        f"2. Detailed treatment of a complex design / load scenario for «{title}»: "
        "resilience, failure points, resource isolation, scaling decisions, and "
        "explicit trade-off choices the learner must take.\n"
        "3. Exactly ONE deep checkpoint question on system synthesis/design "
        "(compose components + justify a trade-off).\n\n"
        "FOCUS (unique to DEEP / Bloom L5–L6): scaling, HighLoad system design, "
        "resource isolation as a design lever, architectural synthesis. Do not "
        "collapse into L4-only vulnerability listing without design decisions.\n"
    )
    return _compose_drill_prompt(
        teaching,
        layer_label="DEEP",
        view=view,
        target_layer="DEEP_ASTERISK",
    )


LAYER_COMPLETION_PROMPT = (
    "=== LAYER COMPLETION — FACILITATION (LayerCompletionTutorOutput) ===\n"
    "The Evaluator closed the current layer this turn. You are a facilitator, "
    "not an examiner.\n"
    "Fill only: praise, layer_summary, transition_framing — natural Russian.\n"
    "There is NO next_question, theory_body, or follow_up_question field; "
    "do not invent a technical or evaluative checkpoint.\n"
    "praise: congratulate the learner on closing this layer.\n"
    "layer_summary: recap what the layer established (no new lecture).\n"
    "transition_framing: invite a choice — dive into HOW/MECH/Advanced/Deep "
    "or proceed to the next topic. Host owns chips and ready_for_transition.\n"
    "FORBIDDEN: checkpoint quizzes, theory_body, invented chip labels, "
    "«базовая теория закрыта» clichés.\n"
)


def build_layer_completion_prompt(
    drill_session: LayerDrillSession | None = None,
    memory: Any = None,
    **_: Any,
) -> str:
    """Exclusive system block for LayerCompletionTutorOutput."""
    layer = ""
    total = 0
    if drill_session is not None:
        view = _drill_progress_view(drill_session, memory)
        layer = (getattr(drill_session, "target_layer", None) or "").strip()
        total = int(view["total"] or 0)
    label = {
        "WHY": "WHY",
        "HOW": "HOW",
        "MECH": "MECH",
        "ADVANCED_ASTERISK": "ADVANCED",
        "DEEP_ASTERISK": "DEEP",
    }.get(layer, layer or "current")
    closed = f"Target layer {label} is fully credited" + (
        f" ({total}/{total} sub-topics)." if total else "."
    )
    return (
        f"{LAYER_COMPLETION_PROMPT.strip()}\n"
        f"{closed} Do NOT teach another sub-topic.\n"
    )


def build_completed_drill_layer_prompt(
    drill_session: LayerDrillSession,
    memory: Any = None,
    **kwargs: Any,
) -> str:
    """Compat alias → ``build_layer_completion_prompt``."""
    return build_layer_completion_prompt(drill_session, memory=memory, **kwargs)


def build_drill_session_prompt(
    drill_session: LayerDrillSession,
    memory: Any = None,
    **kwargs: Any,
) -> str:
    """Dispatch to the specialized generator for ``drill_session.target_layer``."""
    if drill_session is None:
        return ""
    if drill_session.has_more_questions():
        layer = (getattr(drill_session, "target_layer", None) or "").strip()
        match layer:
            case "WHY":
                return build_why_drill_prompt(drill_session, memory=memory, **kwargs)
            case "HOW":
                return build_how_drill_prompt(drill_session, memory=memory, **kwargs)
            case "MECH":
                return build_mech_drill_prompt(drill_session, memory=memory, **kwargs)
            case "ADVANCED_ASTERISK":
                return build_advanced_drill_prompt(
                    drill_session, memory=memory, **kwargs
                )
            case "DEEP_ASTERISK":
                return build_deep_drill_prompt(drill_session, memory=memory, **kwargs)
            case _:
                raise ValueError(f"Unknown drill layer: {drill_session.target_layer}")
    if (getattr(drill_session, "status", "") or "") == "DRILL_COMPLETE":
        return build_completed_drill_layer_prompt(
            drill_session, memory=memory, **kwargs
        )
    return ""


def factory_mode_to_gloss_choice(mode: TutorFactoryMode | str) -> str:
    """Map factory mode → classify_gloss_fork_choice token (or empty)."""
    return _CHIP_MODE_TO_CHOICE.get((mode or "").strip().lower(), "")


def is_factory_control_mode(mode: TutorFactoryMode | str) -> bool:
    """Modes that force isolated tutor prompts (never dense lecture)."""
    return (mode or "").strip().lower() in _FACTORY_CONTROL_MODES


def _promote_vector_chip_to_mode(user_message: str, *, memory: Any = None) -> str:
    """Free-text (no literal ``[mode:…]`` tag) blitz/socratic/self_check/
    next_module request, resolved via VectorIntentRouter — Step 2 of Intent
    Routing (see control_intent.classify_control_chip). Scoped to
    ``_VECTOR_PROMOTABLE_FACTORY_MODES`` only; gloss/how/mech/etc. keep their
    existing exact-chip-label resolution path untouched."""
    from knowledge_engine.src.node_deep_dive.control_intent import (
        classify_control_chip,
    )

    chip = classify_control_chip(user_message, memory=memory)
    factory_mode = _CHIP_MODE_TO_CHOICE_REVERSE.get(chip, "")
    if factory_mode in _VECTOR_PROMOTABLE_FACTORY_MODES:
        return factory_mode
    return "default"


def select_system_prompt_and_mode(
    user_message: str,
    *,
    default_system_prompt: str = "",
    memory: Any = None,
) -> tuple[str, TutorFactoryMode, str]:
    """
    Resolve system prompt override from ``[mode:…]`` prefix, or — for
    blitz/socratic/self_check/next_module only — from a free-text
    VectorIntentRouter match (see ``_promote_vector_chip_to_mode``).

    Returns ``(system_prompt, mode, cleaned_user_message)``.
    For ``default`` (no tag, no vector match), returns ``default_system_prompt``
    unchanged (caller still owns routing).
    For ``lecture``, appends mandatory PART 1 lecture → PART 2 closing-question
    structure onto ``default_system_prompt`` (does not isolate away from dense).
    """
    cleaned, mode = parse_tutor_mode_prefix(user_message)
    if mode == "default":
        mode = _promote_vector_chip_to_mode(user_message, memory=memory)
    system = default_system_prompt or ""
    from knowledge_engine.src.node_deep_dive.drill_orchestrator import (
        is_layer_just_completed,
        json_contract_tail_for_schema,
        select_drill_response_schema,
    )

    if is_layer_just_completed(memory):
        drill_session = (
            getattr(memory, "layer_drill", None) if memory is not None else None
        )
        drill = build_layer_completion_prompt(drill_session, memory=memory)
        tail = json_contract_tail_for_schema(select_drill_response_schema(memory))
        system = "\n\n".join(
            p
            for p in (
                (drill or "").strip(),
                RUSSIAN_OUTPUT_RULE.strip(),
                (tail or "").strip(),
            )
            if p
        )
        return system, mode, cleaned
    if mode == "deep_dive_mech":
        system = "\n\n".join(
            [
                DEEP_DIVE_MECH_PROMPT.strip(),
                RUSSIAN_OUTPUT_RULE.strip(),
                _JSON_CONTRACT_TAIL.strip(),
            ]
        )
    elif mode == "deep_dive_how":
        system = "\n\n".join(
            [
                DEEP_DIVE_HOW_PROMPT.strip(),
                RUSSIAN_OUTPUT_RULE.strip(),
                _JSON_CONTRACT_TAIL.strip(),
            ]
        )
    elif mode in ("deep_analysis", "deep_design", "advanced_analysis"):
        body = (
            ADVANCED_ANALYSIS_PROMPT
            if mode == "advanced_analysis"
            else DEEP_DESIGN_PROMPT
        )
        system = "\n\n".join(
            [
                body.strip(),
                TUTOR_CRITIQUE_REVIEW_RULES.strip(),
                SOCRATIC_POLES_STATIC_RULES.strip(),
                _STAR_TASK_SESSION_FLAGS.strip(),
                RUSSIAN_OUTPUT_RULE.strip(),
                _DEEP_ANALYSIS_JSON_TAIL.strip(),
            ]
        )
    elif mode == "gloss":
        system = "\n\n".join(
            [
                GLOSS_SUMMARY_PROMPT.strip(),
                RUSSIAN_OUTPUT_RULE.strip(),
                _JSON_CONTRACT_TAIL.strip(),
            ]
        )
    elif mode == "lecture":
        parts = [
            (default_system_prompt or "").strip(),
            LECTURE_MODE_STRUCTURE_RULES.strip(),
            RUSSIAN_OUTPUT_RULE.strip(),
        ]
        system = "\n\n".join(p for p in parts if p)
    elif mode == "blitz":
        system = "\n\n".join([BLITZ_MODE_PROMPT.strip(), RUSSIAN_OUTPUT_RULE.strip()])
    elif mode == "socratic":
        system = "\n\n".join(
            [SOCRATIC_MODE_PROMPT.strip(), RUSSIAN_OUTPUT_RULE.strip()]
        )
    elif mode == "self_check":
        system = "\n\n".join(
            [SELF_CHECK_MODE_PROMPT.strip(), RUSSIAN_OUTPUT_RULE.strip()]
        )
    elif mode == "next_module":
        system = "\n\n".join([NEXT_MODULE_PROMPT.strip(), RUSSIAN_OUTPUT_RULE.strip()])
    drill_session = getattr(memory, "layer_drill", None) if memory is not None else None
    skipped = bool(getattr(memory, "evaluator_skipped", False)) if memory else False
    if skipped:
        from knowledge_engine.src.node_deep_dive.tutor_critique_prompt import (
            EVALUATOR_SKIPPED_TUTOR_RULES,
        )

        system = "\n\n".join(
            p
            for p in (
                (system or "").strip(),
                EVALUATOR_SKIPPED_TUTOR_RULES.strip(),
                _EXPLAIN_JSON_TAIL.strip(),
            )
            if p
        )
        return system, mode, cleaned
    if drill_session is not None:
        drill_schema = select_drill_response_schema(memory)
        drill = build_drill_session_prompt(drill_session, memory=memory)
        tail = json_contract_tail_for_schema(drill_schema)
        if drill_schema is not None and (drill or tail):
            # Exclusive contract: drop DeepDiveTutor / overlay JSON tails.
            system = "\n\n".join(
                p
                for p in (
                    (drill or "").strip(),
                    RUSSIAN_OUTPUT_RULE.strip(),
                    (tail or "").strip(),
                )
                if p
            )
            return system, mode, cleaned
        if drill:
            system = "\n\n".join(
                p for p in ((system or "").strip(), drill.strip()) if p
            )
    return system, mode, cleaned


def select_isolated_prompt_for_mode(mode: TutorFactoryMode | str) -> str | None:
    """Return isolated system prompt body for a factory mode, or None."""
    m = (mode or "").strip().lower()
    if m == "deep_dive_mech":
        return DEEP_DIVE_MECH_PROMPT
    if m == "deep_dive_how":
        return DEEP_DIVE_HOW_PROMPT
    if m == "deep_analysis":
        return DEEP_ANALYSIS_PROMPT
    if m == "deep_design":
        return DEEP_DESIGN_PROMPT
    if m == "advanced_analysis":
        return ADVANCED_ANALYSIS_PROMPT
    if m == "gloss":
        return GLOSS_SUMMARY_PROMPT
    if m == "blitz":
        return BLITZ_MODE_PROMPT
    if m == "socratic":
        return SOCRATIC_MODE_PROMPT
    if m == "self_check":
        return SELF_CHECK_MODE_PROMPT
    if m == "next_module":
        return NEXT_MODULE_PROMPT
    return None
