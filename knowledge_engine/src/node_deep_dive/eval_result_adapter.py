"""Adapter: EvaluatorCritiqueContract → tutor feedback / Phase-2 critique JSON."""

from __future__ import annotations

import json
from typing import Any

from knowledge_engine.schemas.llm_contracts.tutor import (
    SubConceptGapEvalContract,
    SubConceptStatusUpdate,
)
from knowledge_engine.schemas.llm_contracts.evaluator_critique import (
    EvaluatedIdea,
    EvaluatorCritiqueContract,
    IdeaStatus,
    OVERLAY_LAYERS,
)


def normalize_overlay_target_layer(raw: str | None) -> str:
    """Map pending_eval_kind / free text → ADVANCED | DEEP (default DEEP)."""
    val = (raw or "").strip().upper()
    if val in OVERLAY_LAYERS:
        return val
    low = (raw or "").strip().lower()
    if low in (
        "advanced_analysis",
        "advanced",
        "adv",
        "l4",
        "advanced_asterisk",
    ) or val in ("ADV", "ADVANCED_ANALYSIS", "L4", "ADVANCED_ASTERISK"):
        return "ADVANCED"
    if val in (
        "DEEP_ANALYSIS",
        "DEEP_DESIGN",
        "STAR",
        "STAR_TASK",
        "ASTERISK_QUESTION",
        "ASTERISK",
        "DESIGN",
        "L5",
        "L6",
        "DEEP_ASTERISK",
    ) or low in ("deep_analysis", "deep_design", "deep", "design", "l5", "l6"):
        return "DEEP"
    return "DEEP"


def critique_to_feedback_text(critique: EvaluatorCritiqueContract) -> str:
    """
    Build tutor-facing transparency text from structured critique.

    Grouped by STRONG / RISK / WEAK + unaccounted edges + verdict.
    """
    lines: list[str] = [
        "[EVALUATOR_CRITIQUE]",
        f"target_layer: {critique.target_layer}",
        f"passes_threshold: {critique.passes_threshold}",
        f"bloom_level_matched: {critique.bloom_level_matched}",
        f"verdict_reason: {(critique.verdict_reason or '').strip()[:800]}",
    ]
    by_status: dict[IdeaStatus, list[EvaluatedIdea]] = {
        IdeaStatus.STRONG: [],
        IdeaStatus.RISK: [],
        IdeaStatus.WEAK: [],
    }
    for idea in critique.analyzed_ideas or []:
        try:
            st = idea.status if isinstance(idea.status, IdeaStatus) else IdeaStatus(idea.status)
        except Exception:
            st = IdeaStatus.WEAK
        by_status.setdefault(st, []).append(idea)

    for status in (IdeaStatus.STRONG, IdeaStatus.RISK, IdeaStatus.WEAK):
        items = by_status.get(status) or []
        if not items:
            continue
        lines.append(f"=== {status.value} ===")
        for idea in items:
            concept = (idea.idea_concept or "").strip()[:200]
            note = (idea.technical_note or "").strip()[:400]
            lines.append(f"- {concept}: {note}")

    edges = [e.strip() for e in (critique.unaccounted_edge_cases or []) if (e or "").strip()]
    if edges:
        lines.append("=== UNACCOUNTED_EDGE_CASES ===")
        for edge in edges[:12]:
            lines.append(f"- {edge[:300]}")

    lines.append(
        "TRANSPARENCY: use STRONG/RISK/WEAK notes and UNACCOUNTED_EDGE_CASES "
        "when writing feedback_on_answer; do not invent new user ideas."
    )
    return "\n".join(lines)


def critique_to_legacy_gap_contract(
    critique: EvaluatorCritiqueContract,
    *,
    concept_id: str,
) -> SubConceptGapEvalContract:
    """
    Soft-compat projection for tooling that still expects gap updates.

    Overlay critiques MUST NOT credit core HOW/MECH via this projection:
    all layer booleans stay False; evidence/focus_hint carry the summary only.
    """
    cid = (concept_id or "").strip() or "unknown"
    focus_parts = [e.strip() for e in (critique.unaccounted_edge_cases or []) if e.strip()]
    focus = "; ".join(focus_parts)[:500] if focus_parts else (
        (critique.verdict_reason or "").strip()[:500]
    )
    evidence = critique_to_feedback_text(critique)[:2000]
    update = SubConceptStatusUpdate(
        id=cid,
        why_passed=False,
        how_passed=False,
        mechanic_passed=False,
        accuracy_grade="NEEDS_CORRECTION",
        detected_errors_or_misconceptions=list(focus_parts[:16]),
        correct_claims=[],
        evidence=evidence,
        focus_hint=focus,
        status=None,
    )
    return SubConceptGapEvalContract(updates=[update])


def critique_to_memory_payload(critique: EvaluatorCritiqueContract) -> dict[str, Any]:
    """JSON-serializable snapshot for SessionMemory / Phase 2."""
    return critique.model_dump(mode="json")


def format_evaluator_critique_for_tutor(
    critique: dict[str, Any] | EvaluatorCritiqueContract | None,
) -> str:
    """
    Full structured critique for the Tutor system/movable payload.

    When present, Tutor MUST pointwise-review analyzed_ideas and
    unaccounted_edge_cases (see TUTOR_CRITIQUE_REVIEW_RULES).
    """
    if critique is None:
        return ""
    if isinstance(critique, EvaluatorCritiqueContract):
        payload: dict[str, Any] = critique.model_dump(mode="json")
    elif isinstance(critique, dict) and critique:
        payload = dict(critique)
    else:
        return ""
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    return (
        "[EVALUATOR_CRITIQUE_JSON]\n"
        "# technical_note / verdict_reason: English tutor-internal. "
        "Render to the learner in Russian.\n"
        f"{body}\n"
        "[/EVALUATOR_CRITIQUE_JSON]"
    )


def should_inject_evaluator_critique(memory: object | None) -> bool:
    """Overlay turns only — do not surface stale critique JSON on core WHY/HOW/MECH."""
    if memory is None:
        return False
    critique = getattr(memory, "last_evaluator_critique", None)
    if not isinstance(critique, dict) or not critique:
        return False
    directive = str(getattr(memory, "last_eval_directive", "") or "").strip()
    if directive in ("DEEP_MASTERY_EARNED", "STAR_TASK_NEEDS_REFINEMENT"):
        return True
    kind = str(getattr(memory, "pending_eval_kind", "") or "").strip().lower()
    from knowledge_engine.src.node_deep_dive.star_task_fsm import is_overlay_eval_kind

    return is_overlay_eval_kind(kind)
