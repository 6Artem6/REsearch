"""Graceful degradation for intent routing, evaluator LLM faults, and FSM hops."""

from __future__ import annotations

from knowledge_engine.src.node_deep_dive.intent_definitions import INTENT_RULES
from knowledge_engine.ui.run_log import trace

MAX_FSM_HOPS_PER_TURN = 5


def classify_intent_from_rules(user_text: str) -> str:
    """
    Degraded fallback: exact chip labels + explicit ``[mode:]`` prefixes only.

    No substring / cue scans over raw text (vector catalog is the soft path).
    Order follows INTENT_RULES so overlay L4/L5 precede generic deep_analysis.
    """
    raw = (user_text or "").strip()
    if not raw:
        return ""
    low = raw.lower()
    for rule in INTENT_RULES:
        for lab in rule.exact_labels:
            if raw == lab or low == lab.lower():
                return rule.intent
        mode = (rule.system_mode or "").strip().lower()
        if mode and low.startswith(mode):
            return rule.intent
    return ""


def degraded_student_message() -> str:
    """Valid student-facing reply when Gemini/LLM 429 / 5xx / timeout."""
    return (
        "Внешняя модель временно недоступна (лимит или таймаут). "
        "Контекст занятия сохранён — повторите ответ, когда будете готовы."
    )


def is_tutor_contract_validation_error(exc: BaseException) -> bool:
    """True when Gemini JSON failed the tutor/drill Pydantic contract.

    The learner turn must still persist (degraded reply) instead of dropping
    the user message from history.
    """
    low = str(exc).lower()
    if "gemini json" in low and "валидац" in low:
        return True
    if "validation error" not in low and "validationerror" not in type(exc).__name__.lower():
        return False
    markers = (
        "theory_body",
        "activedrillstepresponse",
        "completeddrilllayerresponse",
        "deepdivetutorcontract",
        "deepdiveexplaincontract",
        "deepdivedeepanalysiscontract",
    )
    return any(m in low for m in markers)


def is_llm_resilience_error(exc: BaseException) -> bool:
    """True for 429 / 5xx / timeout / quota — host must not 500."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "timeout" in name or "timed out" in msg or "timeout" in msg:
        return True
    if "quota" in name or "quota" in msg or "resourceexhausted" in msg:
        return True
    if "429" in msg or ("rate" in msg and "limit" in msg):
        return True
    if any(code in msg for code in (" 500", " 502", " 503", " 504", "status=500")):
        return True
    if "unavailable" in name or "unavailable" in msg:
        return True
    http = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    try:
        code = int(http)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        code = 0
    return code in {408, 429, 500, 502, 503, 504}


def reset_asterisk_fsm_hops(memory: object) -> None:
    if memory is None:
        return
    try:
        memory.asterisk_fsm_hops = 0  # type: ignore[attr-defined]
    except Exception:
        pass


def note_asterisk_fsm_hop(memory: object) -> bool:
    """
    Count one asterisk-question FSM transition.

    Returns False when the per-turn cap is exhausted (state must freeze).
    """
    if memory is None:
        return False
    n = int(getattr(memory, "asterisk_fsm_hops", 0) or 0)
    if n >= MAX_FSM_HOPS_PER_TURN:
        trace(
            f"NODE_DIVE asterisk_question FSM hop cap | hops={n} "
            f"max={MAX_FSM_HOPS_PER_TURN}"
        )
        return False
    try:
        memory.asterisk_fsm_hops = n + 1  # type: ignore[attr-defined]
    except Exception:
        return False
    return True


def core_ready_for_overlay(memory: object) -> bool:
    """Overlay chips only after 100% core mastery (all core sub-concepts verified)."""
    if memory is None:
        return False
    score = int(getattr(memory, "topic_mastery_score", 0) or 0)
    if score < 100:
        return False
    from knowledge_engine.src.node_deep_dive.concept_map_state import (
        sub_concept_coverage_complete,
    )

    return bool(sub_concept_coverage_complete(memory))  # type: ignore[arg-type]
