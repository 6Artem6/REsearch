"""Control-chip / mode-tag routing for tutor turns.

Fast path: explicit client tags ``[mode|action|intent|begin:…]`` and exact
registered chip labels (0 ms). Soft path: semantic cosine match via
``VectorIntentRouter`` (BGE-M3 ``EMBED_MODEL``) — never substring stems.
"""

from __future__ import annotations

import re
from typing import Literal

from knowledge_engine.src.node_deep_dive.intent_definitions import (
    ACTION_ALIASES,
    EVALUATOR_SKIP_INTENTS,
    FACTORY_MODE_TO_INTENT,
    INTENT_NAMES,
    MODE_SELECTION_SLOT,
    MODE_SELECTION_SLOT_INTENTS,
    MODE_SELECTION_VECTOR_THRESHOLD,
    REGISTERED_CONTROL_CHIPS,
)
from knowledge_engine.ui.run_log import trace

ControlChip = Literal[
    "gloss",
    "how",
    "mech",
    "deep_analysis",
    "advanced_analysis",
    "deep_design",
    "next",
    "lecture",
    "begin",
    "skip",
    "accept_deep",
    "practice",
    "check",
    "",
]

# Soft ceiling still used by non-router heuristics (step_analysis finalize/shift).
# Not used as the sole chip/free-text separator anymore.
CONTROL_CHIP_MAX_LEN = 80

_EXPLICIT_TAG_RE = re.compile(
    r"^\[(?:mode|action|intent|begin)(?::[^\]]+)?\]",
    re.I,
)

_KNOWN_INTENTS = frozenset(INTENT_NAMES)


def has_explicit_control_tag(user_text: str) -> bool:
    """True when message starts with a client/system control tag."""
    raw = (user_text or "").strip()
    if not raw:
        return False
    if raw.lower().startswith("[begin]"):
        return True
    return bool(_EXPLICIT_TAG_RE.match(raw))


def is_too_long_for_control_chip(user_text: str) -> bool:
    """Legacy soft length helper for step_analysis heuristics (not chip routing)."""
    return len((user_text or "").strip()) > CONTROL_CHIP_MAX_LEN


def _trace_control(chip: str, text: str, reason: str) -> None:
    body = (text or "").strip()
    trace(
        f"[INTENT_ROUTER] Triggered control chip '{chip}' "
        f"(length={len(body)}; reason={reason})"
    )


def _classify_explicit_and_exact(user_text: str) -> ControlChip | str:
    """Fast path: tags + exact registered labels. Empty → fall through to vector."""
    raw = (user_text or "").strip()
    if not raw:
        return ""

    from knowledge_engine.src.node_deep_dive.prompt_factory import (
        factory_mode_to_gloss_choice,
        parse_tutor_mode_prefix,
    )

    if raw.lower().startswith("[begin]"):
        _trace_control("begin", raw, "explicit_tag")
        return "begin"

    cleaned, factory_mode = parse_tutor_mode_prefix(raw)
    mapped = factory_mode_to_gloss_choice(factory_mode) or FACTORY_MODE_TO_INTENT.get(
        (factory_mode or "").strip().lower(), ""
    )
    if mapped:
        _trace_control(mapped, raw, f"explicit_mode={factory_mode}")
        return mapped  # type: ignore[return-value]

    m = re.match(r"^\[(action|intent):([^\]]+)\]\s*(.*)$", raw, re.I)
    if m:
        token = (m.group(2) or "").strip().lower().replace("-", "_")
        alias = ACTION_ALIASES.get(token, "")
        if alias:
            _trace_control(alias, raw, f"explicit_{m.group(1).lower()}={token}")
            return alias

    label = (cleaned or raw).strip()
    if label in REGISTERED_CONTROL_CHIPS:
        chip = REGISTERED_CONTROL_CHIPS[label]
        _trace_control(chip, raw, "exact_chip_label")
        return chip
    if raw in REGISTERED_CONTROL_CHIPS:
        chip = REGISTERED_CONTROL_CHIPS[raw]
        _trace_control(chip, raw, "exact_chip_label")
        return chip
    # Case-insensitive exact for short stubs
    low_map = {k.lower(): v for k, v in REGISTERED_CONTROL_CHIPS.items()}
    if label.lower() in low_map:
        chip = low_map[label.lower()]
        _trace_control(chip, raw, "exact_chip_label_ci")
        return chip
    return ""


def _classify_vector(
    user_text: str,
    *,
    allowed_intents: frozenset[str] | None = None,
    threshold: float | None = None,
) -> ControlChip | str:
    from knowledge_engine.src.node_deep_dive.vector_intent_router import (
        get_vector_intent_router,
    )

    router = get_vector_intent_router()
    intent, score = router.classify(
        user_text,
        allowed_intents=allowed_intents,
        threshold=threshold,
    )
    if not intent:
        return ""
    if intent in _KNOWN_INTENTS:
        reason = (
            "fallback_rules"
            if getattr(router, "_degraded", False)
            else f"vector_score={score:.3f}"
        )
        if allowed_intents:
            reason = f"slot_{reason}"
        _trace_control(intent, user_text, reason)
        return intent  # type: ignore[return-value]
    return ""


def is_mode_selection_slot_active(memory: object | None) -> bool:
    """True, если FSM ждёт чип практика / проверка / пропустить."""
    if memory is None:
        return False
    return (getattr(memory, "pending_control_slot", None) or "").strip() == (
        MODE_SELECTION_SLOT
    )


def mark_awaiting_mode_selection(memory: object | None) -> None:
    """Открыть слот выбора ветки после fast-track (без pending_evaluation)."""
    if memory is None:
        return
    memory.pending_control_slot = MODE_SELECTION_SLOT  # type: ignore[attr-defined]
    memory.intro_question_pending = True  # type: ignore[attr-defined]
    memory.pending_evaluation_concept_id = ""  # type: ignore[attr-defined]


def clear_pending_control_slot(memory: object | None) -> None:
    """Закрыть FSM-слот после распознанной ветки или skip."""
    if memory is None:
        return
    if (getattr(memory, "pending_control_slot", None) or "").strip():
        memory.pending_control_slot = ""  # type: ignore[attr-defined]
    if getattr(memory, "intro_question_pending", False):
        memory.intro_question_pending = False  # type: ignore[attr-defined]


def apply_mode_selection_intent(memory: object | None, chip: str) -> bool:
    """
    Применить интент ветки: practice → HOW drill, check → express_blitz.

    skip закрывает слот; сам переход equivalence делает граф.
    """
    if chip not in MODE_SELECTION_SLOT_INTENTS:
        return False
    from knowledge_engine.src.node_deep_dive.learning_loop import set_learning_mode
    from knowledge_engine.src.node_deep_dive.star_task_fsm import start_layer_drill

    clear_pending_control_slot(memory)
    if memory is None:
        return True
    if chip == "practice":
        start_layer_drill(memory, "HOW")  # type: ignore[arg-type]
        set_learning_mode(memory, "socratic_point")  # type: ignore[arg-type]
        _trace_control("practice", "", "slot_apply=HOW_drill")
    elif chip == "check":
        set_learning_mode(memory, "express_blitz")  # type: ignore[arg-type]
        _trace_control("check", "", "slot_apply=express_blitz")
    return True


def classify_exact_control_chip(user_text: str) -> ControlChip | str:
    """Exact tags + registered labels only (no vector / no LanceDB)."""
    return _classify_explicit_and_exact(user_text)


def classify_control_chip_detailed(
    user_text: str,
    *,
    memory: object | None = None,
) -> tuple[str, str]:
    """Вернуть ``(intent, source)``: exact | vector | fallback."""
    raw = (user_text or "").strip()
    if not raw:
        return "", "fallback"
    hit = _classify_explicit_and_exact(raw)
    if hit:
        return hit, "exact"
    slot_active = is_mode_selection_slot_active(memory)
    try:
        if slot_active:
            vec = _classify_vector(
                raw,
                allowed_intents=MODE_SELECTION_SLOT_INTENTS,
                threshold=MODE_SELECTION_VECTOR_THRESHOLD,
            )
            if vec:
                return vec, "vector"
            return "", "fallback"
        vec = _classify_vector(raw)
        if vec:
            from knowledge_engine.src.node_deep_dive.vector_intent_router import (
                get_vector_intent_router,
            )

            src = (
                "fallback"
                if getattr(get_vector_intent_router(), "_degraded", False)
                else "vector"
            )
            return vec, src
        return "", "fallback"
    except Exception:
        from knowledge_engine.src.resilience_manager import classify_intent_from_rules

        fb = classify_intent_from_rules(raw)
        if fb:
            _trace_control(fb, raw, "fallback_rules")
            return fb, "fallback"
        return "", "fallback"


def classify_control_chip(
    user_text: str,
    *,
    memory: object | None = None,
) -> ControlChip | str:
    """
    Classify a user message as a control chip / lecture stub / empty.

    Priority:
      1. Explicit ``[mode|action|intent|begin:…]``
      2. Exact registered chip label (включая слот практика/проверка/пропустить)
      3. Semantic VectorIntentRouter (cosine ≥ threshold);
         при активном ``mode_selection`` — только интенты слота.
    """
    intent, _source = classify_control_chip_detailed(user_text, memory=memory)
    return intent  # type: ignore[return-value]


def is_control_chip_message(
    user_text: str,
    *,
    memory: object | None = None,
) -> bool:
    """True for UI Quick Reply chips that must not be scored by the gap evaluator."""
    chip = classify_control_chip(user_text, memory=memory)
    return chip in EVALUATOR_SKIP_INTENTS


def is_short_accept_deep_dive(user_text: str) -> bool:
    """User accepted optional deep-dive soft pitch (semantic or exact short yes)."""
    raw = (user_text or "").strip()
    if not raw:
        return False
    if has_explicit_control_tag(raw):
        chip = _classify_explicit_and_exact(raw)
        return chip in ("how", "mech", "accept_deep")
    exact_yes = {
        "да",
        "давай",
        "да, давай",
        "углубиться",
        "deep dive",
        "погрузимся",
    }
    if raw.lower() in exact_yes:
        return True
    return classify_control_chip(raw) == "accept_deep"


def is_short_begin_message(user_text: str) -> bool:
    raw = (user_text or "").strip()
    if not raw:
        return False
    if raw.lower().startswith("[begin]"):
        return True
    return classify_control_chip(raw) == "begin"


def is_short_skip_node_message(user_text: str) -> bool:
    return classify_control_chip(user_text) == "skip"


def is_short_lecture_request(user_text: str) -> bool:
    """Dense-lecture ask: explicit ``[mode:lecture]`` or semantic/exact lecture stub."""
    raw = (user_text or "").strip()
    if not raw:
        return False
    if raw.lower().startswith("[mode:lecture]"):
        _trace_control("lecture", raw, "explicit_mode=lecture")
        return True
    return classify_control_chip(raw) == "lecture"
