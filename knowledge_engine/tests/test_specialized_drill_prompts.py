"""Specialized Layer Drill Session prompts — one generator per layer."""

from __future__ import annotations

import pytest

from knowledge_engine.src.node_deep_dive.memory_schemas import (
    LayerDrillSession,
    SessionMemory,
    SubConceptRecord,
)
from knowledge_engine.src.node_deep_dive.prompt_factory import (
    build_advanced_drill_prompt,
    build_deep_drill_prompt,
    build_drill_session_prompt,
    build_how_drill_prompt,
    build_mech_drill_prompt,
    build_why_drill_prompt,
    select_system_prompt_and_mode,
)


def _session(layer: str, *, index: int = 0) -> LayerDrillSession:
    return LayerDrillSession(
        is_active=True,
        target_layer=layer,  # type: ignore[arg-type]
        target_sub_concept_ids=["pyobject", "reference_count", "type_pointer"],
        current_index=index,
        status="DRILL_IN_PROGRESS",
    )


def _memory(session: LayerDrillSession) -> SessionMemory:
    return SessionMemory(
        layer_drill=session,
        sub_concepts=[
            SubConceptRecord(id="pyobject", label="PyObject header"),
            SubConceptRecord(id="reference_count", label="Reference count"),
            SubConceptRecord(id="type_pointer", label="Type pointer"),
        ],
    )


def _all_specialized(session: LayerDrillSession, memory: SessionMemory) -> dict[str, str]:
    return {
        "WHY": build_why_drill_prompt(session, memory=memory),
        "HOW": build_how_drill_prompt(session, memory=memory),
        "MECH": build_mech_drill_prompt(session, memory=memory),
        "ADVANCED": build_advanced_drill_prompt(session, memory=memory),
        "DEEP": build_deep_drill_prompt(session, memory=memory),
    }


def test_each_layer_has_distinct_prompt_templates() -> None:
    session = _session("HOW")
    memory = _memory(session)
    prompts = _all_specialized(session, memory)
    bodies = list(prompts.values())
    assert len(set(bodies)) == 5

    why, how, mech, adv, deep = (
        prompts["WHY"],
        prompts["HOW"],
        prompts["MECH"],
        prompts["ADVANCED"],
        prompts["DEEP"],
    )
    assert "business and system motivation" in why
    assert "architectural reasons for the design choice" in why
    assert "cause-and-effect" in why
    assert "data flow" in how
    assert "step-by-step component interaction" in how
    assert "architectural trade-offs" in how
    assert "C structures" in mech
    assert "CPython/OS macros" in mech
    assert "memory management" in mech
    assert "Bloom L4" in adv
    assert "vulnerabilities" in adv
    assert "cascading failures" in adv
    assert "Bloom L5–L6" in deep or "Bloom L5/L6" in deep
    assert "system design" in deep
    assert "scaling" in deep

    assert "data flow" not in why
    assert "C structures" not in how
    assert "Bloom L4" not in mech
    assert "from-scratch system design" in adv or "green-field" in adv
    assert "vulnerability catalogue" in deep or "L4-only" in deep


def test_all_drill_prompts_enforce_depth_invariant() -> None:
    session = _session("HOW")
    memory = _memory(session)
    for name, text in _all_specialized(session, memory).items():
        low = text.lower()
        assert "300" in text, name
        assert "150" in text, name
        assert "300 russian words" in low or "target ~300" in low, name
        assert "1–2 dry introductory sentences" in text or "1-2 dry" in low, name
        assert "normal lecture" in low, name
        assert "theory_body" in text, name
        assert "next_question" in text, name
        assert "CONTEXT-BOUNDED QUESTION FACTORY" in text, name


def test_dispatcher_routes_to_specialized_generators() -> None:
    memory = _memory(_session("HOW"))
    how = build_drill_session_prompt(_session("HOW"), memory=memory)
    why = build_drill_session_prompt(_session("WHY"), memory=memory)
    mech = build_drill_session_prompt(_session("MECH"), memory=memory)
    adv = build_drill_session_prompt(_session("ADVANCED_ASTERISK"), memory=memory)
    deep = build_drill_session_prompt(_session("DEEP_ASTERISK"), memory=memory)
    assert how == build_how_drill_prompt(_session("HOW"), memory=memory)
    assert why == build_why_drill_prompt(_session("WHY"), memory=memory)
    assert mech == build_mech_drill_prompt(_session("MECH"), memory=memory)
    assert adv == build_advanced_drill_prompt(
        _session("ADVANCED_ASTERISK"), memory=memory
    )
    assert deep == build_deep_drill_prompt(_session("DEEP_ASTERISK"), memory=memory)
    assert "[Слой HOW:" in how
    assert "[Слой WHY:" in why
    assert "[Слой MECH:" in mech
    assert "[Слой ADVANCED:" in adv
    assert "[Слой DEEP:" in deep


def test_dispatcher_rejects_unknown_layer() -> None:
    raw = LayerDrillSession(
        is_active=True,
        target_sub_concept_ids=["pyobject"],
        current_index=0,
        status="DRILL_IN_PROGRESS",
    )
    with pytest.raises(ValueError, match="Unknown drill layer"):
        build_drill_session_prompt(raw)


def test_select_system_prompt_injects_specialized_how_drill() -> None:
    mem = _memory(_session("HOW"))
    system, mode, _ = select_system_prompt_and_mode(
        "[mode:deep_dive_how] Разбери архитектуру темы.",
        default_system_prompt="DEFAULT",
        memory=mem,
    )
    assert mode == "deep_dive_how"
    assert "HOW DRILL — SPECIALIZED ACTIVE TEACHING" in system
    assert "DRILL_ACTIVE" in system
    assert "checked 0/3" in system
    assert "DO NOT declare the node or layer complete" in system
    assert "300" in system
    assert "WHY DRILL — SPECIALIZED" not in system
    assert "JSON OUTPUT (ActiveDrillStepResponse)" in system
    assert "JSON OUTPUT (DeepDiveTutorContract)" not in system
    assert "ANTI-SYCOPHANCY INVARIANTS" in system
    assert "status_header" in system
    assert "theory_body" in system
    assert "next_question" in system
    assert "audit" in system
