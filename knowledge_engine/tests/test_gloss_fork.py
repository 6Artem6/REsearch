"""Layer-aware optional-depth fork at topic completion."""

from __future__ import annotations

from knowledge_engine.src.node_deep_dive.concept_map import (
    classify_gloss_fork_choice,
    gloss_fork_quick_replies,
    open_optional_layers,
)
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    SessionMemory,
    SubConceptRecord,
)
from knowledge_engine.src.node_deep_dive.tutor_behavior_state import (
    _next_action_for_mode,
)


def test_classify_gloss_fork_chips() -> None:
    assert classify_gloss_fork_choice("Хочу Gloss") == "gloss"
    assert classify_gloss_fork_choice("Дожать HOW") == "how"
    assert classify_gloss_fork_choice("Дожать MECH") == "mech"
    assert classify_gloss_fork_choice("Идем дальше") == "next"
    assert classify_gloss_fork_choice("random") == ""
    assert (
        classify_gloss_fork_choice("[mode:deep_dive_mech] Разбери механики и код темы.")
        == "mech"
    )
    assert (
        classify_gloss_fork_choice(
            "[mode:gloss] Сформируй сжатую выжимку (Glossary) по оставшимся слоям."
        )
        == "gloss"
    )


def _complete_memory(**flags: bool) -> SessionMemory:
    return SessionMemory(
        last_eval_directive="PASSED_WITH_GLOSS",
        sub_concepts=[
            SubConceptRecord(
                id="agg",
                label="agg",
                status="verified",
                why_passed=flags.get("why", True),
                how_passed=flags.get("how", True),
                mechanic_passed=flags.get("mech", False),
            )
        ],
    )


def test_open_optional_by_layer() -> None:
    mem = _complete_memory(why=True, how=False, mech=False)
    assert open_optional_layers(mem, "foundation") == ["HOW", "MECHANIC"]
    mem2 = _complete_memory(why=True, how=True, mech=False)
    assert open_optional_layers(mem2, "advanced") == ["MECHANIC"]
    assert open_optional_layers(mem2, "sota") == []


def test_behavior_optional_fork_advanced() -> None:
    mem = _complete_memory(why=True, how=True, mech=False)
    text = _next_action_for_mode(
        "dialogue_feedback",
        "ANSWER",
        "chat",
        "pathway_decision",
        memory=mem,
        user_message="",
        node_layer="advanced",
    )
    assert "OPTIONAL_LAYER FORK" in text
    assert "MECHANIC" in text
    for label in gloss_fork_quick_replies(["MECHANIC"]):
        assert label in text


def test_behavior_full_depth_sota() -> None:
    mem = _complete_memory(why=True, how=True, mech=True)
    mem.last_eval_directive = "PASSED_CLEAN"
    text = _next_action_for_mode(
        "dialogue_feedback",
        "ANSWER",
        "chat",
        "pathway_decision",
        memory=mem,
        user_message="",
        node_layer="sota",
    )
    assert "FULL DEPTH" in text
    assert "100%" in text


def test_behavior_gloss_choice_how() -> None:
    mem = _complete_memory(why=True, how=False, mech=False)
    text = _next_action_for_mode(
        "dialogue_feedback",
        "ANSWER",
        "chat",
        "pathway_decision",
        memory=mem,
        user_message="Дожать HOW",
        node_layer="foundation",
    )
    assert "GLOSS_FORK_CHOICE=how" in text
    assert "ready_for_transition=false" in text
