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


def test_no_optional_chip_before_why_closed() -> None:
    """Реальный баг: «Дожать HOW» появлялся на Foundation-ноде ДО закрытия
    всех WHY-вопросов — open_optional_layers предлагал HOW/MECHANIC, как
    только совокупный how/mech был False, не проверяя why вообще."""
    mem = _complete_memory(why=False, how=False, mech=False)
    layers = open_optional_layers(mem, "foundation")
    assert layers == []
    assert gloss_fork_quick_replies(layers) == []


def test_no_optional_chip_when_only_some_rows_ahead() -> None:
    """Реальный баг (curriculum=indexes_and_data_structures,
    node=b_tree_indexes): у ноды 4 подтемы, why_passed=True только у 2 —
    per-row fallback-цикл видел, что ОДНА из этих двух уже прошла HOW, и
    предлагал «Дожать MECH» по ней одной, хотя у двух других подтем WHY ещё
    не закрыт (WHY не закрыт по ноде в целом)."""
    mem = SessionMemory(
        sub_concepts=[
            SubConceptRecord(
                id="b_tree", label="b_tree", status="verified",
                why_passed=True, how_passed=True, mechanic_passed=True,
            ),
            SubConceptRecord(
                id="balanced_tree", label="balanced_tree", status="verified",
                why_passed=True, how_passed=True, mechanic_passed=False,
            ),
            SubConceptRecord(
                id="range_queries", label="range_queries", status="unchecked",
                why_passed=False, how_passed=False, mechanic_passed=False,
            ),
            SubConceptRecord(
                id="leaf_pages", label="leaf_pages", status="unchecked",
                why_passed=False, how_passed=False, mechanic_passed=False,
            ),
        ],
    )
    layers = open_optional_layers(mem, "foundation")
    assert layers == []
    assert gloss_fork_quick_replies(layers) == []


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
    assert "pathway=optional_fork" in text
    assert "MECHANIC" in text or "open_optional_layers" in text
    # Chips are Host-owned via gloss_fork_quick_replies — not LLM prompt routing.
    chips = gloss_fork_quick_replies(["MECHANIC"])
    assert chips == ["Хочу Gloss", "Дожать MECH", "Идем дальше"]
    assert "Дожать MECHANIC" not in chips
    assert "Концептуальный минимум" not in text
    assert "USER CHOICE HANDLING" not in text
    assert "quick_replies=" not in text


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
    assert "pathway=base_complete" in text
    assert "Базовая теория ноды усвоена" not in text
    assert "Нода полностью освоена" not in text
    assert "quick_replies=" not in text


def test_behavior_deep_analysis_blocks_base_closure_decree() -> None:
    mem = _complete_memory(why=True, how=True, mech=True)
    text = _next_action_for_mode(
        "dialogue_feedback",
        "ANSWER",
        "chat",
        "pathway_decision",
        memory=mem,
        user_message="[mode:deep_analysis] Задачка со звёздочкой",
        node_layer="sota",
    )
    assert "DEEP_ANALYSIS" in text
    assert "node_completed=false" in text
    assert "Базовая теория ноды усвоена" not in text
    assert "Нода полностью освоена" not in text
    assert "pathway=base_complete" not in text


def test_behavior_mech_closed_no_mech_chip() -> None:
    """When MECH is already passed, Python chips must not offer Дожать MECH."""
    mem = _complete_memory(why=True, how=False, mech=True)
    layers = open_optional_layers(mem, "foundation")
    assert "HOW" in layers
    assert "MECHANIC" not in layers
    chips = gloss_fork_quick_replies(layers)
    assert "Дожать HOW" in chips
    assert "Дожать MECH" not in chips
    text = _next_action_for_mode(
        "dialogue_feedback",
        "ANSWER",
        "chat",
        "pathway_decision",
        memory=mem,
        user_message="",
        node_layer="foundation",
    )
    assert "pathway=optional_fork" in text
    assert "HOW" in text
    assert "Концептуальный минимум" not in text
    assert "USER CHOICE HANDLING" not in text
    # next_action must not embed chip recipes for the LLM to invent
    assert "Дожать MECH" not in text
    assert "quick_replies=" not in text


def test_pathway_flag_in_behavior_state() -> None:
    from knowledge_engine.src.node_deep_dive.tutor_behavior_state import (
        build_tutor_behavior_state,
    )

    mem = _complete_memory(why=True, how=True, mech=True)
    mem.last_eval_directive = "PASSED_CLEAN"
    state = build_tutor_behavior_state(
        "ANSWER",
        "chat",
        "chat",
        "pathway_decision",
        "",
        memory=mem,
        node_layer="sota",
    )
    assert state["pathway"] == "base_complete"

    mem2 = _complete_memory(why=True, how=True, mech=False)
    state2 = build_tutor_behavior_state(
        "ANSWER",
        "chat",
        "chat",
        "pathway_decision",
        "",
        memory=mem2,
        node_layer="advanced",
    )
    assert state2["pathway"] == "optional_fork"


def test_behavior_deep_mastery_earned_no_100_percent_decree() -> None:
    mem = _complete_memory(why=True, how=True, mech=True)
    mem.last_eval_directive = "DEEP_MASTERY_EARNED"
    mem.star_task_status = "resolved"
    mem.deep_mastery_concepts = ["agg"]
    text = _next_action_for_mode(
        "dialogue_feedback",
        "ANSWER",
        "chat",
        "pathway_decision",
        memory=mem,
        user_message="",
        node_layer="advanced",
    )
    assert "DEEP_MASTERY_EARNED" in text or "resolved" in text
    assert "Нода полностью освоена" not in text


def test_behavior_star_task_refinement_no_transition() -> None:
    mem = _complete_memory(why=True, how=True, mech=True)
    mem.star_task_status = "needs_refinement"
    mem.pending_eval_kind = "deep_analysis"
    mem.pending_evaluation_concept_id = "agg"
    mem.last_eval_directive = "STAR_TASK_NEEDS_REFINEMENT"
    text = _next_action_for_mode(
        "dialogue_feedback",
        "ANSWER",
        "chat",
        "pathway_decision",
        memory=mem,
        user_message="доработал схему с учётом partition",
        node_layer="advanced",
    )
    assert "needs_refinement" in text
    assert "node_completed=false" in text
    assert "TOPIC_COMPLETE" not in text
    assert "Нода полностью освоена" not in text


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
    assert "quick_replies=" not in text


def _how_gap_memory() -> SessionMemory:
    """Foundation node after WHY gloss: two core rows still missing HOW."""
    return SessionMemory(
        last_eval_directive="PASSED_WITH_GLOSS",
        learning_phase="pathway_decision",
        sub_concepts=[
            SubConceptRecord(
                id="pyobject",
                label="PyObject header",
                status="verified",
                why_passed=True,
                how_passed=True,
                mechanic_passed=False,
            ),
            SubConceptRecord(
                id="reference_count",
                label="Reference count",
                status="verified",
                why_passed=True,
                how_passed=False,
                mechanic_passed=False,
            ),
            SubConceptRecord(
                id="type_pointer",
                label="Type pointer",
                status="verified",
                why_passed=True,
                how_passed=False,
                mechanic_passed=False,
            ),
        ],
    )


def test_set_pending_refuses_verified_without_how_session() -> None:
    from knowledge_engine.src.node_deep_dive.concept_map import (
        resolve_evaluation_target_id,
        set_pending_evaluation_for_tutor_turn,
    )

    mem = _how_gap_memory()
    assert set_pending_evaluation_for_tutor_turn(mem, "reference_count") == ""
    mem.pending_evaluation_concept_id = "reference_count"
    mem.asked_question_sub_concept_id = "reference_count"
    assert resolve_evaluation_target_id(mem) == ""


def test_set_pending_and_eval_target_during_how_session() -> None:
    from knowledge_engine.src.node_deep_dive.concept_map import (
        first_optional_layer_concept_id,
        resolve_evaluation_target_id,
        set_pending_evaluation_for_tutor_turn,
    )

    mem = _how_gap_memory()
    mem.active_optional_layer = "HOW"
    assert first_optional_layer_concept_id(mem, "HOW") == "reference_count"
    cid = set_pending_evaluation_for_tutor_turn(mem, "reference_count")
    assert cid == "reference_count"
    assert resolve_evaluation_target_id(mem) == "reference_count"


def test_orchestrate_how_answer_holds_and_keeps_teaching() -> None:
    from knowledge_engine.src.node_deep_dive.concept_map import (
        host_ready_for_transition,
        orchestrate_tutor_llm_output,
    )
    from knowledge_engine.src.node_deep_dive.schemas import DeepDiveLLMOutput

    mem = _how_gap_memory()
    mem.active_optional_layer = "HOW"
    out = DeepDiveLLMOutput(
        ready_for_transition=True,
        follow_up_question="Где сидит ob_type относительно refcnt?",
        question_sub_concept_id="type_pointer",
        technical_explanation="Разбор arenas / pymalloc — не закрываем ноду.",
    )
    packed = orchestrate_tutor_llm_output(
        mem,
        out,
        user_message="Пулы и арены держат блоки, refcnt в заголовке PyObject.",
        node_layer="foundation",
    )
    assert packed.ready_for_transition is False
    assert packed.suggested_next_step is None
    assert mem.active_optional_layer == "HOW"
    assert host_ready_for_transition(
        mem,
        user_message="Пулы и арены держат блоки, refcnt в заголовке PyObject.",
        node_layer="foundation",
    ) is False


def test_advance_how_session_to_next_subtopic() -> None:
    from knowledge_engine.src.node_deep_dive.concept_map import (
        advance_next_question_after_evaluation,
        select_next_sub_concept,
    )

    mem = _how_gap_memory()
    mem.active_optional_layer = "HOW"
    mem.sub_concepts[1].how_passed = True
    nxt = advance_next_question_after_evaluation(
        mem, evaluated_id="reference_count"
    )
    assert nxt == "type_pointer"
    assert mem.active_optional_layer == "HOW"
    row = select_next_sub_concept(mem)
    assert row is not None
    assert row.id == "type_pointer"


def test_behavior_how_session_continues_without_chip() -> None:
    mem = _how_gap_memory()
    mem.active_optional_layer = "HOW"
    text = _next_action_for_mode(
        "dialogue_feedback",
        "ANSWER",
        "chat",
        "pathway_decision",
        memory=mem,
        user_message="Пулы и арены держат блоки, refcnt в заголовке PyObject.",
        node_layer="foundation",
    )
    assert "GLOSS_FORK_CHOICE=how" in text
    assert "reference_count" in text
    assert "pathway=base_complete" not in text
    assert "pathway=optional_fork" not in text


def test_idle_optional_fork_still_ready_for_chips() -> None:
    from knowledge_engine.src.node_deep_dive.concept_map import (
        host_ready_for_transition,
        orchestrate_tutor_llm_output,
    )
    from knowledge_engine.src.node_deep_dive.schemas import DeepDiveLLMOutput

    mem = _how_gap_memory()
    assert host_ready_for_transition(
        mem, user_message="", node_layer="foundation"
    ) is True
    packed = orchestrate_tutor_llm_output(
        mem,
        DeepDiveLLMOutput(
            ready_for_transition=True,
            follow_up_question="Выбери шаг.",
            technical_explanation="Порог WHY закрыт.",
        ),
        user_message="",
        node_layer="foundation",
    )
    assert packed.ready_for_transition is True
    assert packed.suggested_next_step == "deep_dive_optional"


def test_how_session_releases_when_layer_closed() -> None:
    from knowledge_engine.src.node_deep_dive.concept_map import (
        orchestrate_tutor_llm_output,
    )
    from knowledge_engine.src.node_deep_dive.schemas import DeepDiveLLMOutput

    mem = _how_gap_memory()
    mem.active_optional_layer = "HOW"
    for sc in mem.sub_concepts:
        sc.how_passed = True
    packed = orchestrate_tutor_llm_output(
        mem,
        DeepDiveLLMOutput(
            ready_for_transition=False,
            follow_up_question="Выбери шаг.",
            technical_explanation="HOW закрыт по всем подтемам.",
        ),
        user_message="tp_dealloc в типе, refcnt в заголовке.",
        node_layer="foundation",
    )
    assert mem.active_optional_layer == ""
    assert packed.ready_for_transition is True
    assert packed.suggested_next_step == "deep_dive_optional"


def _mech_gap_memory() -> SessionMemory:
    """Advanced node after WHY+HOW: two core rows still missing MECH."""
    return SessionMemory(
        last_eval_directive="PASSED_WITH_GLOSS",
        learning_phase="pathway_decision",
        sub_concepts=[
            SubConceptRecord(
                id="agg",
                label="Aggregation",
                status="verified",
                why_passed=True,
                how_passed=True,
                mechanic_passed=True,
            ),
            SubConceptRecord(
                id="weights",
                label="Weights",
                status="verified",
                why_passed=True,
                how_passed=True,
                mechanic_passed=False,
            ),
            SubConceptRecord(
                id="timeouts",
                label="Timeouts",
                status="verified",
                why_passed=True,
                how_passed=True,
                mechanic_passed=False,
            ),
        ],
    )


def test_mech_session_binds_pending_and_holds_like_how() -> None:
    from knowledge_engine.src.node_deep_dive.concept_map import (
        first_optional_layer_concept_id,
        host_ready_for_transition,
        orchestrate_tutor_llm_output,
        set_pending_evaluation_for_tutor_turn,
    )
    from knowledge_engine.src.node_deep_dive.schemas import DeepDiveLLMOutput

    mem = _mech_gap_memory()
    mem.active_optional_layer = "MECHANIC"
    assert first_optional_layer_concept_id(mem, "MECHANIC") == "weights"
    assert set_pending_evaluation_for_tutor_turn(mem, "weights") == "weights"
    packed = orchestrate_tutor_llm_output(
        mem,
        DeepDiveLLMOutput(
            ready_for_transition=True,
            follow_up_question="Что если score=0?",
            question_sub_concept_id="timeouts",
            technical_explanation="```python\nclass X: ...\n```",
        ),
        user_message="Веса нормирую, при нуле ставлю epsilon.",
        node_layer="advanced",
    )
    assert packed.ready_for_transition is False
    assert mem.active_optional_layer == "MECHANIC"
    assert host_ready_for_transition(
        mem,
        user_message="Веса нормирую, при нуле ставлю epsilon.",
        node_layer="advanced",
    ) is False


def test_advance_mech_session_to_next_subtopic() -> None:
    from knowledge_engine.src.node_deep_dive.concept_map import (
        advance_next_question_after_evaluation,
        select_next_sub_concept,
    )

    mem = _mech_gap_memory()
    mem.active_optional_layer = "MECHANIC"
    mem.sub_concepts[1].mechanic_passed = True
    nxt = advance_next_question_after_evaluation(mem, evaluated_id="weights")
    assert nxt == "timeouts"
    row = select_next_sub_concept(mem)
    assert row is not None
    assert row.id == "timeouts"


def test_behavior_mech_session_continues_without_chip() -> None:
    mem = _mech_gap_memory()
    mem.active_optional_layer = "MECHANIC"
    text = _next_action_for_mode(
        "dialogue_feedback",
        "ANSWER",
        "chat",
        "pathway_decision",
        memory=mem,
        user_message="Веса нормирую, при нуле ставлю epsilon.",
        node_layer="advanced",
    )
    assert "GLOSS_FORK_CHOICE=mech" in text
    assert "weights" in text
    assert "pathway=base_complete" not in text
