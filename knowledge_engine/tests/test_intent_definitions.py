"""SSOT: INTENT_RULES drives production routing, chips, and the offline probe."""

from __future__ import annotations

from knowledge_engine.src.node_deep_dive.control_intent import (
    ACTION_ALIASES,
    REGISTERED_CONTROL_CHIPS,
    classify_control_chip,
)
from knowledge_engine.src.node_deep_dive.intent_definitions import (
    CHIP_ADVANCED_ANALYSIS,
    CHIP_DEEP_DESIGN,
    CHIP_GLOSS,
    CHIP_HOW,
    CHIP_MECH,
    CHIP_OVERLAY_NEXT,
    INTENT_NAMES,
    INTENT_REFERENCE_PHRASES,
    INTENT_RULES,
    catalog_phrases,
    probe_cues,
    validate_intent_catalog,
)
from knowledge_engine.src.node_deep_dive.star_task_fsm import (
    CHIP_ADVANCED_ANALYSIS as FSM_CHIP_ADVANCED,
    CHIP_DEEP_DESIGN as FSM_CHIP_DEEP,
    CHIP_OVERLAY_NEXT as FSM_CHIP_NEXT,
)
from knowledge_engine.src.node_deep_dive.vector_intent_router import (
    INTENT_REFERENCE_PHRASES as ROUTER_PHRASES,
    VectorIntentRouter,
    iter_reference_entries,
)
from knowledge_engine.tests.intent_embed_probe import lexical_probe_embed


def test_validate_intent_catalog_ok():
    stats = validate_intent_catalog()
    assert stats["ok"] is True
    assert stats["intents"] == len(INTENT_RULES)
    assert stats["phrases"] == sum(len(catalog_phrases(r)) for r in INTENT_RULES)
    assert stats["overlay_order"] == [
        "advanced_analysis",
        "deep_design",
        "deep_analysis",
    ]


def test_overlay_rules_precede_generic_deep_analysis():
    names = list(INTENT_NAMES)
    assert names.index("advanced_analysis") < names.index("deep_analysis")
    assert names.index("deep_design") < names.index("deep_analysis")


def test_router_and_probe_share_the_same_catalog():
    assert ROUTER_PHRASES is INTENT_REFERENCE_PHRASES
    assert set(INTENT_REFERENCE_PHRASES) == set(INTENT_NAMES)
    for rule in INTENT_RULES:
        assert INTENT_REFERENCE_PHRASES[rule.intent] == catalog_phrases(rule)
        assert probe_cues(rule)


def test_host_chips_are_registered_exact_labels():
    assert FSM_CHIP_ADVANCED == CHIP_ADVANCED_ANALYSIS
    assert FSM_CHIP_DEEP == CHIP_DEEP_DESIGN
    assert FSM_CHIP_NEXT == CHIP_OVERLAY_NEXT
    assert REGISTERED_CONTROL_CHIPS[CHIP_GLOSS] == "gloss"
    assert REGISTERED_CONTROL_CHIPS[CHIP_HOW] == "how"
    assert REGISTERED_CONTROL_CHIPS[CHIP_MECH] == "mech"
    assert REGISTERED_CONTROL_CHIPS[CHIP_ADVANCED_ANALYSIS] == "advanced_analysis"
    assert REGISTERED_CONTROL_CHIPS[CHIP_DEEP_DESIGN] == "deep_design"
    assert REGISTERED_CONTROL_CHIPS[CHIP_OVERLAY_NEXT] == "next"
    from knowledge_engine.src.node_deep_dive.intent_definitions import (
        CHIP_CHECK,
        CHIP_PRACTICE,
        CHIP_SKIP,
    )

    assert REGISTERED_CONTROL_CHIPS[CHIP_PRACTICE] == "practice"
    assert REGISTERED_CONTROL_CHIPS[CHIP_CHECK] == "check"
    assert REGISTERED_CONTROL_CHIPS[CHIP_SKIP] == "skip"


def test_action_aliases_cover_factory_modes():
    assert ACTION_ALIASES["deep_dive_how"] == "how"
    assert ACTION_ALIASES["deep_dive_mech"] == "mech"
    assert ACTION_ALIASES["advanced_analysis"] == "advanced_analysis"
    assert ACTION_ALIASES["glossary"] == "gloss"


def test_lancedb_expected_rows_match_ssot_catalog():
    expected = iter_reference_entries()
    assert len(expected) == sum(len(v) for v in INTENT_REFERENCE_PHRASES.values())
    intents = {intent for _eid, intent, _phrase in expected}
    assert intents == set(INTENT_NAMES)


def test_sync_validates_catalog_and_persists_vectors(tmp_path):
    router = VectorIntentRouter(
        threshold=0.82,
        embed_fn=lexical_probe_embed,
        enabled=True,
        persist=True,
        db_path=tmp_path / "intent_lance",
        embed_model="probe-embed",
        auto_sync=False,
    )
    stats = router.sync_and_validate_intents()
    assert stats["catalog_valid"] is True
    assert stats["catalog_intents"] == len(INTENT_RULES)
    assert stats["expected"] == stats["catalog_phrases"]
    assert stats["embedded"] == stats["expected"]
    assert stats["loaded_from_db"] == stats["expected"]

    again = VectorIntentRouter(
        threshold=0.82,
        embed_fn=lexical_probe_embed,
        persist=True,
        db_path=tmp_path / "intent_lance",
        embed_model="probe-embed",
        auto_sync=False,
    )
    stats2 = again.sync_and_validate_intents()
    assert stats2["catalog_valid"] is True
    assert stats2["embedded"] == 0
    assert stats2["loaded_from_db"] == stats["expected"]


def test_chips_and_paraphrases_classify_via_ssot():
    router = VectorIntentRouter(
        threshold=0.82,
        embed_fn=lexical_probe_embed,
        persist=False,
        auto_sync=True,
        enabled=True,
    )
    from knowledge_engine.src.node_deep_dive.vector_intent_router import (
        set_vector_intent_router_for_tests,
    )

    set_vector_intent_router_for_tests(router)
    try:
        assert classify_control_chip(CHIP_HOW) == "how"
        assert classify_control_chip(CHIP_ADVANCED_ANALYSIS) == "advanced_analysis"
        assert classify_control_chip(CHIP_DEEP_DESIGN) == "deep_design"
        assert classify_control_chip("анализ уязвимостей") == "advanced_analysis"
        assert classify_control_chip("задачка со звёздочкой") == "deep_analysis"
        assert classify_control_chip("[mode:lecture] Дай плотный материал") == "lecture"
        assert classify_control_chip("практика") == "practice"
        assert classify_control_chip("проверка") == "check"
    finally:
        set_vector_intent_router_for_tests(None)
