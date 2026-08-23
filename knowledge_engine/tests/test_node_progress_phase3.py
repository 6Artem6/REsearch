"""Phase 3: Core Progress isolated from Overlay (topic_mastery_score)."""

from __future__ import annotations

from knowledge_engine.src.node_deep_dive.concept_map_state import (
    build_coverage_summary,
    core_sub_concepts,
    ensure_sub_concept_map,
    register_deep_mastery,
    sub_concept_coverage_complete,
)
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    SessionMemory,
    SubConceptRecord,
)
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput
from knowledge_engine.src.node_deep_dive.sub_concept_evaluator import (
    apply_threshold_to_sub_concept,
    required_depth_layers,
)
from knowledge_engine.src.node_deep_dive.tiered_memory import (
    derive_node_status,
    engagement_topic_mastery,
    sync_topic_mastery_score,
)


def _node(*, layer: str = "advanced") -> NodeDataInput:
    return NodeDataInput(
        node_id="agg_node",
        title="Weighted aggregation",
        layer=layer,  # type: ignore[arg-type]
        learning_goal="Understand aggregation and consensus scoring.",
        core_concepts=["Weighted aggregation", "Consensus scoring"],
    )


def _credit_required_layers(row: SubConceptRecord, layer: str) -> None:
    req = set(required_depth_layers(layer))
    apply_threshold_to_sub_concept(
        row,
        layer=layer,
        why="WHY" in req,
        how="HOW" in req,
        mechanic="MECHANIC" in req,
        evidence="phase3 core credit",
    )


def test_core_mastery_is_100_percent_without_overlay() -> None:
    layer = "advanced"
    node = _node(layer=layer)
    mem = SessionMemory()
    ensure_sub_concept_map(mem, node)
    core = core_sub_concepts(mem)
    assert len(core) == 2
    assert all(not sc.is_extension for sc in core)

    for sc in core:
        _credit_required_layers(sc, layer)
        assert sc.status == "verified"

    score = sync_topic_mastery_score(mem)
    assert score == 100
    assert mem.topic_mastery_score == 100
    assert mem.deep_mastery_concepts == []
    assert sub_concept_coverage_complete(mem) is True

    added = register_deep_mastery(mem, core[0].id)
    assert added is True
    score_after = sync_topic_mastery_score(mem)
    assert score_after == 100
    cov = build_coverage_summary(mem)
    assert cov is not None
    assert cov.deep_mastery_count == 1
    assert cov.deep_mastery_ids == [core[0].id]
    assert derive_node_status(mem, None) == "mastered"


def test_unverified_extension_does_not_dilute_core_percent() -> None:
    layer = "advanced"
    node = _node(layer=layer)
    mem = SessionMemory()
    ensure_sub_concept_map(mem, node)
    for sc in core_sub_concepts(mem):
        _credit_required_layers(sc, layer)

    mem.sub_concepts.append(
        SubConceptRecord(
            id="star_overlay",
            label="Asterisk overlay design",
            status="unchecked",
            is_extension=True,
        )
    )
    assert engagement_topic_mastery(mem) == 100
    assert sub_concept_coverage_complete(mem) is True
    cov = build_coverage_summary(mem)
    assert cov is not None
    assert cov.total == 2
    assert cov.verified == 2


def test_verified_extension_does_not_inflate_core_percent() -> None:
    layer = "advanced"
    mem = SessionMemory(
        sub_concepts=[
            SubConceptRecord(
                id="core_a",
                label="core_a",
                status="verified",
                why_passed=True,
                how_passed=True,
                mechanic_passed=True,
            ),
            SubConceptRecord(
                id="core_b",
                label="core_b",
                status="unchecked",
            ),
            SubConceptRecord(
                id="ext_star",
                label="ext_star",
                status="verified",
                why_passed=True,
                how_passed=True,
                mechanic_passed=True,
                is_extension=True,
            ),
        ]
    )
    score = engagement_topic_mastery(mem)
    assert score == 50
    assert score < 100
    cov = build_coverage_summary(mem)
    assert cov is not None
    assert cov.total == 2
    assert cov.verified == 1
    assert cov.deep_mastery_count == 0
