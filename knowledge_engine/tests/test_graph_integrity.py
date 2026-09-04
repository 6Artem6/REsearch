"""Knowledge-graph integrity: DAG cycles, orphan ids, overlay refs."""

from __future__ import annotations

from knowledge_engine.src.graph_validator import (
    validate_knowledge_graph_integrity,
)


def _healthy_graph() -> dict:
    return {
        "nodes": [
            {
                "node_id": "hashing",
                "prerequisites": [],
                "sub_concepts": [{"id": "hash_core", "is_extension": False}],
            },
            {
                "node_id": "wal",
                "prerequisites": ["hashing"],
                "sub_concepts": [
                    {"id": "wal_core", "is_extension": False},
                    {
                        "id": "wal_l4",
                        "is_extension": True,
                        "parent_id": "wal_core",
                        "overlay_kind": "advanced_analysis",
                        "overlay_type": "ADVANCED_ASTERISK",
                    },
                ],
                "overlays": [
                    {
                        "concept_id": "wal_l4",
                        "overlay_type": "ADVANCED_ASTERISK",
                    }
                ],
            },
            {
                "node_id": "consensus",
                "prerequisites": ["wal", "hashing"],
                "sub_concepts": [{"id": "raft_core"}],
            },
        ],
        "deep_mastery_concepts": [
            {"concept_id": "wal_l4", "overlay_type": "ADVANCED_ASTERISK"}
        ],
        "pending_evaluation_concept_id": "wal_core",
    }


def test_healthy_graph_is_ok():
    report = validate_knowledge_graph_integrity(_healthy_graph())
    assert report.ok
    assert report.cycles == ()
    assert report.orphan_node_ids == ()
    assert report.orphan_sub_concept_ids == ()
    assert report.overlay_errors == ()
    assert report.node_count == 3
    assert report.sub_concept_count == 4


def test_tarjan_detects_prerequisite_cycle():
    graph = {
        "nodes": [
            {"node_id": "a", "prerequisites": ["c"]},
            {"node_id": "b", "prerequisites": ["a"]},
            {"node_id": "c", "prerequisites": ["b"]},
        ]
    }
    report = validate_knowledge_graph_integrity(graph)
    assert not report.ok
    assert report.cycles
    joined = " ".join("→".join(c) for c in report.cycles)
    assert "a" in joined and "b" in joined and "c" in joined
    assert any("cycle:" in e for e in report.errors)


def test_self_loop_is_a_cycle():
    graph = {"nodes": [{"node_id": "loop", "prerequisites": ["loop"]}]}
    report = validate_knowledge_graph_integrity(graph)
    assert not report.ok
    assert report.cycles == (("loop",),)
    assert any("self-reference" in e for e in report.errors)


def test_orphan_prerequisite_node_id():
    graph = {
        "nodes": [
            {"node_id": "wal", "prerequisites": ["missing_hash"]},
        ]
    }
    report = validate_knowledge_graph_integrity(graph)
    assert not report.ok
    assert "missing_hash" in report.orphan_node_ids
    assert any("prerequisite" in e for e in report.errors)


def test_orphan_sub_concept_and_pending_ref():
    graph = {
        "nodes": [
            {
                "node_id": "wal",
                "prerequisites": [],
                "sub_concepts": [{"id": "wal_core"}],
            }
        ],
        "pending_evaluation_concept_id": "ghost_sc",
        "memory": {
            "asked_question_sub_concept_id": "also_ghost",
            "sub_concepts": [{"id": "wal_core"}],
        },
    }
    report = validate_knowledge_graph_integrity(graph)
    assert not report.ok
    assert "ghost_sc" in report.orphan_sub_concept_ids
    assert "also_ghost" in report.orphan_sub_concept_ids


def test_overlay_broken_concept_and_invalid_type():
    graph = {
        "nodes": [
            {
                "node_id": "wal",
                "prerequisites": [],
                "sub_concepts": [{"id": "wal_core", "is_extension": False}],
                "overlays": [
                    {"concept_id": "no_such_overlay", "overlay_type": "DEEP_ASTERISK"},
                    {"concept_id": "wal_core", "overlay_type": "STAR_BONUS"},
                ],
            }
        ]
    }
    report = validate_knowledge_graph_integrity(graph)
    assert not report.ok
    assert "no_such_overlay" in report.orphan_sub_concept_ids
    assert any("invalid overlay_type" in e for e in report.overlay_errors)
    assert any("no_such_overlay" in e for e in report.overlay_errors)


def test_overlay_extension_parent_must_exist():
    graph = {
        "nodes": [
            {
                "node_id": "wal",
                "prerequisites": [],
                "sub_concepts": [
                    {
                        "id": "wal_l5",
                        "is_extension": True,
                        "parent_id": "missing_core",
                        "overlay_kind": "deep_design",
                        "overlay_type": "DEEP_ASTERISK",
                    }
                ],
            }
        ]
    }
    report = validate_knowledge_graph_integrity(graph)
    assert not report.ok
    assert "missing_core" in report.orphan_sub_concept_ids


def test_pydantic_curriculum_graph_dump_is_accepted():
    from knowledge_engine.src.curriculum.schemas import CurriculumGraph, CurriculumNode

    graph = CurriculumGraph(
        curriculum_id="kg_integrity_demo",
        title="Integrity demo",
        description="Minimal curriculum used only for graph-integrity unit tests.",
        total_nodes=3,
        nodes=[
            CurriculumNode(
                node_id="foundation_one",
                title="Foundation one",
                layer="foundation",
                category="core",
                brief_summary="Introduces hashing primitives for later nodes.",
                core_concepts=["hashing"],
                prerequisites=[],
            ),
            CurriculumNode(
                node_id="advanced_two",
                title="Advanced two",
                layer="advanced",
                category="core",
                brief_summary="Builds a write-ahead log on hashing.",
                core_concepts=["wal"],
                prerequisites=["foundation_one"],
            ),
            CurriculumNode(
                node_id="sota_three",
                title="Sota three",
                layer="sota",
                category="core",
                brief_summary="Consensus on top of the log.",
                core_concepts=["raft"],
                prerequisites=["advanced_two"],
            ),
        ],
    )
    report = validate_knowledge_graph_integrity(graph)
    assert report.ok
    assert report.node_count == 3
