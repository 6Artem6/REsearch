"""Socratic Poles — templated FACT_* payload for asterisk-question deep_analysis."""

from __future__ import annotations

from knowledge_engine.src.node_deep_dive.memory_schemas import (
    DialogueFactManifest,
    SessionMemory,
    SubConceptRecord,
)
from knowledge_engine.src.node_deep_dive.prompt_factory import (
    format_deep_analysis_novelty_block,
    select_system_prompt_and_mode,
)
from knowledge_engine.src.node_deep_dive.socratic_poles import (
    SOCRATIC_POLES_STATIC_RULES,
    build_socratic_poles_payload,
    format_fact_attraction,
    format_fact_repulsion,
    format_socratic_poles_state_block,
)


def test_fact_templates_are_rigid() -> None:
    rep = format_fact_repulsion(
        node="n1",
        concept_id="c_iso",
        claim="Isolation verified via WHY+HOW",
    )
    att = format_fact_attraction(
        node="n1",
        concept_id="c_fail",
        focus_hint="partition recovery gap",
    )
    assert rep.startswith("FACT_REPULSION: [Node: n1]")
    assert 'Concept "c_iso" is VERIFIED' in rep
    assert "Isolation verified" in rep
    assert att.startswith("FACT_ATTRACTION: [Node: n1]")
    assert 'Concept "c_fail" has GAP/BOTTLENECK' in att
    assert "partition recovery gap" in att


def test_build_socratic_poles_payload_local() -> None:
    mem = SessionMemory(
        sub_concepts=[
            SubConceptRecord(
                id="sc_verified",
                label="Verified concept",
                status="verified",
                evidence="WHY+HOW done",
                why_passed=True,
                how_passed=True,
            ),
            SubConceptRecord(
                id="sc_gap",
                label="Gap concept",
                status="gap",
                focus_hint="missing failure modes",
            ),
        ],
        fact_manifest=DialogueFactManifest(
            agreed_concepts=["agreed isolation boundary"],
            open_bottlenecks=["backpressure under fan-out"],
        ),
    )
    payload = build_socratic_poles_payload(
        mem,
        "node_agg",
        curriculum_id="",
        include_cross_node=False,
    )
    assert payload["block"]
    assert "[SOCRATIC_POLES_STATE]" in payload["block"]
    assert "POLARITY: REPULSION" in payload["block"]
    assert "POLARITY: ATTRACTION" in payload["block"]
    assert any("sc_verified" in r["fact_line"] for r in payload["repulsion"])
    assert any("sc_gap" in a["fact_line"] for a in payload["attraction"])
    assert any("backpressure" in a["fact_line"] for a in payload["attraction"])
    assert mem.socratic_poles_snapshot.get("repulsion")


def test_format_poles_state_block_empty() -> None:
    block = format_socratic_poles_state_block([], [])
    assert "[SOCRATIC_POLES_STATE]" in block
    assert "(none yet)" in block


def test_deep_analysis_system_includes_static_poles_rules() -> None:
    system, mode, _ = select_system_prompt_and_mode(
        "[mode:deep_analysis] Задачка со звёздочкой",
        default_system_prompt="DEFAULT",
    )
    assert mode == "deep_analysis"
    assert "SOCRATIC POLES" in system
    assert "FACT_REPULSION" in system
    assert "FACT_ATTRACTION" in system
    assert SOCRATIC_POLES_STATIC_RULES.strip()[:40] in system


def test_novelty_block_orders_poles_then_digests() -> None:
    mem = SessionMemory(
        deep_analysis_prior_digests=["Anatomy: Isolation [R1]"],
        deep_analysis_used_source_ids=["S1"],
    )
    poles = format_socratic_poles_state_block(
        [
            {
                "fact_line": format_fact_repulsion(
                    node="n", concept_id="c1", claim="verified"
                )
            }
        ],
        [
            {
                "fact_line": format_fact_attraction(
                    node="n", concept_id="c2", focus_hint="gap"
                )
            }
        ],
    )
    block = format_deep_analysis_novelty_block(
        mem, rag_exhausted=False, poles_block=poles
    )
    assert block.index("[SOCRATIC_POLES_STATE]") < block.index(
        "[PRIOR_ASTERISK_QUESTION_THESIS_DIGESTS]"
    )
    assert block.index("[PRIOR_ASTERISK_QUESTION_THESIS_DIGESTS]") < block.index(
        "[RAG_COVERAGE_STATE]"
    )
    assert "FACT_REPULSION" in block
    assert "Anatomy: Isolation" in block


def test_novelty_exhausted_inserts_status_before_digests() -> None:
    mem = SessionMemory()
    poles = format_socratic_poles_state_block(
        [],
        [
            {
                "fact_line": format_fact_attraction(
                    node="n", concept_id="c2", focus_hint="latency cascade"
                ),
                "claim": "latency cascade",
            }
        ],
    )
    block = format_deep_analysis_novelty_block(
        mem,
        rag_exhausted=True,
        poles_block=poles,
        attraction_summary="latency cascade",
        registry_empty=True,
        atoms_empty=True,
    )
    assert block.index("[SOCRATIC_POLES_STATE]") < block.index("[RAG_STATUS: EXHAUSTED]")
    assert block.index("[RAG_STATUS: EXHAUSTED]") < block.index(
        "[PRIOR_ASTERISK_QUESTION_THESIS_DIGESTS]"
    )
    assert "latency cascade" in block
