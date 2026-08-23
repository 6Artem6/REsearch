"""Bloom-layer vocabulary (Phase 0) + Evaluator critique JSON contracts (Phase 1).

## Phase 0 — naming (do not conflate)

* ``node.layer`` ∈ {foundation, advanced, sota} — **curriculum difficulty** of the
  node (which WHY/HOW/MECH flags are required for VERIFIED). This is NOT the
  Expert Overlay.
* Core Bloom layers (count toward node 100%): WHY (L1–L2), HOW (L2–L3), MECH (L3).
* Expert Overlay (parallel awards; must NOT flip core ``how_passed`` /
  ``mechanic_passed``): ADVANCED asterisk-question (L4 analysis / edge cases),
  DEEP asterisk-question (L5–L6 design / trade-offs).
  ``pending_eval_kind=advanced_analysis`` → ADVANCED;
  ``deep_design`` / legacy ``deep_analysis`` → DEEP.

Evaluator returns structured critique only — no learner-facing prose.
Tutor (Phase 2+) renders the review from this JSON via the legacy adapter.

Note: lives under ``knowledge_engine/schemas/llm_contracts/`` (project registry).
A package path ``node_deep_dive/schemas/`` would shadow ``schemas.py``.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# Canonical layer tokens for ``target_layer`` (core + overlay).
CoreBloomLayer = Literal["WHY", "HOW", "MECH"]
OverlayBloomLayer = Literal["ADVANCED", "DEEP"]
EvalTargetLayer = Literal["WHY", "HOW", "MECH", "ADVANCED", "DEEP"]

OVERLAY_LAYERS: frozenset[str] = frozenset({"ADVANCED", "DEEP"})
CORE_LAYERS: frozenset[str] = frozenset({"WHY", "HOW", "MECH"})


class IdeaStatus(str, Enum):
    STRONG = "STRONG"  # Architecturally sound and well argued
    RISK = "RISK"  # Hidden risks / bottlenecks / P99-class issues
    WEAK = "WEAK"  # Irrelevant, wrong, or redundant


class EvaluatedIdea(BaseModel):
    idea_concept: str = Field(
        ...,
        min_length=1,
        max_length=400,
        description="Short name of the user-proposed idea / mechanism",
    )
    status: IdeaStatus = Field(
        ...,
        description="STRONG | RISK | WEAK",
    )
    technical_note: str = Field(
        ...,
        min_length=1,
        max_length=1200,
        description=(
            "Dry technical note in English for the Tutor "
            "(why this status was assigned). Not learner-facing copy."
        ),
    )


class EvaluatorCritiqueContract(BaseModel):
    """Pure JSON critique from the Evaluator — no user-facing Russian prose."""

    target_layer: EvalTargetLayer = Field(
        ...,
        description="Layer under test: WHY | HOW | MECH | ADVANCED | DEEP",
    )
    passes_threshold: bool = Field(
        ...,
        description=(
            "Whether the answer meets criteria for last_tutor_question "
            "(not encyclopedic depth beyond the asked layer)."
        ),
    )
    bloom_level_matched: bool = Field(
        ...,
        description=(
            "Depth matches expected Bloom band "
            "(L1–L3 core; L4–L6 overlay ADVANCED/DEEP)"
        ),
    )
    analyzed_ideas: list[EvaluatedIdea] = Field(
        default_factory=list,
        max_length=16,
        description="Per-idea breakdown of the user answer",
    )
    unaccounted_edge_cases: list[str] = Field(
        default_factory=list,
        max_length=12,
        description=(
            "Critical nuances implied by last_tutor_question that the user "
            "omitted. FORBIDDEN: extra details from unasked deeper or "
            "adjacent layers."
        ),
    )
    verdict_reason: str = Field(
        ...,
        min_length=1,
        max_length=1500,
        description="Short technical verdict summary for the Tutor (English)",
    )
    cleared_weakness_tags: list[str] = Field(
        default_factory=list,
        max_length=16,
        description=(
            "Prior weakness_tags this overlay answer closed "
            "(host ledger; empty if none)."
        ),
    )
