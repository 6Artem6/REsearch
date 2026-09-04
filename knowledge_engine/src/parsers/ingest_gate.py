"""Inbound ingest gate: article quality from pass-2 vector traits + Map-Reduce admit."""

from __future__ import annotations

from dataclasses import dataclass, field

from knowledge_engine.src.parsers.paper_structure_schema import (
    INFORMATION_DENSITY_WEIGHT,
    SEMANTIC_LEVEL_WEIGHT,
    TECHNICAL_CORRECTNESS_WEIGHT,
    InformationDensity,
    ParagraphAnalysis,
    ParagraphCredibility,
    SemanticLevel,
    TechnicalCorrectness,
)

INGEST_GATE_REJECT_REASON = "Failed parametric credibility score"

_MISSING_PASS2 = ParagraphCredibility(
    paragraph_id=0,
    semantic_level=SemanticLevel.CONCEPTUAL_MODEL,
    technical_correctness=TechnicalCorrectness.SIMPLIFIED,
    information_density=InformationDensity.NEUTRAL,
    reason="pass-2 row missing; conservative SIMPLIFIED conceptual",
)


def calculate_paragraph_score(p_cred: ParagraphCredibility) -> float:
    """Host P_i. CONTRADICTION zeros the paragraph regardless of other axes."""
    if p_cred.technical_correctness == TechnicalCorrectness.CONTRADICTION:
        return 0.0
    score = TECHNICAL_CORRECTNESS_WEIGHT[p_cred.technical_correctness] * (
        0.6 * SEMANTIC_LEVEL_WEIGHT[p_cred.semantic_level]
        + 0.4 * INFORMATION_DENSITY_WEIGHT[p_cred.information_density]
    )
    return round(score, 3)


def calculate_article_quality(paragraphs: list[ParagraphAnalysis]) -> float:
    """Weighted mean of P_i by topic_relevance. Unscored set → fail-open 1.0."""
    pairs: list[tuple[ParagraphAnalysis, ParagraphCredibility]] = []
    for row in paragraphs:
        cred = row.as_credibility()
        if cred is None:
            continue
        pairs.append((row, cred))
    if not pairs:
        return 1.0
    total_weight = sum(p_anal.topic_relevance for p_anal, _ in pairs)
    if total_weight == 0:
        return 0.0
    weighted_sum = sum(
        calculate_paragraph_score(p_cred) * p_anal.topic_relevance
        for p_anal, p_cred in pairs
    )
    return round(weighted_sum / total_weight, 3)


@dataclass
class IngestGateResult:
    accepted: bool
    quality: float
    body: str
    reject_reason: str | None = None
    paragraphs: list[ParagraphAnalysis] = field(default_factory=list)


def decide_ingest_gate(
    paragraphs: list[ParagraphAnalysis],
    *,
    quality_min: float | None,
) -> tuple[bool, float, str | None]:
    """Admit or reject the article. quality_min=None → never reject the whole document."""
    quality = calculate_article_quality(paragraphs)
    if quality_min is None:
        return True, quality, None
    if quality < float(quality_min):
        return False, quality, INGEST_GATE_REJECT_REASON
    return True, quality, None


def is_contradiction_paragraph(row: ParagraphAnalysis | None) -> bool:
    return (
        row is not None
        and row.technical_correctness == TechnicalCorrectness.CONTRADICTION
    )


def missing_pass2_placeholder(paragraph_id: int) -> ParagraphCredibility:
    return _MISSING_PASS2.model_copy(update={"paragraph_id": paragraph_id})
